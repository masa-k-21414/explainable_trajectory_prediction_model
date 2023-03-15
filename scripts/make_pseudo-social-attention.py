import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from PIL import Image
import time
import skimage
import skimage.transform
import gc
import logging
import os
import sys
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision

from model.data.loader import test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--d_type', default='test', type=str)# test,val,train
parser.add_argument('--save', action='store_true')
parser.add_argument('--directory_save', action='store_true')
# Setup for data loader
parser.add_argument('--data_dir_path', default='', type=str)
parser.add_argument('--test_txt', default='./test.txt', type=str)
parser.add_argument('--frame', default=20, type=int)
parser.add_argument('--test_data', default=False)
parser.add_argument('--num_samples', default=100000, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--size', default=512, type=float)
parser.add_argument('--gpu_num', default='0', type=str)
parser.add_argument('--loader_num_workers', default=3, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--pin_memory', action='store_false')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--check_so_at', action='store_true')
parser.add_argument('--skip', default=20, type=int)
parser.add_argument('--min_ped', default=1, type=int)
parser.add_argument('--max_ped', default=120, type=int)
parser.add_argument('--delim', default=' ')
parser.add_argument('--norm', default=1, type=int)
parser.add_argument('--large_image', action='store_true')
parser.add_argument('--use_seg', action='store_true')
parser.add_argument('--remake_data', action='store_true')
parser.add_argument('--crop_img_size', default=512, type=int)
parser.add_argument('--center_crop', action='store_true')
parser.add_argument('--social_attention_type', default='simple', type=str)
parser.add_argument('--physical_attention_type', default='simple', type=str)
parser.add_argument('--so_prior_type', default='add', type=str)
parser.add_argument('--ph_prior', default='gd', type=str)
parser.add_argument('--ph_prior_type', default='add', type=str)
parser.add_argument('--physical_attention_dim', default=49, type=int)
parser.add_argument('--social_attention_dim', default=32, type=int)

cm = plt.get_cmap("Spectral")

def update_dot(newd):
    global gt_num_t, fig_size, pd_num
    if gt_num_t <= args.frame:
        ln_data_gt = []
        for i in range(pd_num):
            ln_gt[i].set_data(total_aa[sample][:gt_num_t, i, 0], fig_size-total_aa[sample][:gt_num_t, i, 1])
            ln_data_gt.append(ln_gt[i])
        gt_num_t += 1
        return ln_data_gt[0]

def init():
    ax.set_xlim(00, args.size)
    ax.set_ylim(00, args.size)

def ploot(plt_size, pd_num, gt_path):
    
    global ln_gt, ln_pr, gt_num_t, pr_num_t, fig, ax, fig2, ax2, fig_size, CO
    CO = 0
    ln_gt = []
    gt_num_t = 0
    fig, ax = plt.subplots()
    fig_size = plt_size
    color = []
    im = img[sq_num]
    # print(im)
    w = im.size[0]/plt_size
    h = im.size[1]/plt_size

    for i in range(pd_num):
        ln_gt.append([])
        ln_gt[i], = ax.plot([], [], c=cm(i/(pd_num+1)), label=i, ls='-')
        # print(pd_num, i)
        # print(total_aa[sample][args.obs_len-1, i, 0])
        ax.text(total_aa[sample][args.obs_len-1, i, 0], fig_size-total_aa[sample][args.obs_len-1, i, 1], '{}'.format(i),  size=10, color=cm(i/(pd_num+1)))


    ani = animation.FuncAnimation(fig, update_dot, frames = args.frame, interval = 200, init_func=init())
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(im, extent=[*xlim, *ylim], aspect='auto', alpha=0.6)
    ax.set_aspect('equal')
    ani.save("{}/GT.gif".format(gt_path))
    plt.close(fig)

def gaussian(x,y,sigma, mu):
    return np.exp((-1/2*(((x-mu[0].item())**2)/(sigma[0].item()**2)+(((y-mu[1].item())**2)/(sigma[1].item()**2)))))/ (2 * np.pi * sigma[0].item() * sigma[1].item())

def norm(x, traj, prod):
    xsize = np.linalg.norm(x, ord=2)
    for k in range(traj.shape[0]):
        if xsize * np.linalg.norm(traj[k], ord=2) != 0:
            prod[k] /= xsize * np.linalg.norm(traj[k], ord=2)
    return prod

def trans(traj, rel, prod, mu):
    # print(rel[1],rel[0])
    theta = np.arctan([rel[1],rel[0]])[0]
    x = np.copy(traj)
    x[:,0] = traj[:,0] - mu[0]
    x[:,1] = traj[:,1] - mu[1]
    # x[:,0] = x[:,0]*np.sin(theta) - x[:,0]*np.cos(theta)
    x[:,1] = x[:,1]*np.cos(theta) + x[:,1]*np.sin(theta)
    for k in range(traj.shape[0]):
        if x[k,1] >= 0 and prod[k] < 0:
            prod[k] *= -1
        if x[k,1] < 0:
            prod[k] *= 0
    return prod

def softmax(x,T):
    f_x = np.exp(x/T) / np.sum(np.exp(x/T))
    return f_x


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def differ(l, s, i):
    r = np.ones(l.shape[0])
    for k in range(l.shape[0]):
        r[k] *= np.linalg.norm(s[k]-s[i], ord=2) - np.linalg.norm(l[k]-l[i], ord=2)
        # print(r[k])
    r[i] = np.amax(r)
    if np.amax(r) != 0:
        r /= np.amax(r)*2
    # r = sigmoid(r)
    r[i] = 1
    return r

def evaluate(args, loader, num_samples, plt_size, path, f_path):
    global total_aa, img, sq_num, sample, pd_num
    pd_num = 0
    sample = 0
    total_aa = []
    www = []
    for batch in loader:
        img = batch[-1]
        input_img = batch[-2]
        batch = [tensor.cuda() for tensor in batch[:-2]]
        t_w = img[0].size[0]
        t_h = img[0].size[1]
        size = np.array([t_h/t_w, t_w/t_h])
        sigma = np.array([0.25,0.25]) * size
        sigma2 = np.array([0.5,0.5]) * size
        # print(sigma)

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, _, _) = batch
        obs_traj_gt = obs_traj.clone() 

        if sample < num_samples:
            sq_num = 0
            for (start, end) in seq_start_end:
                if not os.path.exists("{}/sample{}".format(f_path,sample)):
                    os.mkdir("{}/sample{}".format(f_path,sample))
                gt_path = "{}/sample{}".format(f_path, sample)
                pd_num = int(end - start)
                # print(pd_num)
                
                last_traj = obs_traj[-1,start:end,:].cpu().data.numpy() 
                last_traj_rel = obs_traj_rel[-1,start:end,:].cpu().data.numpy() 
                start_traj = obs_traj[0,start:end,:].cpu().data.numpy() 

                if args.directory_save:
                    gt_tr = pred_traj_gt[:,start:end,:].cpu().data.numpy() 
                    gt_tr *= plt_size/args.norm
                    input_tr = obs_traj_gt[:,start:end,:].cpu().data.numpy()
                    input_tr *= plt_size/args.norm
                    gt = np.concatenate((input_tr,gt_tr),axis=0) 
                    total_aa.append(gt)
                    ploot(plt_size, pd_num, gt_path)
                
                for i in range(pd_num):
                    mu = last_traj[i]
                    Z = gaussian(last_traj[:,0], last_traj[:,1], sigma, mu)
                    p = gaussian(last_traj[:,0], last_traj[:,1], sigma2, mu)

                    if pd_num != 1:
                        Y = np.dot(last_traj_rel[i, np.newaxis],last_traj_rel.T).T
                        Y = norm(last_traj_rel[i], last_traj_rel, Y)
                        Y = np.squeeze(trans(last_traj, last_traj_rel[i], Y, mu))
                        Y *= p
                        X = differ(last_traj, start_traj, i)
                        X *= p
                        W = softmax(Z+X*0.2+Y*0.1, 0.5)
                    else:
                        W = softmax(Z)


                    if args.directory_save:
                        fig, ax = plt.subplots()
                        ax.set_xlim([-1, pd_num])
                        ax.set_title('Human: {}'.format(i))
                        ax.bar(np.arange(pd_num), W[:pd_num])
                        fig.savefig(gt_path + '/attention_{}.png'.format(i))
                        plt.close(fig)

                    if args.save:
                        W = np.array(W)
                        W = np.pad(W, [0, 120-pd_num])
                        www.append(W)

                sample += 1   
                sq_num += 1
    if args.save:
        www = np.array(www)
        spath = '{}/pseudo-social_attention.csv'.format(path)
        np.savetxt(spath, www)

def main(args):
    f_path = 0
    if args.d_type == 'train':
        args.skip = 4
        args.data_dir_path = './train.txt'
    elif args.d_type == 'val':
        args.skip = 20
        args.data_dir_path = './val.txt'
    elif args.d_type == 'test':
        args.skip = 1
        args.data_dir_path = './test.txt'

    print('----'+ args.d_type + '----skip:' + str(args.skip) + '----')

    with open(args.data_dir_path) as f:
        data_path = [s.strip() for s in f.readlines()]

    if not os.path.exists("./so_attention"):
            os.mkdir("./so_attention")

    for path in data_path:
        if args.directory_save:
            if not os.path.exists("./so_attention/{}".format(args.d_type)):
                os.mkdir("./so_attention/{}".format(args.d_type))
            if not os.path.exists("./so_attention/{}/{}".format(args.d_type, path.split('/')[-2])):
                os.mkdir("./so_attention/{}/{}".format(args.d_type, path.split('/')[-2]))
            if not os.path.exists("./so_attention/{}/{}/{}".format(args.d_type, path.split('/')[-2], path.split('/')[-1])):
                os.mkdir("./so_attention/{}/{}/{}".format(args.d_type, path.split('/')[-2], path.split('/')[-1]))
            f_path = "./so_attention/{}/{}/{}".format(args.d_type, path.split('/')[-2], path.split('/')[-1])

        dset, loader = test_data_loader(args, [path]) 
        

        evaluate(args, loader, args.num_samples, args.size, path, f_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)