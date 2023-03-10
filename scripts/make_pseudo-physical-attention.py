import argparse
import os
import math
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools as it
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

from model.data.loader import test_data_loader,data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--d_type', default='test', type=str) # test,val,train
parser.add_argument('--save', action='store_true')
parser.add_argument('--directory_save', action='store_true')
parser.add_argument('--pseudo_type', default='mask', type=str) # mask,fan_mask,leaky_mask,gauss
parser.add_argument('--var', default='', type=str) # Settings for using Gaussian distribution. There are settings from 1 to 4, and the settings get larger and larger.
# Setup for data loader
parser.add_argument('--data_dir_path', default='', type=str)
parser.add_argument('--frame', default=20, type=int)
parser.add_argument('--test_data', default=False)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--num_samples', default=100000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--size', default=1500, type=float)
parser.add_argument('--gpu_num', default='0', type=str)
parser.add_argument('--loader_num_workers', default=3, type=int)
parser.add_argument('--pin_memory', action='store_false')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--use_seg', action='store_true')
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=20, type=int)
parser.add_argument('--min_ped', default=1, type=int)
parser.add_argument('--max_ped', default=120, type=int)
parser.add_argument('--delim', default=' ')
parser.add_argument('--social_attention_type', default='simple', type=str)
parser.add_argument('--physical_attention_type', default='simple', type=str)
parser.add_argument('--so_prior_type', default='add', type=str)
parser.add_argument('--ph_prior', default='gd', type=str)
parser.add_argument('--ph_prior_type', default='add', type=str)
parser.add_argument('--physical_attention_dim', default=49, type=int)
parser.add_argument('--social_attention_dim', default=32, type=int)
parser.add_argument('--check_so_at', action='store_true')
parser.add_argument('--norm', default=1, type=int)
parser.add_argument('--large_image', action='store_true')
parser.add_argument('--remake_data', action='store_true')
parser.add_argument('--crop_img_size', default=512, type=int)
parser.add_argument('--center_crop', action='store_true')

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
        ax.text(total_aa[sample][args.obs_len-1, i, 0], fig_size-total_aa[sample][args.obs_len-1, i, 1], '{}'.format(i),  size=10, color=cm(i/(pd_num+1)))


    ani = animation.FuncAnimation(fig, update_dot, frames = args.frame, interval = 200, init_func=init())
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(im, extent=[*xlim, *ylim], aspect='auto', alpha=0.6)
    ax.set_aspect('equal')
    ani.save("{}/GT.gif".format(gt_path))

def gaussian(x, y, sigma, mu):
    return np.exp((-1/2*(((x-mu[0].item())**2)/(sigma[0].item()**2)+(((y-mu[1].item())**2)/(sigma[1].item()**2)))))/ (2 * np.pi * sigma[0].item() * sigma[1].item())

def norm(x, traj, prod):
    xsize = np.linalg.norm(x, ord=2)
    for k in range(traj.shape[0]):
        if xsize * np.linalg.norm(traj[k], ord=2) != 0:
            prod[k] /= xsize * np.linalg.norm(traj[k], ord=2)
    return prod

def trans(traj, rel, prod):
    # print(rel[1],rel[0])
    theta = np.arctan([rel[1],rel[0]])[0]
    x = np.copy(traj)
    x[:,0] = traj[:,0] - rel[0]
    x[:,1] = traj[:,1] - rel[1]
    # x[:,0] = x[:,0]*np.sin(theta) - x[:,0]*np.cos(theta)
    x[:,1] = x[:,1]*np.cos(theta) + x[:,1]*np.sin(theta)
    for k in range(traj.shape[0]):
        if x[k,1] >= 0 and prod[k] < 0:
            prod[k] *= -1
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
        r /= np.amax(r)
    r = np.tanh(r)
    
    return r

def cut(x, y, mu, r=24, size=56):
    v = np.zeros((size,size))
    for i in range(x.shape[0]):
        if (x[i]-(mu[0]*size))**2 + (y[i]- (mu[1]*size))**2 > r**2:
            v[y[i], x[i]] = -float('inf')
    return v

def cut_cut(W, xy, vector, size=56):
    v = np.zeros((size,size))
    k = True
    # print(xy*55)
    # print(v.shape)
    for i in range(size):
        for j in range(size):
            if W[i,j] == 0:
                a = np.linalg.norm(vector, ord=2)
                vxy = np.array([j/size,i/size])
                b = np.linalg.norm(vxy-xy, ord=2)

                if a == 0:
                    v[i, j] = 0
                    k = False
                elif ((j/size-xy[0])*vector[0]+(i/size-xy[1])*vector[1])/(a*b) < -0.4:
                    v[i, j] = -float('inf')
                else:
                    v[i, j] = 0
                    k = False
            else:
                v[i, j] = -float('inf') 
    if k == True:
        v = W
        print('a')
    return v

def leaky_cut(x, y, mu, r=12, m=0.5, rm=30, size=56):
    v = np.zeros((size,size))
    for i in range(x.shape[0]):
        ra = (x[i]-(mu[0]*size))**2 + (y[i]- (mu[1]*size))**2
        if ra <= r**2:
            v[y[i], x[i]] = 1
        else:
            z = -1*m*(1/(rm**2-r**2))*(ra-rm**2)
            v[y[i], x[i]] = max(0, z)
    return v

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

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, _, _) = batch
        obs_traj_gt = obs_traj.clone() 

        img_num = 0
        if sample < num_samples:
            sq_num = 0
            for (start, end) in seq_start_end:
                if args.directory_save:
                    fig_size = 512
                    if not os.path.exists("{}/sample{}".format(f_path,sample)):
                        os.mkdir("{}/sample{}".format(f_path,sample))
                    gt_path = "{}/sample{}".format(f_path, sample)
                
                pd_num = int(end - start)
                last_traj = obs_traj[-1,start:end,:].cpu().data.numpy() 
                # last_traj_rel = obs_traj_rel[-1,start:end,:].cpu().data.numpy()
                # print(last_traj.shape)
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

                    
                    mu = last_traj[i].copy()
                    bit = 1/56
                    x = np.arange(0, 1, bit)
                    c = np.array(np.meshgrid(x, x)).T.reshape(-1,2)

                    ## G-Attention
                    if args.pseudo_type == 'gauss':
                        if args.ver == '1':
                            si = np.array([0.3,0.3])
                        elif args.ver == '2':
                            si = np.array([0.5,0.5])
                        elif args.ver == '3':
                            si = np.array([0.7,0.7])
                        elif args.ver == '4':
                            si = np.array([1.0,1.0])
                        W = gaussian(c[:,1], c[:,0], si, mu)
                        www.append(W)
                        W_1 = W.reshape(56, 56)

                    if args.pseudo_type == 'mask' or 'fan_mask':
                        W = cut(c[:,1], c[:,0], mu)
                        if args.pseudo_type == 'fan_mask':
                            W = cut_cut(W, mu, last_traj_rel[i])
                        W_1 = W
                        W2 = np.squeeze(W.reshape(56*56))
                        www.append(W2)

                    if args.pseudo_type == 'leaky_mask':
                        W = leaky_cut(c[:,1], c[:,0], mu)
                        W_1 = W
                        W_2 = np.squeeze(W.reshape(56*56, 1))
                        www.append(W2)
                    
                    if args.directory_save:
                        fig_py, ax_py = plt.subplots()
                        ax_py.set_xlim(00, fig_size)
                        ax_py.set_ylim(00, fig_size)
                        xlim_py = ax_py.get_xlim()
                        ylim_py = ax_py.get_ylim()
                        ax_py.imshow(img[img_num], extent=[*xlim_py, *ylim_py], aspect='auto', alpha=0.40)
                        ax_py.text(mu[0]*fig_size, fig_size-(mu[1]*fig_size), '*', size=8, color=(1,0,0))
                        ax_py.imshow(W_1, cmap='jet', extent=[*xlim_py, *ylim_py], alpha=0.45)
                        ax_py.set_aspect('equal')
                        fig_py.savefig(gt_path + '/attention_{}.png'.format(i))
                        plt.close()

                sample += 1   
                sq_num += 1
        img_num += 1

    if args.save:
        www = np.array(www)
        spath = '{}/pseudo-physical_attention{}{}.csv'.format(path, args.pseudo_type, args.var)
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

    for path in data_path:
        if args.directory_save:
            if not os.path.exists("./ph_attention/{}".format(args.d_type)):
                os.mkdir("./ph_attention/{}".format(args.d_type))
            if not os.path.exists("./ph_attention/{}/{}".format(args.d_type, path.split('/')[-2])):
                os.mkdir("./ph_attention/{}/{}".format(args.a_type, path.split('/')[-2]))
            if not os.path.exists("./ph_attention/{}/{}/{}".format(args.a_type, path.split('/')[-2], path.split('/')[-1])):
                os.mkdir("./ph_attention/{}/{}/{}".format(args.a_type, path.split('/')[-2], path.split('/')[-1]))

            dset, loader = test_data_loader(args, [path]) 
            f_path = "./ph_attention/{}/{}/{}".format(args.d_type, path.split('/')[-2], path.split('/')[-1])

        evaluate(args, loader, args.num_samples, args.size, path, f_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)