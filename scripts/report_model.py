import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from scipy import interpolate
from scipy.interpolate import griddata

from model.data.loader import test_data_loader
from model.models import TrajectoryGenerator, TrajectoryLSTM
from model.losses import displacement_error, final_displacement_error
from model.utils import relative_to_abs
from model.data.loader import test_data_loader
from model.models import TrajectoryGenerator, TrajectoryLSTM
from model.losses import displacement_error, final_displacement_error
from model.utils import relative_to_abs


parser = argparse.ArgumentParser()
parser.add_argument('--model_num', default='444', type=str)
parser.add_argument('--data_dir_path', default='./test.txt', type=str)
parser.add_argument('--frame', default=20, type=int)
parser.add_argument('--test_data', default=False)
parser.add_argument('--num_samples', default=1000, type=int)
parser.add_argument('--size', default=1500, type=float)
parser.add_argument('--obs_len', default=8, type=int)

cm = plt.get_cmap("Spectral")

def update_dot(newd):
    global gt_num_t, fig_size
    if gt_num_t <= args.frame:
        ln_data_gt = []
        for i in range(pd_num):
            ln_gt[i].set_data(total_aa[sample][:gt_num_t, i, 0], fig_size-total_aa[sample][:gt_num_t, i, 1])
            ln_data_gt.append(ln_gt[i])
        gt_num_t += 1
        return ln_data_gt[0]

def update_dot2(newd):
    global pr_num_t, fig_size
    if pr_num_t <= args.frame:
        ln_data_pr = []
        for i in range(pd_num):
            ln_pr[i].set_data(total_bb[sample][:pr_num_t, i, 0], fig_size-total_bb[sample][:pr_num_t, i, 1])
            ln_data_pr.append(ln_pr[i])
        pr_num_t += 1
        return ln_data_pr[0]


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def ploot(model_num, plt_size, img_size, center_crop, At_p, At_s, video_num, pd_num, social_attention_type, ge_type, gcn_Attention=0, start=0, end=0):
    img_path = "./video/checkpoint_with_model_{}/{}/sq-{}".format(model_num, video_num, sample)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    img_path = "./video/checkpoint_with_model_{}/{}/sq-{}/result".format(model_num, video_num, sample)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    
    global ln_gt, ln_pr, gt_num_t, pr_num_t, fig, ax, fig2, ax2, fig_size, CO
    CO = 0
    ln_gt = []
    ln_pr = []
    gt_num_t = 0
    pr_num_t = 0
    
    fig_size = plt_size
    color = []
    im = img[sq_num]
    # print(im)
    w = im.size[0]/plt_size
    h = im.size[1]/plt_size
    cm = plt.get_cmap("Spectral")

    DX = []
    DY = []

    for n in range(pd_num):
        fig, ax = plt.subplots()
        x = np.arange(0, 1500, 1) #x軸の描画範囲の生成。0から10まで0.05刻み。
        y = np.arange(0, 1500, 1) #y軸の描画範囲の生成。0から10まで0.05刻み。

        X, Y = np.meshgrid(x, y)
        Z = []
        D = []
        for k in range(20):
            for l in range (512):
                xx, yy = int(pred[l,k,n,0]), int(pred[l,k,n,1])
                
                if k >= 8:
                    D.append(np.array([xx,yy]))
                    nnn = int(k / 4)-2
                    Z.append(int(k / 4)-2)
                    plt.plot(pred[l,k,n,0], pred[l,k,n,1], marker='.', markersize=1, color=cm(nnn/5)) 

        D = np.array(D)
        Z = np.array(Z)
        
        Z = griddata(points=D, values=Z, xi=(X, Y), method='cubic')

        ax.contourf(X,Y,Z, alpha=0, cmap='jet') 
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.imshow(im, extent=[*xlim, *ylim], alpha=0.6)
        ax.plot(total_aa[sample][:,n,0], 1500-total_aa[sample][:,n,1], color='r')
        ax.plot(total_bb[sample][:8,n,0],1500-total_bb[sample][:8,n,1], color='b')
        # ax.plot(total_bb[sample][8:,n,0], 1500-total_bb[sample][8:,n,1], color='b',  ls='--')
        ax.set_aspect('equal')
        
        # ax = plt.colorbar(label="contour level")  
        fig.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/result/report_{}.png'.format(model_num, video_num, sample,n))
        plt.close(fig)
        plt.close()

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])

    if args.LSTM:
        generator = TrajectoryLSTM(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            g_type=args.ge_type,
            recurrent_graph=args.recurrent_graph,
            visualize=True)
    else:
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.g_mlp_dim,
            n_max=args.n_max,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            att_ph_dim=args.physical_attention_dim,
            att_so_dim=args.social_attention_dim,
            center_crop=args.center_crop,
            crop_img_size=args.crop_img_size,
            norm=args.norm,
            kernel_size=args.kernel_size,
            multiplier=args.multiplier,
            physical_pos_embed=args.physical_pos_embed,
            physical_img_embed=args.physical_img_embed,
            social_pos_embed=args.social_pos_embed,
            attention_type=args.attention_type,
            social_attention_type=args.social_attention_type,
            physical_attention_type=args.physical_attention_type,
            so_prior_type=args.so_prior_type,
            ph_prior_type=args.ph_prior_type,
            recurrent_attention=args.recurrent_attention,
            recurrent_physical_attention=args.recurrent_physical_attention,
            input_recurrent_attention=args.input_recurrent_attention,
            cell_pad=args.cell_pad,
            phy_tempreture=args.physical_tempreture, 
            so_tempreture=args.social_tempreture,
            input_to_decoder=args.input_to_decoder,
            ge_type=args.ge_type,
            gd_type=args.gd_type,
            recurrent_graph=args.recurrent_graph,
            setting_image=args.large_image,
            compress_attention=args.compress_attention,
            concat_state=args.concat_state,
            use_vgg=args.not_use_vgg, 
            use_seg=args.use_seg,
            vgg_train=args.vgg_train, 
            usefulness=args.usefulness,
            artificial_social_attention=args.artificial_social_attention,
            visualize=True)
    
    
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.eval()
    # logger.info('Here is the generator:')
    # logger.info(generator)
    return generator, args.physical_attention_dim, args.social_attention_dim, only_LSTM, social_attention_type, args.ge_type, large_image, args.recurrent_attention, args.ph_prior

def evaluate(args, loader, generator, model_num, num_samples, plt_size, img_size, center_crop, At_p, At_s, LSTM, video_num, social_attention_type, ge_type, recurrent_attention):
    with torch.no_grad():
        global total_aa, total_bb, sq_num, pd_num, img, ph_at, so_at, so_pd_num, sample, pred
        ade_outer, fde_outer = [], []
        total_traj = 0
        total_aa, total_bb = [], []
        so_pd_num = []
        gt, tr = [], []
        sample = 0
        for batch in loader:
            img = batch[-1]
            # print('img')
            input_img = batch[-2]
            t_w = img[0].size[0]
            t_h = img[0].size[1]
            if args.norm == 0:
                R = np.array([plt_size/t_w, plt_size/t_h])
                L = torch.tensor([1,1]).cuda()
            else:
                R = np.array([plt_size/args.norm, plt_size/args.norm])
                L = torch.tensor([t_w/args.norm, t_h/args.norm]).cuda()
            # print(t_w, t_h)
            batch = [tensor.cuda() for tensor in batch[:-2]]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, social_prior_attention, physical_prior_attention) = batch

            tr_gt = pred_traj_gt.clone()
            obs_traj_gt = obs_traj.clone() 
            if social_attention_type == 'sophie':
                obs_traj_gt=obs_traj_gt[:,:,0,:]
            # pred_traj_gt *= plt_size/args.norm
            GCNAttention = 0
            ade, fde = [], []
            pred = []
            total_traj += pred_traj_gt.size(1)

            # for _ in range(1):
            sq_num = 0
            min_ade = 999999
            if sample < num_samples:
                for sampling_num in range(512):
                    if not LSTM:
                        pred_traj_fake, pred_traj_fake_rel, social_attention, physical_attention, agent_num = generator(input_img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
                    if At_s != 0:
                        if social_attention_type == 'simple':
                            social_attention = social_attention.cpu().data.numpy()
                            agent_num = agent_num.cpu().data.numpy()
                        elif social_attention_type == 'self_attention' or social_attention_type == 'prior'  or social_attention_type == 'sophie':
                            social_attention = social_attention.cpu().data.numpy() 
                        elif social_attention_type == 'gcn_attention':
                            GCNAttention = social_attention
                            ge_type = 'embed_gcn' 

                    else:
                        if ge_type == 'gcn' or ge_type == 'embed_gcn' or ge_type == 'embed_gcn_traj':
                            pred_traj_fake, pred_traj_fake_rel, GCNAttention = generator(input_img, obs_traj, obs_traj_rel, seq_start_end)
                            # print(GCNAttention)
                        else:
                            pred_traj_fake, pred_traj_fake_rel = generator(input_img, obs_traj, obs_traj_rel, seq_start_end)

                    # print(pred_traj_fake[:,0])

                    fake = pred_traj_fake.clone()
                    (start, end) = seq_start_end[0]
                    pd_num = end - start
                    # print(pd_num)
                    ade_s = displacement_error(fake[:,start:end,:]*L, tr_gt[:,start:end,:]*L, mode='sum') / (pd_num * args.pred_len)
                    fde_s = final_displacement_error(fake[-1,start:end,:]*L, tr_gt[-1,start:end,:]*L, mode='sum') / (pd_num)
                    gt_tr = pred_traj_gt[:,start:end,:].cpu().data.numpy() * R
                    input_tr = obs_traj_gt[:,start:end,:].cpu().data.numpy() * R
                    predict_tr = pred_traj_fake[:,start:end,:].cpu().data.numpy() * R
                    gt = np.concatenate((input_tr,gt_tr),axis=0)
                    pr = np.concatenate((input_tr,predict_tr),axis=0)
                    a = np.concatenate((input_tr,predict_tr),axis=0)
                    a[:,:,1] = 1500- a[:,:,1]
                    pred.append(a)
                    
                    if min_ade >= ade_s:
                        min_ade = ade_s
                        if sampling_num == 0:        
                            total_aa.append(gt)
                            total_bb.append(pr)
                        else:
                            total_aa[sample] = gt
                            total_bb[sample] = pr
                    sampling_num += 1
                pred = np.array(pred)
                ploot(model_num, plt_size, img_size, center_crop, 0, 0, video_num, pd_num.cpu(), social_attention_type, ge_type, GCNAttention, start, end)
                sq_num += 1
                print('[INFO: visualize_model.py]: test_{} is generated!'.format(sample))
                sample += 1
                if sample >= num_samples:
                    break

def main(args):
    with open(args.data_dir_path) as f:
        test_data_path = [s.strip() for s in f.readlines()]    
            
    paths = ['../checkpoint/checkpoint_{}.pt'.format(i) for i in args.model_num.split(',')]
    
    data_number = 0
    for test_path in test_data_path:
        global video_num, recurrent_attention 
        video_num = test_path.split('/')[-2] + '_' +test_path.split('/')[-1]
        query = 0
        p_args = 0
        for path in paths:
            global model_num
            model_num = args.model_num.split(',')[query]
            print('[INFO: visualize_model.py]: data: {}  model: {}.'.format(video_num, model_num))
            checkpoint = torch.load(path)
            generator, At_p, At_s, LSTM, social_attention_type, ge_type, large_image, recurrent_attention, ph = get_generator(checkpoint)
                
            _args = AttrDict(checkpoint['args'])
            _args.skip = 8
            _args.batch_size = 1
            if p_args == 0 or p_args !=  [_args.center_crop, _args.norm, _args.min_ped, _args.max_ped, large_image, ph]:
                print('[INFO: visualize_model.py]: Data loader. large_image:{}.'.format(large_image))
                _, loader = test_data_loader(_args, [test_path])  

            if not os.path.exists("./video/checkpoint_with_model_{}".format(model_num)):
                os.mkdir("./video/checkpoint_with_model_{}".format(model_num))
            if not os.path.exists("./video/checkpoint_with_model_{}/{}".format(model_num, video_num)):
                os.mkdir("./video/checkpoint_with_model_{}/{}".format(model_num, video_num))

            # evaluate(_args, loader, generator, args.num_samples, args.size, 256, False)
            evaluate(_args, loader, generator, model_num, args.num_samples, args.size,  _args.crop_img_size, _args.center_crop, At_p, At_s, LSTM, video_num, social_attention_type, ge_type, recurrent_attention)
            query += 1
            
            p = './video/checkpoint_with_model_{}/args.txt'.format(model_num)
            fa = open(p, "w")
            for k, v in checkpoint['args'].items():
                fa.write('{}  :  {}. \n'.format(k, v))
            fa.close()
            p_args = [_args.center_crop, _args.norm, _args.min_ped, _args.max_ped, large_image, ph]
        data_number += 1

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)