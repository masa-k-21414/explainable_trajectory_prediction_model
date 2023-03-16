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

from attrdict import AttrDict

from model.data.loader import test_data_loader
from model.models import TrajectoryGenerator, TrajectoryLSTM
from model.losses import displacement_error, final_displacement_error
from model.utils import relative_to_abs
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.INFO)
# print(animation._log.getEffectiveLevel() > logging.DEBUG) 

parser = argparse.ArgumentParser()
parser.add_argument('--model_num', default='444', type=str)
parser.add_argument('--data_dir_path', default='./test.txt', type=str)
parser.add_argument('--frame', default=20, type=int)
parser.add_argument('--data', default=100, type=int)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--size', default=1512, type=float)
parser.add_argument('--obs_len', default=8, type=int)

cm = plt.get_cmap("Spectral")

def init():
    ax.set_xlim(00, args.size)
    ax.set_ylim(00, args.size)

def init2():
    ax2.set_xlim(00, args.size)
    ax2.set_ylim(00, args.size)

def update_dot(newd):
    global gt_num_t, fig_size
    if gt_num_t <= args.frame:
        ln_data_gt = []
        for i in range(pd_num):
            ln_gt[i].set_data(total_aa[sample][:gt_num_t, i, 0], fig_size-total_aa[sample][:gt_num_t, i, 1])
            ln_data_gt.append(ln_gt[i])
        gt_num_t += 1
        return ln_data_gt[0]
    # else:
    #     ln_data_gt = []
    #     for i in range(pd_num):
    #         ln_gt2[i].set_data(total_aa[sample][:gt_num_t, i, 0], fig_size-total_aa[sample][:gt_num_t, i, 1])
    #         ln_data_gt.append(ln_gt2[i])
    #     gt_num_t += 1
    #     return ln_data_gt[0]

def update_dot2(newd):
    global pr_num_t, fig_size
    if pr_num_t <= args.obs_len:
        ln_data_pr = []
        for i in range(pd_num):
            ln_pr[i].set_data(total_bb[sample][:pr_num_t, i, 0], fig_size-total_bb[sample][:pr_num_t, i, 1])
            ln_data_pr.append(ln_pr[i])
        pr_num_t += 1
        return ln_data_pr[0]
    else:
        ln_data_pr = []
        for i in range(pd_num):
            ln_pr2[i].set_data(total_bb[sample][:pr_num_t, i, 0], fig_size-total_bb[sample][:pr_num_t, i, 1])
            ln_data_pr.append(ln_pr2[i])
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
    if At_p != 0 and not os.path.exists(img_path+'/physical_attention'):
        os.mkdir(img_path+'/physical_attention')
    if At_s != 0 and not os.path.exists(img_path+'/social_attention'):
        os.mkdir(img_path+'/social_attention')
    
    global ln_gt, ln_pr, ln_gt2, ln_pr2, gt_num_t, pr_num_t, fig, ax, fig2, ax2, fig_size, CO
    CO = 0
    ln_gt = []
    ln_pr = []
    ln_gt2 = []
    ln_pr2 = []
    gt_num_t = 0
    pr_num_t = 0
    fig, ax = plt.subplots(dpi=300)
    fig2, ax2 = plt.subplots(dpi=300)
    fig_size = plt_size
    color = []
    im = img[sq_num]
    # print(im)
    w = im.size[0]/plt_size
    h = im.size[1]/plt_size

    if gcn_Attention!=0:
        At_s = 0


    for i in range(pd_num):
        ln_gt.append([])
        # ln_gt2.append([])
        ln_gt[i], = ax.plot([], [], c=cm(i/(pd_num.item()+1)), label=i, ls='-')
        # ln_gt2[i], = ax.plot([], [], c=cm(i/(pd_num.item()+1)), label=i, ls='--')
        ln_pr.append([])
        ln_pr2.append([])
        ln_pr[i], = ax2.plot([], [], c=cm(i/(pd_num.item()+1)), label=i, ls='-')
        ln_pr2[i], = ax2.plot([], [], c=cm(i/(pd_num.item()+1)), label=i, ls='--')
        # print(pd_num, i)
        a = np.random.randint(0, 8)
        ax.text(total_aa[sample][a, i, 0]+10, fig_size-total_aa[sample][a, i, 1]-10, '{}'.format(i), size=12, color='k')
        ax2.text(total_bb[sample][a, i, 0]+10, fig_size-total_bb[sample][a, i, 1]-10, '{}'.format(i), size=12, color=cm(i/pd_num.item()))
        color.append(cm(i/(pd_num.item()+1)))

    for j in range(pd_num):
        if At_p != 0:
            # print(At_p )
            if not recurrent_attention:
                A_py2 = ph_at[sample][j].cpu().data.numpy()
                size = int(A_py2.shape[0] ** (1/2))
                # print(size)
                A_py2 = A_py2.reshape(size,size) ###サイズ
                fig_py, ax_py = plt.subplots(dpi=300)
                fig_py2, ax_py2 = plt.subplots(dpi=300)
                
                if center_crop:
                    x_1 = total_aa[sample][args.obs_len-1, j, 0]*w-(img_size//2)
                    y_1 = total_aa[sample][args.obs_len-1, j, 1]*h-(img_size//2)
                    x_2 = total_aa[sample][args.obs_len-1, j, 0]*w+(img_size//2)
                    y_2 = total_aa[sample][args.obs_len-1, j, 1]*h+(img_size//2)

                    crop_img = im.crop((x_1, y_1, x_2, y_2))
                    # crop_img.save('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-img-{}.png'.format(model_num, video_num, sample, j), quality=95)
                    # image = plt.imread('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-img-{}.png'.format(model_num, video_num, sample, j))
                    ax_py.imshow(crop_img)

                    xlim_py = ax_py.get_xlim()
                    ylim_py = ax_py.get_ylim()

                else:
                    ax_py.set_xlim(00, fig_size)
                    ax_py.set_ylim(00, fig_size)
                    xlim_py = ax_py.get_xlim()
                    ylim_py = ax_py.get_ylim()

                    ax_py2.set_xlim(00, fig_size)
                    ax_py2.set_ylim(00, fig_size)

                    ax_py.imshow(im, extent=[*xlim_py, *ylim_py], aspect='auto', alpha=0.40)
                    ax_py2.imshow(im, extent=[*xlim_py, *ylim_py], aspect='auto', alpha=0.40)
                    
                    ax_py.text(total_aa[sample][args.obs_len-1, j, 0], fig_size-total_aa[sample][args.obs_len-1, j, 1], '*', size=4, color=(1,0,0))
                    ax_py2.text(total_aa[sample][args.obs_len-1, j, 0], fig_size-total_aa[sample][args.obs_len-1, j, 1], '*', size=4, color=(1,0,0))

                    A_py = skimage.transform.pyramid_expand(A_py2, upscale=fig_size/size, sigma=20)

                ax_py.imshow(A_py, cmap='jet', extent=[*xlim_py, *ylim_py], alpha=0.45)
                ax_py2.imshow(A_py2, cmap='jet', extent=[*xlim_py, *ylim_py], alpha=0.45)

                ax_py.set_title('Human color: {}'.format(j),  color=color[j])
                ax_py.set_aspect('equal')
                ax_py2.set_title('Human color: {}'.format(j),  color=color[j])
                ax_py2.set_aspect('equal')
                # ax_py.get_xaxis().set_visible(False)
                # ax_py.get_yaxis().set_visible(False)
                # ax_py.axis(False)
                fig_py.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-{}.png'.format(model_num, video_num, sample, j))
                plt.close(fig_py)
                fig_py2.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-{}_box.png'.format(model_num, video_num, sample, j))
                plt.close(fig_py2)

            else:
                if not os.path.exists('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-{}'.format(model_num, video_num, sample, j)):
                    os.mkdir('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-{}'.format(model_num, video_num, sample, j))
                for tn in range(args.frame-args.obs_len):
                    # print(len(ph_at))
                    # print(ph_at[sample].shape)
                    A_py = ph_at[sample][tn,j].cpu().data.numpy()
                    # print(A_py.shape)
                    size = int(A_py.shape[0] ** (1/2))
                    # print(size)
                    A_py = A_py.reshape(size,size) ###サイズ
                    fig_py, ax_py = plt.subplots()
                    
                    if center_crop:
                        x_1 = total_aa[sample][args.obs_len-1, j, 0]*w-(img_size//2)
                        y_1 = total_aa[sample][args.obs_len-1, j, 1]*h-(img_size//2)
                        x_2 = total_aa[sample][args.obs_len-1, j, 0]*w+(img_size//2)
                        y_2 = total_aa[sample][args.obs_len-1, j, 1]*h+(img_size//2)

                        crop_img = im.crop((x_1, y_1, x_2, y_2))
                        # crop_img.save('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-img-{}.png'.format(model_num, video_num, sample, j), quality=95)
                        # image = plt.imread('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-img-{}.png'.format(model_num, video_num, sample, j))
                        ax_py.imshow(crop_img)

                        xlim_py = ax_py.get_xlim()
                        ylim_py = ax_py.get_ylim()

                    else:
                        ax_py.set_xlim(00, fig_size)
                        ax_py.set_ylim(00, fig_size)
                        xlim_py = ax_py.get_xlim()
                        ylim_py = ax_py.get_ylim()
                        ax_py.imshow(im, extent=[*xlim_py, *ylim_py], aspect='auto', alpha=0.40)
                        A_py = skimage.transform.pyramid_expand(A_py, upscale=fig_size/size, sigma=20)
                        ax_py.text(total_aa[sample][args.obs_len-1, j, 0], total_aa[sample][args.obs_len-1, j, 1], '*', size=5, color=(1,0,0))

                    ax_py.imshow(A_py, cmap='jet', extent=[*xlim_py, *ylim_py], alpha=0.45)
                    ax_py.set_title('Human color: {}'.format(j),  color=color[j])
                    ax_py.set_aspect('equal')
                    # ax_py.get_xaxis().set_visible(False)
                    # ax_py.get_yaxis().set_visible(False)
                    # ax_py.axis(False)
                    fig_py.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/physical_attention/pd-{}/frame_{}.png'.format(model_num, video_num, sample, j,tn+args.obs_len))
                    plt.close(fig_py)

        if At_s != 0:
            if not recurrent_attention:
                if social_attention_type == 'simple' or social_attention_type == 'sophie':
                    A_so = so_at[sample][j].squeeze()
                    fig_so, ax_so = plt.subplots()
                    ax_so.set_xlim([-1, pd_num.cpu().data.numpy().item()])
                    colorlist = []
                    for ds in so_pd_num[sample][j]:
                        if ds.item() - 1 < 0:
                            continue
                        colorlist.append(color[ds.item()-1])
                    n = len(colorlist)
                    ax_so.set_title('Human color: {}'.format(j),  color=color[j])
                    ax_so.bar(so_pd_num[sample][j][:n]-1, A_so[:n], color=colorlist)
                    fig_so.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}.png'.format(model_num, video_num, sample, j))
                    plt.close(fig_so)

                else:
                    A_so = so_at[sample][j].squeeze()
                    fig_so, ax_so = plt.subplots()
                    
                    ax_so.set_xlim([-1, pd_num.cpu().data.numpy().item()])
                    ax_so.set_title('Human color: {}'.format(j),  color=color[j])
                    ax_so.bar(np.arange(pd_num) , A_so[:pd_num], color=color)
                    ax_so.set_xticks(np.arange(0, pd_num.cpu().data.numpy().item(),5))
                    fig_so.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}.png'.format(model_num, video_num, sample, j))
                    plt.close(fig_so)

            else:
                if not os.path.exists('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}'.format(model_num, video_num, sample, j)):
                        os.mkdir('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}'.format(model_num, video_num, sample, j))
                if social_attention_type == 'simple' or social_attention_type == 'sophie':
                    for tn in range(args.frame-args.obs_len):
                        
                        A_so = so_at[sample][tn,j].squeeze()
                        fig_so, ax_so = plt.subplots()
                        ax_so.set_xlim([-1, pd_num.cpu().data.numpy().item()])
                        colorlist = []
                        for ds in so_pd_num[sample][j]:
                            if ds.item() - 1 < 0:
                                continue
                            colorlist.append(color[ds.item()-1])
                        n = len(colorlist)
                        ax_so.set_title('Human color: {}'.format(j),  color=color[j])
                        ax_so.bar(so_pd_num[sample][j][:n]-1, A_so[:n], color=colorlist)
                        fig_so.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}/frame_{}.png'.format(model_num, video_num, sample, j, tn+args.obs_len))
                        plt.close(fig_so)

                elif social_attention_type == 'self_attention':
                    for tn in range(args.frame-args.obs_len):
                        # print(so_at[sample].shape)
                        # print(so_at[sample][0])
                        # print(len(so_at))
                        A_so = so_at[sample][tn,j].squeeze()
                        # print(A_so.shapes)
                        fig_so, ax_so = plt.subplots()
                        ax_so.set_xlim([-1, pd_num.cpu().data.numpy().item()])
                        ax_so.set_title('Human color: {}'.format(j),  color=color[j])
                        ax_so.bar(np.arange(pd_num) , A_so[:pd_num], color=color)
                        fig_so.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}/frame_{}.png'.format(model_num, video_num, sample, j, tn+args.obs_len))
                        plt.close(fig_so)

            
        else:
            if ge_type == 'gcn' or ge_type == 'embed_gcn' or ge_type == 'embed_gcn_traj':
                if not os.path.exists(img_path+'/social_attention'):
                    os.mkdir(img_path+'/social_attention')
                to_list=[]
                value=[]
                colorlist = []
                fig_so, ax_so = plt.subplots()
                ax_so.set_xlim([-1, pd_num.cpu().data.numpy().item()])
                # print(gcn_Attention[0][0])
                for g in range(len(gcn_Attention[0][0])):
                    # print(start.cpu().data.numpy().item())
                    if gcn_Attention[0][0][g] >= start.cpu().data.numpy().item() and gcn_Attention[0][0][g] < end.cpu().data.numpy().item():
                        if j == gcn_Attention[0][0][g].cpu().data.numpy().item():
                            # print(gcn_Attention[0][0][g])
                            to_list.append(gcn_Attention[0][1][g].cpu().data.numpy().item()-start.cpu().data.numpy().item())
                            value.append(gcn_Attention[1][g].cpu().data.numpy().item())
                            # print(to_list)
                            # print(value)
                            colorlist.append(color[gcn_Attention[0][1][g].cpu().data.numpy().item()])
                        if gcn_Attention[0][0][g] >= end.cpu().data.numpy().item():
                            break
                ax_so.set_title('Human color: {}'.format(j),  color=color[j])
                ax_so.bar(to_list, value, color=colorlist)
                fig_so.savefig('./video/checkpoint_with_model_{}/{}/sq-{}/social_attention/pd-{}.png'.format(model_num, video_num, sample, j))
                plt.close(fig_so)
    
    ani = animation.FuncAnimation(fig, update_dot, frames = args.frame, interval = 200, init_func=init())
    ani2 = animation.FuncAnimation(fig2, update_dot2, frames = args.frame, interval = 200, init_func=init2())
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()
    ax.imshow(im, extent=[*xlim, *ylim], aspect='auto', alpha=0.6)
    ax2.imshow(im, extent=[*xlim2, *ylim2], aspect='auto', alpha=0.6)
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    ani.save("./video/checkpoint_with_model_{}/{}/sq-{}/test_gt.gif".format(model_num, video_num, sample))
    ani2.save("./video/checkpoint_with_model_{}/{}/sq-{}/test_pr.gif".format(model_num, video_num, sample))

    plt.close(fig)
    plt.close(fig2)
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
            kernel_size=3,
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
            use_seg=False,
            vgg_train=args.vgg_train, 
            usefulness=args.usefulness,
            artificial_social_attention=args.artificial_social_attention,
            visualize=True)
    
    
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.eval()
    # logger.info('Here is the generator:')
    # logger.info(generator)
    return generator, args.LSTM, args.social_attention_type, args.ge_type, args.large_image, args.recurrent_attention, args.ph_prior

def evaluate(args, loader, generator, model_num, num_samples, plt_size, img_size, center_crop, At_p, At_s, LSTM, video_num, social_attention_type, ge_type):
    with torch.no_grad():
        global total_aa, total_bb, sq_num, pd_num, img, ph_at, so_at, so_pd_num, sample, pred
        # print(At_p)
        ade_outer, fde_outer = [], []
        total_traj = 0
        total_aa, total_bb = [], []
        so_pd_num = []
        gt, tr = [], []
        ph_at, so_at = [], []
        sample = 0
        for batch in loader:
            img = batch[-1]
            # print('img')
            input_img = batch[-2]
            t_w = img[0].size[0]
            t_h = img[0].size[1]
            if args.norm == 0:
                R_size = 1
            else:
                R_size = torch.tensor([t_w/args.norm, t_h/args.norm]).cuda()
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
            ade_s, fde_s = 0, 0
            pred = []
            total_traj += pred_traj_gt.size(1)

            # for _ in range(1):
            sq_num = 0
            min_ade = 999999
            if sample < num_samples:
                for sampling_num in range(250):
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
                            social_attention = social_attention.cpu().data.numpy() 

                    else:
                        if ge_type == 'gcn' or ge_type == 'embed_gcn' or ge_type == 'embed_gcn_traj':
                            pred_traj_fake, pred_traj_fake_rel, GCNAttention = generator(input_img, obs_traj, obs_traj_rel, seq_start_end)
                            # print(GCNAttention)
                        # else:
                        #     pred_traj_fake, pred_traj_fake_rel = generator(input_img, obs_traj, obs_traj_rel, seq_start_end)

                    size = plt_size/1

                    fake = pred_traj_fake.clone()
                    (start, end) = seq_start_end[0]
                    pd_num = end - start
                    
                    ade_s = displacement_error(fake[:,start:end,:]*R_size, tr_gt[:,start:end,:]*R_size, mode='sum') / (pd_num * args.pred_len)
                    fde_s = final_displacement_error(fake[-1,start:end,:]*R_size, tr_gt[-1,start:end,:]*R_size, mode='sum') / (pd_num)
                    # ade.append(displacement_error(pred_traj_fake*R_size, pred_traj_gt*R_size, mode='raw'))
                    # fde.append(final_displacement_error(pred_traj_fake[-1]*R_size, pred_traj_gt[-1]*R_size, mode='raw'))
                    gt_tr = pred_traj_gt[:,start:end,:].cpu().data.numpy() * size
                    input_tr = obs_traj_gt[:,start:end,:].cpu().data.numpy() * size
                    predict_tr = pred_traj_fake[:,start:end,:].cpu().data.numpy() * size
                    gt = np.concatenate((input_tr,gt_tr),axis=0)
                    pr = np.concatenate((input_tr,predict_tr),axis=0)
                    
                    if min_ade >= ade_s:
                        min_ade = ade_s
                        min_fde = fde_s
                        if sampling_num == 0:        
                            total_aa.append(gt)
                            total_bb.append(pr)
                            if At_p != 0:
                                ph_at.append(physical_attention[start:end])

                            if At_s != 0:
                                if social_attention_type == 'simple' or social_attention_type == 'sophie':
                                    so_at.append(social_attention[start:end])
                                    so_pd_num.append(agent_num[start:end])

                                else:
                                    so_at.append(social_attention[start:end])

                        else:
                            total_aa[sample] = gt
                            total_bb[sample] = pr
                            if At_p != 0:
                                ph_at[sample] = physical_attention[start:end]

                            if At_s != 0:
                                if social_attention_type == 'simple' or social_attention_type == 'sophie':
                                    so_at[sample] = social_attention[start:end]
                                    so_pd_num[sample] = agent_num[start:end]

                                else:
                                    so_at[sample] = social_attention[start:end]
                        ploot(model_num, plt_size, img_size, center_crop, At_p, At_s, video_num, pd_num.cpu(), social_attention_type, ge_type, GCNAttention, start, end)
                    sampling_num += 1

                
                sq_num += 1
                print('[INFO: visualize_model.py]: test_{} is generated!'.format(sample))
                path = './video/checkpoint_with_model_{}/{}/sq-{}/eval.txt'.format(model_num, video_num, sample)
                g = open(path, "w")
                g.write('ade:{}. fde:{}.\n'.format(min_ade, min_fde))
                g.write('ade/fde:{}/{}'.format(round(min_ade.item(),5), round(min_fde.item(),5)))
                g.close()
                sample += 1
        
        #     ade_sum = evaluate_helper(ade, seq_start_end)
        #     fde_sum = evaluate_helper(fde, seq_start_end)
        #     ade_outer.append(ade_sum)
        #     fde_outer.append(fde_sum)

        # ade = sum(ade_outer) / (total_traj * args.pred_len)
        # fde = sum(fde_outer) / (total_traj)

        # path = './video/checkpoint_with_model_{}/{}/eval.txt'.format(model_num, video_num)
        # f = open(path, "w")
        # f.write('ade:{}. fde:{}.\n'.format(ade, fde))
        # f.write('ade/fde:{}/{}'.format(round(ade.item(),5), round(fde.item(),5)))
        # f.close()

def main(args):
    with open(args.data_dir_path) as f:
        if args.data == 100:
            data_path = [s.strip() for s in f.readlines()]
        else:
            data_path = [[s.strip() for s in f.readlines()][args.data]]
            
    paths = ['../checkpoint/checkpoint_{}.pt'.format(i) for i in args.model_num.split(',')]
    
    data_number = 0
    for test_path in data_path:
        global video_num, recurrent_attention 
        video_num = test_path.split('/')[-2] + '_' +test_path.split('/')[-1]
        # video_num = args.video_num.split(',')[data_number]
        query = 0
        p_args = 0
        for path in paths:
            global model_num
            model_num = args.model_num.split(',')[query]
            print('[INFO: visualize_model.py]: data: {}  model: {}.'.format(video_num, model_num))
            checkpoint = torch.load(path)
            generator, LSTM, social_attention_type, ge_type, large_image, recurrent_attention, ph = get_generator(checkpoint)
                
            _args = AttrDict(checkpoint['args'])
            _args.skip = 1
            _args.batch_size = 1
            if p_args == 0 or p_args !=  [_args.center_crop, _args.norm, _args.min_ped, _args.max_ped, large_image, ph]:
                print('[INFO: visualize_model.py]: Data loader. large_image:{}.'.format(large_image))
                _, loader = test_data_loader(_args, [test_path])  
            os.makedirs('./video', exist_ok=True)
            if not os.path.exists("./video/checkpoint_with_model_{}".format(model_num)):
                os.mkdir("./video/checkpoint_with_model_{}".format(model_num))
            if not os.path.exists("./video/checkpoint_with_model_{}/{}".format(model_num, video_num)):
                os.mkdir("./video/checkpoint_with_model_{}/{}".format(model_num, video_num))

            evaluate(_args, loader, generator, model_num, args.num_samples, args.size,  _args.crop_img_size, _args.center_crop, _args.physical_attention_dim, _args.social_attention_dim, LSTM, video_num, social_attention_type, ge_type)
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    main(args)
