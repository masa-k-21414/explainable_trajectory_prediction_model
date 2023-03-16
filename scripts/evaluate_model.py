import argparse
import os
import torch

from attrdict import AttrDict
from model.data.loader import data_loader
from model.models import TrajectoryGenerator, TrajectoryLSTM
from model.losses import displacement_error, final_displacement_error
from model.data.loader import test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_path', default='./test.txt', type=str)
parser.add_argument('--model_num', default='', type=str)

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
            usefulness=args.usefulness,
            use_vgg=args.not_use_vgg, 
            use_seg=args.use_seg, 
            vgg_train=args.vgg_train, 
            add_input=args.add_input,
            #easy=args.easy,
            artificial_social_attention=args.artificial_social_attention,
            visualize=True)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.eval()
    return generator


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

# def evaluate_helper(error, seq_start_end):
#     sum_ = 0
#     error = torch.stack(error, dim=1)

#     for (start, end) in seq_start_end:
#         start = start.item()
#         end = end.item()
#         _error = error[start:end]
#         for k in range(end-start):
#             _e = torch.min(_error[k], dim=0)
#             # print(_error[k], _e)
#             sum_ += _e[0]
#     return sum_

def evaluate(args, loader, generator, num_samples=1):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            img = batch[-1]
            t_w = img[0].size[0]
            t_h = img[0].size[1]
            if args.norm == 0:
                R_size = 1
            else:
                R_size = torch.tensor([t_w/args.norm, t_h/args.norm]).cuda()
                # R_size = max(t_w,t_h)
            
            input_img = batch[-2]
            batch = [tensor.cuda() for tensor in batch[:-2]]

            if not args.easy:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, social_prior_attention, physical_prior_attention) = batch
            else:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                if args.LSTM:
                    pred_traj_fake, _ = generator(input_img, obs_traj, obs_traj_rel, seq_start_end)
                else:
                    if not args.easy:
                        pred_traj_fake, _, _, _, _ = generator(input_img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
                    else:
                        pred_traj_fake, _, _, _, _ = generator(input_img, obs_traj, obs_traj_rel, seq_start_end)
                ade.append(displacement_error(pred_traj_fake*R_size, pred_traj_gt*R_size, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1]*R_size, pred_traj_gt[-1]*R_size, mode='raw'))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    with open(args.data_dir_path) as f:
        data_path = [s.strip() for s in f.readlines()]
            
    paths = ['../checkpoint/checkpoint_{}.pt'.format(i) for i in args.model_num.split(',')]
    
    query = 0
    inst = True
    for path in paths:
        global model_num
        model_num = args.model_num.split(',')[query]
        
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        if inst:
            _args = AttrDict(checkpoint['args'])
            _args.skip = 1
            _args.batch_size = 1
            _, loader = test_data_loader(_args, data_path)  
        if not _args.l2_loss_only:
            ade, fde = evaluate(_args, loader, generator, 20)
        else:
            ade, fde = evaluate(_args, loader, generator)
        print('[INFO: evaluate_model.py]: model: {}.'.format(model_num))
        print('Model: {}, ADE: {:.2f}, FDE: {:.2f}'.format(model_num, ade, fde))
        query += 1
        inst = False

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
