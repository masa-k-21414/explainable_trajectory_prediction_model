import argparse
import gc
import logging
import os
import sys
import time

from torch.optim.lr_scheduler import StepLR
from model.data.loader import data_loader
from model.losses import gan_g_loss, gan_d_loss, l2_loss, a_l2_loss
from model.losses import displacement_error, final_displacement_error
from model.models import TrajectoryGenerator, TrajectoryDiscriminator, TrajectoryLSTM

from model.utils import int_tuple, bool_flag, get_total_norm
from model.utils import relative_to_abs


parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--train_txt', default='./train.txt', type=str)
parser.add_argument('--val_txt', default='./val.txt', type=str)
parser.add_argument('--validation', action='store_false')
parser.add_argument('--loader_num_workers', default=2, type=int)
parser.add_argument('--pin_memory', action='store_false')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=4, type=int)
parser.add_argument('--min_ped', default=1, type=int)
parser.add_argument('--max_ped', default=120, type=int)
parser.add_argument('--delim', default=' ')
parser.add_argument('--norm', default=1, type=int)
parser.add_argument('--large_image', action='store_true')
parser.add_argument('--remake_data', action='store_true')
parser.add_argument('--check_so_at', action='store_true')

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=0, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--scaler', action='store_true')

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)

# Generator Options
parser.add_argument('--LSTM', action='store_true') # 'True' is Simple model consisting of only LSTM
parser.add_argument('--no_Attention', action='store_true')
parser.add_argument('--cell_pad', action='store_true')
parser.add_argument('--ge_type', default='traj_rel', type=str)
parser.add_argument('--gd_type', default='traj_rel', type=str)
parser.add_argument('--g_mlp_dim', default=64, type=int)
parser.add_argument('--encoder_h_dim_g', default=32, type=int)
parser.add_argument('--decoder_h_dim_g', default=113, type=int)
parser.add_argument('--noise_dim', default=0, type=int)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=0.00001, type=float)
parser.add_argument('--center_crop', action='store_true')
parser.add_argument('--crop_img_size', default=512, type=int)
parser.add_argument('--g_steps', default=1, type=int)
parser.add_argument('--input_to_decoder', action='store_true')
parser.add_argument('--recurrent_graph', action='store_true')

# Attention Options
parser.add_argument('--n_max', default=128, type=int)
parser.add_argument('--multiplier', default=3., type=float)
parser.add_argument('--attention_type', default='simple', type=str)
parser.add_argument('--social_attention_type', default='simple', type=str)
parser.add_argument('--physical_attention_type', default='simple', type=str)
parser.add_argument('--so_prior_type', default='add', type=str)
parser.add_argument('--ph_prior', default='gd', type=str)
parser.add_argument('--ph_ver', default=1, type=int)
parser.add_argument('--so_ver', default=1, type=int)
parser.add_argument('--ph_prior_type', default='add', type=str)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--physical_attention_dim', default=49, type=int)
parser.add_argument('--social_attention_dim', default=32, type=int)
parser.add_argument('--physical_tempreture', default=1., type=float)
parser.add_argument('--social_tempreture', default=1., type=float)
parser.add_argument('--concat_state', action='store_false')
parser.add_argument('--usefulness', action='store_true')
parser.add_argument('--not_use_vgg', action='store_false')
parser.add_argument('--use_seg', action='store_true')
parser.add_argument('--vgg_train', action='store_true')
parser.add_argument('--add_input', action='store_true')
parser.add_argument('--physical_pos_embed', action='store_true')
parser.add_argument('--physical_img_embed', action='store_true')
parser.add_argument('--social_pos_embed', action='store_true')
parser.add_argument('--artificial_social_attention', action='store_true')
parser.add_argument('--recurrent_attention', action='store_true')
parser.add_argument('--recurrent_physical_attention', action='store_true')
parser.add_argument('--input_recurrent_attention', action='store_true')
parser.add_argument('--compress_attention', action='store_true')

# Discriminator Options
parser.add_argument('--d_type', default='traj_rel', type=str)
parser.add_argument('--d_mlp_dim', default=1024, type=int)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=0.005, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_only', action='store_true')
parser.add_argument('--l2_loss_weight', default=1, type=float)
parser.add_argument('--l2_loss_type', default='traj_rel')
parser.add_argument('--best_k', default=1, type=int)
parser.add_argument('--d_loss_type', default='bce')

# Output
parser.add_argument('--output_dir', default='../checkpoint/')
parser.add_argument('--print_every', default=1024, type=int)
parser.add_argument('--print_model', action='store_true')
parser.add_argument('--checkpoint_every', default=512, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_num', default=666)
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', action='store_true')
parser.add_argument('--gpu_num', default=" ", type=str)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
from tensorboardX import SummaryWriter
from tensorboard import tensorboard

from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# torch.autograd.set_detect_anomaly(True)
# logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def main(args):
    # print(args.gpu_num, torch.cuda.device_count(), torch.cuda.current_device())
    long_dtype, float_dtype = get_dtypes(args)
    logger.info('Start model checkpoint_num {}'.format(args.checkpoint_num))

    writer = SummaryWriter(log_dir="./logger/checkpoint_{}".format(args.checkpoint_num))
    
    # restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir, '{}_{}.pt'.format(args.checkpoint_name, args.checkpoint_num))
    os.makedirs(args.output_dir, exist_ok=True)
    
    # check the model have not finished learning
    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        epoch = checkpoint['counters']['epoch']
        t = checkpoint['counters']['t']
        checkpoint['restore_ts'].append(t)
        
        if epoch >= args.num_epochs:
            logger.info("Learning is done.") 
            sys.exit()
        else:
            logger.info("Start Learning from {} epochs.".format(epoch)) 

    t_1 = time.time()
    args.skip = 4
    with open(args.train_txt) as f:
        train_path = [s.strip() for s in f.readlines()]
    logger.info("Initializing train dataset : skip = {}".format(args.skip)) 
    train_dset, train_loader = data_loader(args, train_path)

    t_2 = time.time()
    if args.validation:
        args.skip = 20
        with open(args.val_txt) as f:
            val_path = [s.strip() for s in f.readlines()]
        logger.info("Initializing val dataset : skip = {}".format(args.skip))
        _, val_loader = data_loader(args, val_path)
        t_3 = time.time()

        if args.timing:
            logger.info("Train data load Time: {}, Val data loader Time: {}, SUM: {}".format(round(t_2-t_1,3), round(t_3-t_2,3), round(t_3-t_1,3)))
    
    elif args.timing:
        logger.info("Train data load Time: {}".format(round(t_2-t_1,3)))

    if args.l2_loss_only:
        # args.scaler = True
        args.d_steps = 0
        args.g_steps = 1
        discriminator = None
        d_loss_fn = None

    if args.scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
   
    
    if args.LSTM or args.no_Attention:
        args.physical_attention_dim = 0
        args.social_attention_dim = 0
    
    # cons = int(args.g_steps + args.d_steps)
    if not args.l2_loss_only:
        if len(train_dset) / args.batch_size % args.d_steps == 0:
            iterations_per_epoch = int(len(train_dset) / args.batch_size / args.d_steps)
        else:
            iterations_per_epoch = int(len(train_dset) / args.batch_size / args.d_steps) + 1
        
        args.num_iterations = int((len(train_dset) / args.batch_size / args.d_steps) * args.num_epochs)
    else:
        args.num_iterations = int((len(train_dset) / args.batch_size) *  args.num_epochs)
        iterations_per_epoch = int(len(train_dset) / args.batch_size )
        # print(iterations_per_epoch)

    args.print_every = iterations_per_epoch
    args.checkpoint_every = int((iterations_per_epoch * 6) / 5)

    if args.LSTM:
        generator = TrajectoryLSTM(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            g_type=args.ge_type,
            recurrent_graph=args.recurrent_graph)

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
            artificial_social_attention=args.artificial_social_attention)

    generator.apply(init_weights)
    generator.type(float_dtype).train()

    if args.print_model:
        logger.info('Here is the generator:')
        logger.info(generator)

    g_loss_fn = gan_g_loss
    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_g.zero_grad(set_to_none=True)
    scheduler_g = StepLR(optimizer_g, step_size=100, gamma=0.5)

    if args.l2_loss_only == False:
        discriminator = TrajectoryDiscriminator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            h_dim=args.encoder_h_dim_d,
            mlp_dim=args.d_mlp_dim,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
            d_type=args.d_type)

        discriminator.apply(init_weights)
        discriminator.type(float_dtype).train()

        if args.print_model:
            logger.info('Here is the discriminator:')
            logger.info(discriminator)
    
        d_loss_fn = gan_d_loss        
        optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)
        scheduler_d = StepLR(optimizer_d, step_size=100, gamma=0.5)
        
        optimizer_d.zero_grad(set_to_none=True)

    # restore model from checkpoint
    if restore_path is not None and os.path.isfile(restore_path):
        generator.load_state_dict(checkpoint['g_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        scheduler_g.load_state_dict(checkpoint['g_scher_state'])
        if args.l2_loss_only == False:
            discriminator.load_state_dict(checkpoint['d_state'])
            optimizer_d.load_state_dict(checkpoint['d_optim_state'])
            scheduler_d.load_state_dict(checkpoint['d_scher_state'])
        

    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 1, 0
        if args.l2_loss_only:
            checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'g_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'best_t_nl': None,
            }

        else:
            checkpoint = {
                'args': args.__dict__,
                'G_losses': defaultdict(list),
                'D_losses': defaultdict(list),
                'losses_ts': [],
                'metrics_val': defaultdict(list),
                'metrics_train': defaultdict(list),
                'sample_ts': [],
                'restore_ts': [],
                'norm_g': [],
                'norm_d': [],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'g_state': None,
                'g_optim_state': None,
                'g_scher_state': None,
                'd_state': None,
                'd_optim_state': None,
                'd_scher_state': None,
                'g_best_state': None,
                'd_best_state': None,
                'best_t': None,
                'g_best_nl_state': None,
                'd_best_state_nl': None,
                'best_t_nl': None,
            }

    if args.print_model:
        logger.info('Here is some arguments:')
        logger.info(args)
    
    print('______________________________________________')
    logger.info('There are {} data, {} iterations per epoch.'.format(len(train_dset), iterations_per_epoch))
    logger.info('checkpoint_num: {} Iteration: {} '.format(args.checkpoint_num, args.num_iterations))

    update = False
    update_t = False
    t0 = time.time()
    logger.info('Starting epoch from {}. Number of Epochs remaining {}'.format(epoch, args.num_epochs-epoch))
    with tqdm(total=args.num_epochs-epoch, desc='Progress') as pbar:
        while t <= args.num_iterations:
            gc.collect()
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            loss_list = [0., 0., 0., 0.]
            epoch += 1
            
            for batch in train_loader:
                if args.timing:
                    logger.info("Dataloder Time:{}".format(round(time.time()-t0,4)))

                if args.l2_loss_only:
                    if g_steps_left > 0:
                        step_type = 'g'
                        if args.scaler:
                            losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler)
                        else:
                            losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                        checkpoint['norm_g'].append(get_total_norm(generator.parameters()))
                        loss_list[2] += losses_g['G_l2_loss']
                        g_steps_left -= 1

                else:
                    # Decide whether to use the batch for stepping on discriminator or generator; an iteration consists of args.d_steps steps on the discriminator followed by args.g_steps steps on the generator.
                    if d_steps_left > 0:
                        step_type = 'd'
                        if args.scaler:
                            losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, scaler)
                        else:
                            losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                        checkpoint['norm_d'].append(get_total_norm(discriminator.parameters()))
                        loss_list[0] += losses_d['D_data_loss']
                        d_steps_left -= 1
                    
                    elif g_steps_left > 0:
                        step_type = 'g'
                        if args.scaler:
                            losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler, True)
                        else:
                            losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, None, True)
                        checkpoint['norm_g'].append(get_total_norm(generator.parameters()))
                        loss_list[1] += losses_g['G_discriminator_loss']
                        if args.l2_loss_weight != 0.:
                            loss_list[2] += losses_g['G_l2_loss']
                        g_steps_left -= 1

                if args.timing:
                    t0 = time.time()
                
                # Skip the rest if we are not at the end of an iteration
                if (d_steps_left > 0 or g_steps_left > 0) and t != args.num_iterations:
                    continue

                t += 1
                d_steps_left = args.d_steps
                g_steps_left = args.g_steps
            
                # Tensorborad
                # Maybe save loss
                if t % args.print_every == 0:
                    # print(' Print the loss in Epoch {}.'.format(epoch))
                    if args.l2_loss_only == False:
                        for k, v in sorted(losses_d.items()):
                            # logger.info('  [D] {}: {:.3f}'.format(k, v))
                            checkpoint['D_losses'][k].append(v)
                            if k == 'D_data_loss':
                                tensorboard(writer, t, epoch_d_loss=v)
                    for k, v in sorted(losses_g.items()):
                        # logger.info('  [G] {}: {:.3f}'.format(k, v))
                        checkpoint['G_losses'][k].append(v)
                        if k == 'G_discriminator_loss':
                            tensorboard(writer, t, epoch_g_loss=v)
                        if k == 'G_l2_loss':
                            tensorboard(writer, t, epoch_g_l2=v)
                    checkpoint['losses_ts'].append(t)

                # Maybe save a checkpoint
                if t % args.checkpoint_every == 0 or epoch == args.num_epochs:
                    checkpoint['counters']['t'] = t
                    checkpoint['counters']['epoch'] = epoch
                    checkpoint['sample_ts'].append(t)

                    # Check stats on the validation set
                    if args.validation:
                        # logger.info('Checking stats on val ...')
                        metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn)
                        for k, v in sorted(metrics_val.items()):
                            # logger.info('  [val] {}: {:.3f}'.format(k, v))
                            checkpoint['metrics_val'][k].append(v)
                            if k == 'ade':
                                tensorboard(writer, t, epoch_val_ade=v)

                        min_ade = min(checkpoint['metrics_val']['ade'])
                        min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])
                        if metrics_val['ade'] == min_ade:
                            update = True
                            # logger.info('New low for avg_disp_error')
                            checkpoint['best_t'] = t
                            checkpoint['g_best_state'] = generator.state_dict()
                            if args.l2_loss_only == False:
                                checkpoint['d_best_state'] = discriminator.state_dict()
                        if metrics_val['ade_nl'] == min_ade_nl:
                            # logger.info('New low for avg_disp_error_nl')
                            checkpoint['best_t_nl'] = t
                            checkpoint['g_best_nl_state'] = generator.state_dict()
                            if args.l2_loss_only == False:
                                checkpoint['d_best_nl_state'] = discriminator.state_dict()

                        
                        # tensorboard(writer, epoch, epoch_val_adenl=min_ade_nl)
                    
                    # logger.info('Checking stats on train ...')
                    metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn, limit=True)                
                    for k, v in sorted(metrics_train.items()):
                        # logger.info('  [train] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_train'][k].append(v)
                        if k == 'ade':
                            tensorboard(writer, t, epoch_train_ade=v)

                    if metrics_train['ade'] == min(checkpoint['metrics_train']['ade']):
                        update_t = True
                    
                    # t_min_ade = min(checkpoint['metrics_train']['ade'])
                    # t_min_ade_nl = min(checkpoint['metrics_train']['ade_nl'])

                    # tensorboard(writer, epoch, epoch_train_ade=t_min_ade)
                    # tensorboard(writer, epoch, epoch_train_adenl=t_min_ade_nl)
                    
                    # Save a checkpoint with no model weights by making a shallow
                    # copy of the checkpoint excluding some items

                    # checkpoint_path = os.path.join(args.output_dir, '{}_no_model_{}.pt'.format(args.checkpoint_name, args.checkpoint_num))
                    # logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    # key_blacklist = [
                    #     'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    #     'g_optim_state', 'd_optim_state', 'd_best_state',
                    #     'd_best_nl_state'
                    #     ]
                    # small_checkpoint = {}
                    # for k, v in checkpoint.items():
                    #     if k not in key_blacklist:
                    #         small_checkpoint[k] = v
                    # torch.save(small_checkpoint, checkpoint_path)
                    # logger.info('Done.')

                    # Save another checkpoint with model weights and optimizer state
                    checkpoint['g_state'] = generator.state_dict()
                    checkpoint['g_optim_state'] = optimizer_g.state_dict()
                    checkpoint['g_scher_state'] = scheduler_g.state_dict()
                    if args.l2_loss_only == False:
                        checkpoint['d_state'] = discriminator.state_dict()
                        checkpoint['d_optim_state'] = optimizer_d.state_dict()
                        checkpoint['d_scher_state'] = scheduler_d.state_dict()
                    checkpoint_path = os.path.join(args.output_dir, '{}_{}.pt'.format(args.checkpoint_name, args.checkpoint_num))
                    # logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    # logger.info('Done.')

                    if update:
                        checkpoint_path = os.path.join(args.output_dir, '{}_{}_v.pt'.format(args.checkpoint_name, args.checkpoint_num))
                        # logger.info('Update checkpoint to {}'.format(checkpoint_path))
                        torch.save(checkpoint, checkpoint_path)
                        # logger.info('Done.')
                        update = False

                    if update_t:
                        checkpoint_path = os.path.join(args.output_dir, '{}_{}_t.pt'.format(args.checkpoint_name, args.checkpoint_num))
                        # logger.info('Update checkpoint to {}'.format(checkpoint_path))
                        torch.save(checkpoint, checkpoint_path)
                        # logger.info('Done.')
                        update_t = False
                    
                
            scheduler_d.step()
            scheduler_g.step()
            if epoch == args.num_epochs:
                break
            
            pbar.update(1)
            
def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, scaler=None):
    img = batch[-1]
    batch = [tensor.cuda() for tensor in batch[:-1]]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, social_prior_attention, physical_prior_attention) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    if scaler == None:
        if args.LSTM == False:
            generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
        else:
            generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end)
        pred_traj_fake, pred_traj_fake_rel, _ = generator_out

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

        # Compute loss with optional gradient penalty
        data_loss = d_loss_fn(scores_real, scores_fake, args.d_loss_type)
        losses['D_data_loss'] = data_loss.item()
        loss = loss + data_loss
        losses['D_total_loss'] = loss.item()
        
        optimizer_d.zero_grad(set_to_none=True)
        loss.backward()
        if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)

        optimizer_d.step()

    else:
        with torch.cuda.amp.autocast():
            generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
            pred_traj_fake, pred_traj_fake_rel, _ = generator_out

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            # Compute loss with optional gradient penalty
            data_loss = d_loss_fn(scores_real, scores_fake, args.d_loss_type)
            losses['D_data_loss'] = data_loss.item()
            loss = loss + data_loss
            losses['D_total_loss'] = loss.item()
        
        scaler.scale(loss).backward()
        if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
        scaler.step(optimizer_d)
        scaler.update()
        optimizer_d.zero_grad(set_to_none=True)
    return losses

def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler=None, GAN=False):
    img = batch[-1]
    batch = [tensor.cuda() for tensor in batch[:-1]]
    
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, social_prior_attention, physical_prior_attention) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss = []

    loss_mask = loss_mask[:, args.obs_len:]

    if scaler==None:
        if GAN:
            for _ in range(args.best_k):
                if  args.LSTM == False:
                    generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
                else:
                    generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end)
                
                pred_traj_fake, pred_traj_fake_rel, s_attention = generator_out
                # pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                
                if args.l2_loss_weight > 0:
                    if args.l2_loss_type == 'traj':
                        g_l2_loss.append(args.l2_loss_weight * l2_loss(
                            pred_traj_fake,
                            pred_traj_gt,
                            loss_mask,
                            mode='raw'))

                    elif args.l2_loss_type == 'traj_rel':
                        g_l2_loss.append(args.l2_loss_weight * l2_loss(
                            pred_traj_fake_rel,
                            pred_traj_gt_rel,
                            loss_mask,
                            mode='raw'))

            if args.l2_loss_type == 'traj':
                g_l2_loss_sum = torch.zeros(1).to(pred_traj_gt)

            elif args.l2_loss_type == 'traj_rel':
                g_l2_loss_sum = torch.zeros(1).to(pred_traj_gt_rel)

            if args.l2_loss_weight > 0:
                g_l2_loss = torch.stack(g_l2_loss, dim=1)
                for start, end in seq_start_end.data:
                    _g_l2_loss = g_l2_loss[start:end]
                    _g_l2_loss = torch.sum(_g_l2_loss, dim=0)
                    _g_l2_loss = torch.min(_g_l2_loss) / torch.sum(loss_mask[start:end])
                    g_l2_loss_sum = g_l2_loss_sum + _g_l2_loss
                losses['G_l2_loss'] = g_l2_loss_sum.item()
                loss = loss + g_l2_loss_sum

            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            discriminator_loss = g_loss_fn(scores_fake, args.d_loss_type)
            
            loss = loss + discriminator_loss
            losses['G_discriminator_loss'] = discriminator_loss.item()
            losses['G_total_loss'] = loss.item()
        else:
            t1 = time.time()
            if  args.LSTM == False:
                generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
            else:
                generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end)
            t2 = time.time()
            if args.timing:
                logger.info("Forward Time: {}".format(round(t2-t1,4)))
            
            if args.LSTM:
                pred_traj_fake, pred_traj_fake_rel = generator_out
            else:
                pred_traj_fake, pred_traj_fake_rel, s_attention = generator_out

            g_l2_loss_sum = args.l2_loss_weight * l2_loss(
                                pred_traj_fake_rel,
                                pred_traj_gt_rel,
                                loss_mask,
                                mode='sum')
            
            
            losses['G_l2_loss'] = g_l2_loss_sum.item()
            loss = loss + g_l2_loss_sum

        t4 = time.time()
        if args.timing:
            logger.info("Backforward Time: {}".format(round(t4-t3,4)))

        optimizer_g.zero_grad(set_to_none=True)
        loss.backward()
        if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_d)
        optimizer_g.step()

    else:
        with torch.cuda.amp.autocast():
            if GAN:
                for _ in range(args.best_k):
                    if args.LSTM == False:
                        generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
                    else:
                        generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end)
                    
                    pred_traj_fake, pred_traj_fake_rel, s_attention = generator_out
                    # pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                    
                    if args.l2_loss_weight > 0:
                        if args.l2_loss_type == 'traj':
                            g_l2_loss.append(args.l2_loss_weight * l2_loss(
                                pred_traj_fake,
                                pred_traj_gt,
                                loss_mask,
                                mode='raw'))

                        elif args.l2_loss_type == 'traj_rel':
                            g_l2_loss.append(args.l2_loss_weight * l2_loss(
                                pred_traj_fake_rel,
                                pred_traj_gt_rel,
                                loss_mask,
                                mode='raw'))

                if args.l2_loss_type == 'traj':
                    g_l2_loss_sum = torch.zeros(1).to(pred_traj_gt)

                elif args.l2_loss_type == 'traj_rel':
                    g_l2_loss_sum = torch.zeros(1).to(pred_traj_gt_rel)

                if args.l2_loss_weight > 0:
                    g_l2_loss = torch.stack(g_l2_loss, dim=1)
                    for start, end in seq_start_end.data:
                        _g_l2_loss = g_l2_loss[start:end]
                        _g_l2_loss = torch.sum(_g_l2_loss, dim=0)
                        _g_l2_loss = torch.min(_g_l2_loss) / torch.sum(loss_mask[start:end])
                        g_l2_loss_sum = g_l2_loss_sum + _g_l2_loss
                    losses['G_l2_loss'] = g_l2_loss_sum.item()
                    loss = loss + g_l2_loss_sum

                traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
                traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

                scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
                discriminator_loss = g_loss_fn(scores_fake, args.d_loss_type)
                
                loss = loss + discriminator_loss
                losses['G_discriminator_loss'] = discriminator_loss.item()
                losses['G_total_loss'] = loss.item()
            
            else:
            # for _ in range(args.best_k):
                t1 = time.time()
                if args.LSTM == False:
                    generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
                else:
                    generator_out = generator(img, obs_traj, obs_traj_rel, seq_start_end)
                t2 = time.time()
                if args.timing:
                    logger.info("Forward Time: {}".format(round(t2-t1,4)))
                
                if args.LSTM:
                    pred_traj_fake, pred_traj_fake_rel = generator_out
                else:
                    pred_traj_fake, pred_traj_fake_rel, s_attention = generator_out

                g_l2_loss_sum = args.l2_loss_weight * l2_loss(
                                    pred_traj_fake_rel,
                                    pred_traj_gt_rel,
                                    loss_mask,
                                    mode='sum')
                
                
                losses['G_l2_loss'] = g_l2_loss_sum.item()
                loss = loss + g_l2_loss_sum


        t3 = time.time()
        if args.timing:
            logger.info("Loss Calculation Time: {}".format(round(t3-t2,4)))

        scaler.scale(loss).backward()
        t4 = time.time()
        if args.timing:
            logger.info("Backforward Time: {}".format(round(t4-t3,4)))

        if args.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
        scaler.step(optimizer_g)
        scaler.update()
        optimizer_g.zero_grad(set_to_none=True)
    
    return losses

def check_accuracy(args, loader, generator, discriminator, d_loss_fn, limit=False):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()

    with torch.no_grad():
        for batch in loader:
            img = batch[-1] 
            batch = [tensor.cuda() for tensor in batch[:-1]]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, social_prior_attention, physical_prior_attention) = batch

            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            if args.LSTM == False:
                pred_traj_fake, pred_traj_fake_rel, _ = generator(img, obs_traj, obs_traj_rel, seq_start_end, so_prior=social_prior_attention, ph_prior=physical_prior_attention)
            else:
                pred_traj_fake, pred_traj_fake_rel, _ = generator(img, obs_traj, obs_traj_rel, seq_start_end)

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            if args.social_attention_type == 'sophie' or args.physical_attention_type == 'sophie':
                obs_traj = obs_traj[:,:,0,:]
                obs_traj_rel = obs_traj_rel[:,:,0,:]
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            if args.l2_loss_only == False:
                scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
                scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

                d_loss = d_loss_fn(scores_real, scores_fake)
                d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum = loss_mask_sum + torch.numel(loss_mask.data)
            total_traj = total_traj + pred_traj_gt.size(1)
            total_traj_l = total_traj_l + torch.sum(linear_ped).item()
            total_traj_nl = total_traj_nl + torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break
    
    if args.l2_loss_only == False:
        metrics['d_loss'] = sum(d_losses) / len(d_losses)
    
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl

if __name__ == '__main__':
    main(args)
    logger.info("{} is Done.".format(args.checkpoint_num))


            # if args.l2_loss_only:
            #     tensorboard(writer, epoch, epoch_g_l2=loss_list[2])
            #     # logger.info(' [G] Sum of MSE : {:.3f}'.format(loss_list[2]))
            # else:
            #     if args.l2_loss_weight == 0:
            #         tensorboard(writer, epoch, epoch_d_loss=loss_list[0], epoch_g_loss=loss_list[1])
            #         # logger.info(' [D] Data Loss : {:.3f}'.format(loss_list[0]))
            #         # logger.info(' [G] Discriminator Loss : {:.3f}'.format(loss_list[1]))
            #     else:
            #         tensorboard(writer, epoch, epoch_d_loss=loss_list[0], epoch_g_loss=loss_list[1], epoch_g_l2=loss_list[2])
            #         # logger.info(' [D] Data Loss : {:.3f}'.format(loss_list[0]))
            #         # logger.info(' [G] Discriminator Loss : {:.3f}'.format(loss_list[1]))
            #         # logger.info(' [G] Sum of MSE : {:.3f}'.format(loss_list[2]))
