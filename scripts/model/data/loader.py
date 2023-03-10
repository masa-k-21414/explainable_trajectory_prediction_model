import argparse
import gc
import os
import time
from torch.utils.data import DataLoader
from model.data.trajectories import TrajectoryDataset, seq_collate, test_seq_collate, sophie_seq_collate, sophie_test_seq_collate, ph_seq_collate, ph_test_seq_collate

def data_loader(args, path):
    dset = TrajectoryDataset(
        args,
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        center_crop=args.center_crop, 
        crop_img_size=args.crop_img_size, 
        skip=args.skip,
        min_ped=args.min_ped,
        max_ped=args.max_ped,
        delim=args.delim,
        norm=args.norm,
        large_image=args.large_image,
        remake_data=args.remake_data,
        check_so_at=args.check_so_at)
    
    elif args.physical_attention_type == 'prior3' or args.physical_attention_type == 'prior4' or args.physical_attention_type == 'prior5' or args.physical_attention_type == 'prior6'  or args.physical_attention_type == 'self_attention':
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            pin_memory=args.pin_memory,
            collate_fn=ph_seq_collate)
    else:
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            pin_memory=args.pin_memory,
            collate_fn=seq_collate)
        
    return dset, loader

# test
def test_data_loader(args, path):
    dset = TrajectoryDataset(
        args,
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        center_crop=args.center_crop, 
        crop_img_size=args.crop_img_size, 
        skip=args.skip,
        min_ped=args.min_ped,
        max_ped=args.max_ped,
        delim=args.delim,
        norm=args.norm,
        large_image=args.large_image,
        check_so_at=args.check_so_at,
        test=True)

    elif args.physical_attention_type == 'prior3' or args.physical_attention_type == 'prior4' or args.physical_attention_type == 'prior5' or args.physical_attention_type == 'prior6' or args.physical_attention_type == 'self_attention':
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.loader_num_workers,
            pin_memory=args.pin_memory,
            collate_fn=ph_test_seq_collate)
    else:
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.loader_num_workers,
            pin_memory=args.pin_memory,
            collate_fn=test_seq_collate)

    return dset, loader