import logging
import os
import math
import logging
import sys
import glob
from natsort import natsorted

import torchvision
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def cv2pil(image):
    ''' OpenCV -> PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
 
def datadir_to_videodata(data_dir):
    return data_dir.replace('annotations', 'videos') + '/video.mov'

def save_obs_img(movie, img_num, img_list, img_dir):
    Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    for i in range(Fs - 1):
        flag, frame = movie.read()
        if i in img_num:
            image = cv2pil(frame)
            image.save('{}/frame_num_'.format(img_dir)+'{0:04d}'.format(frame_number)+'.png', quality=95)
            frame_number += 1

def save_obs_crop_img(movie, img_num, img_list, img_dir, seq_list, img_size, obs_len, norm):
    Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    per_sum = 0
    for i in range(Fs - 1):
        flag, frame = movie.read()
        if i in img_num:
            image = cv2pil(frame)
            w = image.size[0]/norm
            h = image.size[1]/norm
            k = 0
            for person in seq_list[frame_number]:
                pos = person[:, obs_len-1]
                x_1 = pos[0].item()*w-(img_size//2)
                y_1 = pos[1].item()*h-(img_size//2)
                x_2 = pos[0].item()*w+(img_size//2)
                y_2 = pos[1].item()*h+(img_size//2)
                img_crop = image.crop((x_1, y_1, x_2, y_2))
                img_crop.save('{}/frame_num_'.format(img_dir)+'{0:06d}'.format(per_sum+k)+'.png', quality=95)
                k += 1
            frame_number += 1
            per_sum += k

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list, sa_data, pa_data, img) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    sa_data = torch.cat(sa_data, dim=0)
    pa_data = torch.cat(pa_data, dim=0)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    # img = torch.cat(img, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, sa_data, pa_data, img
    ]
    return tuple(out)

def test_seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list, sa_data, pa_data, img, test_img) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    sa_data = torch.cat(sa_data, dim=0)
    pa_data = torch.cat(pa_data, dim=0)
    # img = torch.cat(img, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, sa_data, pa_data, img, test_img
    ]
    return tuple(out)

def ph_seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list, sa_data, pa_data, img) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    sa_data = torch.cat(sa_data, dim=0)
    pa_data = torch.cat(pa_data, dim=0)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    img = torch.cat(img, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, sa_data, pa_data, img
    ]
    return tuple(out)

def ph_test_seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list, sa_data, pa_data, img, test_img) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    sa_data = torch.cat(sa_data, dim=0)
    pa_data = torch.cat(pa_data, dim=0)
    img = torch.cat(img, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, sa_data, pa_data, img, test_img
    ]
    return tuple(out)

def sophie_seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list, sa_data, pa_data, img) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(3, 0, 1, 2)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(3, 0, 1, 2)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    sa_data = torch.cat(sa_data, dim=0)
    pa_data = torch.cat(pa_data, dim=0)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    # img = torch.cat(img, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, sa_data, pa_data, img
    ]
    return tuple(out)

def sophie_test_seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list, sa_data, pa_data, img, test_img) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(3, 0, 1, 2)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(3, 0, 1, 2)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    sa_data = torch.cat(sa_data, dim=0)
    pa_data = torch.cat(pa_data, dim=0)
    # img = torch.cat(img, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, sa_data, pa_data, img, test_img
    ]
    return tuple(out)

def read_file(_path, img_path, norm=0, skip=1, delim='\t'):
    data = []
    img = Image.open(img_path)
    cord2imcord = np.array([1.,1.])
    
    if norm != 0:
        # print(max(img.size[0], img.size[1]))
        w = img.size[0]/norm
        h = img.size[1]/norm
        # w = k
        # h = k
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if str(line[6]) == '0':
                line = [float(i) for i in line[:-4]]
                if line[5] % skip == 0:
                    if norm != 0:
                        line = [line[5]//skip, line[0], (line[3]+line[1])/2/w, (line[4]+line[2])/2/h]
                        if (line[2])/2/w > norm or (line[3])/2/h > norm:
                            print('error')
                    else:
                        line = [line[5]//skip, line[0], (line[3]+line[1])/2, (line[4]+line[2])/2]
                    data.append(line)
    return np.asarray(data), cord2imcord


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    # print(t, traj[0, -traj_len:])
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, args, data_dir, obs_len=8, pred_len=12, center_crop=False, crop_img_size=512, skip=1, skip_frame=2, threshold=0.002,
        min_ped=1, max_ped=1000, delim='\t', norm=1, large_image=False, remake_data=False, test=False, check_so_at=False
    ):
        """
        Args:
        - data_dir: List of dataset paths. [path1, path2, ...]
        <frame_id> <ped_id> <x> <y>

        **stanford dataset : <ped_id> <x_min> <y_min> <x_max> <y_max> <frame_id> <lost> <occluded> <generate> <label>
            lost: If 1, the annotation is outside of the view screen.
            occluded: If 1, the annotation is occluded.
            generated: If 1, the annotation was automatically interpolated.
            label: The label for this annotation, enclosed in quotation marks.
        
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - center_crop
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - max_ped
        - delim: Delimiter in the dataset files
        - norm
        - remake_data
        - test
        """
        super(TrajectoryDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm = norm
        self.center_crop = center_crop
        self.test = test
        self.check_so_at = check_so_at
        self.ph_type = args.physical_attention_type
        self.large_image = large_image

        if self.large_image:
            self.large_image_size = 640
            self.transform = transforms.Compose(
                [transforms.Resize((self.large_image_size, self.large_image_size)),
                transforms.ToTensor(),
                ]) 
        else:
            size = 224
            self.transform = transforms.Compose(
                [transforms.Resize((size,size)),
                transforms.ToTensor(), 
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]) 

        num_peds_in_seq = []
        seq_list = []
        s_seq_list = []
        seq_list_rel = []
        s_seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        img_list = []
        test_img_list = []
        self.f_skip = 12
        first = True
        sa = False
        pa = False

        for num in range(len(data_dir)):
            self.remake_data = remake_data
            dn = 0

            self.data_dir = data_dir[num]
            self.img_path = self.data_dir + '/reference.jpg'
            # img = Image.open(self.img_path) 
            self.video = cv2.VideoCapture(datadir_to_videodata(self.data_dir))

            img_num = []
            pednum_list = []
            all_files = os.listdir(self.data_dir)
            all_files = [os.path.join(self.data_dir, 'annotations.txt')]

            for path in all_files:
                data, cord2imcord = read_file(path, self.img_path, self.norm, self.f_skip, self.delim)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames)  - self.seq_len + 1) / self.skip))
                # print(len(frame_data))
                # print(num_sequences)

                for idx in range(0, num_sequences * self.skip + 1, self.skip):
                    if len(frame_data[idx:idx + self.seq_len]) < 1:
                        continue
                    # curr_seq_data = np.concatenate(f_data[::self.f_skip], axis=0)
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq),self.seq_len))

                    num_peds_considered = 0
                    _non_linear_ped = []

                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        # print(pad_end,pad_front)
                        if pad_end - pad_front != self.seq_len:
                            continue
                        if curr_ped_seq.shape[0] != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        # print(curr_ped_seq)
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        # print(pad_front,pad_end)
                        # print(curr_ped_seq.shape)
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1
                        # print(num_peds_considered)

                        if num_peds_considered == max_ped:
                            break

                    if num_peds_considered > min_ped:
                        dn += 1
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        # print(seq_list)
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                        # img_list.append(self.transform(img))
                        img_num.append((idx+self.obs_len)*self.f_skip)
                        pednum_list.append(num_peds_considered)
                        if args.social_attention_type == 'sophie' or args.physical_attention_type == 'sophie':
                            num_peds_in_seq.append(num_peds_considered)
                            curr_seq_exp = np.zeros((num_peds_considered, args.n_max, 2, self.seq_len))
                            curr_seq_rel_exp = np.zeros((num_peds_considered, args.n_max, 2, self.seq_len))
                            for i in range(num_peds_considered):
                                curr_seq_exp[i, 0, :, :] = curr_seq[i]
                                curr_seq_exp[i, 1:i+1, :, :] = curr_seq[0:i]
                                curr_seq_exp[i, i+1:num_peds_considered, :, :] = curr_seq[i+1:num_peds_considered]

                                dists = (curr_seq_exp[i, :] - curr_seq_exp[i, 0]) ** 2
                                dists = np.sum(np.sum(dists, axis=2), axis=1)
                                idxs = np.argsort(dists)
                                curr_seq_exp[i, :] = curr_seq_exp[i, :][idxs]
                                curr_seq_rel_exp[i, 0, :, :] = curr_seq_rel[i]
                                curr_seq_rel_exp[i, 1:i+1, :, :] = curr_seq_rel[0:i]
                                curr_seq_rel_exp[i, i+1:num_peds_considered, :, :] = curr_seq_rel[i+1:num_peds_considered]
                                curr_seq_rel_exp[i, :] = curr_seq_rel_exp[i, :][idxs]
                            s_seq_list.append(curr_seq_exp[:num_peds_considered])
                            s_seq_list_rel.append(curr_seq_rel_exp[:num_peds_considered])
                            

            # if self.center_crop:
            #     img_dir = "{}/crop_True_cropsize_{}_skip_{}_obs_{}_pre_{}_minped_{}".format(self.data_dir, crop_img_size, skip, obs_len, pred_len, min_ped)
            # else:
            img_dir = "{}/crop_False_skip_{}_obs_{}_pre_{}_minped_{}_fskip_{}".format(self.data_dir, self.skip, obs_len, pred_len, min_ped, self.f_skip)
            
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
                self.remake_data = True
                
            if self.remake_data:
                logger.info('Remake image data: {}_{}'.format(self.data_dir.split('/')[-2], self.data_dir.split('/')[-1]))
                # if not self.center_crop:
                save_obs_img(self.video, img_num, img_list, img_dir)
                # else:
                #     save_obs_crop_img(self.video, img_num, img_list, img_dir, seq_list, crop_img_size, self.obs_len, norm)
            

            if self.test or args.physical_attention_dim != 0:
                files = glob.glob(os.path.join(img_dir, '*.png'))
            
                if self.ph_type == 'prior3' or self.ph_type == 'prior4' or self.ph_type == 'prior5' or self.ph_type == 'prior6' or self.ph_type == 'self_attention':
                    n = 0   
                    # path = self.data_dir + '/image_2.csv'
                    # img_save = []
                    for path in natsorted(files):
                        ped_num = pednum_list[n]
                        img = Image.open(path) 
                        img2 = self.transform(img)
                        if self.test:
                            test_img_list.append(img)
                        for _ in range(ped_num):
                            img_list.append(img2)
                        img.load()

                        n += 1
                else:
                    for path in natsorted(files):
                        img = Image.open(path) 
                        img_list.append(self.transform(img))
                        if self.test:
                            test_img_list.append(img)
                        img.load()
                
            # if self.test and self.center_crop:
            #     test_img_dir = "{}/crop_False_skip_{}_obs_{}_pre_{}_minped_{}".format(self.data_dir, skip, obs_len, pred_len, min_ped)
            #     test_files = glob.glob(os.path.join(test_img_dir, '*.png'))
            #     for path in natsorted(test_files):
            #         img = Image.open(path) 
            #         test_img_list.append(img)
            #         img.load()

            if args.social_attention_dim != 0 and args.social_attention_type =='prior':
                sa = True
                # f_path = self.data_dir + '/pseudo-social_attention_.csv'
                if args.so_ver == 1:
                    f_path = self.data_dir + '/pseudo-social_attention.csv'
                elif args.so_ver == 2:
                    f_path = self.data_dir + '/pseudo-social_attention_gauss.csv'
                else:
                    print("pseudo social attention version ERROR")

                if first:
                    self.sa_data = np.loadtxt(f_path)
                else:
                    self.sa_data = np.concatenate([self.sa_data , np.loadtxt(f_path)])


            if args.physical_attention_dim != 0 and (args.physical_attention_type !='one_head_attention' and args.physical_attention_type !='simple' and args.physical_attention_type !='one_head_attention2' and args.physical_attention_type !='sat' and args.physical_attention_type !="NICG_sat" and args.physical_attention_type != 'sophie') and args.ph_prior_type != 'one_head_attention' and args.ph_prior_type != 'not_prior':
                pa = True
                if args.ph_ver == 1:
                    g_path = self.data_dir + '/pseudo-physical_attention_mask.csv'
                elif args.ph_ver == 2:
                    g_path = self.data_dir + '/pseudo-physical_attention_fan_mask.csv'
                elif args.ph_ver == 3:
                    g_path = self.data_dir + '/pseudo-physical_attention_leaky_mask.csv'
                elif args.ph_ver == 4:
                    g_path = self.data_dir + '/pseudo-physical_attention_gauss1.csv'
                elif args.ph_ver == 5:
                    g_path = self.data_dir + '/pseudo-physical_attention_gauss2.csv'
                elif args.ph_ver == 6:
                    g_path = self.data_dir + '/pseudo-physical_attention_gauss3.csv'
                elif args.ph_ver == 7:
                    g_path = self.data_dir + '/pseudo-physical_attention_gauss4.csv'
                else:
                    print("pseudo physical attention version ERROR")

                if first:
                    self.pa_data = np.loadtxt(g_path)
                else:
                    self.pa_data = np.concatenate([self.pa_data , np.loadtxt(g_path)])
                

            logger.info('{}/{} is done. {}'.format(self.data_dir.split('/')[-2], self.data_dir.split('/')[-1], dn))  
            first = False

        # print(self.a_data.shape)
        if args.social_attention_type == 'sophie' or args.physical_attention_type == 'sophie':
            s_seq_list = np.concatenate(s_seq_list, axis=0)
            s_seq_list_rel = np.concatenate(s_seq_list_rel, axis=0)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)  
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        
        # Convert numpy -> Torch Tensor
        if args.social_attention_type == 'sophie' or args.physical_attention_type == 'sophie':
            self.obs_traj = torch.from_numpy(s_seq_list[:, :, :, :self.obs_len]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(s_seq_list_rel[:, :, :, :self.obs_len]).type(torch.float)
        else:
            self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        if args.physical_attention_dim != 0:
            self.img_data = torch.stack(img_list, dim=0)
        else:
            self.img_data = self.pred_traj_rel
            
        if self.test:
            self.test_img_data = test_img_list
        
        if sa:
            self.sa_data = torch.from_numpy(self.sa_data).type(torch.float)
        else:
            self.sa_data = self.pred_traj_rel

        if pa:
            self.pa_data = torch.from_numpy(self.pa_data).type(torch.float)
        else:
            self.pa_data = self.pred_traj_rel

        print(self.img_data.shape, self.pa_data.shape, self.sa_data.shape)

        
            
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        # print(self.obs_traj.shape)
        if self.check_so_at:
            pd_num = end - start
            b = True
            # obs_traj: batch, xy, t
            obs_traj = self.obs_traj[start:end, :]
            last_pos = self.obs_traj[start:end, :, -1]
            _obs_traj = torch.zeros(3,2,8)
            _pred_traj = torch.zeros(3,2,12)
            _obs_traj_rel = torch.zeros(3,2,8)
            _pred_traj_rel = torch.zeros(3,2,12)
            _non_linear_ped = torch.zeros(3) 
            _loss_mask = torch.zeros(3, 20) 

            list_pd = np.arange(pd_num) 
            a = 0
            r = 0.005

            while b:
                q = list_pd[a]
                d = (last_pos[:,0] - last_pos[q,0].item())**2 + (last_pos[:,1] - last_pos[q,1].item())**2 
                # print(d.shape)
                dl = d.clone()
                dl = torch.sort(dl)
                # print(dl[1])
                if dl[0][1] <= r:
                    b = False
                else:
                    a += 1
                    if a == pd_num:
                        a = 0
                        r += 0.005
            min = dl[1][1]
            max = dl[1][-1]
            _obs_traj[0] = self.obs_traj[start+a, :].unsqueeze(0)
            _obs_traj[1] = self.obs_traj[start+min, :].unsqueeze(0)
            _obs_traj[2] = self.obs_traj[start+max, :].unsqueeze(0)
            _pred_traj[0] = self.pred_traj[start+a, :].unsqueeze(0)
            _pred_traj[1] = self.pred_traj[start+min, :].unsqueeze(0)
            _pred_traj[2] = self.pred_traj[start+max, :].unsqueeze(0)
            _obs_traj_rel[0] = self.obs_traj_rel[start+a, :].unsqueeze(0)
            _obs_traj_rel[1] = self.obs_traj_rel[start+min, :].unsqueeze(0)
            _obs_traj_rel[2] = self.obs_traj_rel[start+max, :].unsqueeze(0)
            _pred_traj_rel[0] = self.pred_traj_rel[start+a, :].unsqueeze(0)
            _pred_traj_rel[1] = self.pred_traj_rel[start+min, :].unsqueeze(0)
            _pred_traj_rel[2] = self.pred_traj_rel[start+max, :].unsqueeze(0)
            _non_linear_ped[0] = self.non_linear_ped[start+a]
            _non_linear_ped[1] = self.non_linear_ped[start+min]
            _non_linear_ped[2] = self.non_linear_ped[start+max]
            _loss_mask[0] = self.loss_mask[start+a, :].unsqueeze(0)
            _loss_mask[1] = self.loss_mask[start+min, :].unsqueeze(0)
            _loss_mask[2] = self.loss_mask[start+max, :].unsqueeze(0)
            
            if self.test:
                out = [
                        _obs_traj, _pred_traj, _obs_traj_rel, _pred_traj_rel, _non_linear_ped, _loss_mask, 
                        self.sa_data[3*index:3*index+3], self.pa_data[start:start+3, :],
                        self.img_data[index], self.test_img_data[index]
                    ]
            else:
                out = [
                        _obs_traj, _pred_traj, _obs_traj_rel, _pred_traj_rel, _non_linear_ped, _loss_mask, 
                        self.sa_data[3*index:3*index+3], self.pa_data[start:start+3, :], self.img_data[index]
                    ]

        elif self.ph_type == 'prior3' or self.ph_type == 'prior4' or self.ph_type == 'prior5' or self.ph_type == 'prior6' or self.ph_type == 'self_attention':
            if self.test:
                out = [
                    self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                    self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                    self.non_linear_ped[start:end], self.loss_mask[start:end, :], 
                    self.sa_data[start:end, :], self.pa_data[start:end],
                    self.img_data[start:end], self.test_img_data[index]
                ]

            else:
 
                out = [
                    self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                    self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                    self.non_linear_ped[start:end], self.loss_mask[start:end, :], 
                    self.sa_data[start:end, :], self.pa_data[start:end], self.img_data[start:end]
                ]


        else:
            if self.test:
                out = [
                    self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                    self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                    self.non_linear_ped[start:end], self.loss_mask[start:end, :], 
                    self.sa_data[start:end, :], self.pa_data[start:end],
                    self.img_data[index], self.test_img_data[index]
                ]

            else:
 
                out = [
                    self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                    self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                    self.non_linear_ped[start:end], self.loss_mask[start:end, :], 
                    self.sa_data[start:end, :], self.pa_data[start:end], self.img_data[index]
                ]
        return out