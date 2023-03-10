import numpy as np
import copy
import os
import math
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import sys
import torch.nn.modules.conv as conv
import torchvision
import torchvision.models as module
from torch.nn import functional as F
from torchvision import transforms

# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.data import Data
# from torch_geometric.datasets import KarateClub
# # import torch_geometric.transforms as T
# from torch_geometric.utils import to_networkx
# from torch_geometric_temporal.nn.recurrent import GConvLSTM


def preprocess_image_for_segmentation(im, _len, obs_traj, per_pd=True, encoder='resnet101', encoder_weights='imagenet', seg_mask=False, classes=5):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    # print(obs_traj.shape[1])
    
    if per_pd:
        images = torch.ones(obs_traj.shape[1], 3, 256, 256).cuda()
        k = 0
        for start, end in _len:
            num = end - start
            # print(im[k].shape)
            img = preprocessing_fn(np.transpose(im[k],(1, 2, 0)))
            # print(img.shape)
            img = np.transpose(img, (2, 0, 1))
            # print(img.shape)
            # img = torch.Tensor(img)
            images[start:end] = images[start:end] * img.cuda()
            k += 1
    else:
        images = torch.ones(len(_len), 3, 256, 256).cuda()
        for k in range(len(_len)):
            img = preprocessing_fn(np.transpose(im[k],(1, 2, 0)))
            img = np.transpose(img, (2, 0, 1))
            images[k] = images[k] * img.cuda()
    return images

def image_per_person(im, _len, obs_traj, size=256):
    images = torch.ones(obs_traj.shape[1], 3, size, size).cuda()
    k = 0
    for start, end in _len:
        images[start:end] = im[k].cuda()
        k += 1
    return images

def asorted(size, end_pos):
    C_0 = 1 - torch.abs(torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0])
    C_1 = 1 - torch.abs(torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1])
    return C_0 + C_1

def traj_heatmap_v2(obs_traj, size=256):#高速化
    x_bace = torch.linspace(0,1,size).repeat(size,1,1).squeeze().unsqueeze(0).unsqueeze(1).cuda()
    y_bace = torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().unsqueeze(0).unsqueeze(1).cuda()
    X = 1 - torch.abs(x_bace.repeat(obs_traj.shape[0], obs_traj.shape[1], 1, 1) - obs_traj[:,:,0].unsqueeze(2).unsqueeze(3).expand(obs_traj.shape[0], obs_traj.shape[1],size,size))
    Y = 1 - torch.abs(y_bace.repeat(obs_traj.shape[0], obs_traj.shape[1], 1, 1) - obs_traj[:,:,1].unsqueeze(2).unsqueeze(3).expand(obs_traj.shape[0], obs_traj.shape[1],size,size))
    heatmap = (X + Y)/3
    return heatmap.permute(1, 0, 2, 3)

def traj_heatmap(obs_traj, size=256):
    heatmap = torch.zeros(obs_traj.shape[1], obs_traj.shape[0], size, size).cuda()
    for i in range(obs_traj.shape[1]):
        for t in range(obs_traj.shape[0]):
            heatmap[i,t] = asorted(size, obs_traj[t,i])
    return heatmap

def v_sorted(size, end_pos, v):
    C = torch.ones(2, size, size).cuda()
    if torch.abs(v[0]) < 0.001:
        C[0] = torch.zeros(size,size).cuda()
    else:
        C[0] = (torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0]) * v[0]
    if torch.abs(v[1]) < 0.001:
        C[1] = torch.zeros(size,size).cuda()
    else:
        C[1] = (torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1]) * v[1]
    return C

def traj_v_heatmap(obs_traj, size=256):
    heatmap = torch.zeros(obs_traj.shape[1], 3, size, size).cuda()
    for i in range(obs_traj.shape[1]):
        heatmap[i,1:] = v_sorted(size, obs_traj[-1,i], obs_traj[-1,i]-obs_traj[-2,i])
        heatmap[i,0] = asorted(size, obs_traj[-1,i])
    return heatmap

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        else:
            raise NotImplementedError

        return out

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class vgg_layer(nn.Module):
    def __init__(self, nin, nout, k=3):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, k, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class image_feature_extractor(nn.Module):
    def __init__(self, k=3, L=11):
        super(image_feature_extractor, self).__init__()

        # 224 x 224
        self.c1 = nn.Sequential(
                vgg_layer(L, 32, k),
                vgg_layer(32, 32, k)
                )
        
        # 112 x 112
        self.c2 = nn.Sequential(
                vgg_layer(32, 32, k),
                )

        # 56 x 56
        self.c3 = nn.Sequential(
                vgg_layer(32, 32, k),
                )

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h = self.c1(input) 
        h = self.c2(self.mp(h)) 
        h = self.c3(self.mp(h)) 
        return h

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def make_graph(obs_traj, seq_start_end):
    distdict = {}
    from_node = []
    to_node = []
    for k in range(len(seq_start_end)):
        (start, end) = seq_start_end[k]
        num_ped = end - start
        for i in range(num_ped):
            for j in range(num_ped):
                # from_node.append(start+i)
                # to_node.append(start+j)
                if (obs_traj[start+i][0]-obs_traj[start+j][0])**2 + (obs_traj[start+i][1]-obs_traj[start+j][1])**2 <= 0.5:
                    from_node.append(start+i)
                    to_node.append(start+j)
    # from_node = torch.tensor(from_node)
    # to_node = torch.tensor(to_node)
    edge_index = torch.tensor([from_node, to_node], dtype=torch.long).cuda()
    # print(edge_index)
    # print('aaa')
    return edge_index

def get_noise(shape, noise_type='gaussian'):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

class Physical_one_head_Attention(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, 
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True,  attention_type='simple', easy=False):
        super(Physical_one_head_Attention, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.pos_embed = pos_embed

        if attention_type == 'sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        self.easy = easy
        # self.h_dim = hd_dim
        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size = 3136
            self.img_size_dim = 32
            self.feature_extractor = image_feature_extractor(3, 3)
            
        if attention_type=='simple':
            self.h_dim = he_dim
        else:
            self.h_dim = hd_dim

        if not self.setting_image:
            self.em = nn.Linear(self.img_size_dim, out_dim)
        

        if not self.setting_image:
            # self.o = nn.Linear(out_dim, out_dim)
            if self.h_dim != out_dim:
                self.U = nn.Linear(self.h_dim, out_dim)

            if usefulness == True:
                self.W_u = nn.Linear(self.h_dim+out_dim, 1)
                # self.W_u2 = nn.Linear(out_dim, 1)
                self.F_u = nn.Tanh()
                
            # self.model = nn.MultiheadAttention(embed_dim=out_dim,
            #                                 num_heads=1,
            #                                 batch_first=True)
            self.WQ = nn.Linear(self.h_dim, self.out_dim)
            self.WK = nn.Linear(self.img_size_dim, self.out_dim)
            self.WV = nn.Linear(self.img_size_dim, self.out_dim)
            self.W_o = nn.Linear(self.out_dim, self.out_dim)
            self.softmax = nn.Softmax(2)


        else:
            #1280->1600, 640->400, 320->100
            self.img_size = 400
            # self.encoder_vgg19 = nn.Sequential(*list(self.encoder_vgg19.children())[:-1])
            self.img_size_dim = 512
            self.U = nn.Linear(self.h_dim, self.h_dim)
            self.W = nn.Linear(self.img_size_dim, self.h_dim)
            self.v = nn.Linear(self.h_dim, 1)
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(1)
            self.embed_pos = nn.Linear(2, self.h_dim)
            self.H = nn.Linear(self.img_size_dim, out_dim)

    def forward(self, img, states, seq_start_end, end_pos, prior=0, obs_traj=0, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """

        if self.pos_embed:
            pos = self.embedd(end_pos)
            states = states + pos
                
        # Att_Ph = []
        # physical_attention = []
        # img feature
        if self.center_crop:
            img_feature = self.feature_extractor(img.cuda())
        else:
            if self.easy:
                img_feature = self.feature_extractor(img.cuda())
            else:
                img_feature = self.feature_extractor(torch.stack(img, dim=0).cuda())
            
            img_feature = img_feature.permute(0, 2, 3, 1)
            img_feature = img_feature.view(img_feature.size(0), -1, img_feature.size(-1))
            # print(img_feature.shape)

            # img attention
        img_feature = self.em(img_feature)
        # print(img_feature.shape)
        # img_feature = self.conv(img_feature).view(-1, self.img_size_dim)
        if self.h_dim != self.out_dim:
            M = self.U(states).unsqueeze(1)
        else:
            M = states.unsqueeze(1)
        if self.easy:
            # print(M.shape)
            Q = self.WQ(M)
            K = self.WK(img_feature)
            V = self.WV(img_feature)
            logit = torch.bmm(Q, K.transpose(-2,-1))
            logit = logit / math.sqrt(self.out_dim)
            attention_weight = self.softmax(logit)
            output = torch.matmul(attention_weight , V)
            output = self.W_o(output)
            # output, attention_weight = self.model(M, img_feature, img_feature)
            # output = self.o(output)
            Att_Ph = output.squeeze()
            physical_attention = attention_weight.squeeze()
            # print(Att_Ph.shape, physical_attention.shape)

        else:
            batch = states.shape[0]
            # print(states.shape)
            Att_Ph = torch.ones(batch, self.out_dim).cuda()
            physical_attention = torch.ones(batch, self.img_size).cuda()
            for i in range(len(seq_start_end)):
                (start, end) = seq_start_end[i]
                num_ped = end - start
                N = M[start:end]
                I = img_feature[i].repeat(num_ped, 1, 1)
                Q = self.WQ(N)
                K = self.WK(I)
                V = self.WV(I)
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                attention_weight = self.softmax(logit)
                output = torch.matmul(attention_weight , V)
                output = self.W_o(output)

                # print(M.shape, K.shape)
                # output, attention_weight = self.model(N, K, K)
                # output = self.o(output)
                Att_Ph[start:end] = output.squeeze()
                physical_attention[start:end] = attention_weight.squeeze()
        
        return Att_Ph, physical_attention

class Physical_one_head_Attention_v2(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, 
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True,  attention_type='simple', easy=False):
        super(Physical_one_head_Attention_v2, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        # self.h_dim = hd_dim
        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size = 3136
            self.img_size_dim = 32
            self.feature_extractor = image_feature_extractor(3, 11)
            
        if attention_type=='simple':
            self.h_dim = he_dim
        else:
            self.h_dim = hd_dim
        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        if not self.setting_image:
            self.em = nn.Linear(self.img_size_dim, out_dim)
        

        if not self.setting_image:
            # self.o = nn.Linear(out_dim, out_dim)
            if self.h_dim != out_dim:
                self.U = nn.Linear(self.h_dim, out_dim)

            if usefulness == True:
                self.W_u = nn.Linear(self.h_dim+out_dim, 1)
                # self.W_u2 = nn.Linear(out_dim, 1)
                self.F_u = nn.Tanh()
                
            self.model = nn.MultiheadAttention(embed_dim=out_dim,
                                            num_heads=1,
                                            batch_first=True)
        else:
            #1280->1600, 640->400, 320->100
            self.img_size = 400
            # self.encoder_vgg19 = nn.Sequential(*list(self.encoder_vgg19.children())[:-1])
            self.img_size_dim = 512
            self.U = nn.Linear(self.h_dim, self.h_dim)
            self.W = nn.Linear(self.img_size_dim, self.h_dim)
            self.v = nn.Linear(self.h_dim, 1)
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(1)
            self.embed_pos = nn.Linear(2, self.h_dim)
            self.H = nn.Linear(self.img_size_dim, out_dim)

    def forward(self, img, states, seq_start_end, end_pos, prior=0, obs_traj=0, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        if self.pos_embed:
            pos = self.embedd(end_pos)
            states = states + pos

        heat_map = traj_heatmap(obs_traj, 224)
        # seg = preprocess_image_for_segmentation(img, seq_start_end, obs_traj)
        img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
        img_input = torch.cat([img_data, heat_map], dim=1)
        img_feature = self.feature_extractor(img_input)
        
        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = img_feature.view(img_feature.size(0), -1, img_feature.size(-1))
        # print(img_feature.shape)
        if self.h_dim != self.out_dim:
            M = self.U(states).unsqueeze(1)
        else:
            M = states.unsqueeze(1)

        output, attention_weight = self.model(M, img_feature, img_feature)
        Att_Ph  = output.squeeze()
        physical_attention = attention_weight.squeeze() 
        return Att_Ph, physical_attention

class Physical_Self_Attention(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3, 
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, use_seg=False,
                attention_type='simple', ph_prior_type='add', easy=False):
        super(Physical_Self_Attention, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        self.seg = use_seg
        # self.h_dim = hd_dim

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size = 3136
            self.img_size_dim = 32
            if self.ph_prior_type == 'pre_concat' or self.ph_prior_type == 'not_prior':
                self.feature_extractor = image_feature_extractor(k, 11)
            else:
                self.feature_extractor = image_feature_extractor(k, 3)

        if self.ph_prior_type == 'pre_concat' or self.ph_prior_type == 'not_prior':
            mlp_pre_attn_dims = [self.img_size * 32, 512, out_dim]
            self.W_k = nn.Linear(self.img_size_dim, out_dim)
        else:
            mlp_pre_attn_dims = [self.img_size * 40, 512, out_dim]
            self.W_k = nn.Linear(self.img_size_dim+8, out_dim)
        self.W_q = make_mlp(mlp_pre_attn_dims)
        # self.W_q = nn.Linear(self.img_size_dim+8+self.h_dim, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """

        batch = obs_traj.shape[1]
        if self.ph_prior_type == 'pre_concat' or self.ph_prior_type == 'not_prior':
            if self.easy:
                obs_traj[:,:,1] = 1. -  obs_traj[:,:,1]
                heat_map = traj_heatmap_v2(obs_traj, 224)
                img_data = img.cuda()
                img_input = torch.cat([img_data, heat_map], dim=1)
                # print(img_input.shape)
                img_feature = self.feature_extractor(img_input)
                # print(img_feature.shape)
                Fimg = img_feature.permute(0, 2, 3, 1)
                # print(Fimg.shape)
                Fimg = Fimg.view(Fimg.size(0), -1, Fimg.size(-1))

                q = Fimg.reshape(batch, self.img_size*self.img_size_dim)
                Q = self.W_q(q).unsqueeze(1)
                K = self.W_k(Fimg)
                V = self.W_v(Fimg)
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                phy_attn_weights = self.softmax(logit)
                output = torch.matmul(phy_attn_weights, V)
                output = self.W_o(output)
                Att_Ph = output.squeeze()
                physical_attention = phy_attn_weights.squeeze()

            
            else:
                heat_map = traj_heatmap_v2(obs_traj, 224)
                img_data = img.cuda()
                # img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
                img_input = torch.cat([img_data, heat_map], dim=1)
                # print(img_input.shape)
                img_feature = self.feature_extractor(img_input)
                # print(img_feature.shape)
                Fimg = img_feature.permute(0, 2, 3, 1)
                # print(Fimg.shape)
                Fimg = Fimg.view(Fimg.size(0), -1, Fimg.size(-1))

                q = Fimg.reshape(batch, self.img_size*self.img_size_dim)
                Q = self.W_q(q).unsqueeze(1)
                K = self.W_k(Fimg)
                V = self.W_v(Fimg)
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                if self.ph_prior_type == 'not_prior':
                    phy_attn_weights = self.softmax(logit)
                else:
                    phy_attn_weights = self.softmax(logit+prior.unsqueeze(1))
                output = torch.matmul(phy_attn_weights, V)
                output = self.W_o(output)
                Att_Ph = output.squeeze()
                physical_attention = phy_attn_weights.squeeze()
            # print(Fimg.shape)

        else:
            heat_map = traj_heatmap(obs_traj, 56).permute(0, 2, 3, 1)
            heat_map = heat_map.view(heat_map.size(0), -1, heat_map.size(-1))
            img_feature = self.feature_extractor(torch.stack(img, dim=0).cuda())
            Fimg = img_feature.permute(0, 2, 3, 1)
            Fimg = Fimg.view(batch, self.img_size, self.img_size_dim)

            Att_Ph = torch.ones(batch, self.out_dim).cuda()
            physical_attention = torch.ones(batch, self.img_size).cuda()
            for i in range(len(seq_start_end)):
                (start, end) = seq_start_end[i]
                num_ped = end - start
                P = prior[start:end].unsqueeze(1)
                img_ = Fimg[i].repeat(num_ped, 1, 1)
                _img = torch.cat([img_, heat_map[start:end]], dim=2)
                q = _img.view(num_ped, self.img_size*(self.img_size_dim+8))
                Q = self.W_q(q).unsqueeze(1)
                K = self.W_k(_img)
                V = self.W_v(img_)
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                phy_attn_weights = self.softmax(logit+P)
                output = torch.matmul(phy_attn_weights, V)
                output = self.W_o(output)
                Att_Ph[start:end] = output.squeeze()
                physical_attention[start:end] = phy_attn_weights.squeeze()
        return Att_Ph, physical_attention

class Physical_Self_Attention_v2(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, use_seg=False,
                attention_type='simple', ph_prior_type='add', easy=False):
        super(Physical_Self_Attention_v2, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        self.seg = use_seg
        # self.h_dim = hd_dim
        # if attention_type=='sophie':
        #     self.h_dim = hd_dim
        # else:
        #     self.h_dim = he_dim

        # if self.pos_embed:
        #     self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        
        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size = 3136
            self.img_size_dim = 32
            if self.ph_prior_type == 'pre_concat':
                self.feature_extractor = image_feature_extractor(k, 6)
            else:
                self.feature_extractor = image_feature_extractor(k, 3)

        if self.ph_prior_type == 'pre_concat':
            mlp_pre_attn_dims = [self.img_size * 32, 512, out_dim]
            self.W_k = nn.Linear(self.img_size_dim, out_dim)
        else:
            mlp_pre_attn_dims = [self.img_size * 40, 512, out_dim]
            self.W_k = nn.Linear(self.img_size_dim+8, out_dim)
        self.W_q = make_mlp(mlp_pre_attn_dims)
        # self.W_q = nn.Linear(self.img_size_dim+8+self.h_dim, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        batch = obs_traj.shape[1]
        if self.ph_prior_type == 'pre_concat':
            heat_map = traj_v_heatmap(obs_traj, 224)
            img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
            img_input = torch.cat([img_data, heat_map], dim=1)
            img_feature = self.feature_extractor(img_input)
            Fimg = img_feature.permute(0, 2, 3, 1)
            Fimg = Fimg.view(Fimg.size(0), -1, Fimg.size(-1))

            q = Fimg.reshape(batch, self.img_size*self.img_size_dim)
            Q = self.W_q(q).unsqueeze(1)
            K = self.W_k(Fimg)
            V = self.W_v(Fimg)
            logit = torch.bmm(Q, K.transpose(-2,-1))
            logit = logit / math.sqrt(self.out_dim)
            phy_attn_weights = self.softmax(logit+prior.unsqueeze(1))
            output = torch.matmul(phy_attn_weights, V)
            output = self.W_o(output)
            Att_Ph = output.squeeze()
            physical_attention = phy_attn_weights.squeeze()
        else:
            heat_map = traj_heatmap(obs_traj, 56).permute(0, 2, 3, 1)
            heat_map = heat_map.view(heat_map.size(0), -1, heat_map.size(-1))
            img_feature = self.feature_extractor(torch.stack(img, dim=0).cuda())
            Fimg = img_feature.permute(0, 2, 3, 1)
            Fimg = Fimg.view(Fimg.size(0), -1, Fimg.size(-1))

            Att_Ph = torch.ones(batch, self.out_dim).cuda()
            physical_attention = torch.ones(batch, self.img_size).cuda()
            for i in range(len(seq_start_end)):
                (start, end) = seq_start_end[i]
                num_ped = end - start
                P = prior[start:end].unsqueeze(1)
                img_ = Fimg[i].repeat(num_ped, 1, 1)
                _img = torch.cat([img_, heat_map[start:end]], dim=2)
                q = _img.view(num_ped, self.img_size*(self.img_size_dim+8))
                Q = self.W_q(q).unsqueeze(1)
                K = self.W_k(_img)
                V = self.W_v(img_)
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                phy_attn_weights = self.softmax(logit+P)
                output = torch.matmul(phy_attn_weights, V)
                output = self.W_o(output)
                Att_Ph[start:end] = output.squeeze()
                physical_attention[start:end] = phy_attn_weights.squeeze()
        return Att_Ph, physical_attention

class prior_Physical_Attention(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, use_seg=False,
                attention_type='simple', ph_prior_type='add', easy=False):
        super(prior_Physical_Attention, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        self.seg = use_seg
        # self.h_dim = hd_dim
        

        if attention_type=='sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        
        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        elif use_seg:
            self.img_size = 4096
            self.img_size_dim = 32
            self.feature_extractor = image_feature_extractor(k, 3)

        else:
            self.img_size = 3136
            self.img_size_dim = 32
            self.feature_extractor = image_feature_extractor(k, 3)

        self.W_q = nn.Linear(self.h_dim, out_dim)
        if self.ph_prior_type == 'nottraj_add' or self.ph_prior_type == 'nottraj_mul':
            self.W_k = nn.Linear(self.img_size_dim, out_dim)
        elif ph_prior_type == 'traj_add':
            self.W_k = nn.Linear(self.img_size_dim+8, out_dim)
        else:
            self.W_k = nn.Linear(self.img_size_dim+2, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        if ph_prior_type == 'sat':
            self.sat = nn.Linear(self.out_dim, 1)

    def func(self, end_pos, size):
        size = int(size ** (1/2))
        # print(size)
        C = torch.ones(2, size, size).cuda()
        C[0] = 1 - torch.abs(torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0])
        C[1] = 1 - torch.abs(torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1])
        return C

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        # print(end_pos.shape)
        if self.ph_prior_type == 'traj_add':
            heat_map = traj_heatmap(obs_traj, 56).permute(0, 2, 3, 1)
            heat_map = heat_map.view(heat_map.size(0), -1, heat_map.size(-1))
        
        if self.pos_embed:
            pos = self.embedd(end_pos)
            states = states + pos
        # img feature
        if self.easy:
            img_feature = self.feature_extractor(img.cuda())
        elif self.seg:
            seg = preprocess_image_for_segmentation(img, seq_start_end, obs_traj, False)
            img_feature = self.feature_extractor(seg)
            Fimg = img_feature.permute(0, 2, 3, 1)
            Fimg = Fimg.view(Fimg.size(0), -1, Fimg.size(-1))
        else:
            # print(img)
            # print('a',torch.stack(img, dim=0).cuda()[0])
            img_feature = self.feature_extractor(torch.stack(img, dim=0).cuda())
            # print('b',img_feature[0])
            
            Fimg = img_feature.permute(0, 2, 3, 1)
            Fimg = Fimg.view(Fimg.size(0), -1, Fimg.size(-1))
            # print(Fimg)
            
        M = states.unsqueeze(1)
        if self.easy:
            img_feature = Fimg
            
            # print(M.shape)
            # output, attention_weight = self.model(M, img_feature, img_feature)
            Q = self.W_q(M)
            K = self.W_k(img_feature)
            V = self.W_v(img_feature)
            logit = torch.bmm(Q, K.transpose(-2,-1))
            logit = logit / math.sqrt(self.out_dim)
            attn_weights = self.tanh(logit) 
            phy_attn_weights = self.softmax(attn_weights + prior.unsqueeze(1))
            output = torch.matmul(phy_attn_weights, V)

            output = self.W_o(output)
            Att_Ph = output.squeeze()
            physical_attention = phy_attn_weights.squeeze()
            # print(Att_Ph.shape, physical_attention.shape)

        else:
            batch = states.shape[0]
            # print(states.shape)
            Att_Ph = torch.ones(batch, self.out_dim).cuda()
            physical_attention = torch.ones(batch, self.img_size).cuda()
            if self.ph_prior_type == 'nottraj_add':
                for i in range(len(seq_start_end)):
                    (start, end) = seq_start_end[i]
                    num_ped = end - start
                    N = M[start:end]
                    P = prior[start:end].unsqueeze(1)
                    img_ = Fimg[i].repeat(num_ped, 1, 1)
                    
                    Q = self.W_q(N)
                    K = self.W_k(img_)
                    V = self.W_v(img_)

                    logit = torch.bmm(Q, K.transpose(-2,-1))
                    logit = logit / math.sqrt(self.out_dim)
                    # print(logit)
                    # attn_weights = self.softmax(logit)
                    phy_attn_weights = self.softmax(logit+P)
                    # print(phy_attn_weights)
                    # sys.exit()
                    # print((logit+P).max())
                    output = torch.matmul(phy_attn_weights, V)

                    output = self.W_o(output)
                    Att_Ph[start:end] = output.squeeze()
                    physical_attention[start:end] = phy_attn_weights.squeeze()
                
            elif self.ph_prior_type == 'nottraj_mul':
                for i in range(len(seq_start_end)):
                    (start, end) = seq_start_end[i]
                    num_ped = end - start
                    N = M[start:end]
                    P = prior[start:end].unsqueeze(1)
                    img_ = Fimg[i].repeat(num_ped, 1, 1)
                    
                    Q = self.W_q(N)
                    K = self.W_k(img_)
                    V = self.W_v(img_)

                    logit = torch.bmm(Q, K.transpose(-2,-1))
                    logit = logit / math.sqrt(self.out_dim)
                    # print(logit)
                    attn_weights = self.softmax(logit)
                    phy_attn_weights = self.softmax(attn_weights*P/tempreture)
                    # print(phy_attn_weights)
                    # sys.exit()
                    # print((logit+P).max())
                    output = torch.matmul(phy_attn_weights, V)

                    output = self.W_o(output)
                    Att_Ph[start:end] = output.squeeze()
                    physical_attention[start:end] = phy_attn_weights.squeeze()

            else:
                for i in range(len(seq_start_end)):
                    (start, end) = seq_start_end[i]
                    num_ped = end - start
                    N = M[start:end]
                    P = prior[start:end].unsqueeze(1)
        
                    if self.ph_prior_type == 'traj_add':
                        img_ = Fimg[i].repeat(num_ped, 1, 1)
                        K = self.W_k(torch.cat([img_, heat_map[start:end]], dim=2))
                        Q = self.W_q(N)
                        V = self.W_v(img_)
                        logit = torch.bmm(Q, K.transpose(-2,-1))
                        logit = logit / math.sqrt(self.out_dim)
                        phy_attn_weights = self.softmax(logit+P)
                        
                        output = torch.matmul(phy_attn_weights, V)
                        output = self.W_o(output)
                        Att_Ph[start:end] = output.squeeze()
                        physical_attention[start:end] = phy_attn_weights.squeeze()
                    else:
                        img_ = Fimg[i]
                        for j in range(num_ped):
                            C = self.func(end_pos[start+j], self.img_size)
                            C = C.view(-1, C.size(0))
                            # print(C.shape)
                            # print(img_.shape)
                            K = self.W_k(torch.cat([img_, C], dim=1)).unsqueeze(0)
                            Q = self.W_q(N[j].unsqueeze(0))
                            V = self.W_v(img_.unsqueeze(0))

                            if self.ph_prior_type == 'sat':
                                # print(K.shape, Q.shape)
                                logit = self.tanh(K + Q)
                                logit = self.sat(logit).transpose(1,2)
                                # print(logit.shape)
                            else:
                                logit = torch.bmm(Q, K.transpose(-2,-1))
                                logit = logit / math.sqrt(self.out_dim)
                            attn_weights = self.softmax(logit)

                            if self.ph_prior_type == 'add' or self.ph_prior_type == 'sat':
                                phy_attn_weights = self.softmax(logit+P[j].unsqueeze(0))

                            elif self.ph_prior_type == 'mul':
                                phy_attn_weights = self.softmax(attn_weights*P[j].unsqueeze(0))

                            elif self.ph_prior_type == 'one_head_attention':
                                phy_attn_weights = attn_weights

                            output = torch.matmul(phy_attn_weights, V)

                            output = self.W_o(output)
                            Att_Ph[start+j] = output.squeeze()
                            physical_attention[start+j] = phy_attn_weights.squeeze()
                            # physical_attention[start+j] = P[j]

        return Att_Ph, physical_attention

class prior_Physical_Attention_v2(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, attention_type='simple', ph_prior_type='add', easy=False):
        super(prior_Physical_Attention_v2, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        # self.h_dim = hd_dim
        

        if attention_type == 'sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        
        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size = 4096
            self.img_size_dim = 32
            self.feature_extractor = image_feature_extractor(k,11)

        self.W_q = nn.Linear(self.h_dim, out_dim)
        self.W_k = nn.Linear(self.img_size_dim, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        if ph_prior_type == 'sat':
            self.sat = nn.Linear(self.out_dim, 1)

    def func(self, end_pos, size):
        size = int(size ** (1/2))
        # print(size)
        C = torch.ones(2, size, size).cuda()
        C[0] = torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0]
        C[1] = torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1]
        return C

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        # print(end_pos.shape)
        # print(obs_traj.shape)
        
        heat_map = traj_heatmap_v2(obs_traj)
        seg = preprocess_image_for_segmentation(img, seq_start_end, obs_traj)
        img_input = torch.cat([seg, heat_map], dim=1)
        img_feature = self.feature_extractor(img_input)

        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = img_feature.view(img_feature.size(0), -1, img_feature.size(-1))

        M = states.unsqueeze(1)

        # print(M.shape)
        # output, attention_weight = self.model(M, img_feature, img_feature)
        Q = self.W_q(M)
        K = self.W_k(img_feature)
        V = self.W_v(img_feature)
        logit = torch.bmm(Q, K.transpose(-2,-1))/ math.sqrt(self.out_dim)
        logit = (logit + prior.unsqueeze(1)) 
        phy_attn_weights = self.softmax(logit)
        output = torch.matmul(phy_attn_weights, V)

        output = self.W_o(output)
        Att_Ph = output.squeeze()
        physical_attention = phy_attn_weights.squeeze()

        return Att_Ph, physical_attention

class prior_Physical_Attention_v3(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, attention_type='simple', ph_prior_type='not_seg', easy=False):
        super(prior_Physical_Attention_v3, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        # self.h_dim = hd_dim
        
        if attention_type == 'sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size_dim = 32
            if ph_prior_type == 'not_seg' or ph_prior_type == 'not_prior':
                self.img_size = 3136
                l=11
            elif ph_prior_type == 'vector':
                self.img_size = 3136
                l=6
            elif ph_prior_type == 'gauss':
                self.img_size = 3136
                l=4
            else:
                self.img_size = 4096
                l=14
            self.feature_extractor = image_feature_extractor(k,l)
    
        self.W_q = nn.Linear(self.h_dim, out_dim)
        self.W_k = nn.Linear(self.img_size_dim, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        if ph_prior_type == 'sat':
            self.sat = nn.Linear(self.out_dim, 1)

    def func(self, end_pos, size):
        size = int(size ** (1/2))
        # print(size)
        C = torch.ones(2, size, size).cuda()
        C[0] = torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0]
        C[1] = torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1]
        return C

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        if self.pos_embed:
            pos = self.embedd(end_pos)
            states = states + pos
        # print(end_pos.shape)
        # print(obs_traj.shape)
        if self.easy:
            obs_traj[:,:,1] = 1. - obs_traj[:,:,1]
            # print(obs_traj.shape)
        img_data = img.cuda()
        # img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
        if self.ph_prior_type == 'not_seg' or self.ph_prior_type == 'not_prior':
            heat_map = traj_heatmap_v2(obs_traj, 224)
            # print(heat_map.shape, img_data.shape, obj_traj.shape)
            img_input = torch.cat([img_data, heat_map], dim=1)
        elif self.ph_prior_type == 'vector':
            heat_map = traj_v_heatmap(obs_traj, 224)
            img_input = torch.cat([img_data, heat_map], dim=1)
        elif self.ph_prior_type == 'gauss':
            heat_map = gaussian_heatmap(obs_traj, 224)
            img_input = torch.cat([img_data, heat_map], dim=1)
        else:
            heat_map = traj_heatmap(obs_traj)
            seg = preprocess_image_for_segmentation(img, seq_start_end, obs_traj)
            img_input = torch.cat([img_data, seg, heat_map], dim=1)

        img_feature = self.feature_extractor(img_input)
        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = img_feature.view(img_feature.size(0), -1, img_feature.size(-1))

        M = states.unsqueeze(1)
        Q = self.W_q(M)
        K = self.W_k(img_feature)
        V = self.W_v(img_feature)
        logit = torch.bmm(Q, K.transpose(-2,-1))/ math.sqrt(self.out_dim)
        if (self.easy == False) and self.ph_prior_type != 'not_prior':
            # print(prior)
            logit = (logit + prior.unsqueeze(1)) 
        phy_attn_weights = self.softmax(logit)
        output = torch.matmul(phy_attn_weights, V)

        output = self.W_o(output)
        Att_Ph = output.squeeze()
        physical_attention = phy_attn_weights.squeeze()

        return Att_Ph, physical_attention

class prior_Physical_Attention_v4(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, attention_type='simple', ph_prior_type='not_seg', easy=False):
        super(prior_Physical_Attention_v4, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        # self.h_dim = hd_dim
        
        if attention_type == 'sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size_dim = 32
            if ph_prior_type == 'not_seg' or ph_prior_type == 'not_prior':
                self.img_size = 3136
                l=11
            elif ph_prior_type == 'vector':
                self.img_size = 3136
                l=6
            elif ph_prior_type == 'gauss':
                self.img_size = 3136
                l=4
            else:
                self.img_size = 4096
                l=14
            self.feature_extractor = image_feature_extractor(k,l)
            self.feature_extractor2 = image_feature_extractor(k,3)
    
        self.W_q = nn.Linear(self.h_dim, out_dim)
        self.W_k = nn.Linear(self.img_size_dim, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        if ph_prior_type == 'sat':
            self.sat = nn.Linear(self.out_dim, 1)

    def func(self, end_pos, size):
        size = int(size ** (1/2))
        # print(size)
        C = torch.ones(2, size, size).cuda()
        C[0] = torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0]
        C[1] = torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1]
        return C

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        if self.pos_embed:
            pos = self.embedd(end_pos)
            states = states + pos
        # print(end_pos.shape)
        # print(obs_traj.shape)
        if self.easy:
            obs_traj[:,:,1] = 1. - obs_traj[:,:,1]
            # print(obs_traj.shape)
        img_data = img.cuda()
        # img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
        if self.ph_prior_type == 'not_seg' or self.ph_prior_type == 'not_prior':
            heat_map = traj_heatmap_v2(obs_traj, 224)
            # print(heat_map.shape, img_data.shape, obj_traj.shape)
            img_input = torch.cat([img_data, heat_map], dim=1)
        elif self.ph_prior_type == 'vector':
            heat_map = traj_v_heatmap(obs_traj, 224)
            img_input = torch.cat([img_data, heat_map], dim=1)
        elif self.ph_prior_type == 'gauss':
            heat_map = gaussian_heatmap(obs_traj, 224)
            img_input = torch.cat([img_data, heat_map], dim=1)
        else:
            heat_map = traj_heatmap(obs_traj)
            seg = preprocess_image_for_segmentation(img, seq_start_end, obs_traj)
            img_input = torch.cat([img_data, seg, heat_map], dim=1)
        

        img_feature = self.feature_extractor(img_input)
        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = img_feature.view(img_feature.size(0), -1, img_feature.size(-1))

        Value = self.feature_extractor2(img_data)
        Value = Value.permute(0, 2, 3, 1)
        Value = Value.view(img_feature.size(0), -1, Value.size(-1))

        M = states.unsqueeze(1)
        Q = self.W_q(M)
        K = self.W_k(img_feature)
        V = self.W_v(Value)
        logit = torch.bmm(Q, K.transpose(-2,-1))/ math.sqrt(self.out_dim)
        if (self.easy == False) and self.ph_prior_type != 'not_prior':
            # print(prior)
            logit = (logit + prior.unsqueeze(1)) 
        phy_attn_weights = self.softmax(logit)
        output = torch.matmul(phy_attn_weights, V)

        output = self.W_o(output)
        Att_Ph = output.squeeze()
        physical_attention = phy_attn_weights.squeeze()

        return Att_Ph, physical_attention

class prior_Physical_Attention_v5(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, attention_type='simple', ph_prior_type='not_seg', easy=False):
        super(prior_Physical_Attention_v5, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        # self.h_dim = hd_dim
        
        if attention_type == 'sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size_dim = 32
            self.img_size = 3136
            self.feature_extractor = image_feature_extractor(k,8)
            self.feature_extractor2 = image_feature_extractor(k,3)

        if ph_prior_type == 'Q_is_T':
            mlp_pre_attn_dims = [self.img_size * self.img_size_dim, out_dim]
            self.W_q = make_mlp(mlp_pre_attn_dims)
        else:
            self.W_q = nn.Linear(self.h_dim, out_dim)
        self.W_k = nn.Linear(self.img_size_dim*2, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        # if ph_prior_type == 'sat':
        #     self.sat = nn.Linear(self.out_dim, 1)

    def func(self, end_pos, size):
        size = int(size ** (1/2))
        # print(size)
        C = torch.ones(2, size, size).cuda()
        C[0] = torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0]
        C[1] = torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1]
        return C

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        batch = obs_traj.shape[1]
        if self.ph_prior_type != 'Q_is_T' and self.pos_embed:
            pos = self.embedd(end_pos)
            states = states + pos
        # print(end_pos.shape)
        # print(obs_traj.shape)
        if self.easy:
            obs_traj[:,:,1] = 1. - obs_traj[:,:,1]
            # print(obs_traj.shape)
        img_data = img.cuda()
        _input = traj_heatmap_v2(obs_traj, 224)
        # img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
        

        Tr = self.feature_extractor(_input)
        Tr = Tr.permute(0, 2, 3, 1)
        Tr = Tr.view(Tr.size(0), -1, Tr.size(-1))

        Value = self.feature_extractor2(img_data)
        Value = Value.permute(0, 2, 3, 1)
        Value = Value.view(Value.size(0), -1, Value.size(-1))
        
        if self.ph_prior_type != 'Q_is_T':
            M = states
        else:
            M = Tr.reshape(batch, self.img_size*self.img_size_dim)

        Q = self.W_q(M).unsqueeze(1)
        key = torch.cat([Value, Tr], dim=2)
        K = self.W_k(key)
        V = self.W_v(Value)
        logit = torch.bmm(Q, K.transpose(-2,-1))/ math.sqrt(self.out_dim)
        if (self.easy == False):
            # print(prior)
            logit = (logit + prior.unsqueeze(1)) 
        phy_attn_weights = self.softmax(logit)
        output = torch.matmul(phy_attn_weights, V)

        output = self.W_o(output)
        Att_Ph = output.squeeze()
        physical_attention = phy_attn_weights.squeeze()

        return Att_Ph, physical_attention

class prior_Physical_Attention_v6(nn.Module):
    def __init__(self, he_dim, hd_dim, out_dim=49, embedd_dim=16, center_crop=False, norm=1, k=3,
                pos_embed=False, img_embed=False, setting_image=False, usefulness=False,  vgg_train=False, add_input=False, use_vgg=True, attention_type='simple', ph_prior_type='not_seg', easy=False):
        super(prior_Physical_Attention_v6, self).__init__()

        self.center_crop = center_crop
        self.setting_image = setting_image
        self.norm = norm
        self.pos_embed = pos_embed
        self.img_embed = img_embed
        self.out_dim = out_dim
        self.usefulness = usefulness
        self.add_input = add_input
        self.ph_prior_type = ph_prior_type
        # self.h_dim = he_dim
        self.attention_type = attention_type
        self.easy = easy
        self.pos_embed = pos_embed
        # self.h_dim = hd_dim
        
        if attention_type == 'sophie':
            self.h_dim = hd_dim
        else:
            self.h_dim = he_dim

        if self.pos_embed:
            self.embedd = nn.Linear(2, self.h_dim)

        # self.em = nn.Linear(self.img_size_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        if use_vgg:
            self.img_size = 784
            self.img_size_dim = 512
            self.feature_extractor = module.vgg19_bn(pretrained=True).features
            for param in self.feature_extractor.parameters():
                param.requires_grad = vgg_train
        else:
            self.img_size_dim = 32
            self.img_size = 3136
            self.feature_extractor = image_feature_extractor(k,8)
            self.feature_extractor2 = image_feature_extractor(k,3)

        mlp_pre_attn_dims = [self.img_size * self.img_size_dim, out_dim]
        self.W_q = make_mlp(mlp_pre_attn_dims)
        self.W_k = nn.Linear(self.img_size_dim, out_dim)
        self.W_v = nn.Linear(self.img_size_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        # if ph_prior_type == 'sat':
        #     self.sat = nn.Linear(self.out_dim, 1)

    def func(self, end_pos, size):
        size = int(size ** (1/2))
        # print(size)
        C = torch.ones(2, size, size).cuda()
        C[0] = torch.linspace(0,1,size).repeat(size,1,1).squeeze().cuda()  - end_pos[0]
        C[1] = torch.linspace(0,1,size).repeat(size,1,1).T.squeeze().cuda() - end_pos[1]
        return C

    def forward(self, img, states, seq_start_end, end_pos, prior, obs_traj, tempreture=1):
        """
        Inputs:
        - img
        - states
        - seq_start_end
        - end_pos
        Output:
        - Att_Ph
        - physical_attention
        """
        batch = obs_traj.shape[1]
        # print(end_pos.shape)
        # print(obs_traj.shape)
        if self.easy:
            obs_traj[:,:,1] = 1. - obs_traj[:,:,1]
            # print(obs_traj.shape)
        img_data = img.cuda()
        _input = traj_heatmap_v2(obs_traj, 224)
        # img_data = image_per_person(img, seq_start_end, obs_traj, img[0].shape[1])
        

        Tr = self.feature_extractor(_input)
        Tr = Tr.permute(0, 2, 3, 1)
        Tr = Tr.view(Tr.size(0), -1, Tr.size(-1))

        Value = self.feature_extractor2(img_data)
        Value = Value.permute(0, 2, 3, 1)
        Value = Value.view(Value.size(0), -1, Value.size(-1))
  
        M = Tr.reshape(batch, self.img_size*self.img_size_dim)

        Q = self.W_q(M).unsqueeze(1)
        # key = torch.cat([Value, Tr], dim=2)
        K = self.W_k(Value)
        V = self.W_v(Value)
        logit = torch.bmm(Q, K.transpose(-2,-1))/ math.sqrt(self.out_dim)
        if (self.easy == False):
            # print(prior)
            logit = (logit + prior.unsqueeze(1)) 
        phy_attn_weights = self.softmax(logit)
        output = torch.matmul(phy_attn_weights, V)

        output = self.W_o(output)
        Att_Ph = output.squeeze()
        physical_attention = phy_attn_weights.squeeze()

        return Att_Ph, physical_attention

class Social_Self_Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, embedd_dim, out_dim, multiplier=1, pos_embed=False, attention_type='simple', easy=False):
        super(Social_Self_Attention, self).__init__()

        self.out_dim = out_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.pos_embed = pos_embed
        self.easy = easy

        self.o = nn.Linear(encoder_dim, encoder_dim)

        if pos_embed:
            self.embedd = nn.Linear(2, encoder_dim)

        if encoder_dim != out_dim:
            self.v = nn.Linear(encoder_dim, out_dim)
        depth = float(encoder_dim)
        self.model = nn.MultiheadAttention(embed_dim=encoder_dim,
                                            num_heads=1,
                                            batch_first=True)
        if attention_type=='sophie' and decoder_dim != encoder_dim:
            self.e = nn.Linear(decoder_dim, encoder_dim)
        self.attention_type = attention_type

        self.restrict_qk = depth ** (-1 * multiplier)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, encoder_states, decoder_states, seq_start_end, prior=1, end_pos=1, T=1):
        """
        Inputs:
        - V_so
        - H_agent
        - seq_start_end
        - n_max
        Output:
        - Att_so
        - social_attention
        """
        batch = encoder_states.shape[0]
        if self.pos_embed:
            pos = self.embedd(end_pos)
            encoder_states = encoder_states + pos
        if self.attention_type == 'sophie':
            if self.decoder_dim != self.encoder_dim:
                M = self.e(decoder_states).unsqueeze(0)
            else:
                M = decoder_states.unsqueeze(0)
        else:
            M = encoder_states.unsqueeze(0)
        encoder_state = encoder_states.unsqueeze(0)


        social_attention = torch.ones(batch, 100).cuda()
        Att_So = torch.zeros(batch, self.out_dim).cuda()
        for i in range(len(seq_start_end)):
            (start, end) = seq_start_end[i]
            num_ped = end - start
            zero_pad = nn.ZeroPad2d((0, 100-num_ped, 0, 0))
            Q = M[:,start:end]
            K = encoder_state[:,start:end]
            output, attention_weight = self.model(Q, K, K)
            output = output + Q

            social_attention[start:end] =  zero_pad(attention_weight).squeeze()
            if self.encoder_dim != self.out_dim:
                Att_So[start:end] = self.v(output.squeeze())
            else:
                Att_So[start:end] = output.squeeze()

        return Att_So, social_attention

class prior_Social_Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, embedd_dim, out_dim, multiplier=1, pos_embed=False, attention_type='simple', so_prior_type='add'):
        super(prior_Social_Attention, self).__init__()

        self.out_dim = out_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.pos_embed = pos_embed
        self.so_prior_type = so_prior_type
        # self.model = nn.MultiheadAttention(embed_dim=encoder_dim,
        #                             num_heads=1,
        #                             batch_first=True)

        if attention_type=='simple':
            self.h_dim = encoder_dim
        else:
            self.h_dim = decoder_dim

        self.o = nn.Linear(encoder_dim, encoder_dim)

        if pos_embed:
            self.embedd = nn.Linear(2, encoder_dim)

        if encoder_dim != out_dim:
            self.v = nn.Linear(encoder_dim, out_dim)
                                            
        if attention_type =='sophie' and decoder_dim != encoder_dim:
            self.e = nn.Linear(decoder_dim, encoder_dim)
        self.attention_type = attention_type

        self.W_q = nn.Linear(self.h_dim, out_dim)
        self.W_k = nn.Linear(self.h_dim, out_dim)
        self.W_v = nn.Linear(self.h_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.softmax = nn.Softmax(2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoder_states, decoder_states, seq_start_end, prior, end_pos, T):
        """
        Inputs:
        - V_so
        - H_agent
        - seq_start_end
        - n_max
        Output:
        - Att_so
        - social_attention
        """
        batch = encoder_states.shape[0]
        if self.pos_embed:
            pos = self.embedd(end_pos)
            encoder_states = encoder_states + pos

        encoder_state = encoder_states.unsqueeze(0)

        social_attention = torch.ones(batch, 120).cuda()
        Att_So = torch.zeros(batch, self.out_dim).cuda()

        for i in range(len(seq_start_end)):
            (start, end) = seq_start_end[i]
            num_ped = end - start
            zero_pad = nn.ZeroPad2d((0, 120-num_ped, 0, 0))
            N = encoder_state[:,start:end]
            P = prior[start:end, :num_ped].unsqueeze(0)
            # print(P.shape)
            if self.so_prior_type != 'original':
                Q = self.W_q(N)
                K = self.W_k(N)
                
            V = self.W_v(N)

            if self.so_prior_type == 'add':
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                so_attn_weights = self.softmax(logit+P)

            elif self.so_prior_type == 'mul':
                logit = torch.bmm(Q, K.transpose(-2,-1))
                if T == 1:
                    depth = 1
                else:
                    # depth = T
                    if num_ped/5 < 1:
                        depth = 0.25
                    else:
                        r = num_ped/5
                        depth = (T)**(r)
                    if depth < 0.0001:
                        depth = 0.0001

                logit = logit / math.sqrt(self.out_dim)  
                # logit = logit + (torch.abs(torch.min(logit, dim=1)[0]))
                logit = self.softmax(logit)
                so_attn_weights = self.softmax((logit*P)/(depth))

            elif self.so_prior_type == 'mul_key':
                K = torch.matmul(P, K)
                logit = torch.bmm(Q, K.transpose(-2,-1))
                logit = logit / math.sqrt(self.out_dim)
                attn_weights = self.softmax(logit)
                so_attn_weights = attn_weights

            elif self.so_prior_type == 'original':
                so_attn_weights = P

            output = torch.matmul(so_attn_weights, V)
            output = self.W_o(output)
            output = N + output

            Att_So[start:end] = output.squeeze()
            social_attention[start:end] = zero_pad(so_attn_weights).squeeze()
            # social_attention = prior
            # if self.so_prior_type == 'add':
            #     social_attention[start:end] *= zero_pad(logit).squeeze()
            # else:
            #     social_attention[start:end] *= zero_pad(attn_weights).squeeze()

        return Att_So, social_attention

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(self, embedding_dim=64, h_dim=64):
        super(Encoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTMCell(embedding_dim, h_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(batch, self.h_dim).cuda(),
            torch.zeros(batch, self.h_dim).cuda()
        )

    def forward(self, encoder_input, state_tuple):
        """
        Inputs:
        - encoder_input: Tensor of shape (batch, 2)
        - state_tuple
        Output:
        - state
        """
        # Encode observed Trajectory
        state = self.encoder(encoder_input, state_tuple)
        return state

class Middle_Layer(nn.Module):
    def __init__(self, encoder_h_dim=32, decoder_h_dim=64, embedd_dim=16, mlp_dim=128, noise_dim=(0, ), noise_type='gaussian', norm=1, kernel_size=3,
                att_ph_dim=49, att_so_dim=32, phy_tempreture=1, so_tempreture=1, n_max=12, attention_type='simple', social_attention_type='simple', physical_attention_type='simple',
                so_prior_type='add', ph_prior_type='add', multiplier=1, physical_pos_embed=False, physical_img_embed=False, social_pos_embed=False, cell_pad=False,
                center_crop=False, compress_attention=False, concat_state=True, usefulness=False, use_vgg=True, use_seg=False, vgg_train=False, add_input=False, easy=False, setting_image=False, artificial_social_attention=False):

        super(Middle_Layer, self).__init__()

        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim

        self.att_ph_dim = att_ph_dim
        self.att_so_dim = att_so_dim

        self.concat_state = concat_state
        self.cell_pad = cell_pad

        # dimension check
        # if (compress_attention ==False and (concat_state == True and decoder_h_dim != encoder_h_dim + att_ph_dim + att_so_dim) or (concat_state == False and decoder_h_dim != att_ph_dim + att_so_dim)) or (att_ph_dim + att_so_dim == 0 and decoder_h_dim != encoder_h_dim) :
        #     print('[Attention!] Check the dimension of the decoder. ')

        self.social_attention_type = social_attention_type
        self.physical_attention_type = physical_attention_type
        self.attention_type = attention_type
        self.compress_attention = compress_attention

        if att_so_dim == att_ph_dim == 0 and encoder_h_dim != decoder_h_dim:
            self.E2D_embedding = nn.Linear(encoder_h_dim, decoder_h_dim)
        
        if att_ph_dim != 0:
            
            if physical_attention_type == "one_head_attention":
                self.physical_attention = Physical_one_head_Attention(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    easy=easy
                )
            elif physical_attention_type == "one_head_attention2":
                self.physical_attention = Physical_one_head_Attention_v2(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    easy=easy
                )

            elif physical_attention_type == "self_attention":
                self.physical_attention = Physical_Self_Attention(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )

            elif physical_attention_type == "self_attention2":
                self.physical_attention = Physical_Self_Attention_v2(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )
                
            elif physical_attention_type == "prior":
                self.physical_attention = prior_Physical_Attention(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    k=kernel_size,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    use_seg=use_seg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )
            elif physical_attention_type == "prior2":
                self.physical_attention = prior_Physical_Attention_v2(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    k=kernel_size,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )

            elif physical_attention_type == "prior3":
                self.physical_attention = prior_Physical_Attention_v3(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    k=kernel_size,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )
            elif physical_attention_type == "prior4":
                self.physical_attention = prior_Physical_Attention_v4(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    k=kernel_size,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )

            elif physical_attention_type == "prior5":
                self.physical_attention = prior_Physical_Attention_v5(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    k=kernel_size,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )

            elif physical_attention_type == "prior6":
                self.physical_attention = prior_Physical_Attention_v6(
                    he_dim=encoder_h_dim, 
                    hd_dim=decoder_h_dim, 
                    embedd_dim=embedd_dim,
                    out_dim=att_ph_dim,
                    center_crop=center_crop,
                    norm=norm,
                    k=kernel_size,
                    pos_embed=physical_pos_embed,
                    img_embed=physical_img_embed,
                    setting_image=setting_image,
                    usefulness=usefulness,
                    vgg_train=vgg_train, 
                    add_input=add_input,
                    use_vgg=use_vgg,
                    attention_type=attention_type,
                    ph_prior_type=ph_prior_type,
                    easy=easy
                )

        if att_so_dim != 0:
            if social_attention_type == 'self_attention':
                self.social_attention = Social_Self_Attention(
                    encoder_dim=encoder_h_dim,
                    decoder_dim=decoder_h_dim,
                    embedd_dim=embedd_dim,
                    out_dim=att_so_dim,
                    multiplier=multiplier,
                    pos_embed=social_pos_embed,
                    attention_type=attention_type
                )
            
            elif social_attention_type == 'prior':
                self.social_attention = prior_Social_Attention(
                    encoder_dim=encoder_h_dim,
                    decoder_dim=decoder_h_dim,
                    embedd_dim=embedd_dim,
                    out_dim=att_so_dim,
                    multiplier=multiplier,
                    pos_embed=social_pos_embed,
                    attention_type=attention_type,
                    so_prior_type=so_prior_type
                )


        self.noise = False
        if noise_dim == 0:
            noise_first_dim = 0
        else:
            self.noise = True
            self.noise_dim = noise_dim
            noise_first_dim = noise_dim
            self.noise_type = noise_type

        if type(noise_dim) is tuple:
            self.noise = False
        
        if self.compress_attention:
            self.comp = nn.Linear(att_ph_dim + att_so_dim + encoder_h_dim, decoder_h_dim)
        
        self.need_mlp_attention = False
        
        if self.noise:
            if att_ph_dim + att_so_dim + noise_first_dim == decoder_h_dim:
                self.need_mlp_attention = False
            else:
                self.need_mlp_attention = True
                if self.concat_state:
                    mlp_decoder_context_dims = [att_ph_dim + att_so_dim + encoder_h_dim, mlp_dim, decoder_h_dim - noise_first_dim]
                else:
                    mlp_decoder_context_dims = [att_ph_dim + att_so_dim, mlp_dim, decoder_h_dim - noise_first_dim]
                
                self.mlp_attention = make_mlp(mlp_decoder_context_dims)
        
        self.so_tempreture = so_tempreture
        self.phy_tempreture = phy_tempreture
        self.center_crop = center_crop

    def add_noise(self, _input):
        npeds = _input.size(0)
        noise_shape = (self.noise_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, -1).repeat(npeds, 1)
        return torch.cat((_input, vec), dim=1)

    def forward(self, image, encoder_state, seq_start_end, last_pos, obs_traj, decoder_state=None, V_so=0, edge=0, so_prior=0, ph_prior=0, recurrent_attention=False):
        """
        Inputs:
        - image
        - encoder_state
        - decoder_state
        - seq_start_end
        - last_pos
        - Att_ph
        - recurrent_physical_attention
        Output:
        - decoder_state
        - physical_attention
        - social_attention
        - agent_num
        - Att_ph
        """
        physical_attention, social_attention, agent_num = 0, 0, 0
        Att_so = 0
        Att_ph = 0

        if self.att_so_dim == self.att_ph_dim == 0:
            if self.encoder_h_dim != self.decoder_h_dim:
                state = self.E2D_embedding(encoder_state[0])
            else:
                state = encoder_state[0]
            decoder_state = (state, decoder_state[1])
        
        else:
            
            # Social Attention
            if self.att_so_dim != 0:
                Att_so, social_attention = self.social_attention(encoder_state[0], decoder_state[0], seq_start_end, so_prior, last_pos, self.so_tempreture)

            # Physical Attention
            if self.att_ph_dim != 0:
                Att_ph, physical_attention = self.physical_attention(image, encoder_state[0], seq_start_end, last_pos, ph_prior, obs_traj, self.phy_tempreture)

            if decoder_state != None:
                if self.concat_state:
                    if self.att_so_dim == 0:
                        attention = torch.cat([encoder_state[0], Att_ph], dim=1)
                    elif self.att_ph_dim == 0:
                        attention = torch.cat([encoder_state[0], Att_so], dim=1)
                    else:
                        attention = torch.cat([encoder_state[0], Att_so, Att_ph], dim=1)
                else:
                    if self.att_so_dim == 0:
                        attention = Att_ph
                    elif self.att_ph_dim == 0:
                        attention = Att_so
                    else:
                        attention = torch.cat([Att_so, Att_ph], dim=1)
                
                if self.compress_attention:
                    attention = self.comp(attention)
                
                # Noise add
                if not self.noise:
                    noise_tensor = attention
                else:
                    if self.need_mlp_attention:
                        attention = self.mlp_attention(attention)
                    noise_tensor = self.add_noise(attention)
                if self.cell_pad:
                    decoder_state = (noise_tensor, noise_tensor)
                else:
                    decoder_state = (noise_tensor, decoder_state[1])

        return decoder_state, physical_attention, social_attention, agent_num, V_so, edge

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, embedding_dim=64, h_dim=128
    ):
        super(Decoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTMCell(embedding_dim, h_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(batch, self.h_dim).cuda(),
            torch.zeros(batch, self.h_dim).cuda()
        )

    def forward(self, decoder_input, state_tuple):
        """
        Inputs:
        - decoder_input
        - state_tuple: (hh, ch) each tensor of shape (batch, h_dim)
        Output:
        - state
        """
        # Predict Trajectory Decode
        state = self.decoder(decoder_input, state_tuple)
        return state

class TrajectoryLSTM(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, 
        encoder_h_dim=64, g_type='traj', recurrent_graph=False, visualize=False
    ):
        super(TrajectoryLSTM, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.g_type = g_type
        self.recurrent_graph = recurrent_graph
        self.visualize_process = visualize
        self.encoder_h_dim = encoder_h_dim
        self.visualize=visualize

        if g_type == 'traj_both':
            self.encoder_embedding = nn.Linear(4, embedding_dim)
        else:
            self.encoder_embedding = nn.Linear(2, embedding_dim)
        
        if g_type == 'self_attention':
            self.self_attention = Social_Self_Attention(
                encoder_dim=encoder_h_dim,
                decoder_dim=encoder_h_dim,
                embedd_dim=embedding_dim,
                out_dim=encoder_h_dim,
                multiplier=2,
                pos_embed=False
            )
        
        if g_type == 'gcn':
            self.gcn = GCN(encoder_h_dim, encoder_h_dim*2, g_type)
        elif g_type == 'embed_gcn' or g_type =='embed_gcn_traj':
            self.gcn = GCN(2, embedding_dim, g_type)
        
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim
        )
        
        self.predictor = nn.Linear(encoder_h_dim, 2)

    def forward(self, img, obs_traj, obs_traj_rel, seq_start_end):
        """
        Inputs:
        - img
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        relation between different types of noise and outputs.
        Output:
        - pred_traj_fake
        - pred_traj_fake_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        last_pos = obs_traj[-1].clone()
        last_pos_rel = obs_traj_rel[-1].clone()
        if self.g_type == 'traj_both':
            obs_traj_both = torch.cat([obs_traj, obs_traj_rel], dim=2)

        # encoder state
        encoder_state = self.encoder.init_hidden(batch)
        
        # predict output
        pred_traj_fake = []
        pred_traj_fake_rel = []

        for i in range(self.seq_len):
            # print(i)
            if i < self.obs_len:
                if self.g_type == 'gcn' or  self.g_type == 'embed_gcn' or  self.g_type == 'embed_gcn_traj':
                    if self.recurrent_graph:
                        edge = make_graph(obs_traj[i], seq_start_end)
                    elif i == 0 and self.recurrent_graph==False:
                        edge = make_graph(last_pos, seq_start_end)
                else:
                    edge=None
                    # print(edge)
                # Encode seq
                if self.g_type == 'traj' or self.g_type == 'traj_res':
                    encoder_state = self.encoder(self.encoder_embedding(obs_traj[i]), encoder_state)
                
                if self.g_type == 'traj_rel' or self.g_type == 'self_attention':
                    encoder_state = self.encoder(self.encoder_embedding(obs_traj_rel[i]), encoder_state)

                if self.g_type == 'traj_both':
                    encoder_state = self.encoder(self.encoder_embedding(obs_traj_both[i]), encoder_state)
                
                if self.g_type == 'embed_gcn_traj':
                    data = Data(x=obs_traj[i], edge_index=edge).cuda()
                    gcn_state, gcn_attention = self.gcn(data)
                    encoder_state = self.encoder(gcn_state, encoder_state)

                if self.g_type == 'gcn':
                    encoder_state = self.encoder(self.encoder_embedding(obs_traj_rel[i]), encoder_state)
                    data = Data(x=encoder_state[0], edge_index=edge).cuda()
                    gcn_state, gcn_attention = self.gcn(data)
                    encoder_state = (gcn_state, encoder_state[1])
                
                if self.g_type == 'embed_gcn':
                    data = Data(x=obs_traj_rel[i], edge_index=edge).cuda()
                    gcn_state, gcn_attention = self.gcn(data)
                    encoder_state = self.encoder(gcn_state, encoder_state)

            else:
                # Predict Trajectory
                if self.g_type == 'traj':
                    pred_pos = self.predictor(encoder_state[0])
                    pred_traj_fake.append(pred_pos)
                    pred_traj_fake_rel.append(pred_pos - last_pos)

                    last_pos_rel = pred_pos - last_pos
                    last_pos = pred_pos
                    encoder_state = self.encoder(self.encoder_embedding(last_pos), encoder_state)

                if self.g_type == 'traj_rel':
                    last_pos_rel = self.predictor(encoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    
                    last_pos = last_pos + last_pos_rel 
                    encoder_state = self.encoder(self.encoder_embedding(last_pos_rel), encoder_state)
                
                if self.g_type == 'self_attention':
                    Att_so, social_attention = self.self_attention(encoder_state[0], encoder_state[0], seq_start_end, last_pos)
                    last_pos_rel = self.predictor(Att_so)
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    
                    last_pos = last_pos + last_pos_rel 
                    encoder_state = self.encoder(self.encoder_embedding(last_pos_rel), encoder_state)
                
                if self.g_type == 'traj_both':
                    last_pos_rel = self.predictor(encoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 
                    encoder_state = self.encoder(self.encoder_embedding(torch.cat([last_pos, last_pos_rel],dim=1)), encoder_state)

                if self.g_type == 'traj_res':
                    last_pos_rel = self.predictor(encoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 
                    encoder_state = self.encoder(self.encoder_embedding(last_pos), encoder_state)   

                if self.g_type == 'embed_gcn_traj':
                    pred_pos = self.predictor(encoder_state[0])
                    pred_traj_fake.append(pred_pos)
                    pred_traj_fake_rel.append(pred_pos - last_pos)
                    last_pos_rel = pred_pos - last_pos
                    last_pos = pred_pos
                    if self.recurrent_graph:
                        edge = make_graph(last_pos, seq_start_end)
                    data = Data(x=last_pos, edge_index=edge).cuda()
                    gcn_state, _ = self.gcn(data)
                    encoder_state = self.encoder(gcn_state, encoder_state)

                if self.g_type == 'gcn':
                    last_pos_rel = self.predictor(encoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 
                    if self.recurrent_graph:
                        edge = make_graph(last_pos, seq_start_end)
                    encoder_state = self.encoder(self.encoder_embedding(last_pos_rel), encoder_state)
                    data = Data(x=encoder_state[0], edge_index=edge).cuda()
                    gcn_state, _ = self.gcn(data)
                    encoder_state = (gcn_state, encoder_state[1])   
                
                if self.g_type == 'embed_gcn':
                    last_pos_rel = self.predictor(encoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 
                    if self.recurrent_graph:
                        edge = make_graph(last_pos, seq_start_end)
                    data = Data(x=last_pos_rel, edge_index=edge).cuda()
                    gcn_state, _ = self.gcn(data)
                    encoder_state = self.encoder(gcn_state, encoder_state)
        # print(pred_traj_fake[0]==pred_traj_fake[11])
        pred_traj_fake = torch.stack(pred_traj_fake, dim=0).cuda()
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0).cuda()
        # print(pred_traj_fake_rel)
        
        if (self.g_type == 'gcn' or  self.g_type == 'embed_gcn' or  self.g_type == 'embed_gcn_traj') and self.visualize: 
            return pred_traj_fake, pred_traj_fake_rel, gcn_attention
        else:
            return pred_traj_fake, pred_traj_fake_rel

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64, decoder_h_dim=128, mlp_dim=128, noise_dim=(0, ), n_max=32, 
        noise_type='gaussian', noise_mix_type='ped', att_ph_dim=16, att_so_dim=16, center_crop=False, crop_img_size=512, norm=1, kernel_size=3,
        attention_type='simple', social_attention_type='simple',  physical_attention_type='simple', 
        so_prior_type='add',  ph_prior_type='add', multiplier=1, physical_pos_embed=False, physical_img_embed=False, social_pos_embed=False,
        recurrent_attention=False, recurrent_physical_attention=False, input_recurrent_attention=False, cell_pad=False,
        phy_tempreture=1., so_tempreture=1., input_to_decoder=False, ge_type='traj_rel', gd_type='traj_rel', recurrent_graph=False, 
        setting_image=False, compress_attention=False, concat_state=True, usefulness=False, use_vgg=True, use_seg=False, vgg_train=False, add_input=False,
        easy=False, artificial_social_attention=False, visualize=False):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.recurrent_attention = recurrent_attention
        self.recurrent_physical_attention = recurrent_physical_attention
        self.ge_type = ge_type
        self.gd_type = gd_type
        self.input_to_decoder = input_to_decoder
        self.visualize_process = visualize
        self.setting_image = setting_image
        self.concat_state = concat_state
        self.input_recurrent_attention = input_recurrent_attention
        self.recurrent_graph = recurrent_graph
        self.attention_type = attention_type
        # if social_attention_type == 'sophie':
        #     # self.input_to_decoder = True
        self.att_ph_dim = att_ph_dim
        self.att_so_dim = att_so_dim
        self.n_max = n_max
        
        self.s_type = social_attention_type
        self.p_type = physical_attention_type

        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim

        self.decoder_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(decoder_h_dim, 2)
        if self.encoder_h_dim != 0:
            self.encoder = Encoder(
                embedding_dim=embedding_dim,
                h_dim=encoder_h_dim
            )
            if ge_type == 'traj_both':
                self.encoder_embedding = nn.Linear(4, embedding_dim)
            else:
                self.encoder_embedding = nn.Linear(2, embedding_dim)

            if ge_type == 'embed_gcn' or ge_type =='embed_gcn_traj':
                self.gcn = GCN(2, embedding_dim, ge_type)
            if gd_type == 'embed_gcn' or gd_type =='embed_gcn_traj':
                self.d_gcn = GCN(2, embedding_dim, ge_type)


        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim
        )

        self.Middle_Layer = Middle_Layer(
            encoder_h_dim=encoder_h_dim, 
            decoder_h_dim=decoder_h_dim,
            embedd_dim=embedding_dim,
            mlp_dim=mlp_dim,
            noise_dim=noise_dim,
            noise_type=noise_type,
            norm=norm,
            att_ph_dim=att_ph_dim,
            att_so_dim=att_so_dim, 
            phy_tempreture=phy_tempreture,
            so_tempreture=so_tempreture,
            n_max=n_max,
            kernel_size=kernel_size,
            attention_type=attention_type, 
            social_attention_type=social_attention_type,
            physical_attention_type=physical_attention_type,
            so_prior_type=so_prior_type,
            ph_prior_type=ph_prior_type,
            multiplier=multiplier,
            physical_pos_embed=physical_pos_embed,
            physical_img_embed=physical_img_embed,
            social_pos_embed=social_pos_embed,
            cell_pad=cell_pad,
            center_crop=center_crop,
            compress_attention=compress_attention,
            concat_state=concat_state,
            usefulness=usefulness,
            use_vgg=use_vgg,
            use_seg=use_seg,
            vgg_train=vgg_train, 
            add_input=add_input,
            easy=easy,
            setting_image = setting_image,
            artificial_social_attention=artificial_social_attention
            )

    def forward(self, img, obs_traj, obs_traj_rel, seq_start_end, so_prior=0, ph_prior=0):
        """
        Inputs:
        - img
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        relation between different types of noise and outputs.
        Output:
        - pred_traj_fake
        - pred_traj_fake_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)

        last_pos = obs_traj[-1].clone()
        last_pos_rel = obs_traj_rel[-1].clone()

        if self.ge_type == 'traj_both':
            obs_traj_both = torch.cat([obs_traj, obs_traj_rel], dim=2)
        
        # encoder state
        if self.encoder_h_dim == 0:
            encoder_state = obs_traj[-1].clone()
        else:
            if self.s_type == 'sophie' or self.p_type == 'sophie':
                npeds = obs_traj.size(1)
                total = npeds * self.n_max
                # print(obs_traj.view(8, -1, 2).shape)
                encoder_state = self.encoder.init_hidden(total)
                if self.ge_type == 'traj':
                    obs_traj_embedding = self.encoder_embedding(obs_traj.view(8, -1, 2))
                else:
                    obs_traj_embedding = self.encoder_embedding(obs_traj_rel.view(8, -1, 2))
                obs_traj_embedding = obs_traj_embedding.view(-1, total, self.embedding_dim)
            else:
                encoder_state = self.encoder.init_hidden(batch)

        # decoder cell state
        decoder_state = self.decoder.init_hidden(batch)
        if self.recurrent_attention:
            phy_attention = []
            so_attention = []

        # predict output
        pred_traj_fake = []
        pred_traj_fake_rel = []

        if self.encoder_h_dim == 0:
            decoder_state, physical_attention, social_attention, agent_num, V_so, edge = self.Middle_Layer(img, encoder_state, seq_start_end, last_pos, obs_traj, so_prior=so_prior, ph_prior=ph_prior, decoder_state=decoder_state)
            if not self.recurrent_attention:
                phy_attention, so_attention = physical_attention, social_attention
            else:
                phy_attention.append(physical_attention)
                so_attention.append(social_attention)

            for i in range(self.pred_len):
                if self.s_type == 'sophie' or self.p_type == 'sophie':
                    last_pos = obs_traj[-1,:,0].clone()
                    last_pos_rel = obs_traj_rel[-1,:,0].clone()
                # Predict Trajectory seq
                if self.gd_type == 'traj':
                    decoder_state = self.decoder(self.decoder_embedding(last_pos), decoder_state)
                    pred_pos = self.hidden2pos(decoder_state[0])
                    pred_traj_fake.append(pred_pos)
                    pred_traj_fake_rel.append(pred_pos - last_pos)
                    last_pos_rel = pred_pos - last_pos
                    last_pos = pred_pos

                elif self.gd_type == 'traj_rel':
                    decoder_state = self.decoder(self.decoder_embedding(last_pos_rel), decoder_state)
                    last_pos_rel = self.hidden2pos(decoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 

                elif self.gd_type == 'traj_res':
                    decoder_state = self.decoder(self.decoder_embedding(last_pos), decoder_state)
                    last_pos_rel = self.hidden2pos(decoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 

                elif self.gd_type == 'embed_gcn_traj':
                    if self.recurrent_graph:
                        edge = make_graph(last_pos, seq_start_end)
                    data = Data(x=last_pos, edge_index=edge).cuda()
                    gcn_state, _ = self.d_gcn(data)
                    decoder_state = self.decoder(gcn_state, decoder_state)
                    pred_pos = self.hidden2pos(decoder_state[0])
                    pred_traj_fake.append(pred_pos)
                    pred_traj_fake_rel.append(pred_pos - last_pos)
                    last_pos_rel = pred_pos - last_pos
                    last_pos = pred_pos
                
                elif self.gd_type == 'embed_gcn':
                    if self.recurrent_graph:
                        edge = make_graph(last_pos, seq_start_end)
                    data = Data(x=last_pos_rel, edge_index=edge).cuda()
                    gcn_state, _ = self.d_gcn(data)
                    decoder_state = self.decoder(gcn_state, decoder_state)
                    last_pos_rel = self.hidden2pos(decoder_state[0])
                    pred_traj_fake_rel.append(last_pos_rel)
                    pred_traj_fake.append(last_pos + last_pos_rel)
                    last_pos = last_pos + last_pos_rel 

                #recurrnt attention
                if self.recurrent_attention:
                    decoder_state, physical_attention, social_attention, _, _, _ = self.Middle_Layer(img, encoder_state, seq_start_end, last_pos, decoder_state, V_so, edge, self.recurrent_attention)
                    phy_attention.append(physical_attention)
                    so_attention.append(social_attention)
                    if self.input_recurrent_attention:
                        encoder_state = (Att_so, encoder_state[1])
        else:
            for i in range(self.seq_len):
                if i < self.obs_len:
                    if self.ge_type == 'gcn' or  self.ge_type == 'embed_gcn' or  self.ge_type == 'embed_gcn_traj':
                        if self.recurrent_graph:
                            edge = make_graph(obs_traj[i], seq_start_end)
                        elif i == 0 and self.recurrent_graph==False:
                            edge = make_graph(last_pos, seq_start_end)
                    else:
                        edge=None
                    # print('2')
                    # Encode seq
                    if self.s_type == 'sophie' or self.p_type == 'sophie':
                        encoder_state = self.encoder(obs_traj_embedding[i], encoder_state)
                    elif self.ge_type == 'traj':
                        encoder_state = self.encoder(self.encoder_embedding(obs_traj[i]), encoder_state)
                    elif self.ge_type == 'traj_rel':
                        encoder_state = self.encoder(self.encoder_embedding(obs_traj_rel[i]), encoder_state)    
                    elif self.ge_type == 'traj_both':
                        encoder_state = self.encoder(self.encoder_embedding(obs_traj_both[i]), encoder_state)
                    elif self.ge_type == 'embed_gcn_traj':
                        data = Data(x=obs_traj[i], edge_index=edge).cuda()
                        gcn_state, gcn_attention = self.gcn(data)
                        encoder_state = self.encoder(gcn_state, encoder_state)
                    elif self.ge_type == 'embed_gcn':
                        data = Data(x=obs_traj_rel[i], edge_index=edge).cuda()
                        gcn_state, gcn_attention = self.gcn(data)
                        # print('3')
                        encoder_state = self.encoder(gcn_state, encoder_state)
                        # print('4')
                    
                    #input to decoder
                    if self.input_to_decoder == True and i != self.obs_len-1:
                        if self.gd_type == 'traj' or self.gd_type == 'traj_res':
                            decoder_state = self.decoder(self.decoder_embedding(obs_traj[i]), decoder_state)
                        elif self.gd_type == 'traj_rel' :
                            decoder_state = self.decoder(self.decoder_embedding(obs_traj_rel[i]), decoder_state)
                    
                # Middle Layer
                    if i < self.obs_len - 1 and self.input_recurrent_attention:
                        _, _, _, _, _, Att_so = self.Middle_Layer(img, encoder_state, seq_start_end, obs_traj[i], decoder_state=None, recurrent_physical_attention=False)
                        encoder_state = (Att_so, encoder_state[1])

                    if self.attention_type == 'sophie' and i == self.obs_len - 2:
                        decoder_state = self.decoder(self.decoder_embedding(obs_traj[i]), decoder_state)
                    
                    if i == self.obs_len - 1:
                        if self.s_type == 'sophie' or self.p_type == 'sophie':
                            final_h = encoder_state[0]
                            encoder_state = (final_h.view(batch, self.n_max, self.encoder_h_dim), encoder_state[1])
                        decoder_state, physical_attention, social_attention, agent_num, V_so, edge = self.Middle_Layer(img, encoder_state, seq_start_end, last_pos, obs_traj, so_prior=so_prior, ph_prior=ph_prior, decoder_state=decoder_state)
                        if not self.recurrent_attention:
                            phy_attention, so_attention = physical_attention, social_attention
                        else:
                            phy_attention.append(physical_attention)
                            # print(physical_attention.shape)
                            so_attention.append(social_attention)
                else:
                    if self.s_type == 'sophie' or self.p_type == 'sophie':
                        last_pos = obs_traj[-1,:,0].clone()
                        last_pos_rel = obs_traj_rel[-1,:,0].clone()
                    # Predict Trajectory seq
                    if self.gd_type == 'traj':
                        decoder_state = self.decoder(self.decoder_embedding(last_pos), decoder_state)
                        pred_pos = self.hidden2pos(decoder_state[0])
                        pred_traj_fake.append(pred_pos)
                        pred_traj_fake_rel.append(pred_pos - last_pos)
                        last_pos_rel = pred_pos - last_pos
                        last_pos = pred_pos

                    elif self.gd_type == 'traj_rel':
                        decoder_state = self.decoder(self.decoder_embedding(last_pos_rel), decoder_state)
                        last_pos_rel = self.hidden2pos(decoder_state[0])
                        pred_traj_fake_rel.append(last_pos_rel)
                        pred_traj_fake.append(last_pos + last_pos_rel)
                        last_pos = last_pos + last_pos_rel 

                    elif self.gd_type == 'traj_res':
                        decoder_state = self.decoder(self.decoder_embedding(last_pos), decoder_state)
                        last_pos_rel = self.hidden2pos(decoder_state[0])
                        pred_traj_fake_rel.append(last_pos_rel)
                        pred_traj_fake.append(last_pos + last_pos_rel)
                        last_pos = last_pos + last_pos_rel 

                    elif self.gd_type == 'embed_gcn_traj':
                        if self.recurrent_graph:
                            edge = make_graph(last_pos, seq_start_end)
                        data = Data(x=last_pos, edge_index=edge).cuda()
                        gcn_state, _ = self.d_gcn(data)
                        decoder_state = self.decoder(gcn_state, decoder_state)
                        pred_pos = self.hidden2pos(decoder_state[0])
                        pred_traj_fake.append(pred_pos)
                        pred_traj_fake_rel.append(pred_pos - last_pos)
                        last_pos_rel = pred_pos - last_pos
                        last_pos = pred_pos
                    
                    elif self.gd_type == 'embed_gcn':
                        if self.recurrent_graph:
                            edge = make_graph(last_pos, seq_start_end)
                        data = Data(x=last_pos_rel, edge_index=edge).cuda()
                        gcn_state, _ = self.d_gcn(data)
                        decoder_state = self.decoder(gcn_state, decoder_state)
                        last_pos_rel = self.hidden2pos(decoder_state[0])
                        pred_traj_fake_rel.append(last_pos_rel)
                        pred_traj_fake.append(last_pos + last_pos_rel)
                        last_pos = last_pos + last_pos_rel 

                    #recurrnt attention
                    if self.recurrent_attention:
                        if self.ge_type == 'traj':
                            encoder_state = self.encoder(self.encoder_embedding(last_pos), encoder_state)
                        elif self.ge_type == 'traj_rel':
                            encoder_state = self.encoder(self.encoder_embedding(last_pos_rel), encoder_state)
                        elif self.ge_type == 'traj_both':
                            encoder_state = self.encoder(self.encoder_embedding(torch.cat([last_pos, last_pos_rel], dim=1)), encoder_state)
                        decoder_state, physical_attention, social_attention, _, _, _ = self.Middle_Layer(img, encoder_state, seq_start_end, last_pos, decoder_state, V_so, edge, self.recurrent_attention)
                        phy_attention.append(physical_attention)
                        so_attention.append(social_attention)
                        if self.input_recurrent_attention:
                            encoder_state = (Att_so, encoder_state[1])

        pred_traj_fake = torch.stack(pred_traj_fake, dim=0).cuda()
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0).cuda()
        if self.recurrent_attention:
            if self.att_ph_dim != 0:
                phy_attention = torch.stack(phy_attention, dim=0).cuda()
            if self.att_so_dim != 0:
                so_attention =  torch.stack(so_attention, dim=0).cuda()

        if self.visualize_process:
            return pred_traj_fake, pred_traj_fake_rel, so_attention, phy_attention, agent_num
        else:    
            return pred_traj_fake, pred_traj_fake_rel, so_attention

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        activation='leakyrelu', batch_norm=True, dropout=0.0, d_type='traj_rel'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        if self.d_type == 'traj_rel':
            batch = traj_rel.shape[1]
            encoder_state = self.encoder.init_hidden(batch)
            for i in range(self.seq_len):
                encoder_state = self.encoder(self.spatial_embedding(traj_rel[i]), encoder_state)
        
        elif self.d_type == 'traj':
            batch = traj.shape[1]
            encoder_state = self.encoder.init_hidden(batch)
            for i in range(self.seq_len):
                encoder_state = self.encoder(self.spatial_embedding(traj[i]), encoder_state)

        final_h = encoder_state[0]
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.

        classifier_input = final_h.squeeze()
        scores = self.real_classifier(classifier_input)
        return scores

