# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

### DO NOT USE THIS CODE FOR OUR CLIENTS ####


import os,sys,time
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from functools import partial


class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
        ]
        if norm_layer is not None:
          layers.append(norm_layer(C_out))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

class ResnetBasicBlock(nn.Module):
  __constants__ = ['downsample']

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride,bias=bias)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes,bias=bias)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out
  
class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x):
    '''
    @x: (B,N,D)
    '''
    return x + self.pe[:, :x.size(1)]


class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1, loss_func=nn.MSELoss()):
    super().__init__()
    self.cfg = cfg
    if self.cfg['use_BN']:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=484)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )

    self.loss_func = loss_func

    # Time debug stuff
    self.resnet_time = 0
    self.transformer_time = 0
    self.n_iter = 0

  def stream(self, A, B, batch_norms, T_src=None):
    """
    @A: (B,C,H,W)
    """
    bs = len(A)
    output = {}
    t0 = time.time()

    x = torch.cat([A,B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a,b),1).contiguous()
    ab = self.encodeAB(ab)  #(B,C,H,W)

    # Time debug stuff
    self.resnet_time += time.time() - t0
    t0 = time.time()

    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))

    output['trans'] = self.trans_head(ab).mean(dim=1)
    output['rot'] = self.rot_head(ab).mean(dim=1)

    # Time debug stuff
    self.transformer_time += time.time() - t0
    self.n_iter += 1

    return torch.cat((output['trans'], output['rot']), dim=1), ab

  def forward(self, A, B, T_src=None):
    AB = self.stream(A, B, None, T_src=T_src)
    return AB

  def loss(self, predictions, targets):
      loss = self.loss_func(predictions[0], targets[0])
      return loss
  
  def print_time(self):
    print(f"Resnet mean time: {round(self.resnet_time/self.n_iter, 4)}")
    print(f"Transformer mean time: {round(self.transformer_time/self.n_iter, 4)}")