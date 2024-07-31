"""
Created on Mon June  10 10:08:17 2024
@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import CNN, PointCNN
from models.fpn import FPN
from models.modules import *
        
class Network(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Create backbone    
        if cfg.net.type=='MLP':
            self.net = make_mlp(dim_list=[cfg.data.img_height*cfg.data.img_width, cfg.net.hidden_dim, cfg.net.hidden_dim])  
        elif cfg.net.type=='CNN':
            self.net = CNN(1, cfg.net.hidden_dim)
        elif cfg.net.type=='PointCNN':
            self.net = PointCNN(1, cfg.net.hidden_dim)
        elif cfg.net.type=='FPN':
            self.net = FPN(1, cfg.net.hidden_dim, cfg.net.hidden_dim)
        else:
            raise Exception(f"Network type {cfg.net.type} not supported!!" )

        # Create heads
        if cfg.task=='classification':
            self.head = ClassificationHead(cfg.net.hidden_dim, cfg.num_classes)
        elif cfg.task=='detection':
            self.head = DetectionHead(cfg.net.hidden_dim, cfg.net.hidden_dim, cfg.num_classes)
        elif cfg.task == 'clustering':
            self.head = ClusterHead(cfg.net.hidden_dim, cfg.num_classes)
        else:
            raise Exception(f"Head type {cfg.task} not supported!!" )

        self.num_params = self.model_parameters()

    def reset_head_parameters(self):
        for layer in self.head.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def model_parameters(self, verbose=0):
        if verbose>0:
            print('{:<30} {:<10} {:}'.format('Parame Name', 'Total Param', 'Param Shape'))
        total_params=0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if verbose>0:
                    print('{:<30} {:<10} {:}'.format(name, param.numel(), tuple(param.shape)))
                total_params+=param.nelement()
        if verbose>0:
            print('Total Trainable Parameters :{:<10}'.format(total_params))
        return total_params

    def forward(self, x):
        x = self.net(x)
        # print(x.shape)
        
        if self.cfg.net.type!='MLP':
            # perform global pooling
            x = torch.mean(x, dim=[2, 3], keepdim=False)
        
        x = self.head(x)
        # print(x.shape)
        
        return x


