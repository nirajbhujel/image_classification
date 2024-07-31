"""
Created on Mon June  10 10:08:17 2024
@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):
    layers = [nn.Flatten(start_dim=1)]
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
    
class ClassificationHead(nn.Module):
    """
    Simple MLP network with dropout.
    """
    def __init__(self, in_channels=512, out_channels=2, act_layer=nn.ReLU, dropout=0.0, bias=True):
        super().__init__()

        self.flat = nn.Flatten(start_dim=1)
        self.act = act_layer()
        self.drop = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(in_channels, in_channels//2, bias=bias)
        self.fc2 = nn.Linear(in_channels//2, in_channels//4, bias=bias)
        self.fc3 = nn.Linear(in_channels//4, out_channels, bias=bias)

    def forward(self, x):
        
        x = self.flat(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class DetectionHead(nn.Module):
    """
    MLP network predicting class label give a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(self, in_channels=512, hidden_channels=512, out_channels=2, act_layer=nn.ReLU, dropout=0.0, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels=out_channels
        
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=bias)
            
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x

    
class ClusterHead(nn.Module):

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, in_channels))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.in_channels))

    def forward(self, x, alpha=None, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_clusters={self.n_classes})"

