"""
Created on Mon June  10 10:08:17 2024
@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN encoder, used to extract point features from the input images.
    """
    OUTPUT_SUBSAMPLE = 8
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.conv1 = nn.Conv2d(in_channels, 32, 7, 2, 3, bias=False)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256, out_channels, 3, 1, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x)) # 1/1 [480, 848]
        x = F.relu(self.conv2(x)) # 1/2 [240, 424]
        x = F.relu(self.conv3(x)) # 1/4 [120, 212]
        x = F.relu(self.conv4(x)) # 1/8 [60, 106]
        x = F.relu(self.conv5(x)) # 1/8 [60, 106]

        return x

class PointCNN(nn.Module):
    """
    CNN encoder, used to extract point features from the input images.
    """
    OUTPUT_SUBSAMPLE = 8
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.conv1 = nn.Conv2d(in_channels, 32, 7, 2, 3, bias=False)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, out_channels, 1, 1, 0)

    def forward(self, x):

        x = F.relu(self.conv1(x)) # 1/1 [480, 848]
        x = F.relu(self.conv2(x)) # 1/2 [240, 424]
        x = F.relu(self.conv3(x)) # 1/4 [120, 212]
        res = F.relu(self.conv4(x)) # 1/8 [60, 106]

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x