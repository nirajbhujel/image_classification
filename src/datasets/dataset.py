"""
Created on Wed June  6 10:57:22 2024
@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk)
"""

import os
import glob
import numpy as np
import random
import math
import warnings
import collections
import matplotlib.pyplot as plt

from os.path import join
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from datasets.preprocess import create_train_val_split
from datasets.random_augment import RandAugment
from datasets.random_erasing import RandomErasing
from utils.misc import load_file, create_new_dir, listdir

class LaserImageDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(self, 
                 data_cfg,
                 phase,
                 img_transform,
                 num_classes=1,
                 **kwargs):
        super().__init__()
        
        self.data_cfg = data_cfg
        self.phase = phase
        self.num_classes=num_classes
        self.balanced_class=kwargs.get('balanced_class', 1)
        self.shuffle_labels = kwargs.get('shuffle_labels', 1)

        self.data_dir = data_cfg.data_dir

        self.img_dir = join(self.data_dir, "images")
        self.label_dir = join(self.data_dir, "labels")

        self.img_transform = img_transform
        
        train_img_labels, val_img_labels = create_train_val_split(data_cfg.data_dir, 
            data_cfg.datasets, data_cfg.val_split, self.balanced_class, self.shuffle_labels)
        
        img_labels = train_img_labels if phase=='train' else val_img_labels
        counter_dict = collections.Counter([l[1] for l in img_labels])
        for c, class_count in counter_dict.items():
            print(f"Class {c}: {class_count}/{len(img_labels)} {phase} samples")
        
        self.images = []
        self.labels = [] 
        self.img_files = []
        for img, label in img_labels:
            self.labels.append(np.eye(num_classes)[int(label)])
            self.img_files.append(img)
            if data_cfg.prefetch:
                tmp_img = Image.open(join(self.img_dir, img))
                self.images.append(tmp_img.copy())
                tmp_img.close()
                    
                    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        if self.data_cfg.prefetch:
            img = self.images[idx]
        else:
            img = Image.open(join(self.img_dir, self.img_files[idx])) 

        if len(self.images)>0:
            img = self.images[idx]

        img = self.img_transform(img)
        
        label = torch.from_numpy(self.labels[idx])

        return img.to(torch.float32), label.to(torch.float32), self.img_files[idx]
    
    def sample_batch(self, samples, batch_size):
        print(f"Sampling {batch_size} {self.phase} images for single batch ... ")
        self.img_files = self.img_files[:batch_size]
        self.labels = self.labels[:batch_size]
        
    
def get_transforms(data_cfg, phase):

    # image transform applied for both train and test set
    if data_cfg.n_channels==1:
        normalize = T.Normalize(mean=[0.1], std=[0.2])
    else:
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    image_transform = T.Compose([
                        T.Resize((data_cfg.img_height, data_cfg.img_width), TF.InterpolationMode.NEAREST),
                        T.ToTensor(),
                        normalize,
                        ])
    if phase=="train":
        geometric_transforms = T.Compose([
            T.RandomApply([
                        # T.CenterCrop((data_cfg.img_height, data_cfg.img_width)),
                        T.RandomResizedCrop(size=(data_cfg.img_height, data_cfg.img_width), scale=(0.8, 1.0)),
                        T.RandomRotation(10),
                        T.RandomHorizontalFlip(p=0.5),
                        ], data_cfg.aug_geome)
            ])

        photometric_transforms = T.Compose([
            T.RandomApply([RandAugment(num_ops=2, magnitude=9)], data_cfg.aug_photo)
            ])
        
        return T.Compose([image_transform, geometric_transforms, photometric_transforms])
    
    else:
        return image_transform

    
