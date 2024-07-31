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

from datasets.preprocess import create_train_val_split, load_labels
from datasets.random_augment import RandAugment
from datasets.random_erasing import RandomErasing
from utils.misc import load_file, create_new_dir, listdir
    
class LaserImageDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(self, 
                 cfg,
                 phase,
                 img_labels,
                 img_transform,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.phase = phase
        self.img_transform = img_transform

        self.img_dir = join(cfg.data.data_dir, "images")
        self.label_dir = join(cfg.data.data_dir, "labels")

        counter_dict = collections.Counter([np.argmax(label) for (img, label) in img_labels])
        for c, class_count in counter_dict.items():
            print(f"Class {c}: {class_count}/{len(img_labels)} {phase} samples")

        if cfg.train.single_batch:
            img_labels = img_labels[:cfg.train.batch_size]
            
        self.images = []
        self.labels = []
        self.img_files =  []
        for img, label in img_labels:
            self.labels.append(label)
            self.img_files.append(img)
            
            if cfg.data.prefetch:
                img = Image.open(join(self.img_dir, img))
                self.images.append(img.copy())
                img.close()
   
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if self.cfg.data.prefetch:
            img = self.images[idx]
        else:
            img = Image.open(join(self.img_dir, self.img_files[idx])) 

        img = self.img_transform(img)
        
        label = torch.from_numpy(self.labels[idx])

        return img.to(torch.float32), label.to(torch.float32), self.img_files[idx]
    
    def sample_batch(self, batch_size):
        print(f"Sampling {batch_size} {self.phase} images for single batch ... ")
        self.img_files = self.img_files[:batch_size]
        self.labels = self.labels[:batch_size]
        self.images = self.images[:batch_size]
        
    
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

def create_datasets(cfg, single_batch=False):
    train_img_labels, val_img_labels = create_train_val_split(cfg.data.data_dir, cfg.data.train_datasets, 
                                                              val_split=cfg.data.val_split, 
                                                              balance_class=cfg.data.balance_class, 
                                                              shuffle=True)
                                                      
    train_dataset = LaserImageDataset(cfg,
                                      phase='train',
                                      img_labels = train_img_labels,
                                      img_transform=get_transforms(cfg.data, "train"),
                                     )
    
    val_dataset = LaserImageDataset(cfg,
                                    phase='val',
                                    img_labels=val_img_labels,
                                    img_transform=get_transforms(cfg.data, "val"),
                                   )


    test_dataset = LaserImageDataset(cfg,
                                     phase='test',
                                     img_labels=load_labels(cfg.data.data_dir, cfg.data.test_datasets),
                                     img_transform=get_transforms(cfg.data, "test"),
                                    )
    return train_dataset, val_dataset, test_dataset
    
