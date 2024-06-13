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

from datasets.random_augment import RandAugment
from datasets.random_erasing import RandomErasing
from utils.misc import load_file, create_new_dir

class LaserImageDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(self, 
                 cfg,
                 phase,
                 img_transform,
                 num_classes=1,
                 **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.phase = phase
        self.num_classes=num_classes

        self.data_dir = cfg.data_dir

        self.img_dir = join(self.data_dir, "images")
        self.label_dir = join(self.data_dir, "labels")

        self.img_transform = img_transform
        
        train_img_labels, val_img_labels = self.create_split(cfg.data_dir, cfg.datasets, cfg.val_split)
        if phase=='train':
            self.img_labels = train_img_labels
        else:
            self.img_labels = val_img_labels

        class_labels = defaultdict(int)
        for img, label in self.img_labels:
            class_labels[label] += 1
        for c, count in class_labels.items():
            print(f"Class: {c}, {count}/{len(self.img_labels)} {phase} samples")

    def create_split(self, data_dir, datasets, val_split=0.3):
        
        train_labels = []
        val_labels = []
        for dset in datasets:
            for file in sorted(os.listdir(f"{data_dir}/labels/{dset}")):

                # Separate the labels into classes
                cls_dict = defaultdict(list)
                for img_label in load_file(f"{data_dir}/labels/{dset}/{file}"):
                    img, label = img_label.split(',')
                    cls_dict[label].append((img, int(label)))

                # Split each class labels into training and validation sets                           
                for c, class_labels in cls_dict.items():
                    # print(f"{c=}, {len(class_labels)} images ")
                    split_indx = int(len(class_labels)*(1-val_split))
                    train_labels.extend(class_labels[:split_indx])
                    val_labels.extend(class_labels[split_indx:])

        return train_labels, val_labels      
                    
                    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_file, label = self.img_labels[idx]

        label = F.one_hot(torch.tensor(label), self.num_classes)

        img = Image.open(join(self.img_dir, img_file))
        img = self.img_transform(img)

        return img.to(torch.float32), label.to(torch.float32), img_file
    
    def sample_batch(self, batch_size):
        print(f"Sampling {batch_size} {self.phase} images for single batch ... ")
        self.img_labels = self.img_labels[:batch_size]
        
class DirectoryDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()

        self.img_dir = join(root, "image")
        self.label_dir = join(root, "labels")

        self.transform = transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))

        assert len(self.img_files) > 0

        if os.path.exists(join(root, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(self.label_files)
        else:
            self.label_files = None

    def __getitem__(self, idx):
        # Load image.
        # img = io.imread(join(self.img_dir, self.img_files[idx]))
        img = Image.open(join(self.img_dir, self.img_files[idx]))

        seed = np.random.randint(2147483647)
        set_random_seed(seed)
        img = self.transform(img)

        if self.label_files is not None:
            label = Image.open(join(self.label_dir, self.label_files[idx]))
            set_random_seed(seed)
            label = self.target_transform(label).squeeze(0)
        else:
            label = 0

        return img, label, self.img_files[idx]

    def __len__(self):
        return len(self.img_files)
    
def get_transforms(cfg, phase):

    # image transform applied for both train and test set
    if cfg.n_channels==1:
        normalize = T.Normalize(mean=[0.4], std=[0.25])
    else:
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    image_transform = T.Compose([
                        T.Resize((cfg.img_height, cfg.img_width), TF.InterpolationMode.NEAREST),
                        T.ToTensor(),
                        normalize,
                        ])
    if phase=="train":
        geometric_transforms = T.Compose([
            T.RandomApply([
                        # T.CenterCrop((cfg.img_height, cfg.img_width)),
                        T.RandomResizedCrop(size=(cfg.img_height, cfg.img_width), scale=(0.8, 1.0)),
                        T.RandomRotation(10),
                        T.RandomHorizontalFlip(p=0.5),
                        ], cfg.aug_geome)
            ])

        photometric_transforms = T.Compose([
            T.RandomApply([RandAugment(num_ops=2, magnitude=9)], cfg.aug_photo)
            ])
        
        return T.Compose([image_transform, geometric_transforms, photometric_transforms])
    
    else:
        return image_transform

    
