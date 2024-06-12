import io

import os
import math
import time
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

def set_random_seed(seed):
    #setup seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def torch_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def clip_gradients(model, clip_value):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip_value / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2

unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prep_for_plot(img, unnormalize=True, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    if unnormalize:
        img = unnorm(img)

    plot_img = img.squeeze(0).permute(1, 2, 0).cpu()
    
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())

    return plot_img

def add_plot(writer, name, step):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    image = T.ToTensor()(image)
    writer.add_image(name, image, step)
    plt.clf()
    plt.close()


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def force_cudnn_initialization(s=16):
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def model_parameters(model, verbose=0):
    total_params=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params+=param.nelement()
    return total_params
    
def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
            print(f.shape)
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

def non_contributing_params(model, outputs=[]):
    contributing_parameters = set.union(*[set(get_contributing_params(out)) for out in outputs])
    all_parameters = set(model.parameters())
    non_contributing_parameters = all_parameters - contributing_parameters
    if len(non_contributing_parameters)>0:
        print([p.shape for p in non_contributing_parameters])  # returns the [999999.0] tensor
    return sum([np.product(list(p.shape)) for p in non_contributing_parameters])


