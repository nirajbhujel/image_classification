"""
Created on Sat Aug 22 15:37:55 2020

@author: Niraj

Imported from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""

import os
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, metric_name = 'val_loss', last_epoch=0, trace_func=print):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_val_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.metric_name = metric_name
        self.last_epoch = last_epoch
        
    def __call__(self, metric_val, epoch=None):

        score = -metric_val

        if self.best_score is None:
            self.trace_func("{} improved from {:.6f} to {:.6f}.".format(self.metric_name, self.metric_val_min, metric_val))
            self.metric_val_min = metric_val
            self.best_score = score

        elif score < (self.best_score + self.delta):
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch

            self.trace_func("{} improved from {:.6f} to {:.6f}.".format(self.metric_name, self.metric_val_min, metric_val))

            self.best_score = score
            self.metric_val_min = metric_val
            self.counter = 0

        else:
            if epoch is None:
                self.counter += 1
            else:
                self.counter = (epoch-self.last_epoch)

            self.trace_func("{} didn't improve from {:.6f}. Early stopping counter: {}/{}".format(
                self.metric_name, self.metric_val_min, self.counter, self.patience))
            
            if self.counter>self.patience:
                self.early_stop = True


        