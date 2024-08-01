"""
Created on Sat June 10 15:40:55 2024
@author: Niraj Bhujel (niraj.bhujel@stf.ac.uk)
"""

import os
import numpy as np
import torch

class ModelCheckpoint:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, ckpt_dir, ckpt_name = None, monitor = None, mode='min', 
        trace_func=print, verbose=True, debug=False, logger=None):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = (ckpt_name or f"best_{monitor}")
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_val = None
        self.metric_val_min = np.Inf
        self.delta = 1e-6
        self.trace_func = trace_func
        self.debug = debug

    def __call__(self, metric_val, epoch=None):

        current_val = metric_val
        if not self.mode=='min':
            current_val *= -1

        # save on first call
        if self.best_val is None:
            # self.trace_func("{} improved from {:.6f} to {:.6f}.".format(self.monitor, self.metric_val_min, metric_val))
            self.best_val = current_val
            self.metric_val_min = metric_val
            self.save_checkpoint()

        elif current_val < (self.best_val + self.delta):
            # self.trace_func("{} improved from {:.6f} to {:.6f}.".format(self.monitor, self.metric_val_min, metric_val))
            self.best_val = current_val
            self.metric_val_min = metric_val
            self.save_checkpoint()

    def save_checkpoint(self,):
        '''Saves model when validation loss decrease.'''
        
        if not self.debug:
            save_path = os.path.join(self.ckpt_dir, self.ckpt_name + '.pth')
            self.trace_func("Saving model checkpoint ... ")
            try:
                torch.save(self.model.module.state_dict(), save_path)
            except:
                torch.save(self.model.state_dict(), save_path)
        