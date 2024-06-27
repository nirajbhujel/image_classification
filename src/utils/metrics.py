import cv2
import math
import numpy as np
from collections import defaultdict
from tabulate import tabulate

import torch

def _binary_stat_scores(preds, target):
    """Compute the statistics. Follow torchmetrics.classification.stat_scores.py"""
    tp = ((target == preds) & (target == 1)).sum()
    fn = ((target != preds) & (target == 1)).sum()
    fp = ((target != preds) & (target == 0)).sum()
    tn = ((target == preds) & (target == 0)).sum()
    return tp, fp, tn, fn

def binary_accuracy(y_true, y_pred, threshold=0.5):
        
    y_pred = (y_pred>0.5).to(y_true.dtype)

    return torch.mean((y_pred==y_true).to(y_true.dtype), dim=-1)
    

class MetricLogger:
    def __init__(self, name="binary_accuracy"):
        
        self.name = name
        
        self.reset()
    
    def __len__(self, ):
        return self.count
    
    def reset(self, ):
        self.count = 0
        self.hist = []
        
        self.stats = {}
        self.table = ''

        self._reset_state()

    def _reset_state(self, ):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def _update_state(self, tp, fp, tn, fn):
        self.tp += tp
        self.fp += fp 
        self.tn += tn  
        self.fn += fn

    def __call__(self, y_pred, y_true):
        
        self.count += 1

        if self.name=='binary_accuracy':
            y_pred = (y_pred>0.5).to(y_true.dtype)

            tp, fp, tn, fn = _binary_stat_scores(y_pred, y_true)
            
            self._update_state(tp, fp, tn, fn)

        acc = (tp + tn) / (tp + tn + fp + fn)

        return acc

    def _compute_stats(self, ):
        
        stats = {}

        stats['tp'] = self.tp.item()
        stats['fp'] = self.fp.item()
        stats['tn'] = self.tn.item()
        stats['fn'] = self.fn.item()
        stats['accuracy'] = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)
        stats['precision'] = self.tp / (self.tp + self.fn)

        return stats
   
    
    def compute(self, verbose=0):

        if not len(self)>0:
            raise Exception("Nothing to compute. Please update metrics first")
            
        self.results = self._compute_stats()
        
        print(self.tabulate_data(self.results))                                       
                                               
    def tabulate_data(self, results, tablefmt='simple'):
            
        headers = ["Metric"] + list(results.keys())
        
        data = []
        
        # metric row
        row = [self.name] + [f"{v:.2f}" for k, v in results.items()]
        data.append(row)
        
        # add bottom hline
        data.append(['-------'] * len(headers))
        
        tabular_data = tabulate(data, headers=headers, tablefmt=tablefmt)

        return tabular_data

            
    
            
        
                                              
                                              
        
        
    
                