import cv2
import math
import numpy as np
from collections import defaultdict
from tabulate import tabulate

import torch

def binary_accuracy(y_true, y_pred, threshold=0.5):
        
    y_pred = (y_pred>0.5).to(y_true.dtype)
    acc = torch.mean((y_pred==y_true).to(y_true.dtype), dim=-1)

    return acc


class MetricLogger:
    def __init__(self, name="binary_accuracy", sample_weight=None, reduction='mean'):
        
        self.name = name
        self.sample_weight = sample_weight
        self.reduction = reduction
        
        self.count = 0
        self.accuracy = []
        
        self.stats = {}
        self.table = ''
    
    def __len__(self, ):
        return len(self.gt_poses)
    
    def update(self, y_true, y_pred):
        
        if self.name=='binary_accuracy':
            acc = binary_accuracy(y_true, y_pred)

        if self.sample_weight is not None:
            acc = self.sample_weight * acc

        if self.reduction=='sum':
            acc_red = torch.sum(acc)
        else:
            acc_red = torch.mean(acc)
        
        self.accuracy.append(acc_red)
        self.count += 1

        return acc_red

    def compute_metric_stats(self, metric):
        
        stats = {}
        
        stats["min"] = np.min(metric).round(3)
        stats["max"] = np.max(metric).round(3)
        stats["mean"] = np.mean(metric).round(3)
        # stats["med"] = np.median(metric).round(3)
        # stats["rmse"] = np.sqrt(np.mean(np.square(metric))).round(3)

        return stats
   
    
    def compute(self, ):

        if not len(self)>0:
            raise Exception("Nothing to compute. Please update metrics first")
            
        self.stats = self.compute_metric_stats(self.accuracy)
        
        self.table = self.tabulate_data()
        print(self.table)
                                               
                                               
    def tabulate_data(self, tablefmt='simple'):
            
        headers = ["Metric", "Min", "Max", "Mean"]
        
        data = []
        name_row = [''] * len(headers)
        name_row[0] = self.name
        data.append(name_row)
        
        # add a hline
        data.append(['-------'] * len(headers))
        
        # metric row
        row = [f"{e}"] + [f"{v:.3f}" for k, v in self.stats.items()]
        data.append(row)
            
        tabular_data = tabulate(data, headers=headers, tablefmt=tablefmt)

        return tabular_data

            
    
            
        
                                              
                                              
        
        
    
                