import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BCE_Loss:
	def __init__(self, reduction='mean'):
		super().__init__()
		self.func = nn.BCELoss(reduction=reduction)

	def compute(self, y_true, y_pred):
		assert y_pred.shape==y_true.shape, f"{y_pred.shape=} is not equal to {y_true.shape=}"
		y_pred = F.softmax(y_pred, dim=-1)
		return self.func(y_pred, y_true)