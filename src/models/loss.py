import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BCE_Loss(nn.Module):
	def __init__(self, cls_weights=None, reduction='mean'):
		super().__init__()

		if cls_weights is not None:
			cls_weights = torch.tensor(cls_weights)
			reduction = 'none'

		self.cls_weights = cls_weights
		self.func = nn.BCELoss(reduction=reduction)

	def forward(self, y_true, y_pred):
		B, C = y_true.shape
		assert y_pred.shape==y_true.shape, f"{y_pred.shape=} is not equal to {y_true.shape=}"
		
		y_pred = F.softmax(y_pred, dim=-1)
		
		loss = self.func(y_pred, y_true)

		if self.cls_weights is not None:
			weight = torch.ones_like(loss)
			weight[y_true.argmax(-1)==0] = self.cls_weights[0]
			weight[y_true.argmax(-1)==1] = self.cls_weights[1]

			loss = (weight*loss).mean()

		return loss