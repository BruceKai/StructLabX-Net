""" 
  @Author: Zhiwen.Cai  
  @Date: 2022-09-15 14:56:08  
  @Last Modified by: Zhiwen.Cai  
  @Last Modified time: 2022-09-15 14:56:08  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryTanimotoLoss(nn.Module):
    """Tanimoto loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-4, p=2, reduction='mean'):
        super(BinaryTanimotoLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        num_1 = torch.sum(torch.mul(1-predict, 1-target), dim=1) + self.smooth
        den_1 = torch.sum((1-predict).pow(self.p) + (1-target).pow(self.p), dim=1) + self.smooth

        loss = 1 - 0.5*num_1/(den_1-num_1+ self.smooth) - 0.5*num / (den-num+ self.smooth)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))