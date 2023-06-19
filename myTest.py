import torch
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps
import numpy as np
from torch.nn.modules.utils import _pair
import cv2
from mmdet.models.utils import build_linear_layer
import torch.nn  as nn
import cv2
class mlp:
    def __init__(self):
        self.roi_feat_size = 7
        self.pooling = nn.AvgPool2d(self.roi_feat_size)
        self.fc1 = build_linear_layer(dict(type='Linear'),256,128)
        self.relu = nn.ReLU()
        self.fc2 = build_linear_layer(dict(type='Linear'),128,64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

def BatchNorm(x,gamma,beta):
    """

    Args:
        x:  shape of (B,C,H,W)
        gamma:
        beta:
        bn_param: running_mean ans running_var
    compute  a layer of one batch
    Returns:

    """

    result = 0.
    eps = 1e-5

    x_mean = torch.mean(x,dim = 1,keepdim=True)
    x_var = torch.var(x,dim=1,keepdim=True)
    x_normalization = (x-x_mean)/torch.sqrt(x_var+eps)
    result = x_normalization*gamma+beta

    return result

def LayerNorm(x,gamma,beta):
    result = 0.
    eps = 1e-5
    """layernorm compute a instance of one batch 's all pixels"""
    x_mean = torch.mean(x, dim=0, keepdim=True)
    x_var = torch.var(x, dim=0, keepdim=True)
    x_normalization = (x - x_mean) / torch.sqrt(x_var + eps)
    result = x_normalization * gamma + beta

    return result

def InstanceNorm(x,gamma,beta):
    result = 0.
    eps = 1e-5
    x_mean = torch.mean(x, dim=(0, 1), keepdim=True)
    print(x_mean.shape)
    x_var = torch.var(x, dim=(0, 1), keepdim=True)
    x_normalization = (x - x_mean) / torch.sqrt(x_var + eps)
    result = x_normalization * gamma + beta
    return result

def GroupNorm(x,gamma,beta):
    result = 0.
    eps = 1e-5

    x = x.reshape()

    x_mean = torch.mean(x, dim=(0, 1), keepdim=True)
    print(x_mean.shape)
    x_var = torch.var(x, dim=(0, 1), keepdim=True)
    x_normalization = (x - x_mean) / torch.sqrt(x_var + eps)
    result = x_normalization * gamma + beta
    return result
a = torch.rand((2,2,3,3))
b = InstanceNorm(a,0.5,0.1)
print(a,b)
