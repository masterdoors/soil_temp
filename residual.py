
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from bisect import bisect
import torch.nn.functional as F

class ResBlock(nn.Module):
    pass

class ResNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):    
        super(ResNetModel, self).__init__()
        self.input = nn.Linear(input_size, hidden_size,dtype=torch.float64,bias=True)
        self.blocks = nn.ModuleList([nn.Linear(hidden_size, output_size,dtype=torch.float64,bias=False)]) 

class ResNet:
    def __init__(self,):
        pass

    def fit(X,y):
        pass

    def predict(X):
        pass    