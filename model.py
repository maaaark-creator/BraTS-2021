import os
import sys

# 【第一步】必须在所有科学计算库导入之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 如果还是报错，可以加上下面这行强制单线程
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim



class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        

        self.conv5 = nn.Conv3d(128 + 64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.conv6 = nn.Conv3d(64 + 32, 32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(32)
        self.conv7 = nn.Conv3d(32 + 16, 16, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(16)
        self.conv8 = nn.Conv3d(16, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
  
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool(c1)
        c2 = self.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool(c2)
        c3 = self.relu(self.bn3(self.conv3(p2)))
        p3 = self.pool(c3)
        c4 = self.relu(self.bn4(self.conv4(p3)))
        
  
        u5 = self.up(c4)
        cat5 = torch.cat([u5, c3], dim=1)  
        c5 = self.relu(self.bn5(self.conv5(cat5)))
        
        u6 = self.up(c5)
        cat6 = torch.cat([u6, c2], dim=1) 
        c6 = self.relu(self.bn6(self.conv6(cat6)))
        
        u7 = self.up(c6)
        cat7 = torch.cat([u7, c1], dim=1)  
        c7 = self.relu(self.bn7(self.conv7(cat7)))
        
        output = self.conv8(c7)
        return output