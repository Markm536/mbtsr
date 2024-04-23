import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock2D(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, drop_prob=0.0, max_pool_size=2):
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.drop = nn.Dropout(p=drop_prob)

        self.max_pool_size = max_pool_size
        if max_pool_size:
            self.max_pool = nn.MaxPool2d((max_pool_size, max_pool_size))

    def forward(self, x):
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.drop(x)
        if self.max_pool_size:
            x = self.max_pool(x)

        return x

class ResBlock2D(torch.nn.Module):
    def __init__(self, input_channels, kernel_size=3, stride=1, padding=1, drop_prob=0.0):
        super().__init__()
        
        self.conv1 = ConvBlock2D(input_channels, input_channels, kernel_size=3, stride=1, padding=1, drop_prob=0.0, max_pool_size=0)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBlock2D(input_channels, input_channels, kernel_size=3, stride=1, padding=1, drop_prob=0.0, max_pool_size=0)
        self.relu2 = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x0 = x
        x = self.conv1(x0)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm(x)
        x = self.drop(x)

        return x + x0
    
class DownsampleModel(nn.Module):
    pass

class SuperResModel(nn.Module):
    pass