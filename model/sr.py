import torch
from torch import nn
import torchvision
import math
import torch.nn.functional as F
from functools import partial

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super(ConvolutionalBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        ))

        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == 'ReLU':
            layers.append(nn.ReLU())
        elif activation == 'PReLU':
            layers.append(nn.PReLU())
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'Tanh':
            layers.append(nn.Tanh())
        
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)

        return output


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, scaling_factor=2, mode='linear'):
        super(UpsampleBlock, self).__init__()

        self.conv = ConvolutionalBlock(n_channels, n_channels * (scaling_factor**2), kernel_size)
        if mode == 'shuffle':
            self.scaling = nn.PixelShuffle(upscale_factor=scaling_factor)
        else:
            self.conv = nn.Identity()
            self.scaling = nn.Upsample(scale_factor=scaling_factor)
        
        self.prelu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)

        output = self.scaling(output)        
        output = self.prelu(output)          

        return output
    
class DownsampleBlock(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, scaling_factor=2, mode='shuffle'):
        super(DownsampleBlock, self).__init__()

        self.scaling = nn.PixelUnshuffle(downscale_factor=scaling_factor)
        if mode == 'shuffle':
            self.conv = ConvolutionalBlock(n_channels, n_channels // (scaling_factor**2), kernel_size)
        else:
            self.conv = nn.Identity()
            self.scaling = partial(F.interpolate, scale_factor= 1 / scaling_factor, mode='bilinear')

        self.prelu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)             
        output = self.scaling(output)           
        output = self.prelu(output)             

        return output


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(ResidualBlock, self).__init__()

        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
            batch_norm=True, activation=None
        )

        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
            batch_norm=True, activation='ReLU'
        )

    def forward(self, input):
        residual = input                    
        output = self.conv_block1(input)    
        output = self.conv_block2(output)   
        output = output + residual          

        return output


class DownsampleModel(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(DownsampleModel, self).__init__()
        self.conv_block1 = ConvolutionalBlock(
            in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
            batch_norm=False, activation='ReLU'
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)]
        )

        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels, out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=True, activation=None
        )

        n_sample_blocks = int(math.log2(scaling_factor))
        self.shuffle_block = nn.Sequential(
            *[DownsampleBlock(n_channels=n_channels, kernel_size=small_kernel_size, scaling_factor=2) for i
              in range(n_sample_blocks)]
        )

        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
            batch_norm=False, activation='Tanh'
        )
    
    def forward(self, x):
        output = self.conv_block1(x)                        
        residual = output                                   
        output = self.residual_blocks(output)               
        output = self.conv_block2(output)                   
        output = output + residual                          
        output = self.shuffle_block(output)                 
        sr_imgs = self.conv_block3(output)                  
        sr_imgs = torch.clip(sr_imgs, -1, 1) 

        return sr_imgs # in range [-1, 1]

class UpsampleModel(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(UpsampleModel, self).__init__()

        self.conv_block1 = ConvolutionalBlock(
            in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
            batch_norm=False, activation='ReLU'
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)]
        )

        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels, out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=True, activation='ReLU'
        )

        n_sample_blocks = int(math.log2(scaling_factor))
        self.shuffle_block = nn.Sequential(
            *[UpsampleBlock(n_channels=n_channels, kernel_size=small_kernel_size, scaling_factor=2) for i
              in range(n_sample_blocks)]
        )

        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
            batch_norm=False, activation='Tanh'
        )

    def forward(self, x):
        output = self.conv_block1(x)                        
        residual = output                                   
        output = self.residual_blocks(output)               
        output = self.conv_block2(output)                   
        output = output + residual                          
        output = self.shuffle_block(output)
        sr_imgs = self.conv_block3(output)
        sr_imgs = torch.clip(sr_imgs, -1, 1)
        
        return sr_imgs # in range [-1, 1]


class Discriminator(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, n_blocks=8, fc_size=1024):
        super(Discriminator, self).__init__()

        in_channels = 3

        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(ConvolutionalBlock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=1 if i % 2 == 0 else 2, batch_norm=i != 0, activation='LeakyReLU')
            )
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit