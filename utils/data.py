from PIL import Image
import numpy as np
from pathlib import Path
import os
import cv2

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from tqdm import tqdm
import torchvision.transforms.functional as FT

# from CompressAI.compressai.transforms.transforms import RGB2YCbCr, YCbCr2RGB, YUV444To420, YUV420To444
from torchvision.transforms import ToTensor

def read_image(filepath: str) -> torch.Tensor: 
    img = Image.open(filepath).convert("RGB")
    return ToTensor()(img)

def psnr(x, y):
    x = convert_image(x, '[-1, 1]', '[0, 1]')
    y = convert_image(y, '[-1, 1]', '[0, 1]')
    psnr = -10 * torch.log10(((x - y)**2).mean())
    return psnr

def cnt_bpp(res, num_pix):
    bpp = sum(len(s[0]) for s in res["strings"]) * 8.0 / num_pix
    return bpp

class ImagesDataloader(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.imgs_names = sorted(os.listdir(str(img_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, index):
        img_path = self.img_dir / self.imgs_names[index]
        img_mat = read_image(img_path)

        if self.transform:
            img_mat = self.transform(img_mat)

        # img_mat = img_mat.transpose(2, 0, 1)
        img_mat = convert_image(img_mat, '[0, 1]', '[-1, 1]')
        return img_mat
    
def convert_image(img, source, target, imagenet_mean_cuda = None, imagenet_std_cuda = None):
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    return img