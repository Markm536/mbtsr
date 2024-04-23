from PIL import Image
import numpy as np
from pathlib import Path
import os
import cv2

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# import albumentations as A
from tqdm import tqdm

from CompressAI.compressai.transforms.transforms import RGB2YCbCr, YCbCr2RGB, YUV444To420, YUV420To444
from torchvision.transforms import ToTensor

def read_image(filepath: str) -> torch.Tensor: 
    """
    Read filepath image to torch.Tensor in range [0.0, 1.0]
    """
    # assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return ToTensor()(img)

def human_order(x: torch.Tensor):
    return x.squeeze(0).permute(1, 2, 0)

def tonp(img: torch.Tensor):
    return img.detach().numpy()

def tomat(img: torch.Tensor) -> cv2.Mat:
    res = cv2.Mat((tonp(img) * 255)[..., ::-1].astype(np.uint8))
    return res

def topil(img: torch.Tensor) -> cv2.Mat:
    res = Image.fromarray((tonp(img) * 255).astype(np.uint8))
    return res

def cv2pil(x):
    res = np.array(x)[..., ::-1]
    res = cv2.Mat(res)
    return res

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
        return img_mat