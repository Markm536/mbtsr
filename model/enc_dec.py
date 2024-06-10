from CompressAI.compressai.models.google import MeanScaleHyperprior
from CompressAI.compressai.zoo.image import mbt2018_mean

from typing import Tuple
import torchvision
import torch.nn as nn

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def setup_codec(quality: int = 3, metric: str = "mse"):
    model: MeanScaleHyperprior
    model = mbt2018_mean(quality, metric, True)
    freeze_model(model)
    return model
    
class TruncatedVGG19(nn.Module):
    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()

        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        output = self.truncated_vgg19(input)
        return output