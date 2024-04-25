from CompressAI.compressai.models.google import MeanScaleHyperprior
from CompressAI.compressai.zoo.image import mbt2018_mean

from typing import Tuple

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def setup_codec(quality: int = 3, metric: str = "mse"):
    model: MeanScaleHyperprior
    model = mbt2018_mean(quality, metric, True)
    freeze_model(model)
    return model
    
