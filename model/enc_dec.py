from CompressAI.compressai.models.google import MeanScaleHyperprior
from CompressAI.compressai.zoo.image import mbt2018_mean

from typing import Tuple

def setup_codec(quality: int = 3, metric: str = "mse"):
    model: MeanScaleHyperprior
    model = mbt2018_mean(quality, metric, True)
    return model
    
