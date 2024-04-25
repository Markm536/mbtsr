import torch 
import torch.nn as nn

from model.enc_dec import setup_codec
from CompressAI.compressai.models.google import MeanScaleHyperprior
from model.sr import DownsampleModel, SRGenerator


class SRCompress(nn.Module):
    def __init__(self, use_residuals=False):
        super().__init__()
        self.down = DownsampleModel()
        self.ups = SRGenerator()

        self.codec: MeanScaleHyperprior
        self.codec = setup_codec()
        
        self.use_res = use_residuals

    def forward(self, x):
        x_down = self.down(x).permute(0, 3, 1, 2)
        x_down_dec = self.codec(x_down)['x_hat'].permute(0, 2, 3, 1)
        x_ups_dec = self.ups(x_down_dec).permute(0, 3, 1, 2)

        if self.use_res == False:
            return x_ups_dec
        
        # ???
        res = x - x_ups_dec
        # from [-1, 1] to [0, 1]
        res = (res + 1) / 2
        res_dec = self.codec(res_dec)
        res = res * 2 - 1
        res = x_ups_dec + res
        return res  