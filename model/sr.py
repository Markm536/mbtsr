import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.model = nn.Sequential(
            # TODO: add layers
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(filters, momentum=0.8),
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters, momentum=0.8)
        )

    def forward(self, z):
        return self.model(z) + z

class UpsamplingBlock(nn.Module):
    def __init__(self, filters: int, input_filters: int = None):
        super().__init__()

        if input_filters is None:
          input_filters = filters

        self.model = nn.Sequential(
            # TODO: add layers
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_filters, filters, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, z):
        return self.model(z)

class SRGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=9 // 2, padding_mode="zeros"),
            nn.ReLU(inplace=True),
        )
        self.residual_chain = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)],
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2, padding_mode="zeros"),
            nn.BatchNorm2d(64, momentum=0.8)
        )
        self.upsample_conv = nn.Sequential(
            UpsamplingBlock(256, input_filters=64),
            UpsamplingBlock(256),
            nn.Conv2d(256, 3, kernel_size=9, padding=9 // 2, padding_mode="zeros"),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        conv = self.init_conv(x)
        x = self.residual_chain(conv) + conv
        img = self.upsample_conv(x)
        # make channel axis last
        img = img.permute(0, 2, 3, 1)
        return img

class DBlock(nn.Module):
    def __init__(
        self,
        input_filters: int,
        filters: int,
        kernels: int,
        strides: int = 1,
        batch_norm: bool = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(input_filters, filters, kernel_size=kernels, stride=strides, padding=kernels // 2, padding_mode="zeros"),
            nn.LeakyReLU(0.2)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(filters, momentum=0.8))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

class SRDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            DBlock(3, 64, 1, True),
            DBlock(64, 64, 2),
            DBlock(64, 128, 1),
            DBlock(128, 128, 2),
            DBlock(128, 256, 1),
            DBlock(256, 256, 2),
            DBlock(256, 512, 1),
            DBlock(512, 512, 2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = z.permute(0, 3, 1, 2)
        return self.model(x)
    
class DownsampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = DBlock(3, 8, 3)
        self.x2 = DBlock(8, 16, 3)
        self.x3 = DBlock(16, 32, 3)
        self.x4 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DBlock(59, 32, 1),
            nn.Sequential(DBlock(32, 16, 3), DBlock(16, 16, 3)),
            nn.MaxPool2d((2, 2)),
            nn.Sequential(DBlock(16, 16, 3), DBlock(16, 8, 3), DBlock(8, 8, 1), DBlock(8, 3, 1)),
        )
    
    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2) 
        x = torch.concat([x, x1, x2, x3], dim=1)
        x = self.x4(x)
        x = x.permute(0, 2, 3, 1)
        return x