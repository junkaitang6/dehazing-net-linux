import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialPriorAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.spatial_att(x)
        return x * att


class LargeKernelBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=21, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv_dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, dilation=dilation)
        self.conv_pw = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv_pw(self.conv_dw(x))))

class LSNet_Dehaze(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.enc1 = LargeKernelBlock(3, base_ch, kernel_size=21)
        self.enc2 = LargeKernelBlock(base_ch, base_ch*2, kernel_size=21)
        self.enc3 = LargeKernelBlock(base_ch*2, base_ch*4, kernel_size=31)

        self.spa1 = SpatialPriorAttention(base_ch)
        self.spa2 = SpatialPriorAttention(base_ch*2)
        self.spa3 = SpatialPriorAttention(base_ch*4)

        self.dec3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1)
        self.dec1 = nn.Conv2d(base_ch, 3, 3, 1, 1)

    def forward(self, x):
        e1 = self.spa1(self.enc1(x))
        e2 = self.spa2(self.enc2(F.max_pool2d(e1, 2)))
        e3 = self.spa3(self.enc3(F.max_pool2d(e2, 2)))

        d3 = F.relu(self.dec3(e3) + e2)
        d2 = F.relu(self.dec2(d3) + e1)
        out = torch.tanh(self.dec1(d2)) * 0.5 + 0.5
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lightdehaze_net(model_type="original", **kwargs):
    if model_type == "original":
        return LSNet_Dehaze()


