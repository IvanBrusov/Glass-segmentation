import torch
import torch.nn as nn
import torch.nn.functional as F


class CNA(nn.Module):
    def __init__(self, in_inc, out_nc, stride=1, dropout=0.5):
        super().__init__()

        self.conv = nn.Conv2d(in_inc, out_nc, 3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_nc)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        return out


class UnetUnit(nn.Module):
    def __init__(self, in_ic, inner_nc, out_nc, inner_unit=None, dropout=0.5):
        super().__init__()

        self.conv1 = CNA(in_ic, inner_nc, stride=2, dropout=dropout)
        self.conv2 = CNA(inner_nc, inner_nc, dropout=dropout)
        self.inner_unit = inner_unit
        self.conv3 = CNA(inner_nc, inner_nc, dropout=dropout)
        self.conv_cat = nn.Conv2d(in_ic+inner_nc, out_nc, 3, padding=1)


    def forward(self, x):
        _, _, h, w = x.shape

        inner = self.conv1(x)
        inner = self.conv2(inner)
        if self.inner_unit is not None:
            inner = self.inner_unit(inner)
        inner = self.conv3(inner)

        inner = F.interpolate(inner, size=(h, w), mode='bilinear')
        inner = torch.cat((x, inner), axis=1)
        out = self.conv_cat(inner)
        return out


class UNet(nn.Module):
    def __init__(self, in_nc=1, nc=32, out_nc=1, dropout=0.5):
        super().__init__()

        self.cna1 = CNA(in_nc, nc)
        self.cna2 = CNA(nc, nc)

        unet_unit = UnetUnit(8*nc, 8*nc, 8*nc, dropout=dropout)
        unet_unit = UnetUnit(4*nc, 8 * nc, 4*nc, unet_unit, dropout=dropout)
        unet_unit = UnetUnit(2*nc, 4 * nc, 2*nc, unet_unit, dropout=dropout)
        self.unet_unit = UnetUnit(nc, 2 * nc, nc, unet_unit, dropout=dropout)

        self.cna3 = CNA(nc, nc, dropout=dropout)

        self.conv_last = nn.Conv2d(nc, out_nc, 3, padding=1)


    def forward(self, x):
        out = self.cna1(x)
        out = self.cna2(out)
        out = self.unet_unit(out)
        out = self.cna3(out)
        out = self.conv_last(out)
        return out
