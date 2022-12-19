from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from Attention import PositionAttention, ChannelAttention
from ResNet import ResNet101


class DAModule(nn.Module):
    def __init__(self, in_c, num_classes):
        super(DAModule, self).__init__()
        ratio = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c // ratio, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c // ratio),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c // ratio, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c // ratio),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c // ratio, in_c // ratio, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c // ratio),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_c // ratio, in_c // (ratio * 2), 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c // (ratio * 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // (ratio * 2), num_classes, 3, 1, 1, bias=False)
        )
        self.pos_att = PositionAttention(in_c // ratio)
        self.cha_att = ChannelAttention(in_c // ratio)

    def forward(self, x):
        x_PA = self.conv1(x)
        x_CA = self.conv2(x)

        PA = self.pos_att(x_PA)
        CA = self.cha_att(x_CA)

        out = self.conv3(PA + CA)
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.conv4(out)
        return out


class DANet(nn.Module):
    def __init__(self, num_classes, dilated_scale=8, use_aux=True):
        super(DANet, self).__init__()
        self.backbone = ResNet101()
        self.use_aux = use_aux
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")
        self.master_decoder = DAModule(2048, num_classes)
        if use_aux:
            self.aux_decoder = DAModule(1024, num_classes)

    def forward(self, x):
        _, _, f3, f4 = self.backbone(x)
        out = self.master_decoder(f4)
        if self.use_aux:
            aux_out = self.aux_decoder(f3)
            return aux_out, out
        else:
            return out

    def change_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == "__main__":
    x = torch.ones(1, 3, 320, 320)
    net = DANet(21, use_aux=False)
    y = net(x)
    for i in y:
        print(i.shape)
