from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from ResNet import ResNet101


class GCBlock(nn.Module):
    def __init__(self, in_c, ratio=64):
        super(GCBlock, self).__init__()
        out_c = in_c // ratio

        self.conv_a = nn.Conv2d(in_c, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv_v = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, in_c, 1, bias=False)
        )

    def forward(self, x):
        n, c, h, w = x.shape
        # n 1 h w -> n 1 h*w -> n h*w 1
        att = self.softmax(self.conv_a(x).view(n, 1, -1).permute(0, 2, 1).view(n, -1, 1))
        # n c h*w
        V = x.view(n, c, h * w)
        out = torch.bmm(V, att).view(n, c, 1, 1)
        # n c 1 1
        out = self.conv_v(out)
        out = x + out
        return out


class GCNet(nn.Module):
    def __init__(self, num_classes, dilated_scale=8):
        super(GCNet, self).__init__()
        self.backbone = ResNet101()
        self.dilated_scale = dilated_scale
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")
        self.GC = GCBlock(2048)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=self.dilated_scale, mode='bilinear', align_corners=True)
        )
        self.cls_seg = nn.Conv2d(256, num_classes, 3, 1, 1, bias=False)

    def forward(self, x):
        _, _, _, f4 = self.backbone(x)
        f4 = self.GC(f4)
        f4 = self.conv1(f4)
        f4 = self.cls_seg(f4)
        return f4

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
    x = torch.ones(2, 3, 320, 320)
    GCBlock = GCNet(21)
    out = GCBlock(x)
    print(out.shape)
