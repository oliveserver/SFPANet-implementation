from functools import partial

import torch
from torch import nn

from Attention import CrissCrossAttention
from ResNet import ResNet101


class RCCAModule(nn.Module):
    def __init__(self, num_classes, in_c, recurrence=2):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        ratio = 32
        self.in_c = in_c
        self.inter_c = in_c // ratio
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_c, self.inter_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.inter_c),
            nn.ReLU(inplace=True)
        )
        self.CCA = CrissCrossAttention(self.inter_c)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_c, self.inter_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.inter_c),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.in_c + self.inter_c, self.inter_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.inter_c),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(self.inter_c, num_classes, 1, bias=False)
        )

    def forward(self, x):
        out = self.conv_in(x)
        for i in range(self.recurrence):
            out = self.CCA(out)
        out = self.conv_out(out)
        out = torch.cat([x, out], dim=1)
        out = self.cls_seg(out)
        return out


class CCNet(nn.Module):
    def __init__(self, num_classes, dilated_scale=8, use_aux=True):
        super(CCNet, self).__init__()
        self.backbone = ResNet101()
        self.use_aux = use_aux
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")
        self.master_decoder = RCCAModule(num_classes, 2048)
        if use_aux:
            self.aux_decoder = RCCAModule(num_classes, 1024)

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
    net = CCNet(21)
    x = torch.ones(1, 3, 320, 320)
    y = net(x)
    for i in y:
        print(i.shape)
