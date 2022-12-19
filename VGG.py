from functools import partial

import torch
from torch import nn
import torchvision
import torch.nn.functional as F

vgg16 = torchvision.models.vgg16(pretrained=True)


class VGG(nn.Module):
    def __init__(self, use_f=True):
        super(VGG, self).__init__()
        self.use_f = use_f
        self.l1 = nn.Sequential(*list(vgg16.features.children())[:12])
        self.l2 = nn.Sequential(*list(vgg16.features.children())[12:19])
        self.l3 = nn.Sequential(*list(vgg16.features.children())[19:24])
        self.l4 = nn.Sequential(*list(vgg16.features.children())[24:])

    def forward(self, x):
        x = self.l1(x)
        f1 = x
        x = self.l2(x)
        f2 = x
        x = self.l3(x)
        f3 = x
        x = self.l4(x)
        f4 = x
        f3 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, scale_factor=4, mode='bilinear', align_corners=True)
        if self.use_f:
            return [f1, f2, f3, f4]
        else:
            return x

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
    net = VGG()
    y = net(x)
    for i in y:
        print(i.shape)
    # print(net.l1)
    # print(net.l2)
    # print(net.l3)
    # print(net.l4)
