from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from ResNet import ResNet101


class ASPP(nn.Module):
    def __init__(self, in_c, stride=8):
        super(ASPP, self).__init__()
        assert stride in [8, 16], 'stride must be 8 or 16!'
        if stride == 8:
            dilations = [1, 6, 12, 18]
        else:
            dilations = [1, 12, 24, 36]
        self.aspp1 = self.make_aspp(in_c, 256, 1, dilations[0])
        self.aspp2 = self.make_aspp(in_c, 256, 3, dilations[1])
        self.aspp3 = self.make_aspp(in_c, 256, 3, dilations[2])
        self.aspp4 = self.make_aspp(in_c, 256, 3, dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        n, c, h, w = x.size()
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(h, w), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat([x1, x2, x3, x4, x5], dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        return x

    def make_aspp(self, in_c, out_c, kernel_size, dilation):
        padding = 0 if kernel_size == 1 else dilation
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


class Decoder(nn.Module):
    def __init__(self, low_c, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_c, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, bias=False),
        )

    def forward(self, x, low_x):
        low_x = self.conv1(low_x)
        low_x = self.relu(self.bn1(low_x))
        n, c, h, w = low_x.shape
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = self.output(torch.cat([x, low_x], dim=1))
        return x


class DeepLab(nn.Module):
    def __init__(self, num_classes, dilated_scale=16, output_stride=8):
        super(DeepLab, self).__init__()
        self.backbone = ResNet101()
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")
        low_c = 256
        self.ASPP = ASPP(2048, output_stride)
        self.decoder = Decoder(low_c, num_classes)

    def forward(self, x):
        n, c, h, w = x.shape
        f1, f2, f3, f4 = self.backbone(x)
        out = self.ASPP(f4)
        out = self.decoder(out, f1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
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
    net = DeepLab(21)
    x = torch.ones(2, 3, 320, 320)
    y = net(x)
    for i in y:
        print(i.shape)
