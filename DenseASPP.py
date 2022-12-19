import torch
from torch import nn
import torch.nn.functional as F
from ResNet import ResNet101
from functools import partial


class DenseASPPconv(nn.Module):
    def __init__(self, in_c, inter_c, out_c, dilate, dropout=0.1):
        super(DenseASPPconv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, inter_c, 1, bias=False),
            nn.BatchNorm2d(inter_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_c, out_c, 3, padding=dilate, dilation=dilate, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layer(x)


class DenseASPPBlock(nn.Module):
    def __init__(self, in_c, inter_c1, inter_c2):
        super(DenseASPPBlock, self).__init__()
        self.aspp_3 = DenseASPPconv(in_c, inter_c1, inter_c2, 3, 0.1)
        self.aspp_6 = DenseASPPconv(in_c + inter_c2, inter_c1, inter_c2, 6, 0.1)
        self.aspp_12 = DenseASPPconv(in_c + inter_c2 * 2, inter_c1, inter_c2, 12, 0.1)
        self.aspp_18 = DenseASPPconv(in_c + inter_c2 * 3, inter_c1, inter_c2, 18, 0.1)
        self.aspp_24 = DenseASPPconv(in_c + inter_c2 * 4, inter_c1, inter_c2, 24, 0.1)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        return x


class DenseASPPNet(nn.Module):
    def __init__(self, num_classes, dilated_scale=8):
        super(DenseASPPNet, self).__init__()
        self.backbone = ResNet101()
        self.dilated_scale = dilated_scale
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")
        self.layer = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.denseaspp = DenseASPPBlock(512, 64, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(512 + 128 * 5, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1, bias=False)
        )

    def forward(self, x):
        _, _, _, f4 = self.backbone(x)
        out = self.layer(f4)
        out = self.denseaspp(out)
        out = self.decoder(out)
        out = F.interpolate(out, scale_factor=self.dilated_scale, mode='bilinear', align_corners=True)
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
    net = DenseASPPNet(21)
    x = torch.ones(2, 3, 320, 320)
    y = net(x)
    for i in y:
        print(i.shape)
