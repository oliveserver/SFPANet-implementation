from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from ResNet import ResNet101


class EAModule(nn.Module):
    def __init__(self, in_c):
        super(EAModule, self).__init__()
        self.k = 64
        self.conv1 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, in_c, 1, bias=False),
            nn.BatchNorm2d(in_c)
        )
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        n, c, h, w = out.shape
        out = out.view(n, c, h * w)
        att = self.linear_0(out)
        att = self.softmax(att)
        att = att / (1e-9 + att.sum(dim=1, keepdim=True))
        out = self.linear_1(att)
        out = out.view(n, c, h, w)
        out = self.layer(out)
        out = out + x
        out = self.relu(out)
        return out


class EANet(nn.Module):
    def __init__(self, num_classes, dilated_scale=8):
        super(EANet, self).__init__()
        self.backbone = ResNet101()
        self.dilated_scale = dilated_scale
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")
        self.layer1 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.att = EAModule(512)
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.upsample = nn.Upsample(scale_factor=self.dilated_scale, mode='bilinear', align_corners=True)
        self.layer3 = nn.Conv2d(256, num_classes, 1, bias=False)

    def forward(self, x):
        _, _, _, f4 = self.backbone(x)
        out = self.layer1(f4)
        out = self.att(out)
        out = self.layer2(out)
        out = self.upsample(out)
        out = self.layer3(out)
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
    net = EANet(21)
    x = torch.ones(2, 3, 320, 320)
    y = net(x)
    for i in y:
        print(i.shape)
