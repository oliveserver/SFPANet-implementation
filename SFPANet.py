import torch
from torch import nn
import torch.nn.functional as F
from ResNet import ResNet101
from functools import partial
from VGG import VGG

d1 = 0.5
d2 = 0.1


class PPconv(nn.Module):
    def __init__(self, in_c, out_c, dilate):
        super(PPconv, self).__init__()
        kernel_size = 1 if dilate == 1 else 3
        padding = 0 if dilate == 1 else dilate
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilate, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(d1)
        )

    def forward(self, x):
        return self.layer(x)


class DASPPModule(nn.Module):
    def __init__(self, in_c, out_c):
        super(DASPPModule, self).__init__()
        self.pp_1 = PPconv(in_c, out_c, 1)
        self.pp_6 = PPconv(in_c + out_c, out_c, 6)
        self.pp_12 = PPconv(in_c + out_c * 2, out_c, 12)
        self.pp_18 = PPconv(in_c + out_c * 3, out_c, 18)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_c + out_c * 5, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(d1)
        )

    def forward(self, x):
        n, c, h, w = x.shape
        y = self.avg_pool(x)
        y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)

        pp1 = self.pp_1(x)
        x = torch.cat([x, pp1], dim=1)

        pp6 = self.pp_6(x)
        x = torch.cat([x, pp6], dim=1)

        pp12 = self.pp_12(x)
        x = torch.cat([x, pp12], dim=1)

        pp18 = self.pp_18(x)
        x = torch.cat([x, pp18], dim=1)

        x = torch.cat([x, y], dim=1)
        x = self.layer(x)
        return x


class CPAttentionModule(nn.Module):
    def __init__(self, in_c):
        super(CPAttentionModule, self).__init__()
        # k 需要调整
        self.k = 192
        self.conv1 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, in_c, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Dropout(d2)
        )
        self.softmax = nn.Softmax(dim=2)

        # num 需要调整
        self.num = 64
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.MLP = nn.Sequential(
            nn.Linear(in_c, self.num, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.num, in_c, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        n, c, h, w = out.shape
        out = out.view(n, c, h * w)
        print(out.shape)
        att = self.linear_0(out)
        print(att.shape)
        att = self.softmax(att)
        att = att / (1e-5 + att.sum(dim=1, keepdim=True))
        out = self.linear_1(att)
        out = out.view(n, c, h, w)
        out = self.layer(out)
        avg_out = self.avg_pool(out).squeeze()
        max_out = self.max_pool(out).squeeze()
        avg_out = self.MLP(avg_out)
        max_out = self.MLP(max_out)
        att_out = avg_out + max_out
        att_out = self.sigmoid(att_out).view(n, c, 1, 1)
        out = out * att_out
        out = out + x
        return out


class SFPANet(nn.Module):
    def __init__(self, num_classes):
        super(SFPANet, self).__init__()
        self.backbone = nn.ModuleList([ResNet101(), VGG()])
        self.backbone[0].layer3.apply(partial(self.change_dilate, dilate=2))
        self.backbone[0].layer4.apply(partial(self.change_dilate, dilate=4))
        self.backbone[1].l3.apply(partial(self.change_dilate, dilate=2))
        self.backbone[1].l4.apply(partial(self.change_dilate, dilate=4))
        self.c = 512
        self.conv11 = nn.Conv2d(512, self.c, 1, bias=False)
        self.conv12 = nn.Conv2d(1024, self.c, 1, bias=False)
        self.conv13 = nn.Conv2d(2048, self.c, 1, bias=False)
        self.conv21 = nn.Conv2d(512, self.c, 1, bias=False)
        self.conv22 = nn.Conv2d(512, self.c, 1, bias=False)
        self.conv23 = nn.Conv2d(512, self.c, 1, bias=False)
        self.inter_c = 256
        self.out_c = 256
        self.pp = DASPPModule(self.c * 6, self.out_c)
        self.att = CPAttentionModule(self.out_c)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.out_c, num_classes, 1, bias=False),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        resnet = self.backbone[0]
        vgg = self.backbone[1]
        _, f12, f13, f14 = resnet(x)
        print( f12.shape, f13.shape, f14.shape)
        _, f22, f23, f24 = vgg(x)
        # print(f21.shape, f22.shape, f23.shape, f24.shape)
        f12 = self.conv11(f12)
        f13 = self.conv12(f13)
        f14 = self.conv13(f14)
        f22 = self.conv21(f22)
        f23 = self.conv22(f23)
        f24 = self.conv23(f24)
        # print(f12.shape, f13.shape, f14.shape, f22.shape, f23.shape, f24.shape)
        f = torch.cat([f12, f13, f14, f22, f23, f24], dim=1)
        # print(f.shape)
        f = self.pp(f)
        f = self.att(f)
        f = self.cls_seg(f)
        return f

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


if __name__ == '__main__':
    x = torch.ones(2, 3, 320, 320)
    net = SFPANet(21)
    y = net(x)
    print(y.shape)
