import torch
from torch import nn
from Attention import CBAM
from SFPANet import DASPPModule, CPAttentionModule
from ResNet import ResNet101
from functools import partial
from VGG import VGG


# using ResNet101 as backbone
class AB1(nn.Module):
    def __init__(self, num_classes):
        super(AB1, self).__init__()
        self.backbone = ResNet101()
        self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        self.out_c = 512
        self.c = 512
        self.conv1 = nn.Conv2d(512, self.c, 1, bias=False)
        self.conv2 = nn.Conv2d(1024, self.c, 1, bias=False)
        self.conv3 = nn.Conv2d(2048, self.c, 1, bias=False)
        self.pp = DASPPModule(self.c * 3, self.out_c)
        self.att = CPAttentionModule(self.out_c)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.out_c, num_classes, 1, bias=False),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        _, f1, f2, f3 = self.backbone(x)
        # print(f1.shape,f2.shape,f3.shape)
        f1=self.conv1(f1)
        f2=self.conv2(f2)
        f3=self.conv3(f3)
        f = torch.concat([f1, f2, f3], dim=1)
        out = self.pp(f)
        out = self.att(out)
        out = self.cls_seg(out)
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

# MF structure with CBAM attention
class AB2(nn.Module):
    def __init__(self, num_classes):
        super(AB2, self).__init__()
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
        self.att = CBAM(self.out_c, 3)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.out_c, num_classes, 1, bias=False),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        resnet = self.backbone[0]
        vgg = self.backbone[1]
        _, f12, f13, f14 = resnet(x)
        # print( f12.shape, f13.shape, f14.shape)
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



# eliminate DSPP module

class AB3(nn.Module):
    def __init__(self, num_classes):
        super(AB3, self).__init__()
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
        self.conv=nn.Conv2d(3072,self.out_c,3,1,1,bias=False)
        self.att = CBAM(self.out_c, 3)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.out_c, num_classes, 1, bias=False),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        resnet = self.backbone[0]
        vgg = self.backbone[1]
        _, f12, f13, f14 = resnet(x)
        # print( f12.shape, f13.shape, f14.shape)
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
        f=self.conv(f)
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



# using ResNet101 as backbone, eliminate DSPP module
class AB4(nn.Module):
    def __init__(self, num_classes):
        super(AB4, self).__init__()
        self.backbone = ResNet101()
        self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        self.out_c = 512
        self.c = 512
        self.conv1 = nn.Conv2d(512, self.c, 1, bias=False)
        self.conv2 = nn.Conv2d(1024, self.c, 1, bias=False)
        self.conv3 = nn.Conv2d(2048, self.c, 1, bias=False)
        self.conv=nn.Conv2d(self.c*3,self.out_c,3,1,1,bias=False)
        self.att = CPAttentionModule(self.out_c)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.out_c, num_classes, 1, bias=False),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        _, f1, f2, f3 = self.backbone(x)
        # print(f1.shape,f2.shape,f3.shape)
        f1=self.conv1(f1)
        f2=self.conv2(f2)
        f3=self.conv3(f3)
        f = torch.concat([f1, f2, f3], dim=1)
        out = self.conv(f)
        out = self.att(out)
        out = self.cls_seg(out)
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



if __name__ == '__main__':
    net=AB4(21)
    x=torch.ones(2,3,320,320)
    y=net(x)
    print(y.shape)