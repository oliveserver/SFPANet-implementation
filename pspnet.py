from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from ResNet import ResNet101


class PSPModule(nn.Module):
    def __init__(self, in_c, pool_sizes):
        super(PSPModule, self).__init__()
        out_c = in_c // len(pool_sizes)
        self.stages = nn.ModuleList([self.make_layer(in_c, out_c, size) for size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c + out_c * len(pool_sizes), out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def make_layer(self, in_c, out_c, size):
        return nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=size),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        pyramids = [x]
        pyramids.extend(
            [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        out = self.bottleneck(torch.cat(pyramids, dim=1))
        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes, dilated_scale=8, use_aux=True):
        super(PSPNet, self).__init__()
        self.backbone = ResNet101()
        self.use_aux = use_aux
        if dilated_scale == 8:
            self.backbone.layer3.apply(partial(self.change_dilate, dilate=2))
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=4))
        elif dilated_scale == 16:
            self.backbone.layer4.apply(partial(self.change_dilate, dilate=2))
        else:
            raise ValueError("Invalid dilated_scale it must be 8 or 16.")

        aux_c = 1024
        out_c = 2048
        self.master_branch = nn.Sequential(
            PSPModule(out_c, (1, 2, 3, 6)),
            nn.Conv2d(out_c // 4, num_classes, 1, bias=False)
        )

        if use_aux:
            self.aux_branch = nn.Sequential(
                nn.Conv2d(aux_c, out_c // 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_c // 8, num_classes, 1, bias=False)
            )

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        _, _, f3, f4 = self.backbone(x)
        # print(f1.shape,f2.shape,f3.shape,f4.shape)
        aux_out = f3
        out = f4
        out = self.master_branch(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        if self.use_aux:
            aux_out = self.aux_branch(aux_out)
            aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
            return aux_out, out
        else:
            return out

    # padding 与 dilation抵消，shape不变，本质上就是将stride由2变为1，去掉了下采样，加入空洞卷积
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
    net = PSPNet(21, use_aux=True)
    print(net)
    # x = torch.ones(2, 3, 320, 320)
    # y = net(x)
    # for i in y:
    #     print(i.shape)
