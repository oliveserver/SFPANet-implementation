import torch
from torch import nn

from bilinear_kernel import bilinear_kernel
from ResNet import ResNet101


# FCN using backbone: ResNet101
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.backbone = ResNet101()
        self.conv1 = nn.Conv2d(512, num_classes, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(1024, num_classes, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(2048, num_classes, 3, 1, 1, bias=False)

        self.upsample_2x_1 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1)
        self.upsample_2x_2 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1)

        self.upsample_8x = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1),
            nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1),
            nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1)
        )

        for i in self.modules():
            if isinstance(i, nn.ConvTranspose2d):
                i.weight.data.copy_(bilinear_kernel(i.in_channels, i.out_channels, i.kernel_size[0]))

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
        f2 = self.conv1(f2)
        f3 = self.conv2(f3)
        f4 = self.conv3(f4)

        f4 = self.upsample_2x_1(f4)
        f3 += f4

        f3 = self.upsample_2x_1(f3)
        f2 += f3

        f2 = self.upsample_8x(f2)
        return f2


if __name__ == "__main__":
    x=torch.ones(1,3,320,320)
    net=FCN(21)
    print(net(x).shape)