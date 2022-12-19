import torch
from torch import nn
import torchvision

resnet50 = torchvision.models.resnet50(pretrained=True)
resnet101 = torchvision.models.resnet101(pretrained=True)


class ResNet50(nn.Module):
    def __init__(self, use_f=True):
        super(ResNet50, self).__init__()
        self.use_f = use_f
        # self.add_conv0 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        # self.add_bn0 = nn.BatchNorm2d(64)
        # self.add_relu0 = nn.ReLU(inplace=True)
        # self.add_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        # nn.init.kaiming_normal_(self.add_conv0.weight)
        # nn.init.kaiming_normal_(self.add_conv1.weight)

    def forward(self, x):
        # x = self.add_conv0(x)
        # x = self.add_bn0(x)
        # x = self.add_relu0(x)
        # x = self.add_conv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x
        if self.use_f:
            return [f1, f2, f3, f4]
        else:
            return x


class ResNet101(nn.Module):
    def __init__(self, use_f=True):
        super(ResNet101, self).__init__()
        self.use_f = use_f
        # self.add_conv0 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        # self.add_bn0 = nn.BatchNorm2d(64)
        # self.add_relu0 = nn.ReLU(inplace=True)
        # self.add_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv1 = resnet101.conv1
        self.bn1 = resnet101.bn1
        self.relu = resnet101.relu
        self.maxpool = resnet101.maxpool
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4
        # nn.init.kaiming_normal_(self.add_conv0.weight)
        # nn.init.kaiming_normal_(self.add_conv1.weight)

    def forward(self, x):
        # x = self.add_conv0(x)
        # x = self.add_bn0(x)
        # x = self.add_relu0(x)
        # x = self.add_conv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x
        if self.use_f:
            return [f1, f2, f3, f4]
        else:
            return x



if __name__ == '__main__':
    net = ResNet50()
    x = torch.ones(1, 3, 320, 320)
    y = net(x)
    for i in y:
        print(i.shape)
    # print(net)
