import os.path

import numpy as np
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image


class Processor:
    def __init__(self, size=(320, 320)):
        self.colormap = np.array([[0, 0, 0], [0, 128, 192], [128, 0, 0], [64, 0, 128],
                                  [192, 192, 128], [64, 64, 128], [64, 64, 0], [128, 64, 128],
                                  [0, 0, 192], [192, 128, 128], [128, 128, 128], [128, 128, 0]])
        self.map = torch.zeros(256 ** 3, dtype=torch.uint8)
        self.size = size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1),
        ])
        for i, m in enumerate(self.colormap):
            t = (m[0] * 256 + m[1]) * 256 + m[2]
            self.map[t] = i

    def from_3d_to_2d(self, label):
        label = np.array(label, dtype=np.int32)
        t = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]
        return self.map[t]

    def from_2d_to_3d(self, pred):
        colormap = torch.tensor(self.colormap, dtype=torch.int32)
        x = torch.tensor(pred, dtype=torch.long)
        return colormap[x, :].data.cpu().numpy()

    # 这里要修改！！！！！
    def rand_crop(self, img, label):
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=self.size)
        img = torchvision.transforms.functional.crop(img, i, j, h, w)
        label = torchvision.transforms.functional.crop(label, i, j, h, w)
        return img, label

    def random_flip(self, img, label):
        p = np.random.choice([0, 1])
        q = np.random.choice([0, 1])
        t = torchvision.transforms.RandomHorizontalFlip(p)
        img = t(img)
        label = t(label)
        t = torchvision.transforms.RandomVerticalFlip(q)
        img = t(img)
        label = t(label)
        return img, label


class CamVidDataset(torch.utils.data.Dataset):
    def __init__(self, size=(320, 320), istrained=True):
        super(CamVidDataset, self).__init__()
        self.processor = Processor(size=size)
        self.root = r'../CamVid'
        if istrained:
            with open(os.path.join(self.root, 'camvid_traintest.txt')) as f:
                l = f.readlines()
                self.image_list = [i.split(' ')[0] for i in l]
                self.label_list = [i.split(' ')[1][:-1] for i in l]

            self.image_list = list(map(lambda x: self.root + '/' + str(x), self.image_list))
            self.label_list = list(map(lambda x: self.root + '/' + str(x), self.label_list))
        else:
            with open(os.path.join(self.root, 'camvid_val.txt')) as f:
                l = f.readlines()
                self.image_list = [i.split(' ')[0] for i in l]
                self.label_list = [i.split(' ')[1][:-1] for i in l]

            self.image_list = list(map(lambda x: self.root + '/' + str(x), self.image_list))
            self.label_list = list(map(lambda x: self.root + '/' + str(x), self.label_list))
        print('using CamVidDataset... istrained: {}  num of samples: {}'.format(istrained,len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img = Image.open(self.image_list[item]).convert('RGB')
        label = Image.open(self.label_list[item]).convert('RGB')
        img, label = self.processor.rand_crop(img, label)
        img, label = self.processor.random_flip(img, label)
        img = self.processor.transform(img)
        label = np.array(label).astype('int32')
        label = self.processor.from_3d_to_2d(label)
        label = label.long()
        return img, label


if __name__ == '__main__':
    data = CamVidDataset(size=(720, 960), istrained=True)
    # img, label= data.__getitem__(72)
    # label = np.array(label)
    # fig, ax = plt.subplots(1, 2)
    # t1 = data.processor.from_2d_to_3d(label)
    # ax[0].imshow(t1)
    # ax[1].imshow(ll)
    # plt.show()
