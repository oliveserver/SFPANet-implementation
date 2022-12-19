import os.path
import re
import torchvision
import numpy as np
import torch
from PIL import Image
import random

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class Processor:
    def __init__(self, size=(256, 256), num_classes=21):
        self.num_classes = num_classes
        self.colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                  [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                  [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                                  [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                                  [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                                  [0, 64, 128]])
        self.VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
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
        t = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]
        return self.map[t]

    def from_2d_to_3d(self, pred):
        colormap = torch.tensor(self.colormap, dtype=torch.int32)
        x = torch.tensor(pred, dtype=torch.long)
        return colormap[x, :].data.cpu().numpy()


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, size=(256, 256), istrained=True):
        super(VOCDataset, self).__init__()
        self.root = r'../VOC2012'
        self.processor = Processor(size)
        self.size = size
        self.istrained = istrained
        if istrained:
            path = os.path.join(self.root, 'trains.txt')
        else:
            path = os.path.join(self.root, 'vals.txt')
        with open(path) as f:
            self.image_list = f.readlines()
        self.image_list = list(map(lambda x: x[:-1], self.image_list))

        self.image_list = list(
            filter(lambda x: int(x.split(' ')[1]) > size[0] and int(x.split(' ')[2]) > size[1], self.image_list))

        self.image_list = list(map(lambda x: x.split(' ')[0], self.image_list))
        print('using VOCDataset... istrained: {}  num of samples: {}'.format(istrained, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        source_path = os.path.join(self.root, "img", str(self.image_list[idx]) + ".jpg")
        label_path = os.path.join(self.root, "label", str(self.image_list[idx]) + ".png")
        img = Image.open(source_path)
        label = Image.open(label_path).convert('RGB')
        img, label = self.voc_rand_crop(img, label)
        img, label = self.random_flip(img, label)
        img = self.processor.transform(img)
        label = np.array(label).astype('int32')
        label = self.processor.from_3d_to_2d(label)
        label = label.long()
        return img, label

    def voc_rand_crop(self, img, label):
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


if __name__ == "__main__":
    d = VOCDataset((1,1),False)
    # img, label = d.__getitem__(2)
    # label = d.processor.from_2d_to_3d(label)
    # plt.imshow(label)
    # plt.show()

    # train_loader = DataLoader(d, 16, shuffle=False, drop_last=True)

