import numpy as np
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt


class Processor:
    def __init__(self, size=(320, 320)):
        self.colormap = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                                  [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                                  [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                                  [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                                  [0, 0, 230], [119, 11, 32]])
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


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, size=(320, 320), istrained=True):
        super(CityscapesDataset, self).__init__()
        self.processor = Processor(size=size)
        if istrained:
            self.dataset = torchvision.datasets.Cityscapes('../cityscapes', 'train', 'fine', 'color')
        else:
            self.dataset = torchvision.datasets.Cityscapes('../cityscapes', 'val', 'fine', 'color')
        print('using CityscapesDataset... istrained: {}  num of samples: {}'.format(istrained, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        img = img.convert('RGB')
        label = label.convert('RGB')
        img, label = self.processor.rand_crop(img, label)
        img, label = self.processor.random_flip(img, label)
        img = self.processor.transform(img)
        label = np.array(label).astype('int32')
        label = self.processor.from_3d_to_2d(label).long()
        return img, label


if __name__ == '__main__':
    size = (1024, 2048)
    dataset = CityscapesDataset(istrained=True, size=size)
    img1, label1 = dataset.__getitem__(200)
    label1 = label1.numpy()
    label1 = dataset.processor.from_2d_to_3d(label1)
    plt.imshow(label1)
    plt.show()
