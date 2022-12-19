import torch.optim
from torch.utils.data import DataLoader

from FCN import FCN
from DANet import DANet
from CCNet import CCNet
from DenseASPP import DenseASPPNet
from EANet import EANet
from SFPANet import SFPANet
from GCNet import GCNet
from deeplab import DeepLab
from pspnet import PSPNet
from Train import *
from VOCDataset import VOCDataset
from CityscapesDataset import CityscapesDataset
from CamVidDataset import CamVidDataset
import torch.distributed as dist
import argparse


def set_up_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_up_seed(43)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', help='local device id currently', type=int)
args = parser.parse_args()

classes = [21, 19, 12]

use_aux = False
using_data = 0  # 0 ->VOC 1 ->Cityscape 2 ->CamVid
total_epoch = 200
# 0 -> FCN 1 ->PSPNet 2 ->DANet 3 ->CCNet 4 -> DeepLab 5 -> DenseASPPNet
# 6 -> GCNet 7 ->EANet 8 ->SFPANet
using_net_num = 0
n_gpus = 1
num_workers = 4 * n_gpus
num_classes = classes[using_data]
lr = n_gpus * 5e-4

torch.distributed.init_process_group('nccl', world_size=n_gpus, rank=args.local_rank)
torch.cuda.set_device(args.local_rank)

if using_data == 0:
    size = (224, 224)
    batch_size = 128
    train_data = VOCDataset(istrained=True, size=size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_data = VOCDataset(istrained=False, size=size)
    train_loader = DataLoader(train_data, batch_size, shuffle=False, drop_last=True, num_workers=num_workers,
                              sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True, num_workers=num_workers,
                             pin_memory=True)
elif using_data == 1:
    size = (480, 480)
    batch_size = 8
    train_data = CityscapesDataset(istrained=True, size=size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_data = CityscapesDataset(istrained=False, size=size)
    train_loader = DataLoader(train_data, batch_size, shuffle=False, drop_last=True, num_workers=num_workers,
                              sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True, num_workers=num_workers,
                             pin_memory=True
                             )
elif using_data == 2:
    size = (384, 480)
    batch_size = 8
    train_data = CamVidDataset(istrained=True, size=size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_data = CamVidDataset(istrained=False, size=size)
    train_loader = DataLoader(train_data, batch_size, shuffle=False, drop_last=True, num_workers=num_workers,
                              sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True, num_workers=num_workers,
                             pin_memory=True)

if using_net_num == 0:
    net = FCN(num_classes)
elif using_net_num == 1:
    net = PSPNet(num_classes, use_aux=use_aux)
elif using_net_num == 2:
    net = DANet(num_classes, use_aux=use_aux)
elif using_net_num == 3:
    net = CCNet(num_classes, use_aux=use_aux)
elif using_net_num == 4:
    net = DeepLab(num_classes)
elif using_net_num == 5:
    net = DenseASPPNet(num_classes)
elif using_net_num == 6:
    net = GCNet(num_classes)
elif using_net_num == 7:
    net = EANet(num_classes)
else:
    net = SFPANet(num_classes)

base_params = net.backbone.parameters()
base_id = list(map(id, base_params))
other_params = list(filter(lambda p: id(p) not in base_id, net.parameters()))
# print(len(list(net.backbone.parameters())), len(list(other_params)),len(list(net.parameters())))
param_groups = [
    {"params": net.backbone.parameters(), "lr": n_gpus * 2e-5},
    {"params": other_params}
]

# optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=2e-5)
optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=2e-5)
loss_func = torch.nn.CrossEntropyLoss().cuda(args.local_rank)
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)

if torch.cuda.device_count() > 1 and args.local_rank == 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

train(net=net,
      train_dataloader=train_loader,
      test_dataloader=test_loader,
      optimizer=optimizer,
      loss_function=loss_func,
      total_epoch=total_epoch,
      lr=lr,
      args=args,
      train_sampler=train_sampler,
      num_classes=num_classes,
      size=size,
      use_aux=use_aux,
      # starting_epoch=120,
      # pretrained='model/MyFCN120.pth'
      )
# python -m torch.distributed.launch --nproc_per_node=1 main.py
