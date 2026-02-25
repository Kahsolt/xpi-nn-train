#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2025/05/03

# ddp测试脚本: 研究如何用torch自带的DDP实现多机CPU分布式训练 (rk3399 * 6)
# https://www.cnblogs.com/zhangxianrong/p/18379062

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

import os
import socket
from pathlib import Path
from argparse import ArgumentParser
from time import time

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as T
from torchvision.models import MobileNetV3, MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.datasets import CIFAR10

if 'setup env':
  hostname = socket.gethostname()
  rank = 0 if hostname.startswith('fmx') else (int(hostname[-1]) + 1)

  os.environ['MASTER_ADDR'] = 'fmx1'
  os.environ['MASTER_PORT'] = '8888'
  os.environ['WORLD_SIZE'] = '6'
  os.environ['RANK'] = str(rank)
  os.environ['LOCAL_WORLD_SIZE'] = '1'
  os.environ['LOCAL_RANK'] = '0'

  print(f'>> Hostname: {hostname}')
  print('>> Env:')
  for name in [
    'MASTER_ADDR',
    'MASTER_PORT',
    'WORLD_SIZE',
    'RANK',
    'LOCAL_WORLD_SIZE',
    'LOCAL_RANK',
  ]:
    print(f'  {name}: {os.getenv(name)}')

BASE_PATH = Path(__file__).parent
DATA_ROOT = Path(r'F:\Datasets\Image').expanduser()
assert DATA_ROOT.is_dir()
NUM_CLASS = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)  # NOTE: 使用正确版本的std


def get_model() -> MobileNetV3:
  model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
  linear_old: Linear = model.classifier[-1]
  linear_new = Linear(linear_old.in_features, NUM_CLASS, bias=linear_old.bias is not None)
  model.classifier[-1] = linear_new
  model.requires_grad_(True)
  return model


def run(args):
  #model = get_model()
  model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
  model = DistributedDataParallel(model)
  optim = Adam(model.parameters(), lr=args.learning_rate)

  transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
  ])
  transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
  ])
  train_dataset = CIFAR10(DATA_ROOT, train=True,  transform=transform_train)
  test_dataset  = CIFAR10(DATA_ROOT, train=False, transform=transform_test)
  train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,  drop_last=True)
  test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
  train_loader  = DataLoader(train_dataset, sampler=train_sampler, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
  test_loader   = DataLoader(test_dataset,  sampler=test_sampler,  pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)

  if rank == 0:
    #print(model)
    print('param_cnt:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f'train_samples: {len(train_dataset)}, train_batches: {len(train_loader)}')
    print(f'test_samples: {len(train_dataset)}, test_batches: {len(test_loader)}')

  step = 0
  ts_init = time()
  for epoch in range(args.epochs):
    ts_epoch = time()

    ''' Train '''
    train_sampler.set_epoch(epoch)
    model.train()
    for i, (x, y) in enumerate(train_loader):
      ts_step = time()

      logits = model(x)
      loss = F.cross_entropy(logits, y)
      optim.zero_grad()
      loss.backward()
      optim.step()

      step += 1
      if rank == 0 and step % 30 == 0:
        print(f'[Step {step}] loss: {loss.item():.5f} ({time() - ts_step:.5f} s/b)')

    ''' Test '''
    tot, ok = 0, 0
    test_sampler.set_epoch(epoch)
    model.eval()
    for i, (x, y) in enumerate(test_loader):
      logits = model(x)
      pred = torch.argmax(logits, -1)
      ok  += torch.sum(y == pred).item()
      tot += x.size(0)

    if rank == 0:
      print(f'[Epoch {epoch}] Acc: {ok / tot:.3%} ({time() - ts_epoch:.5f}s)')
    dist.barrier()

  (BASE_PATH / 'out').mkdir(exist_ok=True)
  torch.save({
    'model': model.state_dict(),
    'optim': optim.state_dict(),
  }, BASE_PATH / 'out' / 'ckpt.pth')

  if rank == 0:
    print(f'Done: {time() - ts_init:.5f}s')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-E', '--epochs',         default=50,   type=int,   help='Number of epochs (default: %(default)s)')
  parser.add_argument('-B', '--batch_size',     default=128,  type=int,   help='Batch size (default: %(default)s)')
  parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate (default: %(default)s)')
  parser.add_argument('-j', '--num_workers',    default=0,    type=int,   help='Num workers (default: %(default)s)')
  parser.add_argument('--load')
  args = parser.parse_args()

  dist.init_process_group('gloo', init_method='env://')
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  print(f">> rank = {rank} is initialized")
  if rank == 0:
    print(args)
  try:
    run(args)
  finally:
    dist.destroy_process_group()
