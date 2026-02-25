#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2025/05/03

# ddp测试脚本的对比实验: 直接在单个高性能CPU上跑 (i5-12400F)

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

from pathlib import Path
from argparse import ArgumentParser
from time import time

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import MobileNetV3, MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.datasets import CIFAR10

BASE_PATH = Path(__file__).parent
DATA_ROOT = Path(r'F:\Datasets\Image')
assert DATA_ROOT.is_dir()
NUM_CLASS = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)  # 这才是正确版本的std

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


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
  model = model.to(device)
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
  train_dataset = CIFAR10(DATA_ROOT, train=True,  download=True, transform=transform_train)
  test_dataset  = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform_test)
  train_loader  = DataLoader(train_dataset, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
  test_loader   = DataLoader(test_dataset,  pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)

  #print(model)
  print('param_cnt:', sum(p.numel() for p in model.parameters() if p.requires_grad))
  print(f'train_samples: {len(train_dataset)}, train_batches: {len(train_loader)}')
  print(f'test_samples: {len(train_dataset)}, test_batches: {len(test_loader)}')

  step = 0
  ts_init = time()
  for epoch in range(args.epochs):
    ts_epoch = time()

    ''' Train '''
    model.train()
    for i, (x, y) in enumerate(train_loader):
      x = x.to(device).float()
      y = y.to(device).long()

      ts_step = time()

      logits = model(x)
      loss = F.cross_entropy(logits, y)
      optim.zero_grad()
      loss.backward()
      optim.step()

      step += 1
      if step % 30 == 0:
        print(f'[Step {step}] loss: {loss.item():.5f} ({time() - ts_step:.5f} s/b)')

    ''' Test '''
    tot, ok = 0, 0
    model.eval()
    for i, (x, y) in enumerate(test_loader):
      x = x.to(device).float()
      y = y.to(device).long()

      logits = model(x)
      pred = torch.argmax(logits, -1)
      ok  += torch.sum(y == pred).item()
      tot += x.size(0)

    print(f'[Epoch {epoch}] Acc: {ok / tot:.3%} ({time() - ts_epoch:.5f}s)')

  print(f'Done: {time() - ts_init:.5f}s')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-E', '--epochs',         default=50,   type=int,   help='Number of epochs (default: %(default)s)')
  parser.add_argument('-B', '--batch_size',     default=128,  type=int,   help='Batch size (default: %(default)s)')
  parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate (default: %(default)s)')
  parser.add_argument('-j', '--num_workers',    default=0,    type=int,   help='Num workers (default: %(default)s)')
  parser.add_argument('--load')
  args = parser.parse_args()

  print(args)
  run(args)
