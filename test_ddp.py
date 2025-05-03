#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2025/05/03

# ddp测试脚本: 研究如何用torch自带的DDP实现多机CPU分布式训练
# https://www.cnblogs.com/zhangxianrong/p/18379062

import os
import socket
from time import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

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


class RandDataset(Dataset):

  def __len__(self):
    return 1000

  def __getitem__(self, idx:int):
    x = np.random.normal(0, 1, [1024]).astype(np.float32)
    y = np.random.normal(0, 1, [10]).astype(np.float32)
    return x, y


if __name__ == '__main__':
  dist.init_process_group('gloo', init_method='env://')
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  print(f">> rank = {rank} is initialized")

  model = nn.Linear(1024, 10)
  model = DistributedDataParallel(model)
  if rank == 0: print(model)
  optim = Adam(model.parameters(), lr=2e-4)
  dataset = RandDataset()
  sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)

  ts_init = time()
  for epoch in range(10):
    sampler.set_epoch(epoch)
    ts_epoch = time()
    for i, (x, y) in enumerate(dataloader):
      ts_step = time()
      out = model(x)
      loss = F.mse_loss(out, y)
      optim.zero_grad()
      loss.backward()
      optim.step()

      if rank == 0 and (i + 1) % 10 == 0:
        print(f'[Step {i}] {time() - ts_step:.5f}s')
    if rank == 0:
      print(f'[Epoch {epoch}] {time() - ts_epoch:.5f}s')
    dist.barrier()
  if rank == 0:
    print(f'Done: {time() - ts_init:.5f}s')

  dist.destroy_process_group()
