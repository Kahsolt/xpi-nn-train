#!/usr/bin/env python3

import os
import json
import socket
from time import time
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist

from xpi_nn_train.datasets.torchvision import DataLoader
from xpi_nn_train.utils import seed_everything, ValueWindow, safe_serialize, get_next_torch_log_folder


def setup_ddp_env() -> int:
  '''
  I have an RK3399 cluster consisting of 1 * FMX1-Pro and 5 * MRK3399:
    fmx1  master
    mrk0  node0
    mrk1  node1
    mrk2  node2
    mrk3  node3
    mrk4  node4
  Just hardcode it so far :(
  '''

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

  return rank


# https://www.cnblogs.com/zhangxianrong/p/18379062
def run_train_ddp(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, args:Namespace):
  torch.set_num_threads(torch.get_num_interop_threads())
  seed_everything(args.seed)

  dtype = torch.float32  # fp16/bf16 on most SoC is slow
  rank = setup_ddp_env()

  if rank == 0:
    print('[args]')
    print(args)
    print('[param_cnt]')
    print('   trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('   total:', sum(p.numel() for p in model.parameters()))
    print('[dataset]')
    print(f'  train: {len(train_loader.dataset)} samples, {len(train_loader)} batches')
    print(f'  test: {len(test_loader.dataset)} samples, {len(test_loader)} batches')

  optim = Adam(model.parameters(), lr=args.learning_rate)
  if args.load:
    print(f'>> Resume from ckpt {args.load}')
    ckpt = torch.load(args.load)
    model.load_state_dict(ckpt['model'])
    optim.load_state_dict(ckpt['optim'])
  else:
    print('>> Fresh train!')

  vw = ValueWindow(100)
  train_acc_list: list[float] = []
  test_acc_list: list[float] = []
  step = 1
  ts_init = time()
  for epoch in range(1, 1 + args.epochs):
    ts_epoch = time()

    ''' Train '''
    tot, ok = 0, 0
    model.train()
    train_loader.sampler.set_epoch(epoch)
    for x, y in train_loader:
      ts_step = time()
      x = x.to(dtype)
      y = y.long()
      logits = model(x)
      loss = F.cross_entropy(logits, y)
      optim.zero_grad()
      loss.backward()
      optim.step()
      vw.append(time() - ts_step)

      with torch.inference_mode():
        pred = torch.argmax(logits, -1)
        ok  += torch.sum(y == pred).item()
        tot += x.size(0)

      step += 1
      if step % vw.nlen == 0:
        print(f'[Step {step}] loss: {loss.item():.5f}, acc: {ok / tot:.3%} ({1 / vw.mean:.5f} it/s)')

    train_acc_list.append(ok / tot)

    ''' Test '''
    with torch.inference_mode():
      tot, ok = 0, 0
      model.eval()
      test_loader.sampler.set_epoch(epoch)
      for x, y in test_loader:
        x = x.to(dtype)
        y = y.long()
        logits = model(x)
        pred = torch.argmax(logits, -1)
        ok  += torch.sum(y == pred).item()
        tot += x.size(0)

    test_acc_list.append(ok / tot)
    print(f'[Epoch {epoch}] Test Acc: {ok / tot:.3%} ({time() - ts_epoch:.5f}s)\n')

    dist.barrier()

  if rank == 0:
    ts_done = time() - ts_init
    print(f'Done: {ts_done:.5f}s')

    save_path = get_next_torch_log_folder()
    print(f'>> Save to {save_path}')
    torch.save({
      'model': model.state_dict(),
      'optim': optim.state_dict(),
      'train_acc_list': train_acc_list,
      'test_acc_list': test_acc_list,
      'ts': ts_done,
    }, save_path / 'final.pth')
    with open(save_path / 'args.json', 'w', encoding='utf-8') as fh:
      json.dump(vars(args), fh, indent=2, default=safe_serialize)
