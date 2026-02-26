#!/usr/bin/env python3

import json
from time import time
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from xpi_nn_train.datasets.torchvision import DataLoader
from xpi_nn_train.utils import seed_everything, ValueWindow, safe_serialize, get_next_torch_log_folder


def run_train_native(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, args:Namespace):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('medium')
    dtype = torch.bfloat16  # bf16 is better on cuda
  else:
    torch.set_num_threads(torch.get_num_interop_threads())
    dtype = torch.float32  # fp16/bf16 on most SoC is slow
  seed_everything(args.seed)

  print('[model]')
  print(model)
  print('[args]')
  print(args)
  print('[param_cnt]')
  print('   trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
  print('   total:', sum(p.numel() for p in model.parameters()))
  print('[dataset]')
  print(f'  train: {len(train_loader.dataset)} samples, {len(train_loader)} batches')
  print(f'  test: {len(test_loader.dataset)} samples, {len(test_loader)} batches')
  print('device:', device)
  print('dtype:', dtype)

  model = model.to(device, dtype)
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
    optim.zero_grad()
    for x, y in train_loader:
      ts_step = time()
      x = x.to(device, dtype)
      y = y.to(device).long()
      logits = model(x)
      loss = F.cross_entropy(logits, y)
      if args.grad_acc > 1:
        loss = loss / args.grad_acc
      loss.backward()
      if step % args.grad_acc == 0:
        optim.step()
        optim.zero_grad()
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
      for x, y in test_loader:
        x = x.to(device, dtype)
        y = y.to(device).long()

        logits = model(x)
        pred = torch.argmax(logits, -1)
        ok  += torch.sum(y == pred).item()
        tot += x.size(0)

    test_acc_list.append(ok / tot)
    print(f'[Epoch {epoch}] Test Acc: {ok / tot:.3%} ({time() - ts_epoch:.5f}s)\n')

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
