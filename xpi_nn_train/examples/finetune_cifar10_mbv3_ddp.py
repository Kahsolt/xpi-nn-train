#!/usr/bin/env python3

# 用MobidleNet_v3_small微调CIFAR10数据集 (多机CPU分布式训练)

import os
import socket

from xpi_nn_train.models.torchvision import get_mbv3_small
from xpi_nn_train.trainers import run_train, get_trainer_args


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


if __name__ == '__main__':
  setup_ddp_env()

  args = get_trainer_args()
  args.backend = 'torch_ddp'
  model = get_mbv3_small(10)
  run_train(model, 'CIFAR10')
