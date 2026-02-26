#!/usr/bin/env python3

# 用简单MLP/CNN训练NMIST数据集

from argparse import ArgumentParser

import torch.nn as nn

from xpi_nn_train.models.mlp import MLP
from xpi_nn_train.models.lenet import LeNet5
from xpi_nn_train.trainers import run_train


def get_model() -> MLP | LeNet5:
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='MLP', choices=['MLP', 'MLP-SiLU', 'LeNet'], help='model arch')
  args, _ = parser.parse_known_args()
  model_arch = args.model
  print(f'>> Running with model_arch={model_arch}')

  if model_arch == 'MLP':
    model = MLP(28*28, 10, [128])
  elif model_arch == 'MLP-SiLU':
    model = MLP(28*28, 10, [128], act=nn.SiLU)
  elif model_arch == 'LeNet':
    model = LeNet5(10)
  return model


if __name__ == '__main__':
  model = get_model()
  run_train(model, 'MNIST')
