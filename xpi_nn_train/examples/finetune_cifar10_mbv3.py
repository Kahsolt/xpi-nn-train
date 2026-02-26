#!/usr/bin/env python3

# 用MobidleNet_v3_small微调CIFAR10数据集

from argparse import ArgumentParser

from xpi_nn_train.models.torchvision import get_mbv3_small, MobileNetV3
from xpi_nn_train.trainers import run_train


def get_pretrained_mbv3_with_freeze() -> MobileNetV3:
  parser = ArgumentParser()
  parser.add_argument('-F', '--freeze_level', default=-1, type=int, help='finetune with different freeze levels')
  args, _ = parser.parse_known_args()
  freeze_level = args.freeze_level
  print(f'>> Running with freeze_level={freeze_level}')

  model = get_mbv3_small(10)
  if freeze_level == -1:
    # 全量微调
    model.requires_grad_(True)
  elif freeze_level == 0:
    # 仅微调最后一层Linear
    model.requires_grad_(False)
    model.classifier[-1].requires_grad_(True)
  elif freeze_level == 1:
    # 仅微调第一层Conv和最后一层Linear
    model.requires_grad_(False)
    model.features[0].requires_grad_(True)
    model.classifier.requires_grad_(True)
  else:
    raise NotImplementedError(f'unknown freeze_level {freeze_level}')
  return model


if __name__ == '__main__':
  model = get_pretrained_mbv3_with_freeze()
  run_train(model, 'CIFAR10')
