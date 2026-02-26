#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LeNet5(nn.Module):

  input_size = (-1, 1, 32, 32)

  def __init__(self, num_classes:int=10):
    super().__init__()

    self.img_size = self.input_size[-2:]

    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
      nn.BatchNorm2d(6),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Linear(400, 120),
      nn.ReLU(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, num_classes),
    )

  def forward(self, x: Tensor) -> Tensor:
    if x.shape[-2:] != self.img_size:
      x = F.interpolate(x, size=self.img_size, mode='bilinear')
    x = self.layer1(x)
    x = self.layer2(x)
    x = x.flatten(start_dim=1)
    x = self.classifier(x)
    return x
