#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor


class ESPCN(nn.Module):

  def __init__(self, upscale_factor:int=2, in_channels:int=3, out_channels:int=3, channels:int=64):
    super().__init__()

    hidden_channels = channels // 2
    out_channels *= upscale_factor ** 2

    self.feature_maps = nn.Sequential(
      nn.Conv2d(in_channels, channels, kernel_size=5, padding=2),
      nn.Tanh(),
      nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
      nn.Tanh(),
      nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
    )
    self.sub_pixel = nn.PixelShuffle(upscale_factor)

    self.reset_weights()

  def reset_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight.data, 0.0, 0.001)
        nn.init.zeros_(module.bias.data)

  def forward(self, x: Tensor) -> Tensor:
    x = self.feature_maps(x)
    x = self.sub_pixel(x)
    x = torch.clamp_(x, 0.0, 1.0)
    return x
