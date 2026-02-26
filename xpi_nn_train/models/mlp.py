#!/usr/bin/env python3

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):

  def __init__(self, in_dim:int=784, out_dim:int=10, hid_dims:list[int]=[128], act:type[nn.Module]=nn.ReLU):
    super().__init__()

    dims = [in_dim] + hid_dims + [out_dim]
    self.layers = nn.ModuleList()
    for d_in, d_out in zip(dims, dims[1:]):
      layer = nn.Linear(d_in, d_out, bias=True)
      self.layers.append(layer)
    self.act = act()

  def forward(self, x: Tensor) -> Tensor:
    if len(x.shape) > 2:
      x = x.flatten(start_dim=1)
    x = self.layers[0](x)
    for layer in self.layers[1:]:
      x = self.act(x)
      x = layer(x)
    return x
