#!/usr/bin/env python3

import os
from pathlib import Path

BASE_PATH = Path(__file__).parent   # base path for pypi package
ROOT_PATH = BASE_PATH.parent        # root path for git project
IS_PYPI = 'site-packages' in str(BASE_PATH)

# decide datasets & logging folder
if IS_PYPI:
  DATA_ROOT = BASE_PATH / 'DATA_ROOT'
  LOG_PATH = Path('.') / 'logs'
else:
  DATA_ROOT = ROOT_PATH / 'DATA_ROOT'
  LOG_PATH = ROOT_PATH / 'logs'
# override by envvar
if 'DATA_ROOT' in os.environ:
  DATA_ROOT = Path.expanduser(os.environ['DATA_ROOT'])
  assert DATA_ROOT.is_dir(), f"DATA_ROOT detected in envvar, but path {DATA_ROOT} not exists"
if 'LOG_PATH' in os.environ:
  LOG_PATH = Path.expanduser(os.environ['LOG_PATH'])
  assert LOG_PATH.is_dir(), f"LOG_PATH detected in envvar, but path {LOG_PATH} not exists"


def seed_everything(seed:int=114514):
  import random
  import numpy as np
  import torch

  os.environ["PL_GLOBAL_SEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


SERIALIZERS = [
  str,
  repr,
  lambda x: f'Error serialize type {type(x)}',
]

def safe_serialize(obj:object):
  for serializer in SERIALIZERS:
    try:
      return serializer(obj)
    except:
      pass


def get_next_log_folder(parent:Path=LOG_PATH) -> Path:
  i = 0
  dp = parent / f'version_{i}'
  while dp.is_dir():
    i += 1
    dp = parent / f'version_{i}'
  dp.mkdir(parents=True)
  return dp


class ValueWindow:

  def __init__(self, nlen:int=100):
    self.nlen = nlen
    self.vals: list[float] = []
    self.sum = 0.0

  def append(self, v:float):
    if len(self.vals) >= self.nlen:
      self.sum -= self.vals.pop(0)
    self.vals.append(v)
    self.sum += v

  @property
  def mean(self) -> float:
    if not self.vals: return 0.0
    return self.sum / len(self.vals)
