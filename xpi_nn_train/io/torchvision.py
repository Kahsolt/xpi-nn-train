#!/usr/bin/env python3

import torchvision.datasets as D
from torchvision.datasets import VisionDataset

from xpi_nn_train.utils import DATA_ROOT


def get_dataset(name:str, split:str='train') -> VisionDataset:
  if name == 'mnist':
    dataset = D.MNIST()
  
  return dataset
