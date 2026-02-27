#!/usr/bin/env python3

from inspect import signature
from typing import Callable

from torch.utils.data import DataLoader, DistributedSampler
import torchvision.datasets as D
from torchvision.datasets import VisionDataset, FakeData
from torchvision import transforms as T

from xpi_nn_train.utils import DATA_ROOT

notify_data_root = True


def list_datasets():
  blacklist = [
    VisionDataset,
    FakeData,
    D.DatasetFolder,
    D.ImageFolder,
  ]
  return [k for k, v in vars(D).items() if v not in blacklist and type(v) == type(object) and issubclass(v, VisionDataset)]


def get_dataset(name:str, split:str='train', is_train:bool=None, transform:Callable=None) -> VisionDataset:
  assert name in list_datasets(), f"dataset {name} not in supported list"
  global notify_data_root
  if notify_data_root:
    print(f'DATA_ROOT: {DATA_ROOT}')
    notify_data_root = False

  # fix for EMNIST
  is_train = split == 'train' if is_train is None else is_train

  # add well-known transform
  if transform is None:
    if name == 'MNIST':
      # https://www.kaggle.com/code/berrywell/calculating-mnist-inverted-mnist-mean-std
      MNIST_MEAN = (0.1306604762738431,)
      MNIST_STD  = (0.3081078038564622,)
      if split == 'train':
        transform = T.Compose([
          T.RandomCrop(28, padding=2),
          T.ToTensor(),
          T.Normalize(MNIST_MEAN, MNIST_STD),
        ])
      else:
        transform = T.Compose([
          T.ToTensor(),
          T.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    elif name == 'CIFAR10':
      # https://github.com/kuangliu/pytorch-cifar/issues/16
      CIFAR10_MEAN = (0.49137255, 0.48235294, 0.44666667)
      CIFAR10_STD  = (0.24705882, 0.24352941, 0.26156863)
      if split == 'train':
        # https://cloud.tencent.com/developer/article/2075677
        transform = T.Compose([
          T.RandomCrop(32, padding=4),
          T.RandomHorizontalFlip(),
          T.ToTensor(),
          T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
      else:
        transform = T.Compose([
          T.ToTensor(),
          T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
      transform = T.Compose([
        T.ToTensor(),
      ])

  # build args from params
  dataset_builder = getattr(D, name)
  params = signature(dataset_builder).parameters
  has_kwargs = 'kwargs' in params
  args = {}
  if 'root' in params or has_kwargs:
    args['root'] = DATA_ROOT
  if 'split' in params or has_kwargs:
    args['split'] = split
  if 'train' in params or has_kwargs:
    args['train'] = is_train
  if 'transform' in params or has_kwargs:
    args['transform'] = transform
  if 'download' in params or has_kwargs:
    args['download'] = True

  dataset = dataset_builder(**args)
  return dataset


def get_dataloader(name:str, split:str='train', is_train:bool=None, batch_size:int=16, transform:Callable=None, num_workers:int=0, world_size:int=1, rank:int=-1) -> DataLoader:
  # fix for EMNIST
  is_train = split == 'train' if is_train is None else is_train

  dataset = get_dataset(name, split, is_train, transform)
  if world_size > 1:
    assert rank >= 0, 'must set "rank" in distributed scenario'
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=is_train)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers, sampler=sampler)
  else:
    dataloader = DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train, pin_memory=True, num_workers=num_workers)
  return dataloader
