#!/usr/bin/env python3

from lightning import LightningDataModule

from xpi_nn_train.datasets.torchvision import get_dataloader


class LitData(LightningDataModule):

  def __init__(self, name:str, batch_size:int=16, num_workers:int=0):
    super().__init__()

    self.name = name
    self.batch_size = batch_size
    self.num_workers = num_workers

  def train_dataloader(self):
    return get_dataloader(self.name, split='train', batch_size=self.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    return get_dataloader(self.name, split='val', batch_size=self.batch_size, num_workers=self.num_workers)
