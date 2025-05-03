#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2025/03/14

# 用简单两层 MLP 训练 NMIST 数据集

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torchmetrics.classification import MulticlassAccuracy

from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_num_threads(torch.get_num_interop_threads())

BASE_PATH = Path(__file__).parent
DATA_ROOT = Path(os.getenv('DATA_ROOT', '~/Datasets')).expanduser()
print('DATA_ROOT:', DATA_ROOT)
NUM_CLASS = 10


class MLP(nn.Module):

  def __init__(self, d_in:int, d_out:int, d_hid:int):
    super().__init__()

    self.mlp = nn.Sequential(
      nn.Linear(d_in,  d_hid),
      nn.SiLU(inplace=True),
      nn.Linear(d_hid, d_out),
    )

  def forward(self, x:Tensor) -> Tensor:
    x = torch.flatten(x, start_dim=1)
    return self.mlp(x)


class LitData(LightningDataModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def train_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
        ])
        dataset = MNIST(DATA_ROOT, train=True, transform=transform, download=True)
        return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)

    def val_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
        ])
        dataset = MNIST(DATA_ROOT, train=False, transform=transform, download=True)
        return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False)


class LitModel(LightningModule):

    def __init__(self, model:nn.Module, args=None):
        super().__init__()

        self.model = model
        self.args = args
        if args is not None:
            self.save_hyperparameters(args)

        self.train_acc1 = MulticlassAccuracy(num_classes=NUM_CLASS, top_k=1)
        self.train_acc3 = MulticlassAccuracy(num_classes=NUM_CLASS, top_k=3)
        self.val_acc1   = MulticlassAccuracy(num_classes=NUM_CLASS, top_k=1)
        self.val_acc3   = MulticlassAccuracy(num_classes=NUM_CLASS, top_k=3)

    def configure_optimizers(self):
        return Adam(self.model.named_parameters(), lr=self.args.learning_rate)

    def forward_step(self, batch:Tuple[Tensor], split:str) -> Tensor:
        x, y = batch
        o = self.model(x)
        loss = F.cross_entropy(o, y)

        with torch.inference_mode():
            is_train = split == 'train'
            if is_train:
                acc1, acc3 = self.train_acc1, self.train_acc3
            else:
                acc1, acc3 = self.val_acc1, self.val_acc3
            acc1(o, y)
            acc3(o, y)

            self.log_dict({
                f'{split}_loss': loss,
                f'{split}_acc_top1': acc1,
                f'{split}_acc_top3': acc3,
            }, on_step=is_train, on_epoch=True)

        return loss

    def training_step(self, batch:Tuple[Tensor]) -> Tensor:
        return self.forward_step(batch, 'train')

    @torch.inference_mode
    def validation_step(self, batch:Tuple[Tensor]) -> Tensor:
        return self.forward_step(batch, 'val')


def run(args):
    seed_everything(args.seed)

    model = MLP(28*28, 10, 128)
    litmodel = LitModel(model, args)
    if args.load:
        print(f'>> Resume from ckpt {args.load}')
        litmodel.load_from_checkpoint(args.load, model=model)
    else:
        print('>> Fresh train!')

    litdata = LitData(args)
    trainer = Trainer(
        precision='32-true',
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_acc,
        benchmark=True,
        deterministic=False,
        enable_checkpointing=True,
        val_check_interval=0.3,
        callbacks=[
            ModelCheckpoint(monitor="val_loss")
        ],
    )
    trainer.fit(litmodel, datamodule=litdata)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-E', '--epochs',         default=100,   type=int,   help='Number of epochs (default: %(default)s)')
    parser.add_argument('-B', '--batch_size',     default=32,    type=int,   help='Batch size (default: %(default)s)')
    parser.add_argument('-g', '--grad_acc',       default=4,     type=int,   help='Grad accumulate steps (default: %(default)s)')
    parser.add_argument('-lr', '--learning_rate', default=2e-4,  type=float, help='Learning rate (default: %(default)s)')
    parser.add_argument('-j', '--num_workers',    default=0,     type=int,   help='Num workers (default: %(default)s)')
    parser.add_argument('--load')
    parser.add_argument('--seed', default=114514, type=int)
    args = parser.parse_args()

    run(args)
