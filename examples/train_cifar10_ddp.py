#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2025/05/03

# 用简单 MobidleNet_v3_small 分布式训练 CIFAR10 数据集
# https://www.cnblogs.com/zhangxianrong/p/18379062

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, MobileNetV3
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchmetrics.classification import MulticlassAccuracy

from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_num_threads(torch.get_num_interop_threads())

BASE_PATH = Path(__file__).parent
DATA_ROOT = BASE_PATH / 'data'
NUM_CLASS = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)  # NOTE: 使用正确版本的std


def get_model(args) -> MobileNetV3:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    linear_old: nn.Linear = model.classifier[-1]
    linear_new = nn.Linear(linear_old.in_features, NUM_CLASS, bias=linear_old.bias is not None)
    model.classifier[-1] = linear_new

    if args.phase == -1:
        model.requires_grad_(True)
    elif args.phase == 0:
        model.requires_grad_(False)
        model.classifier[-1].requires_grad_(True)
    elif args.phase == 1:
        model.requires_grad_(False)
        model.features[0].requires_grad_(True)
        model.classifier.requires_grad_(True)
    else:
        raise NotImplementedError(f'unknown phase {args.phase}')

    return model


class LitData(LightningDataModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def train_dataloader(self):
        # https://cloud.tencent.com/developer/article/2075677
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        dataset = CIFAR10(DATA_ROOT, train=True, transform=transform, download=True)
        return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)

    def val_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        dataset = CIFAR10(DATA_ROOT, train=False, transform=transform, download=True)
        return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False)


class LitModel(LightningModule):

    def __init__(self, model:MobileNetV3, args=None):
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
        return Adam(self.model.parameters(), lr=self.args.learning_rate)

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

    model = get_model(args)
    litmodel = LitModel(model, args)
    if args.load:
        print(f'>> Resume from ckpt {args.load}')
        LitModel.load_from_checkpoint(args.load, model=model)
    else:
        print('>> Fresh train!')

    litdata = LitData(args)
    trainer = Trainer(
        accelerator='cpu',
        strategy='ddp',
        num_nodes=6,
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
    parser.add_argument('--phase', default=-1, type=int, help='carefully do multi-staging finetune')
    parser.add_argument('--seed', default=114514, type=int)
    args = parser.parse_args()

    print(f'>> running phase: {args.phase}')

    run(args)
