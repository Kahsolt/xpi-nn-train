#!/usr/bin/env python3

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from xpi_nn_train.datasets.lightning import LitData


class LitModelCls(LightningModule):

  def __init__(self, model:nn.Module, args:Namespace):
    super().__init__()

    self.model = model
    if args is not None:
      self.save_hyperparameters(args)
    self.num_classes = getattr(args, 'num_classes', None)
    self.learning_rate = getattr(args, 'learning_rate', 2e-4)

    if self.num_classes:
      self.train_acc1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
      self.train_acc3 = MulticlassAccuracy(num_classes=self.num_classes, top_k=3)
      self.val_acc1   = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
      self.val_acc3   = MulticlassAccuracy(num_classes=self.num_classes, top_k=3)
    else:
      print('WARN: missing args "num_classes", this model is infer-only')

  def configure_optimizers(self):
    return Adam(self.model.parameters(), lr=self.learning_rate)

  def training_step(self, batch:tuple[Tensor]) -> Tensor:
    return self.forward_step(batch, 'train')

  @torch.inference_mode
  def validation_step(self, batch:tuple[Tensor]) -> Tensor:
    return self.forward_step(batch, 'val')

  def forward_step(self, batch:tuple[Tensor], split:str) -> Tensor:
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
        f'{split}_acc@1': acc1,
        f'{split}_acc@3': acc3,
      }, on_step=is_train, on_epoch=True)

    return loss


def run_train_lit(litmodel:LitModelCls, litdata:LitData, args:Namespace) -> Trainer:
  if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
  else:
    torch.set_num_threads(torch.get_num_interop_threads())
  seed_everything(args.seed)

  if args.load:
    print(f'>> Resume from ckpt {args.load}')
    LitModelCls.load_from_checkpoint(args.load, model=litmodel.model)
  else:
    print('>> Fresh train!')

  if args.lora > 0:
    from peft import get_peft_model
    from peft.tuners.lora import LoraConfig

    print(f'>> Enable peft: LoRA rank={args.lora}')
    peft_cfg = LoraConfig(
      r=args.lora,
      lora_dropout=0.2,
      target_modules='all-linear',
      bias='none',
    )
    litmodel.model = get_peft_model(litmodel.model, peft_cfg)

  trainer = Trainer(
    precision='32-true',  # fp16/bf16 on most SoC is slow
    max_epochs=args.epochs,
    accumulate_grad_batches=args.grad_acc,
    benchmark=True,
    deterministic=False,
    enable_checkpointing=True,
    val_check_interval=1.0,
    callbacks=[
      ModelCheckpoint(monitor='val_loss'),
    ],
  )
  trainer.fit(litmodel, datamodule=litdata)
  return trainer
