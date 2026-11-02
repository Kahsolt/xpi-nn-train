#!/usr/bin/env python3

# 用torchvision训练伪数据集或CIFAR10，用以快速评估硬件性能

from time import perf_counter, sleep
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

from xpi_nn_train.models.torchvision import get_model_builder
from xpi_nn_train.datasets.torchvision import DataLoader, get_dataloader
from xpi_nn_train.trainers import get_trainer_args
from xpi_nn_train.utils import merge_namespaces, seed_everything, ValueWindow
from xpi_nn_train.torch_utils import resolve_device, resolve_dtype, torch_backends_setup, torch_backends_info

MODELS = [
  'mobilenet_v3_small',
  'mobilenet_v3_large',
  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'squeezenet1_1',
  'efficientnet_v2_s',
  'vgg16_bn',
  'resnet18',
  'resnet50',
  'densenet121',
  'resnext50_32x4d',
  'convnext_tiny',
  'vit_b_16',
  'vit_b_32',
  'swin_t',
  'swin_v2_t',
]
DATASET_TO_NUM_CLASSES = {
  'FAKE': 1000,
  'CIFAR10': 10,
}


class MyFakeData(Dataset):

  def __init__(self, size:int=1000, image_size:tuple[int]=(3, 224, 224), num_classes:int=1000, device:str='cpu', dtype:torch.dtype=torch.float32):
    super().__init__()

    self.size = size
    self.image_size = image_size
    self.num_classes = num_classes
    self.device = device
    self.dtype = dtype

  def __len__(self) -> int:
    return self.size

  def __getitem__(self, index:int) -> tuple:
    img = torch.randn(*self.image_size, dtype=self.dtype, device=self.device)
    lbl = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long, device=self.device).item()
    return img, lbl


def run(args) -> float:
  '''Init'''
  device = resolve_device(args.device)
  dtype = resolve_dtype(args.dtype)
  if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True         # 开了会快一些，但第0个epoch显著慢
    torch.backends.cudnn.deterministic = False    # 开了会慢一些
    # TF32累加 (Ampere+)，会有一点精度损失但相对安全
    # https://docs.pytorch.ac.cn/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    # 新版API:
    #torch.backends.fp32_precision = 'tf32'
    #torch.backends.cuda.matmul.fp32_precision = 'tf32'
    #torch.backends.cudnn.fp32_precision = 'tf32'
    #torch.backends.cudnn.conv.fp32_precision = 'tf32'
    #torch.backends.cudnn.rnn.fp32_precision = 'tf32'
    # 旧版API:
    #torch.set_float32_matmul_precision('medium')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True
    # FP16/BF16规约；训练时可能数值溢出，建议用于推理时
    # https://docs.pytorch.ac.cn/docs/2.14/notes/cuda.html#reduced-precision-reduction-in-fp16-gemms
    #torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True   # 默认False
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True   # 默认True
    # FP16累加 (Volta+)；训练时可能数值溢出，建议用于推理时
    #torch.backends.cuda.matmul.allow_fp16_accumulation = True                  # 默认False
    nbatch = 80
  else:
    # 默认已是物理核心数，均不能超过物理核心数
    #import os
    #n_phys_cpu = (os.cpu_count() // 2) or 1
    #torch.set_num_threads(n_phys_cpu)
    #torch.set_num_interop_threads(1)   # 理论上设置为1更好，默认值是物理核心数
    nbatch = 4
  seed_everything(args.seed)

  '''Model'''
  model_builder = get_model_builder(args.model)
  model = model_builder(num_classes=args.num_classes)
  model = model.to(device, dtype)
  optim = Adam(model.parameters(), lr=args.learning_rate)

  '''Data'''
  if args.dataset == 'CIFAR10':
    dataloader = get_dataloader(args.dataset, split='test', batch_size=args.batch_size, num_workers=args.num_workers)
  elif args.dataset == 'FAKE':
    dataset = MyFakeData(size=args.batch_size*nbatch, image_size=(3, 224, 224), num_classes=1000, device=device, dtype=dtype)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=args.num_workers)
  else:
    raise ValueError(f'>> unknown dataset: {args.dataset}')

  #print(model)
  print(args)
  print('[Run config]')
  print(f'  model:', args.model)
  print('     trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
  print('     total:', sum(p.numel() for p in model.parameters()))
  print(f'  dataset:', args.dataset)
  print(f'    {len(dataloader.dataset)} samples, {len(dataloader)} batches')
  print(f'  device:', device)
  print(f'  dtype:', dtype)

  '''Train'''
  vw_ema = ValueWindow(args.epochs - 1)
  ts_init = perf_counter()
  for epoch in range(args.epochs):
    torch.cuda.synchronize()
    ts_epoch = perf_counter()

    vw = ValueWindow(10000)
    nsamples = 0
    model.train()
    for x, y in dataloader:
      torch.cuda.synchronize()
      ts_step = perf_counter()

      x = x.to(device, dtype)
      y = y.to(device).long()
      optim.zero_grad()
      logits = model(x)
      loss = F.cross_entropy(logits, y)
      loss.backward()
      optim.step()

      torch.cuda.synchronize()
      tc_step = perf_counter() - ts_step

      nsample = len(y)
      nsamples += nsample
      vw.append(nsample / tc_step)

    torch.cuda.synchronize()
    tc_epoch = perf_counter() - ts_epoch
    print(f'[epoch {epoch}] elapsed_e2e: {tc_epoch:.2f}s, speed_e2e_avg: {nsamples / tc_epoch:.2f} fps, speed: {vw.mean:.2f} fps')

    if epoch > 0: # 舍弃epoch=0作为warmup
      vw_ema.append(vw.mean)

  tc_done = perf_counter() - ts_init
  print(f'Done: {tc_done:.5f}s, speed_avg: {vw_ema.mean:.2f} fps')

  return vw_ema.mean


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', default='FAKE', choices=['FAKE', 'CIFAR10'], help='dataset')
  parser.add_argument('-M', '--model', default='mobilenet_v3_small', choices=MODELS, help='model arch')
  parser.add_argument('-X', '--run_all', action='store_true')
  bench_args, _ = parser.parse_known_args()

  train_args = get_trainer_args()
  train_args.backend = 'torch'
  train_args.num_classes = DATASET_TO_NUM_CLASSES[bench_args.dataset]
  train_args.epochs = 4

  args = merge_namespaces(bench_args, train_args)

  torch_backends_setup()
  torch_backends_info()

  if args.run_all:
    results = []
    for model in MODELS:
      args.model = model
      fps = run(args)
      results.append((model, fps))
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
      torch.cuda.synchronize()
      sleep(10)
    print()
    print('=' * 72)
    print('| ' + ' | '.join([model for model, _ in results]) + ' |')
    print('|' + f' :-: |' * len(results))
    print('| ' + ' | '.join([f'{fps:.2f}' for _, fps in results]) + ' |')
  else:
    run(args)
