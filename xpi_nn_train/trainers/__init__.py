import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch.nn as nn


def list_trainer_backends() -> list[str]:
  return [fp.stem for fp in Path(__file__).parent.iterdir() if fp.is_file() and fp.suffix == '.py' and not fp.stem.startswith('_')]


def get_trainer_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend', default='torch', choices=list_trainer_backends(), help='Trainer backend (default: %(default)s)')
  parser.add_argument('-N', '--num_classes', default=10, type=int, help='Num classes for cls model (default: %(default)s)')
  parser.add_argument('-E', '--epochs', default=5, type=int, help='Number of epochs (default: %(default)s)')
  parser.add_argument('-B', '--batch_size', default=32, type=int, help='Batch size (default: %(default)s)')
  parser.add_argument('-g', '--grad_acc', default=4, type=int, help='Grad accumulate steps (default: %(default)s)')
  parser.add_argument('-lr', '--learning_rate', default=2e-4, type=float, help='Learning rate (default: %(default)s)')
  parser.add_argument('-r', '--lora', default=0, type=int, help='LoRA rank (default: %(default)s)')
  parser.add_argument('-j', '--num_workers', default=0, type=int, help='Num workers (default: %(default)s)')
  parser.add_argument('--load', help='resume from pretrained checkpoint')
  parser.add_argument('--seed', default=114514, type=int)
  args, _ = parser.parse_known_args()

  assert args.grad_acc >= 1, "--grad_acc must >= 1"

  return args


def run_train(model:nn.Module, dataset:str, args:Namespace=None):
  args = args or get_trainer_args()

  if args.backend == 'torch':
    from xpi_nn_train.datasets.torchvision import get_dataloader
    from xpi_nn_train.trainers.torch import run_train_native

    train_loader = get_dataloader(dataset, split='train', batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = get_dataloader(dataset, split='val', batch_size=args.batch_size, num_workers=args.num_workers)
    run_train_native(model, train_loader, test_loader, args)

  elif args.backend == 'lightning':
    from xpi_nn_train.datasets.lightning import LitData
    from xpi_nn_train.trainers.lightning import LitModelCls, run_train_lit

    litmodel = LitModelCls(model, args)
    litdata = LitData(dataset, args.batch_size, args.num_workers)
    run_train_lit(litmodel, litdata, args)

  elif args.backend == 'torch_ddp':
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from xpi_nn_train.datasets.torchvision import get_dataloader
    from xpi_nn_train.trainers.torch_ddp import setup_ddp_env, run_train_ddp

    try:
      setup_ddp_env()
      dist.init_process_group('gloo', init_method='env://')
      world_size = dist.get_world_size()
      rank = dist.get_rank()
      print(f">> rank = {rank} is initialized")

      model = DistributedDataParallel(model)
      train_loader = get_dataloader(dataset, 'train', batch_size=args.batch_size, num_workers=args.num_workers, world_size=world_size, rank=rank)
      test_loader = get_dataloader(dataset, 'val', batch_size=args.batch_size, num_workers=args.num_workers, world_size=world_size, rank=rank)
      run_train_ddp(model, train_loader, test_loader)
    except Exception as e:
      print(e)
    finally:
      dist.destroy_process_group()
