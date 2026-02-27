# xpi-nn-train

    A lightweight toolbox for NN model train on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.

----

**⚠ In most cases you'll NEVER train a model on low-end SoC(s), this repo is mainly for CPU benchmark purpose.** ;)

### Installation

You can either install from PyPI:

```shell
# create venv (optional but recommended!!)
conda create -n xpi python==3.13    # py3.10 or later
conda activate xpi
# install minimal version (torch, recommened!!)
pip install xpi-nn-train
# install full version (+lightning+peft)
pip install xpi-nn-train[full]
```

or install locally:

```shell
# create venv (optional but recommended!!)
conda create -n xpi python==3.13    # py3.10 or later
conda activate xpi
# clone this repo
git clone https://gitee.com/kahsolt/xpi-nn-train.git
cd xpi-nn-train
# install dependencies
pip install -r requirements.txt
# install locally
pip install -e .
```

### Usage

⚪ Use via API

Note that `xpi-nn-train` is opt for **image classification** benchmarking,

- for the basics, see [tutorial.ipynb](./examples/tutorial.ipynb)
- for other tasks please refer to [espcn.ipynb](./examples/espcn.ipynb)

⚪ Use via command line

```shell
# run simple examples
python -m xpi_nn_train.examples.train_mnist -K torch -M MLP
python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K torch -F 1
# run LoRA finetune (need full version)
python -m xpi_nn_train.examples.train_mnist -K lightning -M LeNet
python -m xpi_nn_train.examples.train_mnist -K lightning -M LeNet -r 4 --load ./lightning_logs/version_1/checkpoints/epoch=4-step=2157.ckpt
# run ddp (hardcoded, temporarily only works on my machine 😈
python -m xpi_nn_train.examples.finetune_cifar10_mbv3_ddp
```

### Configurations

ℹ We focus on CV models implemented in PyTorch & TorchVision

#### Envvars

- DATA_ROOT: folder path for auto-downloaded datasets, defaults to `<site-packages>/xpi_nn_train/DATA_ROOT` (PyPI install) or `<xpi-nn-train>/DATA_ROOT` (local install)
- LOG_PATH: folder path for training logs, defaults to `./logs` (PyPI install) or `<xpi-nn-train>/logs` (local install)

#### Models providers

| name | comment |
| :-: | :-: |
| torchvision    | clf |
| LeNet          | MNIST clf |
| ESPCN          | lightweight sr |
| MLP            |  |
| (user-defined) |  |

#### Trainer backends

| name | distributed | peft (LoRA etc.) |
| :-: | :-: | :-: |
| torch     | √ | x |
| lightning | x | √ |

### Tested Devices

- BCM2837: RaspberryPi 3B
- H618: OrangePi Zero 3
- RK3576: LubanCat3
- RK3399: FMX1 Pro, MRK3399
