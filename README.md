# xpi-nn-train

    A lightweight toolbox for NN model train on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.
    Or just run a quick performance benchmark on any backends!

----

**⚠ You'll NEVER wanna train a model on the low-end SoC(s), this repo is mainly for CPU/GPU performance benchmark purpose.** ;)

### Installation

You can either install from PyPI:

```shell
# create venv (optional but recommended!!)
mamba create -n xpi python==3.13    # py3.10 or later
mamba activate xpi
# install minimal version (only torch+torchvision, recommened!!)
uv pip install xpi-nn-train
# install full version (+lightning+peft)
uv pip install xpi-nn-train[full]
```

or install locally:

```shell
# create venv (optional but recommended!!)
mamba create -n xpi python==3.13    # py3.10 or later
mamba activate xpi
# clone this repo
git clone https://github.com/Kahsolt/xpi-nn-train.git
cd xpi-nn-train
# install dependencies
uv pip install -r requirements.txt
# install locally
uv pip install -e .
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

ℹ So far we only focus on CV models implemented in PyTorch & TorchVision

#### Envvars

- DATA_ROOT: folder path for auto-downloaded datasets, defaults to `<site-packages>/xpi_nn_train/DATA_ROOT` (PyPI install) or `<xpi-nn-train>/DATA_ROOT` (local install)
- LOG_PATH: folder path for training logs, defaults to `./logs` (PyPI install) or `<xpi-nn-train>/logs` (local install)

#### Models providers

| name | comment |
| :-: | :-: |
| torchvision    | ImageNet clf |
| LeNet          | MNIST clf |
| ESPCN          | lightweight sr |
| MLP            |  |
| (user-defined) |  |

#### Trainer backends

| name | distributed | peft (LoRA etc.) |
| :-: | :-: | :-: |
| torch     | √ | x |
| lightning | x | √ |

### ref

- xpi-nn-infer: https://github.com/Kahsolt/xpi-nn-infer
