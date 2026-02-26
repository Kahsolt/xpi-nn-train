# xpi-nn-train

    A lightweight toolbox for NN model train on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.

----

**⚠ In most cases you'll NEVER train a model on low-end SoC(s), this repo is mainly for CPU benchmark purpose.** ;)

### Quickstart

```shell
# clone this repo
git clone https://gitee.com/kahsolt/xpi-nn-train.git
cd xpi-nn-train

# create venv (optional but recommended!!)
conda create -n xpi python==3.13
conda create xpi
# install dependencies
pip install -r requirements.txt

# run simple examples
python -m xpi_nn_train.examples.train_mnist -K torch -M MLP
python -m xpi_nn_train.examples.train_mnist -K lightning -M LeNet
python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K torch -F 1
# run LoRA finetune
python -m xpi_nn_train.examples.train_mnist -K lightning -M LeNet -r 4 --load ./lightning_logs/version_1/checkpoints/epoch=4-step=2157.ckpt
# run ddp (hardcoded, temporarily only works on my machine 😈
python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K torch_ddp
```

### Backends

ℹ We focus on CV models implemented in PyTorch & TorchVision

#### Models providers

| name | comment |
| :-: | :-: |
| torchvision    | clf/seg/det |
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
