# xpi-nn-train

    A lightweight toolbox for NN model train on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.

----

**⚠ In most cases you'll NEVER train a model on low-end SoC(s), this repo is mainly for CPU benchmark purpose.** ;)

### Quickstart

```
pip install xpi-nn-train
python -m xpi_nn_train.tools.install_default_backends -y
python -m xpi_nn_train.tools.test_train_mnist
```

### Tested Devices

- BCM2837: RaspberryPi 3B
- H618: OrangePi Zero 3
- RK3576: LubanCat3
- RK3399: FMX1 Pro, MRK3399

### Backends

ℹ xpi-nn-train is a bare framework, you need configure backends to make it work.

#### Model providers

Thanks to all the open-source model providers 🎉

| name | supported | comment |
| :-: | :-: | :-: |
| torchvision    |  |  |
| (user-defined) |  |  |

#### Trainer backends

| name | distributed | peft (LoRA etc.) | comment |
| :-: | :-: | :-: |
| torch     | √ | √ |  |
| lightning | √ | √ |  |

#### IO backends

| name | type | comment |
| :-: | :-: | :-: |
| torchvision | img |  |
| datatsets   | img |  |
