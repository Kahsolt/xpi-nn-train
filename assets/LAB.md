# Lab-server

----

### RTX3060

`python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K torch`
bf16: [Step 1500] loss: 0.48633, acc: 19.699% (67.71590 it/s)

`python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K lightning`
35.69it/s

### 40HX

⚠ 为啥这卡跑训练还不如P104，内存带宽问题？

`python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K torch`
bf16: [Step 1500] loss: 0.50391, acc: 19.784% (12.94878 it/s)
fp32: [Step 1500] loss: 0.41771, acc: 26.830% (27.55788 it/s)

`python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K lightning`
14.15it/s

### P104 (单卡)

`python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K torch`
bf16: [Step 1500] loss: 0.52734, acc: 19.982% (20.73861 it/s)
fp32: [Step 1500] loss: 0.42460, acc: 26.155% (46.20991 it/s)

`python -m xpi_nn_train.examples.finetune_cifar10_mbv3 -K lightning`
28.21it/s
