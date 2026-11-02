### performance benchmark (Embedded)

    仅关心训练速度，用于评估硬件性能 :)

----

ℹ 评测条件: `从头训练MNIST/CIFAR10，取第2个epoch的平均速度`
ℹ 数据类型: fp32

### dryrun: 内存随机数据，无磁盘IO

| hardware | MLP | LeNet | mbnetv3_s (MNIST) | mbnetv3_s (CIFAR10) | espcn (CIFAR10) |
| :-: | :-: | :-: | :-: | :-: |
| RK3576 (lbc3)     |  |  |  |  |  |
| RK3399 (FMX1)     |  |  |  |  |  |
| RK3399 (MRK)      |  |  |  |  |  |
| H618 (opiz3)      |  |  |  |  |  |
| BCM2837B0 (rpi3b) |  |  |  |  |  |

### realdata: 加载数据集，有磁盘IO (不同机器上差异可能较大)
