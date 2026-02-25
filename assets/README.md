# mnist-playground

    Train toy neural networks on Orange Pi Zero 3 :)

----

### MNIST on opiz3

⚪ train_mnist

```
[version_0] 朴素MLP从头训练
  cmd: python train_mnist.py
  model: MLP(28*28, 10, 128)
  epoch: 100 (46.85k steps)
  bs: 32
  grad_acc: 4
  lr: 2e-4
  train_acc: 0.999
  val_acc: 0.977
  comment: 25k步后过拟合：val_loss上升，但val_acc_top1持平、val_acc_top3仍在上升

| Device/Host | dtype | Timecost |
| :-: | :-: | :-: |
| opiz3  | bf16 | 3h46m13s |
| opiz3  | fp32 | 3h28m26s |
| opiz3  | bf16 | 5h4m39s  |
| rk3399 | fp32 | 2h40m    |

[version_1] LoRA微调
  cmd: python train_mnist.py -E 50 -lr 1e-5 --load ./lightning_logs/version_0/checkpoints/epoch=50-step=23731.ckpt --lora 8
  epoch: 50 (23.45k steps, 1h52m11s)
  bs: 32
  grad_acc: 4
  train_acc: 0.9956
  val_acc: 0.977
  comment: 由于原模型已经收敛，原地微调后反而稍差；学出来的矩阵值域量级约为 A: 0.05~0.1, B ~ 0.0057~0.0072
```

⚪ train_cifar10

分阶段微调

```
[version_2] 仅训练最后一层Linear
  cmd: python train_cifar10.py -E 50 -lr 2e-4 --phase 0
  epoch: 50 (19.55k steps, 2.33it/s, 14h5m20s)
  bs: 32
  grad_acc: 4
  train_acc: ~0.27
  val_acc: ~0.3
  comment: 6k步时训练已收敛

[version_3] 仅微调第一个Conv2dNormActivation块和最后的classififer块
  cmd: python train_cifar10.py -E 20 -lr 2e-4 --phase 1 --load ./lightning_logs/version_2/checkpoints/epoch=36-step=14193.ckpt
  epoch: 20 (7.819k steps, 0.83it/s, 11h38m7s)
  train_acc: 0.3144
  val_acc: 0.3581
  comment: 验证集精度直线上涨但非常慢，似乎这样训不太行啊 :(

[version_4] 全量微调
  cmd: python train_cifar10.py -E 50 -lr 2e-4 --phase -1 --load ./lightning_logs/version_3/checkpoints/epoch=18-step=7389.ckpt
  epoch: 50 (19.51k steps, 0.4it/s, 2d5h15m11s)
  train_acc: 0.7736
  val_acc: 0.7448
  comment: 仍未完全收敛，但涨幅空间不大了
```

直接全量微调

```
[version_5] 直接全量微调
  cmd: python train_cifar10.py -E 50 -lr 2e-4 --phase -1
  epoch: 50 (19.51k steps, 0.4it/s; 0.5it/s on fmx1)
```

⚪ test_ddp

```
[rk3399集群 (6卡)] 直接全量微调
  cmd: python test_ddp.py
  epoch: 50 (3300 steps)
    - rk3399 * 6: 2.47536 s/b, 8799.64s=2.44h
    - i5-12400F:  0.14061 s/b, 3417.02s=56.95min
    - RTX 3060:   0.01562 s/b,  867.01s=14.45min
  bs: 128
  lr: 1e-3
  test_acc: 71.146%
```

----
2025年3月14日
by Armit
