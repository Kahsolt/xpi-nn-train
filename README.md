# mnist-playground

    Train toy neural networks on Orange Pi Zero 3 :)

----

⚪ train_mnist

```
[version_0] 朴素MLP从头训练]
  cmd: python train_mnist.py
  model: MLP(28*28, 10, 128)
  epoch: 100 (46.85k steps, 3h46m13s)
  bs: 32
  grad_acc: 4
  lr: 2e-4
  train_acc: 0.999
  val_acc: 0.977
  comment: 25k步后过拟合：val_loss上升，但val_acc_top1持平、val_acc_top3仍在上升

[version_1] LoRA微调
  cmd: python train_mnist.py -E 50 -lr 1e-5 --load ./lightning_logs/version_0/checkpoints/epoch=50-step=23731.ckpt --lora 8
  epoch: 50 (23.45k steps, 1h52m11s)
  bs: 32
  grad_acc: 4
  train_acc: 0.9956
  val_acc: 0.977
  comment: 由于原模型已经收敛，原地微调后反而稍差；学出来的矩阵值域量级约为 A: 0.05~0.1, B ~ 0.0057~0.0072
```

----
2025年3月14日
by Armit
