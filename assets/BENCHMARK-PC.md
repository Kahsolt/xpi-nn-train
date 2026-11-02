### performance benchmark (PC: desktop & mobile)

    仅关心训练速度，用于评估硬件性能 :)

----

ℹ 评测条件: `用torchvision模型从头训练随机数据(224x224/1000分类/bs=32)，取3个epoch的平均速度`

⚪ fp32

| hardware | device | mobilenet_v3_small | mobilenet_v3_large | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | squeezenet1_1 | efficientnet_v2_s | vgg16_bn | resnet18 | resnet50 | densenet121 | resnext50_32x4d | convnext_tiny | vit_b_16 | vit_b_32 | swin_t | swin_v2_t |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| RTX3060 (GB@170W)     | cuda | 1351.83 | 579.14 | 1381.22 | 1170.63 | 1139.58 | 159.28 | 89.55 | 515.52 | 189.89 | 170.08 | 141.02 | 108.04 | 53.59 | 199.26 | 122.13 | 97.26 |
| RTX3060 (CF@100W LHR) | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40HX                  | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P104-100 (PCIe x1)    | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R780M                 | rocm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| U5 235H               | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 需绑核排除小小核
| i5-12400F             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| i7-11700K             | cpu  |   65.52 |  21.25 |  120.84 |   54.66 |   43.01 |   6.54 |  3.72 |  23.71 |   6.93 |   6.70 |   5.73 |   8.67 |  3.17 |  12.32 |   6.48 |  4.89 |
| E5-2683v4             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R9 8945HS             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R5 1600X              | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

⚪ fp16

| hardware | device | mobilenet_v3_small | mobilenet_v3_large | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | squeezenet1_1 | efficientnet_v2_s | vgg16_bn | resnet18 | resnet50 | densenet121 | resnext50_32x4d | convnext_tiny | vit_b_16 | vit_b_32 | swin_t | swin_v2_t |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| RTX3060 (GB@170W)     | cuda | 1466.60 | 853.14 | 1343.89 | 1269.64 | 1731.55 | 257.32 | 144.94 | 904.05 | 332.05 | 277.02 | 274.25 | 392.64 | 154.25 | 493.25 | 295.37 | 210.05 |
| RTX3060 (CF@100W LHR) | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40HX                  | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P104-100 (PCIe x1)    | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R780M                 | rocm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

⚪ bf16

| hardware | device | mobilenet_v3_small | mobilenet_v3_large | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | squeezenet1_1 | efficientnet_v2_s | vgg16_bn | resnet18 | resnet50 | densenet121 | resnext50_32x4d | convnext_tiny | vit_b_16 | vit_b_32 | swin_t | swin_v2_t |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| RTX3060 (GB@170W)     | cuda | 1462.62 | 765.55 | 1378.91 | 1296.88 | 1688.27 | 233.43 | 147.97 | 860.43 | 327.02 | 279.24 | 263.48 | 389.88 | 151.73 | 490.83 | 291.02 | 211.62 |
| RTX3060 (CF@100W LHR) | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40HX                  | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P104-100 (PCIe x1)    | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R780M                 | rocm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
