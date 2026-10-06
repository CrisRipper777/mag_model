# MAG 协同 NC pilot v1

本轮只覆盖 Movies、Grocery、Reddit-S 的 train/validation。test 推理与指标、真实 LP、Toys/ele-fashion 扩展均未运行。三个图的 validation 已被此前多轮开发使用，seed62 是校准，63–65 是本轮描述性确认；不作显著性或因果宣称。

- 标准校准：90/90 成功；标准失败尝试 0。
- 扩展校准：0 成功；失败尝试 0。
- Confirmation：45/45 成功；失败 0。
- LR 范围扩展：无。

## Calibration range decision

标准六配置网格 90/90 成功。selection 中有 9 个图×模型选到 LR 边界；这 9 个获选配置均在 300 步前因 patience 停止。唯一到达 300 步上限的是 Grocery/B1 的 LR=0.001（网格内部值）。因此没有证据显示边界 LR 在达到步数上限时仍持续优化，本轮不扩展 LR。逐项选择依据见 `search_decision.json`。

- 协议锁：`/hdd1/DataInHere/YHF/mag_model/docs/collab_nc_pilot_v1/protocol_lock.json`；training source commit：`1bed18a05a6bf667f61640c29ec098aefcd411b3`。

## Confirmation train / validation 表

Acc 与 Macro-F1 为百分比。均值±SD 使用 seed63/64/65 且 ddof=0；seed62 校准 checkpoint 不并入确认均值。train−val 是均值差，用于描述拟合差距。

| Dataset | Model | Val Acc | Val Macro-F1 | Train Acc | Train Macro-F1 | Train−val Acc/F1 | LR | WD | best steps | step median ms | peak alloc GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Movies | B0 | 51.37 ± 0.21 | 40.28 ± 1.45 | 61.87 | 56.69 | +10.51/+16.40 | 0.003 | 0.0001 | 42,32,31 | 9.56 | 0.408 |
| Movies | B1 | 53.77 ± 0.35 | 43.03 ± 1.83 | 74.07 | 69.58 | +20.30/+26.54 | 0.001 | 0.001 | 46,57,50 | 44.84 | 1.439 |
| Movies | B2 | 53.06 ± 0.13 | 39.71 ± 2.61 | 73.16 | 65.04 | +20.11/+25.33 | 0.001 | 0.0001 | 73,71,81 | 26.72 | 1.357 |
| Movies | M0 | 51.62 ± 0.28 | 38.84 ± 1.22 | 72.95 | 63.24 | +21.33/+24.40 | 0.0003 | 0.001 | 83,77,60 | 274.41 | 4.330 |
| Movies | M1 | 51.76 ± 0.19 | 39.32 ± 1.49 | 73.25 | 63.57 | +21.49/+24.25 | 0.0003 | 0.0001 | 86,80,60 | 256.82 | 4.327 |
| Grocery | B0 | 76.52 ± 0.14 | 68.56 ± 0.61 | 92.18 | 88.68 | +15.67/+20.11 | 0.001 | 0.0001 | 70,54,55 | 7.29 | 0.518 |
| Grocery | B1 | 81.40 ± 0.39 | 74.42 ± 0.79 | 99.97 | 99.97 | +18.57/+25.55 | 0.001 | 0.0001 | 138,133,206 | 40.75 | 1.530 |
| Grocery | B2 | 80.73 ± 0.16 | 72.05 ± 0.54 | 97.32 | 95.54 | +16.58/+23.49 | 0.003 | 0.001 | 70,65,101 | 24.25 | 1.424 |
| Grocery | M0 | 80.43 ± 0.70 | 72.08 ± 1.03 | 99.99 | 99.99 | +19.56/+27.91 | 0.003 | 0.001 | 177,204,211 | 247.29 | 4.148 |
| Grocery | M1 | 79.86 ± 0.72 | 71.31 ± 0.53 | 97.87 | 96.64 | +18.01/+25.33 | 0.003 | 0.0001 | 38,248,188 | 236.88 | 4.103 |
| Reddit-S | B0 | 92.51 ± 0.09 | 87.44 ± 0.27 | 98.82 | 97.57 | +6.30/+10.13 | 0.001 | 0.0001 | 40,71,50 | 7.20 | 0.597 |
| Reddit-S | B1 | 95.91 ± 0.27 | 91.93 ± 0.26 | 99.95 | 99.88 | +4.04/+7.95 | 0.003 | 0.001 | 161,247,43 | 33.39 | 1.846 |
| Reddit-S | B2 | 95.99 ± 0.21 | 92.18 ± 0.65 | 99.97 | 99.91 | +3.97/+7.73 | 0.003 | 0.0001 | 104,75,73 | 25.47 | 1.880 |
| Reddit-S | M0 | 95.83 ± 0.17 | 91.71 ± 0.16 | 99.97 | 99.92 | +4.14/+8.21 | 0.003 | 0.0001 | 64,88,105 | 409.98 | 6.836 |
| Reddit-S | M1 | 95.65 ± 0.23 | 91.74 ± 0.44 | 99.97 | 99.91 | +4.32/+8.17 | 0.001 | 0.0001 | 125,184,75 | 405.04 | 6.833 |

## Seed-level confirmation

| Dataset | Model | Seed | LR | WD | Train Acc/F1 | Val Acc/F1 | Train/Val CE | Best step | Wall min | Median step ms | Peak alloc/reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | B0 | 63 | 0.003 | 0.0001 | 64.72/61.26 | 51.56/41.23 | 1.0205/1.4903 | 42 | 0.11 | 9.57 | 0.408/0.562 |
| Movies | B0 | 64 | 0.003 | 0.0001 | 59.54/52.40 | 51.08/38.23 | 1.1715/1.4780 | 32 | 0.09 | 9.40 | 0.408/0.564 |
| Movies | B0 | 65 | 0.003 | 0.0001 | 61.36/56.40 | 51.47/41.40 | 1.1221/1.4745 | 31 | 0.08 | 9.71 | 0.408/0.564 |
| Movies | B1 | 63 | 0.001 | 0.001 | 73.16/68.24 | 54.14/45.55 | 0.8064/1.4516 | 46 | 0.28 | 46.08 | 1.440/1.619 |
| Movies | B1 | 64 | 0.001 | 0.001 | 78.29/75.74 | 53.30/42.28 | 0.6607/1.5074 | 57 | 0.26 | 43.78 | 1.439/1.627 |
| Movies | B1 | 65 | 0.001 | 0.001 | 70.77/64.75 | 53.87/41.27 | 0.8417/1.4646 | 50 | 0.26 | 44.66 | 1.439/1.627 |
| Movies | B2 | 63 | 0.001 | 0.0001 | 72.38/64.75 | 53.24/41.26 | 0.8295/1.4997 | 73 | 0.32 | 26.10 | 1.357/1.848 |
| Movies | B2 | 64 | 0.001 | 0.0001 | 73.74/65.83 | 52.97/41.83 | 0.8106/1.5191 | 71 | 0.33 | 26.66 | 1.357/1.848 |
| Movies | B2 | 65 | 0.001 | 0.0001 | 73.38/64.52 | 52.97/36.03 | 0.7702/1.6320 | 81 | 0.34 | 27.40 | 1.357/1.848 |
| Movies | M0 | 63 | 0.0003 | 0.001 | 76.85/68.77 | 51.26/40.55 | 0.7293/1.5494 | 83 | 1.13 | 272.91 | 4.331/6.016 |
| Movies | M0 | 64 | 0.0003 | 0.001 | 77.91/68.94 | 51.95/38.22 | 0.7023/1.6188 | 77 | 1.09 | 275.81 | 4.330/6.016 |
| Movies | M0 | 65 | 0.0003 | 0.001 | 64.10/52.02 | 51.65/37.76 | 1.0763/1.4750 | 60 | 0.95 | 274.50 | 4.330/6.016 |
| Movies | M1 | 63 | 0.0003 | 0.0001 | 79.04/72.02 | 51.56/40.96 | 0.6819/1.5764 | 86 | 0.99 | 263.60 | 4.327/6.016 |
| Movies | M1 | 64 | 0.0003 | 0.0001 | 76.63/67.21 | 52.01/39.64 | 0.7328/1.5829 | 80 | 0.93 | 254.25 | 4.327/6.016 |
| Movies | M1 | 65 | 0.0003 | 0.0001 | 64.09/51.48 | 51.71/37.36 | 1.0712/1.4784 | 60 | 0.76 | 252.60 | 4.327/6.016 |
| Grocery | B0 | 63 | 0.001 | 0.0001 | 96.36/94.43 | 76.69/69.32 | 0.1843/0.8849 | 70 | 0.13 | 7.17 | 0.518/6.016 |
| Grocery | B0 | 64 | 0.001 | 0.0001 | 90.10/85.82 | 76.34/67.83 | 0.3578/0.8445 | 54 | 0.12 | 7.15 | 0.518/6.016 |
| Grocery | B0 | 65 | 0.001 | 0.0001 | 90.09/85.78 | 76.52/68.54 | 0.3609/0.8434 | 55 | 0.14 | 7.57 | 0.518/6.016 |
| Grocery | B1 | 63 | 0.001 | 0.0001 | 99.97/99.97 | 81.93/74.36 | 0.0024/1.2957 | 138 | 0.52 | 34.51 | 1.530/6.016 |
| Grocery | B1 | 64 | 0.001 | 0.0001 | 99.94/99.94 | 81.26/75.42 | 0.0071/1.1223 | 133 | 0.52 | 43.46 | 1.530/6.016 |
| Grocery | B1 | 65 | 0.001 | 0.0001 | 99.99/99.99 | 81.00/73.48 | 0.0006/1.5664 | 206 | 0.59 | 44.27 | 1.530/6.016 |
| Grocery | B2 | 63 | 0.003 | 0.001 | 97.50/96.15 | 80.91/72.49 | 0.0995/0.8697 | 70 | 0.27 | 24.55 | 1.424/6.016 |
| Grocery | B2 | 64 | 0.003 | 0.001 | 94.55/90.67 | 80.76/72.36 | 0.1863/0.7691 | 65 | 0.27 | 24.92 | 1.424/6.016 |
| Grocery | B2 | 65 | 0.003 | 0.001 | 99.89/99.80 | 80.53/71.29 | 0.0071/1.3319 | 101 | 0.32 | 23.30 | 1.424/6.016 |
| Grocery | M0 | 63 | 0.003 | 0.001 | 99.99/99.99 | 79.82/72.32 | 0.0003/1.9760 | 177 | 1.57 | 243.76 | 4.148/6.016 |
| Grocery | M0 | 64 | 0.003 | 0.001 | 99.98/99.98 | 80.06/70.72 | 0.0004/2.0081 | 204 | 1.71 | 253.12 | 4.148/6.016 |
| Grocery | M0 | 65 | 0.003 | 0.001 | 100.00/100.00 | 81.41/73.21 | 0.0001/1.7523 | 211 | 1.90 | 244.98 | 4.148/6.016 |
| Grocery | M1 | 63 | 0.003 | 0.0001 | 93.66/89.96 | 78.86/70.67 | 0.2315/0.8259 | 38 | 0.53 | 244.27 | 4.102/6.016 |
| Grocery | M1 | 64 | 0.003 | 0.0001 | 100.00/100.00 | 80.47/71.96 | 0.0000/2.1253 | 248 | 1.87 | 233.43 | 4.104/6.016 |
| Grocery | M1 | 65 | 0.003 | 0.0001 | 99.95/99.96 | 80.26/71.30 | 0.0009/2.0203 | 188 | 1.50 | 232.96 | 4.104/6.016 |
| Reddit-S | B0 | 63 | 0.001 | 0.0001 | 97.61/95.32 | 92.42/87.23 | 0.0931/0.2648 | 40 | 0.10 | 6.79 | 0.597/6.016 |
| Reddit-S | B0 | 64 | 0.001 | 0.0001 | 99.93/99.82 | 92.64/87.27 | 0.0141/0.2901 | 71 | 0.13 | 7.70 | 0.597/6.016 |
| Reddit-S | B0 | 65 | 0.001 | 0.0001 | 98.91/97.56 | 92.48/87.81 | 0.0597/0.2711 | 50 | 0.11 | 7.10 | 0.597/6.016 |
| Reddit-S | B1 | 63 | 0.003 | 0.001 | 99.97/99.92 | 96.07/92.11 | 0.0007/0.3179 | 161 | 0.43 | 33.48 | 1.846/6.016 |
| Reddit-S | B1 | 64 | 0.003 | 0.001 | 99.97/99.92 | 96.13/92.12 | 0.0005/0.2940 | 247 | 0.58 | 33.29 | 1.846/6.016 |
| Reddit-S | B1 | 65 | 0.003 | 0.001 | 99.92/99.79 | 95.53/91.56 | 0.0042/0.2351 | 43 | 0.18 | 33.40 | 1.846/6.016 |
| Reddit-S | B2 | 63 | 0.003 | 0.0001 | 99.97/99.92 | 96.29/93.07 | 0.0006/0.2871 | 104 | 0.32 | 25.62 | 1.880/6.016 |
| Reddit-S | B2 | 64 | 0.003 | 0.0001 | 99.96/99.89 | 95.91/91.92 | 0.0007/0.2885 | 75 | 0.24 | 25.64 | 1.880/6.016 |
| Reddit-S | B2 | 65 | 0.003 | 0.0001 | 99.97/99.92 | 95.78/91.55 | 0.0008/0.2908 | 73 | 0.28 | 25.15 | 1.880/6.016 |
| Reddit-S | M0 | 63 | 0.003 | 0.0001 | 99.97/99.92 | 95.72/91.49 | 0.0006/0.2814 | 64 | 1.09 | 400.70 | 6.836/9.270 |
| Reddit-S | M0 | 64 | 0.003 | 0.0001 | 99.97/99.92 | 96.07/91.85 | 0.0005/0.2813 | 88 | 1.39 | 415.24 | 6.836/9.270 |
| Reddit-S | M0 | 65 | 0.003 | 0.0001 | 99.97/99.92 | 95.69/91.79 | 0.0005/0.3392 | 105 | 1.51 | 414.01 | 6.836/9.270 |
| Reddit-S | M1 | 63 | 0.001 | 0.0001 | 99.97/99.92 | 95.47/91.23 | 0.0005/0.3165 | 125 | 1.58 | 406.41 | 6.833/9.270 |
| Reddit-S | M1 | 64 | 0.001 | 0.0001 | 99.97/99.92 | 95.97/92.30 | 0.0005/0.2967 | 184 | 2.00 | 404.32 | 6.833/9.270 |
| Reddit-S | M1 | 65 | 0.001 | 0.0001 | 99.96/99.89 | 95.50/91.70 | 0.0009/0.2852 | 75 | 1.25 | 404.39 | 6.833/9.270 |

## Path-head metrics at each selected checkpoint

来源头只在对应原始 availability 子集评估；表中给出分母。B0 无 path head。

| Dataset | Model | Seed | Partition | Head | n | Acc | Macro-F1 | CE |
|---|---|---:|---|---|---:|---:|---:|---:|
| Movies | B1 | 63 | train | image_private | 10003 | 48.89 | 23.91 | 1.6139 |
| Movies | B1 | 63 | train | shared | 10003 | 48.90 | 22.02 | 1.5710 |
| Movies | B1 | 63 | train | text_private | 10003 | 40.22 | 10.64 | 1.9776 |
| Movies | B1 | 63 | validation | image_private | 3334 | 43.61 | 18.61 | 1.8279 |
| Movies | B1 | 63 | validation | shared | 3334 | 43.70 | 17.95 | 1.7504 |
| Movies | B1 | 63 | validation | text_private | 3334 | 40.64 | 10.59 | 1.9797 |
| Movies | B1 | 64 | train | image_private | 10003 | 52.09 | 26.66 | 1.5194 |
| Movies | B1 | 64 | train | shared | 10003 | 53.09 | 25.25 | 1.4502 |
| Movies | B1 | 64 | train | text_private | 10003 | 41.92 | 13.79 | 1.8741 |
| Movies | B1 | 64 | validation | image_private | 3334 | 45.17 | 21.43 | 1.7816 |
| Movies | B1 | 64 | validation | shared | 3334 | 46.52 | 20.31 | 1.7057 |
| Movies | B1 | 64 | validation | text_private | 3334 | 40.43 | 11.97 | 1.9362 |
| Movies | B1 | 65 | train | image_private | 10003 | 48.92 | 23.63 | 1.6468 |
| Movies | B1 | 65 | train | shared | 10003 | 50.23 | 23.23 | 1.5512 |
| Movies | B1 | 65 | train | text_private | 10003 | 40.37 | 11.05 | 1.9677 |
| Movies | B1 | 65 | validation | image_private | 3334 | 44.24 | 20.18 | 1.8634 |
| Movies | B1 | 65 | validation | shared | 3334 | 46.46 | 17.43 | 1.7305 |
| Movies | B1 | 65 | validation | text_private | 3334 | 40.13 | 10.87 | 1.9968 |
| Movies | B2 | 63 | train | image | 10003 | 51.84 | 24.79 | 1.5682 |
| Movies | B2 | 63 | train | text | 10003 | 42.92 | 13.24 | 1.8494 |
| Movies | B2 | 63 | validation | image | 3334 | 45.95 | 18.63 | 1.7390 |
| Movies | B2 | 63 | validation | text | 3334 | 41.54 | 11.73 | 1.8988 |
| Movies | B2 | 64 | train | image | 10003 | 52.40 | 23.95 | 1.5755 |
| Movies | B2 | 64 | train | text | 10003 | 42.55 | 12.29 | 1.8308 |
| Movies | B2 | 64 | validation | image | 3334 | 46.61 | 18.18 | 1.7467 |
| Movies | B2 | 64 | validation | text | 3334 | 41.72 | 11.61 | 1.8840 |
| Movies | B2 | 65 | train | image | 10003 | 50.91 | 20.17 | 1.5527 |
| Movies | B2 | 65 | train | text | 10003 | 42.08 | 12.56 | 1.8663 |
| Movies | B2 | 65 | validation | image | 3334 | 45.65 | 15.57 | 1.7266 |
| Movies | B2 | 65 | validation | text | 3334 | 41.12 | 11.55 | 1.9028 |
| Movies | M0 | 63 | train | image_private | 10003 | 44.20 | 15.51 | 1.8984 |
| Movies | M0 | 63 | train | shared | 10003 | 46.93 | 17.54 | 1.7228 |
| Movies | M0 | 63 | train | text_private | 10003 | 40.73 | 10.62 | 2.0003 |
| Movies | M0 | 63 | validation | image_private | 3334 | 39.02 | 11.99 | 2.0761 |
| Movies | M0 | 63 | validation | shared | 3334 | 40.82 | 13.62 | 1.9565 |
| Movies | M0 | 63 | validation | text_private | 3334 | 39.95 | 10.09 | 2.0127 |
| Movies | M0 | 64 | train | image_private | 10003 | 43.91 | 14.77 | 1.8713 |
| Movies | M0 | 64 | train | shared | 10003 | 47.36 | 14.75 | 1.7617 |
| Movies | M0 | 64 | train | text_private | 10003 | 40.19 | 11.02 | 1.9914 |
| Movies | M0 | 64 | validation | image_private | 3334 | 40.40 | 13.49 | 2.0287 |
| Movies | M0 | 64 | validation | shared | 3334 | 41.90 | 12.36 | 1.9740 |
| Movies | M0 | 64 | validation | text_private | 3334 | 40.04 | 10.48 | 2.0280 |
| Movies | M0 | 65 | train | image_private | 10003 | 40.10 | 13.24 | 2.0459 |
| Movies | M0 | 65 | train | shared | 10003 | 44.12 | 12.53 | 1.9018 |
| Movies | M0 | 65 | train | text_private | 10003 | 38.30 | 8.23 | 2.1628 |
| Movies | M0 | 65 | validation | image_private | 3334 | 37.19 | 11.52 | 2.1582 |
| Movies | M0 | 65 | validation | shared | 3334 | 43.16 | 11.80 | 1.9863 |
| Movies | M0 | 65 | validation | text_private | 3334 | 38.21 | 8.32 | 2.1710 |
| Movies | M1 | 63 | train | image_private | 10003 | 44.57 | 15.47 | 1.8818 |
| Movies | M1 | 63 | train | shared | 10003 | 47.91 | 18.39 | 1.6906 |
| Movies | M1 | 63 | train | text_private | 10003 | 40.85 | 10.89 | 1.9986 |
| Movies | M1 | 63 | validation | image_private | 3334 | 39.59 | 12.29 | 2.0724 |
| Movies | M1 | 63 | validation | shared | 3334 | 40.94 | 13.21 | 1.9410 |
| Movies | M1 | 63 | validation | text_private | 3334 | 40.34 | 10.56 | 2.0133 |
| Movies | M1 | 64 | train | image_private | 10003 | 43.67 | 14.61 | 1.8851 |
| Movies | M1 | 64 | train | shared | 10003 | 46.16 | 14.18 | 1.7944 |
| Movies | M1 | 64 | train | text_private | 10003 | 39.84 | 10.98 | 1.9995 |
| Movies | M1 | 64 | validation | image_private | 3334 | 40.37 | 13.44 | 2.0316 |
| Movies | M1 | 64 | validation | shared | 3334 | 41.39 | 12.35 | 1.9689 |
| Movies | M1 | 64 | validation | text_private | 3334 | 40.07 | 10.58 | 2.0317 |
| Movies | M1 | 65 | train | image_private | 10003 | 40.13 | 13.21 | 2.0470 |
| Movies | M1 | 65 | train | shared | 10003 | 44.09 | 13.27 | 1.9010 |
| Movies | M1 | 65 | train | text_private | 10003 | 38.35 | 8.31 | 2.1651 |
| Movies | M1 | 65 | validation | image_private | 3334 | 37.13 | 11.45 | 2.1595 |
| Movies | M1 | 65 | validation | shared | 3334 | 43.19 | 12.01 | 1.9870 |
| Movies | M1 | 65 | validation | text_private | 3334 | 38.24 | 8.39 | 2.1729 |
| Grocery | B1 | 63 | train | image_private | 10244 | 89.71 | 80.70 | 0.3235 |
| Grocery | B1 | 63 | train | shared | 10244 | 97.14 | 94.14 | 0.1136 |
| Grocery | B1 | 63 | train | text_private | 10244 | 83.13 | 73.73 | 0.5461 |
| Grocery | B1 | 63 | validation | image_private | 3415 | 73.41 | 62.99 | 1.1883 |
| Grocery | B1 | 63 | validation | shared | 3415 | 79.74 | 70.86 | 0.9105 |
| Grocery | B1 | 63 | validation | text_private | 3415 | 75.43 | 64.27 | 0.9078 |
| Grocery | B1 | 64 | train | image_private | 10244 | 84.11 | 72.62 | 0.5003 |
| Grocery | B1 | 64 | train | shared | 10244 | 92.37 | 85.61 | 0.2469 |
| Grocery | B1 | 64 | train | text_private | 10244 | 79.54 | 69.34 | 0.6676 |
| Grocery | B1 | 64 | validation | image_private | 3415 | 70.63 | 58.51 | 1.1581 |
| Grocery | B1 | 64 | validation | shared | 3415 | 78.18 | 69.03 | 0.8415 |
| Grocery | B1 | 64 | validation | text_private | 3415 | 72.65 | 60.92 | 0.9737 |
| Grocery | B1 | 65 | train | image_private | 10244 | 96.40 | 93.94 | 0.1295 |
| Grocery | B1 | 65 | train | shared | 10244 | 99.83 | 99.65 | 0.0232 |
| Grocery | B1 | 65 | train | text_private | 10244 | 87.84 | 81.25 | 0.3918 |
| Grocery | B1 | 65 | validation | image_private | 3415 | 73.12 | 62.68 | 1.4917 |
| Grocery | B1 | 65 | validation | shared | 3415 | 79.80 | 71.75 | 1.1530 |
| Grocery | B1 | 65 | validation | text_private | 3415 | 76.25 | 66.83 | 0.8934 |
| Grocery | B2 | 63 | train | image | 10244 | 70.90 | 60.38 | 0.9959 |
| Grocery | B2 | 63 | train | text | 10244 | 71.15 | 57.99 | 1.0094 |
| Grocery | B2 | 63 | validation | image | 3415 | 62.96 | 51.65 | 1.2868 |
| Grocery | B2 | 63 | validation | text | 3415 | 67.03 | 53.64 | 1.1571 |
| Grocery | B2 | 64 | train | image | 10244 | 69.39 | 58.47 | 1.0383 |
| Grocery | B2 | 64 | train | text | 10244 | 68.60 | 55.06 | 1.0854 |
| Grocery | B2 | 64 | validation | image | 3415 | 63.10 | 51.53 | 1.2883 |
| Grocery | B2 | 64 | validation | text | 3415 | 65.04 | 51.30 | 1.2080 |
| Grocery | B2 | 65 | train | image | 10244 | 74.17 | 67.26 | 0.8618 |
| Grocery | B2 | 65 | train | text | 10244 | 75.37 | 63.99 | 0.8433 |
| Grocery | B2 | 65 | validation | image | 3415 | 64.83 | 56.03 | 1.2560 |
| Grocery | B2 | 65 | validation | text | 3415 | 69.02 | 56.52 | 1.0551 |
| Grocery | M0 | 63 | train | image_private | 10244 | 99.40 | 98.76 | 0.0323 |
| Grocery | M0 | 63 | train | shared | 10244 | 99.63 | 99.57 | 0.0225 |
| Grocery | M0 | 63 | train | text_private | 10244 | 99.94 | 99.92 | 0.0049 |
| Grocery | M0 | 63 | validation | image_private | 3415 | 72.94 | 63.19 | 2.3987 |
| Grocery | M0 | 63 | validation | shared | 3415 | 75.17 | 66.29 | 1.8347 |
| Grocery | M0 | 63 | validation | text_private | 3415 | 75.26 | 67.65 | 2.1041 |
| Grocery | M0 | 64 | train | image_private | 10244 | 99.77 | 99.52 | 0.0159 |
| Grocery | M0 | 64 | train | shared | 10244 | 99.71 | 99.64 | 0.0228 |
| Grocery | M0 | 64 | train | text_private | 10244 | 99.93 | 99.95 | 0.0076 |
| Grocery | M0 | 64 | validation | image_private | 3415 | 70.89 | 59.49 | 2.6055 |
| Grocery | M0 | 64 | validation | shared | 3415 | 75.02 | 65.45 | 1.7626 |
| Grocery | M0 | 64 | validation | text_private | 3415 | 74.58 | 63.61 | 2.2730 |
| Grocery | M0 | 65 | train | image_private | 10244 | 99.70 | 99.45 | 0.0207 |
| Grocery | M0 | 65 | train | shared | 10244 | 98.19 | 97.71 | 0.0728 |
| Grocery | M0 | 65 | train | text_private | 10244 | 97.72 | 95.97 | 0.0996 |
| Grocery | M0 | 65 | validation | image_private | 3415 | 72.74 | 62.70 | 2.1694 |
| Grocery | M0 | 65 | validation | shared | 3415 | 76.19 | 66.07 | 1.5692 |
| Grocery | M0 | 65 | validation | text_private | 3415 | 75.87 | 66.35 | 1.2490 |
| Grocery | M1 | 63 | train | image_private | 10244 | 75.57 | 57.72 | 0.8366 |
| Grocery | M1 | 63 | train | shared | 10244 | 73.13 | 58.75 | 0.9018 |
| Grocery | M1 | 63 | train | text_private | 10244 | 60.86 | 43.54 | 1.3370 |
| Grocery | M1 | 63 | validation | image_private | 3415 | 68.52 | 51.83 | 1.2081 |
| Grocery | M1 | 63 | validation | shared | 3415 | 68.20 | 54.46 | 1.1310 |
| Grocery | M1 | 63 | validation | text_private | 3415 | 58.33 | 40.31 | 1.4230 |
| Grocery | M1 | 64 | train | image_private | 10244 | 99.93 | 99.92 | 0.0059 |
| Grocery | M1 | 64 | train | shared | 10244 | 99.92 | 99.93 | 0.0077 |
| Grocery | M1 | 64 | train | text_private | 10244 | 100.00 | 100.00 | 0.0016 |
| Grocery | M1 | 64 | validation | image_private | 3415 | 71.16 | 58.69 | 2.8350 |
| Grocery | M1 | 64 | validation | shared | 3415 | 75.23 | 66.67 | 2.1261 |
| Grocery | M1 | 64 | validation | text_private | 3415 | 75.23 | 64.16 | 2.4324 |
| Grocery | M1 | 65 | train | image_private | 10244 | 99.63 | 99.46 | 0.0255 |
| Grocery | M1 | 65 | train | shared | 10244 | 99.63 | 99.60 | 0.0215 |
| Grocery | M1 | 65 | train | text_private | 10244 | 99.57 | 99.47 | 0.0196 |
| Grocery | M1 | 65 | validation | image_private | 3415 | 72.97 | 62.88 | 2.1140 |
| Grocery | M1 | 65 | validation | shared | 3415 | 76.02 | 66.01 | 1.9310 |
| Grocery | M1 | 65 | validation | text_private | 3415 | 74.85 | 65.44 | 2.1426 |
| Reddit-S | B1 | 63 | train | image_private | 9536 | 97.64 | 97.59 | 0.0829 |
| Reddit-S | B1 | 63 | train | shared | 9536 | 99.96 | 99.89 | 0.0018 |
| Reddit-S | B1 | 63 | train | text_private | 9536 | 99.93 | 99.83 | 0.0053 |
| Reddit-S | B1 | 63 | validation | image_private | 3179 | 93.17 | 89.35 | 0.3805 |
| Reddit-S | B1 | 63 | validation | shared | 3179 | 95.78 | 92.09 | 0.2833 |
| Reddit-S | B1 | 63 | validation | text_private | 3179 | 90.28 | 84.47 | 0.6083 |
| Reddit-S | B1 | 64 | train | image_private | 9536 | 97.71 | 97.79 | 0.0801 |
| Reddit-S | B1 | 64 | train | shared | 9536 | 99.97 | 99.92 | 0.0008 |
| Reddit-S | B1 | 64 | train | text_private | 9536 | 99.97 | 99.92 | 0.0038 |
| Reddit-S | B1 | 64 | validation | image_private | 3179 | 93.08 | 89.55 | 0.4155 |
| Reddit-S | B1 | 64 | validation | shared | 3179 | 96.26 | 92.85 | 0.2747 |
| Reddit-S | B1 | 64 | validation | text_private | 3179 | 91.03 | 85.92 | 0.5580 |
| Reddit-S | B1 | 65 | train | image_private | 9536 | 96.13 | 93.83 | 0.1320 |
| Reddit-S | B1 | 65 | train | shared | 9536 | 97.78 | 96.11 | 0.0692 |
| Reddit-S | B1 | 65 | train | text_private | 9536 | 77.88 | 69.97 | 0.6833 |
| Reddit-S | B1 | 65 | validation | image_private | 3179 | 92.67 | 87.89 | 0.3048 |
| Reddit-S | B1 | 65 | validation | shared | 3179 | 93.99 | 89.54 | 0.2382 |
| Reddit-S | B1 | 65 | validation | text_private | 3179 | 74.90 | 65.62 | 0.7959 |
| Reddit-S | B2 | 63 | train | image | 9536 | 94.24 | 92.51 | 0.2014 |
| Reddit-S | B2 | 63 | train | text | 9536 | 91.46 | 88.62 | 0.3509 |
| Reddit-S | B2 | 63 | validation | image | 3179 | 89.27 | 84.42 | 0.3688 |
| Reddit-S | B2 | 63 | validation | text | 3179 | 69.90 | 64.41 | 1.0510 |
| Reddit-S | B2 | 64 | train | image | 9536 | 92.92 | 89.79 | 0.2411 |
| Reddit-S | B2 | 64 | train | text | 9536 | 80.62 | 75.99 | 0.6453 |
| Reddit-S | B2 | 64 | validation | image | 3179 | 89.46 | 84.50 | 0.3673 |
| Reddit-S | B2 | 64 | validation | text | 3179 | 70.18 | 63.77 | 1.0126 |
| Reddit-S | B2 | 65 | train | image | 9536 | 93.05 | 90.29 | 0.2416 |
| Reddit-S | B2 | 65 | train | text | 9536 | 79.69 | 74.37 | 0.6836 |
| Reddit-S | B2 | 65 | validation | image | 3179 | 89.34 | 83.69 | 0.3675 |
| Reddit-S | B2 | 65 | validation | text | 3179 | 68.67 | 62.01 | 1.0206 |
| Reddit-S | M0 | 63 | train | image_private | 9536 | 97.00 | 96.18 | 0.1046 |
| Reddit-S | M0 | 63 | train | shared | 9536 | 99.65 | 99.13 | 0.0193 |
| Reddit-S | M0 | 63 | train | text_private | 9536 | 95.44 | 92.07 | 0.1533 |
| Reddit-S | M0 | 63 | validation | image_private | 3179 | 92.10 | 86.89 | 0.3550 |
| Reddit-S | M0 | 63 | validation | shared | 3179 | 94.72 | 90.32 | 0.2381 |
| Reddit-S | M0 | 63 | validation | text_private | 3179 | 87.01 | 79.32 | 0.4961 |
| Reddit-S | M0 | 64 | train | image_private | 9536 | 97.37 | 96.99 | 0.0960 |
| Reddit-S | M0 | 64 | train | shared | 9536 | 99.56 | 98.95 | 0.0173 |
| Reddit-S | M0 | 64 | train | text_private | 9536 | 98.92 | 98.05 | 0.0479 |
| Reddit-S | M0 | 64 | validation | image_private | 3179 | 92.61 | 87.16 | 0.3339 |
| Reddit-S | M0 | 64 | validation | shared | 3179 | 95.03 | 89.87 | 0.2246 |
| Reddit-S | M0 | 64 | validation | text_private | 3179 | 88.52 | 81.57 | 0.5371 |
| Reddit-S | M0 | 65 | train | image_private | 9536 | 97.64 | 97.62 | 0.0841 |
| Reddit-S | M0 | 65 | train | shared | 9536 | 99.88 | 99.71 | 0.0061 |
| Reddit-S | M0 | 65 | train | text_private | 9536 | 99.86 | 99.70 | 0.0084 |
| Reddit-S | M0 | 65 | validation | image_private | 3179 | 92.23 | 87.14 | 0.3858 |
| Reddit-S | M0 | 65 | validation | shared | 3179 | 95.03 | 90.80 | 0.2819 |
| Reddit-S | M0 | 65 | validation | text_private | 3179 | 88.68 | 82.35 | 0.7213 |
| Reddit-S | M1 | 63 | train | image_private | 9536 | 97.14 | 96.45 | 0.1044 |
| Reddit-S | M1 | 63 | train | shared | 9536 | 99.81 | 99.54 | 0.0144 |
| Reddit-S | M1 | 63 | train | text_private | 9536 | 98.47 | 97.49 | 0.0686 |
| Reddit-S | M1 | 63 | validation | image_private | 3179 | 92.10 | 87.57 | 0.3555 |
| Reddit-S | M1 | 63 | validation | shared | 3179 | 94.21 | 89.05 | 0.2859 |
| Reddit-S | M1 | 63 | validation | text_private | 3179 | 88.02 | 80.85 | 0.5606 |
| Reddit-S | M1 | 64 | train | image_private | 9536 | 97.56 | 97.35 | 0.0880 |
| Reddit-S | M1 | 64 | train | shared | 9536 | 99.96 | 99.89 | 0.0031 |
| Reddit-S | M1 | 64 | train | text_private | 9536 | 99.95 | 99.87 | 0.0048 |
| Reddit-S | M1 | 64 | validation | image_private | 3179 | 92.26 | 87.03 | 0.3395 |
| Reddit-S | M1 | 64 | validation | shared | 3179 | 94.75 | 90.42 | 0.2985 |
| Reddit-S | M1 | 64 | validation | text_private | 3179 | 89.21 | 83.64 | 0.6902 |
| Reddit-S | M1 | 65 | train | image_private | 9536 | 96.25 | 94.37 | 0.1341 |
| Reddit-S | M1 | 65 | train | shared | 9536 | 98.94 | 97.74 | 0.0430 |
| Reddit-S | M1 | 65 | train | text_private | 9536 | 86.64 | 79.66 | 0.4314 |
| Reddit-S | M1 | 65 | validation | image_private | 3179 | 91.82 | 86.55 | 0.3289 |
| Reddit-S | M1 | 65 | validation | shared | 3179 | 93.99 | 89.21 | 0.2642 |
| Reddit-S | M1 | 65 | validation | text_private | 3179 | 82.26 | 73.15 | 0.6223 |

## Active parameters and runtime

参数按活动 backbone 与 NC task heads 分开；B0 的 640-D 主头含零填充槽，表中同时列完整 head 参数与有效 head 参数，不声称容量配平。

| Dataset | Model | Active backbone | Task heads total/effective | Zero-pad head params | Total active | Wall min mean | Normal peak alloc/reserved GiB | Diagnostic peak alloc GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | B0 | 247040 | 12820/7700 | 5120 | 259860 | 0.09 | 0.408/0.564 | 0.164 |
| Movies | B1 | 893056 | 18000/18000 | 0 | 911056 | 0.27 | 1.439/1.624 | 0.237 |
| Movies | B2 | 811744 | 7740/7740 | 0 | 819484 | 0.33 | 1.357/1.848 | 0.191 |
| Movies | M0 | 1103600 | 18000/18000 | 0 | 1121600 | 1.06 | 4.330/6.016 | 0.227 |
| Movies | M1 | 903408 | 18000/18000 | 0 | 921408 | 0.89 | 4.327/6.016 | 0.223 |
| Grocery | B0 | 247040 | 12820/7700 | 5120 | 259860 | 0.13 | 0.518/6.016 | 0.267 |
| Grocery | B1 | 893056 | 18000/18000 | 0 | 911056 | 0.54 | 1.530/6.016 | 0.344 |
| Grocery | B2 | 811744 | 7740/7740 | 0 | 819484 | 0.29 | 1.424/6.016 | 0.294 |
| Grocery | M0 | 1103600 | 18000/18000 | 0 | 1121600 | 1.73 | 4.148/6.016 | 0.331 |
| Grocery | M1 | 903408 | 18000/18000 | 0 | 921408 | 1.30 | 4.103/6.016 | 0.327 |
| Reddit-S | B0 | 247040 | 12820/7700 | 5120 | 259860 | 0.11 | 0.597/6.016 | 0.364 |
| Reddit-S | B1 | 893056 | 18000/18000 | 0 | 911056 | 0.40 | 1.846/6.016 | 0.438 |
| Reddit-S | B2 | 811744 | 7740/7740 | 0 | 819484 | 0.28 | 1.880/6.016 | 0.392 |
| Reddit-S | M0 | 1103600 | 18000/18000 | 0 | 1121600 | 1.33 | 6.836/9.270 | 0.425 |
| Reddit-S | M1 | 903408 | 18000/18000 | 0 | 921408 | 1.61 | 6.833/9.270 | 0.422 |

## Calibration cells

每个候选配置都保留在 `run_results.jsonl` 与 registry；下表列出全部候选的验证正确数/F1/best step，便于复核配置选择。

| Dataset | Model | Config | LR | WD | Val correct | Val Macro-F1 | Best step | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Movies | B0 | 0 | 0.0003 | 0.0001 | 1664 | 0.35384910737483677 | 78 | success |
| Movies | B0 | 1 | 0.0003 | 0.001 | 1664 | 0.35384910737483677 | 78 | success |
| Movies | B0 | 2 | 0.001 | 0.0001 | 1700 | 0.40271787663329073 | 45 | success |
| Movies | B0 | 3 | 0.001 | 0.001 | 1700 | 0.40271787663329073 | 45 | success |
| Movies | B0 | 4 | 0.003 | 0.0001 | 1703 | 0.4127996642056953 | 35 | success |
| Movies | B0 | 5 | 0.003 | 0.001 | 1703 | 0.4127996642056953 | 35 | success |
| Movies | B1 | 0 | 0.0003 | 0.0001 | 1786 | 0.4275679757864708 | 87 | success |
| Movies | B1 | 1 | 0.0003 | 0.001 | 1788 | 0.4308451669542414 | 84 | success |
| Movies | B1 | 2 | 0.001 | 0.0001 | 1784 | 0.4292416738352216 | 56 | success |
| Movies | B1 | 3 | 0.001 | 0.001 | 1790 | 0.43040459534103687 | 56 | success |
| Movies | B1 | 4 | 0.003 | 0.0001 | 1771 | 0.4469452944318489 | 46 | success |
| Movies | B1 | 5 | 0.003 | 0.001 | 1776 | 0.44706190602503304 | 38 | success |
| Movies | B2 | 0 | 0.0003 | 0.0001 | 1705 | 0.3635473617779911 | 146 | success |
| Movies | B2 | 1 | 0.0003 | 0.001 | 1704 | 0.2944835321002268 | 124 | success |
| Movies | B2 | 2 | 0.001 | 0.0001 | 1761 | 0.41052266812964955 | 76 | success |
| Movies | B2 | 3 | 0.001 | 0.001 | 1747 | 0.4033760145004546 | 69 | success |
| Movies | B2 | 4 | 0.003 | 0.0001 | 1753 | 0.3732928190015008 | 54 | success |
| Movies | B2 | 5 | 0.003 | 0.001 | 1741 | 0.3781636302420533 | 83 | success |
| Movies | M0 | 0 | 0.0003 | 0.0001 | 1726 | 0.39099168825925795 | 69 | success |
| Movies | M0 | 1 | 0.0003 | 0.001 | 1728 | 0.3914529649606525 | 69 | success |
| Movies | M0 | 2 | 0.001 | 0.0001 | 1714 | 0.40806452570807183 | 36 | success |
| Movies | M0 | 3 | 0.001 | 0.001 | 1713 | 0.40953504308828614 | 36 | success |
| Movies | M0 | 4 | 0.003 | 0.0001 | 1727 | 0.4395905973500704 | 41 | success |
| Movies | M0 | 5 | 0.003 | 0.001 | 1722 | 0.4337959188409939 | 45 | success |
| Movies | M1 | 0 | 0.0003 | 0.0001 | 1738 | 0.407016997768144 | 72 | success |
| Movies | M1 | 1 | 0.0003 | 0.001 | 1736 | 0.40696584449147755 | 72 | success |
| Movies | M1 | 2 | 0.001 | 0.0001 | 1732 | 0.40705697139922703 | 40 | success |
| Movies | M1 | 3 | 0.001 | 0.001 | 1732 | 0.4089118410439461 | 40 | success |
| Movies | M1 | 4 | 0.003 | 0.0001 | 1725 | 0.4232286258922541 | 35 | success |
| Movies | M1 | 5 | 0.003 | 0.001 | 1726 | 0.4236248689177836 | 35 | success |
| Grocery | B0 | 0 | 0.0003 | 0.0001 | 2622 | 0.6773036200541476 | 112 | success |
| Grocery | B0 | 1 | 0.0003 | 0.001 | 2622 | 0.6773036200541476 | 112 | success |
| Grocery | B0 | 2 | 0.001 | 0.0001 | 2624 | 0.6937284678532908 | 78 | success |
| Grocery | B0 | 3 | 0.001 | 0.001 | 2614 | 0.689985612346961 | 76 | success |
| Grocery | B0 | 4 | 0.003 | 0.0001 | 2621 | 0.687209458210715 | 51 | success |
| Grocery | B0 | 5 | 0.003 | 0.001 | 2618 | 0.691078253490307 | 50 | success |
| Grocery | B1 | 0 | 0.0003 | 0.0001 | 2741 | 0.7308960731485123 | 178 | success |
| Grocery | B1 | 1 | 0.0003 | 0.001 | 2751 | 0.7370722756755319 | 235 | success |
| Grocery | B1 | 2 | 0.001 | 0.0001 | 2786 | 0.7479688322263783 | 286 | success |
| Grocery | B1 | 3 | 0.001 | 0.001 | 2782 | 0.7518388441305401 | 221 | success |
| Grocery | B1 | 4 | 0.003 | 0.0001 | 2727 | 0.7239901176033524 | 45 | success |
| Grocery | B1 | 5 | 0.003 | 0.001 | 2726 | 0.7217019063694814 | 45 | success |
| Grocery | B2 | 0 | 0.0003 | 0.0001 | 2704 | 0.687673937020503 | 277 | success |
| Grocery | B2 | 1 | 0.0003 | 0.001 | 2705 | 0.6860918596050996 | 249 | success |
| Grocery | B2 | 2 | 0.001 | 0.0001 | 2744 | 0.7282385257505775 | 121 | success |
| Grocery | B2 | 3 | 0.001 | 0.001 | 2739 | 0.7277045634661188 | 130 | success |
| Grocery | B2 | 4 | 0.003 | 0.0001 | 2750 | 0.7299367950862312 | 79 | success |
| Grocery | B2 | 5 | 0.003 | 0.001 | 2761 | 0.7370682304737811 | 85 | success |
| Grocery | M0 | 0 | 0.0003 | 0.0001 | 2643 | 0.6824463834183312 | 96 | success |
| Grocery | M0 | 1 | 0.0003 | 0.001 | 2644 | 0.6796411347400685 | 95 | success |
| Grocery | M0 | 2 | 0.001 | 0.0001 | 2687 | 0.6903404247091766 | 44 | success |
| Grocery | M0 | 3 | 0.001 | 0.001 | 2688 | 0.687175745080981 | 46 | success |
| Grocery | M0 | 4 | 0.003 | 0.0001 | 2691 | 0.7057626303733154 | 42 | success |
| Grocery | M0 | 5 | 0.003 | 0.001 | 2753 | 0.7278068965391322 | 224 | success |
| Grocery | M1 | 0 | 0.0003 | 0.0001 | 2648 | 0.6848786947711207 | 90 | success |
| Grocery | M1 | 1 | 0.0003 | 0.001 | 2652 | 0.6874046591899117 | 104 | success |
| Grocery | M1 | 2 | 0.001 | 0.0001 | 2681 | 0.6946235551075337 | 45 | success |
| Grocery | M1 | 3 | 0.001 | 0.001 | 2687 | 0.6906089232734063 | 44 | success |
| Grocery | M1 | 4 | 0.003 | 0.0001 | 2760 | 0.7186433233441245 | 172 | success |
| Grocery | M1 | 5 | 0.003 | 0.001 | 2758 | 0.7203930327193226 | 173 | success |
| Reddit-S | B0 | 0 | 0.0003 | 0.0001 | 2935 | 0.8700607870949486 | 130 | success |
| Reddit-S | B0 | 1 | 0.0003 | 0.001 | 2935 | 0.8700607870949486 | 130 | success |
| Reddit-S | B0 | 2 | 0.001 | 0.0001 | 2943 | 0.8744527081021827 | 45 | success |
| Reddit-S | B0 | 3 | 0.001 | 0.001 | 2943 | 0.8744527081021827 | 45 | success |
| Reddit-S | B0 | 4 | 0.003 | 0.0001 | 2941 | 0.8780869932687281 | 26 | success |
| Reddit-S | B0 | 5 | 0.003 | 0.001 | 2941 | 0.8780869932687281 | 26 | success |
| Reddit-S | B1 | 0 | 0.0003 | 0.0001 | 3045 | 0.9201806506427538 | 137 | success |
| Reddit-S | B1 | 1 | 0.0003 | 0.001 | 3045 | 0.9201806506427538 | 137 | success |
| Reddit-S | B1 | 2 | 0.001 | 0.0001 | 3048 | 0.9223474539064475 | 117 | success |
| Reddit-S | B1 | 3 | 0.001 | 0.001 | 3048 | 0.9223474539064475 | 117 | success |
| Reddit-S | B1 | 4 | 0.003 | 0.0001 | 3054 | 0.9185678792882863 | 186 | success |
| Reddit-S | B1 | 5 | 0.003 | 0.001 | 3057 | 0.920489375380692 | 216 | success |
| Reddit-S | B2 | 0 | 0.0003 | 0.0001 | 3032 | 0.9121530046039418 | 89 | success |
| Reddit-S | B2 | 1 | 0.0003 | 0.001 | 3032 | 0.9121530046039418 | 89 | success |
| Reddit-S | B2 | 2 | 0.001 | 0.0001 | 3040 | 0.9214851756655754 | 115 | success |
| Reddit-S | B2 | 3 | 0.001 | 0.001 | 3038 | 0.9209919719218427 | 116 | success |
| Reddit-S | B2 | 4 | 0.003 | 0.0001 | 3048 | 0.9205643275324974 | 38 | success |
| Reddit-S | B2 | 5 | 0.003 | 0.001 | 3048 | 0.9205643275324974 | 38 | success |
| Reddit-S | M0 | 0 | 0.0003 | 0.0001 | 3036 | 0.9175641477910664 | 181 | success |
| Reddit-S | M0 | 1 | 0.0003 | 0.001 | 3036 | 0.9175641477910664 | 181 | success |
| Reddit-S | M0 | 2 | 0.001 | 0.0001 | 3049 | 0.9189349384938623 | 133 | success |
| Reddit-S | M0 | 3 | 0.001 | 0.001 | 3049 | 0.9189349384938623 | 133 | success |
| Reddit-S | M0 | 4 | 0.003 | 0.0001 | 3056 | 0.9225372054718788 | 60 | success |
| Reddit-S | M0 | 5 | 0.003 | 0.001 | 3055 | 0.9220536076802688 | 60 | success |
| Reddit-S | M1 | 0 | 0.0003 | 0.0001 | 3041 | 0.9211361327948223 | 216 | success |
| Reddit-S | M1 | 1 | 0.0003 | 0.001 | 3041 | 0.9211361327948223 | 216 | success |
| Reddit-S | M1 | 2 | 0.001 | 0.0001 | 3053 | 0.9225424614239193 | 197 | success |
| Reddit-S | M1 | 3 | 0.001 | 0.001 | 3053 | 0.9225424614239193 | 197 | success |
| Reddit-S | M1 | 4 | 0.003 | 0.0001 | 3048 | 0.9132912338127863 | 65 | success |
| Reddit-S | M1 | 5 | 0.003 | 0.001 | 3049 | 0.9142005830873796 | 65 | success |

## Matched descriptive deltas

同数据集、同 confirmation training seed 的候选差，单位百分点；先对三个 seed 求均值，再对三个图等权平均。差值是描述性比较，不作显著性或因果结论。

| Model contrast | Dataset | Mean Δ Acc | Mean Δ Macro-F1 | Seed-level Acc deltas | Seed-level F1 deltas |
|---|---|---:|---:|---|---|
| M0−B1 | Movies | -2.15 | -4.19 | -2.88,-1.35,-2.22 | -5.01,-4.06,-3.50 |
| M0−B1 | Grocery | -0.97 | -2.34 | -2.11,-1.20,+0.41 | -2.04,-4.70,-0.27 |
| M0−B1 | Reddit-S | -0.08 | -0.22 | -0.35,-0.06,+0.16 | -0.62,-0.27,+0.22 |
| M0−B1 | Equal-graph mean | -1.07 | -2.25 | — | — |
| M1−B1 | Movies | -2.01 | -3.71 | -2.58,-1.29,-2.16 | -4.59,-2.64,-3.90 |
| M1−B1 | Grocery | -1.53 | -3.11 | -3.07,-0.79,-0.73 | -3.68,-3.46,-2.18 |
| M1−B1 | Reddit-S | -0.26 | -0.19 | -0.60,-0.16,-0.03 | -0.88,+0.18,+0.14 |
| M1−B1 | Equal-graph mean | -1.27 | -2.34 | — | — |
| M0−B2 | Movies | -1.44 | -0.86 | -1.98,-1.02,-1.32 | -0.71,-3.61,+1.73 |
| M0−B2 | Grocery | -0.30 | +0.04 | -1.08,-0.70,+0.88 | -0.17,-1.64,+1.92 |
| M0−B2 | Reddit-S | -0.17 | -0.47 | -0.57,+0.16,-0.09 | -1.58,-0.07,+0.24 |
| M0−B2 | Equal-graph mean | -0.64 | -0.43 | — | — |
| M1−B2 | Movies | -1.30 | -0.38 | -1.68,-0.96,-1.26 | -0.30,-2.19,+1.34 |
| M1−B2 | Grocery | -0.87 | -0.73 | -2.05,-0.29,-0.26 | -1.82,-0.40,+0.01 |
| M1−B2 | Reddit-S | -0.35 | -0.44 | -0.82,+0.06,-0.28 | -1.85,+0.38,+0.15 |
| M1−B2 | Equal-graph mean | -0.84 | -0.52 | — | — |
| M0−M1 | Movies | -0.14 | -0.48 | -0.30,-0.06,-0.06 | -0.42,-1.42,+0.40 |
| M0−M1 | Grocery | +0.57 | +0.77 | +0.97,-0.41,+1.14 | +1.65,-1.24,+1.91 |
| M0−M1 | Reddit-S | +0.18 | -0.03 | +0.25,+0.09,+0.19 | +0.27,-0.45,+0.08 |
| M0−M1 | Equal-graph mean | +0.20 | +0.09 | — | — |

## Descriptive reading

- M0-B1: equal-graph means are -1.07 pp Acc and -2.25 pp Macro-F1 (lower on the two metrics in these three fitted seeds). This is descriptive, not a significance or causal result.
- M1-B1: equal-graph means are -1.27 pp Acc and -2.34 pp Macro-F1 (lower on the two metrics in these three fitted seeds). This is descriptive, not a significance or causal result.
- M0-B2: equal-graph means are -0.64 pp Acc and -0.43 pp Macro-F1 (lower on the two metrics in these three fitted seeds). This is descriptive, not a significance or causal result.
- M1-B2: equal-graph means are -0.84 pp Acc and -0.52 pp Macro-F1 (lower on the two metrics in these three fitted seeds). This is descriptive, not a significance or causal result.
- M0-M1: equal-graph means are +0.20 pp Acc and +0.09 pp Macro-F1 (higher on the two metrics in these three fitted seeds). This is descriptive, not a significance or causal result.

M0−M1 is the task-only versus weak pre-graph reconstruction/pair objective comparison for this implementation. B1/B2 comparisons include their specified operator and, for B2, legacy edge canonicalization. Path-head scores and router values describe trained computations; they do not establish semantic roles, causal utility, or information proportions. Train−validation gaps can describe fit/generalization differences but cannot by themselves identify overfitting, shared-embedding effects, gate competition, or another optimizer cause.


## Outcome and engineering cost

在这三个已多次开发使用的图上，B1 同模态宽 mean 图在 Movies 与 Grocery 的确认 Val Acc 最高；Reddit-S 上 B2 为 95.99%，B1 为 95.91%，差异很小。M0 相对 B1 的等图均值为 −1.07 pp Acc/−2.25 pp Macro-F1，M1 相对 B1 为 −1.27/−2.34 pp。相对 B2，M0 为 −0.64/−0.43 pp，M1 为 −0.84/−0.52 pp。当前完整双轴骨干没有显示出超过强简单/旧 R1 参照的验证收益。M0−M1 的等图均值只有 +0.20/+0.09 pp，且逐图并非同方向，不能据此认为弱图前 SSL 有稳定收益。

M0/M1 的计算代价明显高于 B1：活动 backbone 参数约 1.104M/0.903M，对比 B1 的 0.893M；确认时训练 step 中位数约 237–410 ms，对比 B1 约 33–45 ms，且 M0/M1 的峰值 allocated memory 约 4.1–6.8 GiB，对比 B1 约 1.4–1.8 GiB。因此就本轮这些验证表现而言，额外双轴能力尚未证明值得作为默认骨干承担该代价；这只是本轮描述性结论，不是否定后续机制研究。

本轮实际加载的 R1 canonical edge set 与 loader edge set 在 Movies、Grocery、Reddit-S 的对称差均为 0；边数和图 hash 见 `data_profile.json`。资源统计中每 run 的 peak allocated 是本次峰值；PyTorch CUDA allocator 会跨顺序运行保留 cache，peak reserved 因而含此前运行保留的 reservation 高水位，不能把表中 reserved 数字解释成每个模型独立冷启动的显存。

## Scope, cost, and interpretation

所有正式训练使用全图 train-only supervision；全图特征和拓扑按既有传导协议参与 forward。指标只索引固定 train/validation。B2 保持旧 R1 的图 canonicalization，M0/B1 用 loader 原边，因此 B2 对比需连同 edge-set 差异解释。参数量不代表计算量匹配。

路径头、路由/NULL/π与状态范数是计算诊断，不是因果效用、信息占比或真实语义角色。M0/M1 的差异可描述弱图前 SSL 是否与任务目标共存；B0/B1/B2 的差异也不能单独定位某一结构因素。过拟合、共享表示、gate竞争和优化因素没有因果判定。

## Evidence

- Registry: `/hdd1/DataInHere/YHF/mag_model/reports/collab_nc_pilot_v1/run_registry.json`
- 逐单元结果: `/hdd1/DataInHere/YHF/mag_model/reports/collab_nc_pilot_v1/run_results.jsonl`
- Protocol lock: `/hdd1/DataInHere/YHF/mag_model/docs/collab_nc_pilot_v1/protocol_lock.json`
- Raw checkpoints, logs, train/validation argmax: `/hdd1/DataInHere/YHF/mag_model/outputs/collab_nc_pilot_v1`
- 本文由冻结 train/validation argmax 和指标重新汇总；审查性质为自审，不是独立审查。
