# 机器学习课程设计实验一～四 代码

本仓库为东北大学《机器学习》课程设计四个实验的全部源码。

实验报告（Word）另附，本仓库用于代码复查。

## 目录

| 目录 | 实验 | 任务类型 | 线上成绩 |
|---|---|---|---|
| `lab1_seedling_classification/` | 实验一 植物幼苗图像分类 | 小样本细粒度分类 | 首次 0.22（提交格式错误）→ 已修复 |
| `lab2_facial_emotion/` | 实验二 人脸表情识别（FER2013 六类） | 图像分类 | 0.417（标签编码错误）→ **0.682** |
| `lab3_fovea_localization/` | 实验三 视网膜黄斑中心凹定位 | 关键点回归 | 11 px（采用既有版本） |
| `lab4_lesion_segmentation/` | 实验四 眼底病灶分割 | 语义分割 | **Dice 0.82** |

## 环境

```
Python 3.14.7
torch 2.14.1+cpu, torchvision 0.29.1+cpu
numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1
Pillow, matplotlib
```

全部实验在 CPU 上完成（28 逻辑核，无 GPU）。

## 快速运行

### 实验一　植物幼苗分类

```bat
python train_linear.py resnet18 resnet50 efficientnet_b0 densenet121 --size 224
python finetune.py --backbone resnet18 --all-folds --epochs 45 --tag ft_r18
python finetune.py --backbone resnet50 --all-folds --epochs 40 --lr 2.5e-4 --tag ft_r50
python finetune.py --backbone efficientnet_b0 --all-folds --epochs 45 --tag ft_effb0
finalize.cmd
```

关键脚本：
- `train_linear.py` — 路线 A：冻结特征 + 9 种分类头 × 5 折 CV
- `finetune.py` — 路线 B：端到端微调 + 8 视图 TTA
- `validate_ensemble.py` — **集成方法的嵌套验证**（A 半选权重、B 半评估）
- `verify_submission.py` — **提交文件独立核验**（重新推理逐张核对 ID 配对）

### 实验二　人脸表情识别

```bat
python train2.py --net resnet_fer --epochs 80 --seed 42 --tag v2_res_s42
python train2.py --net vgg_fer    --epochs 80 --seed 42 --tag v2_vgg_s42
python retest_tta.py --apply
python ensemble2.py --prefix v --encoding fer_order
python verify2.py --n 12
```

关键脚本：
- `nets.py` — VGG_FER / ResNet_FER / Wide_CNN 三个网络结构
- `infer_encoding.py` — **从线上分数反推标签数字编码**（5040 种排列搜索）

### 实验三　黄斑中心凹定位

```bat
python d6_fast.py
```

关键脚本：
- `d1_classic.py` — 经典图像处理检测器与评估
- `d3_unet.py` — U-Net 热图回归（64×64 热图 + 亚像素质心解码）
- `d4_twostage.py` — 两阶段：粗定位 + 原分辨率裁剪精修

### 实验四　病灶分割

```bat
python t4_unet.py
```

```powershell
.\rle_tool.ps1 -Input submission.csv -Output modified.csv -Dilate 1
```

关键脚本：
- `t4_unet.py` — 小型 U-Net + 4 折 CV + 官方 RLE 编码
- `rle_tool.ps1` — RLE 解析、合法性校验与可控膨胀

## 数据准备

数据集未包含在本仓库中（体积大且版权归课程方）。请将课程下发的压缩包解压后按脚本内的路径常量放置，或修改各 `common*.py` / `t4_unet.py` 顶部的路径变量。

## 说明

- 所有实验均固定随机种子，便于复现。
- 报告中所有数值均可在各实验 `result/` 目录下的日志与 JSON 中找到对应记录。
- 未对任何失败或负面实验做修饰。
