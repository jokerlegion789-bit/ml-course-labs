# -*- coding: utf-8 -*-
"""task2 模型库：为 48x48 灰度 FER 设计的轻量 CNN（CPU 可训练）。

设计原则：
  - 尽早下采样（stem 用 stride=2），把算力花在通道数而不是空间分辨率上；
  - 全部用 3x3 卷积 + BatchNorm + ReLU，GAP 收尾，几乎无全连接层；
  - 单图前向约 20-30M MACs，保证 CPU 上每轮可控制在 1-2 分钟。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class VGG_FER(nn.Module):
    """紧凑 VGG：stem s2 -> [32, 64, 128, 256] 逐级下采样 -> GAP。"""

    def __init__(self, n_class=6, width=1.0, p_drop=0.4):
        super().__init__()
        w = lambda c: max(8, int(c * width))
        self.f = nn.Sequential(
            conv_bn(1, w(32), s=2),          # 24
            conv_bn(w(32), w(32)),
            conv_bn(w(32), w(64), s=2),      # 12
            conv_bn(w(64), w(64)),
            conv_bn(w(64), w(128), s=2),     # 6
            conv_bn(w(128), w(128)),
            conv_bn(w(128), w(256), s=2),    # 3
            conv_bn(w(256), w(256)),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(w(256), n_class))

    def forward(self, x):
        return self.head(self.f(x))


class BasicBlock(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.short = None
        if s != 1 or i != o:
            self.short = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + idt, inplace=True)


class ResNet_FER(nn.Module):
    """小型 ResNet：stem s2，三阶段各 2 个残差块。"""

    def __init__(self, n_class=6, p_drop=0.35):
        super().__init__()
        self.stem = conv_bn(1, 32, s=2)                      # 24
        self.s1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))          # 24
        self.s2 = nn.Sequential(BasicBlock(32, 64, s=2), BasicBlock(64, 64))     # 12
        self.s3 = nn.Sequential(BasicBlock(64, 128, s=2), BasicBlock(128, 128))  # 6
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(p_drop), nn.Linear(128, n_class))

    def forward(self, x):
        x = self.stem(x)
        x = self.s3(self.s2(self.s1(x)))
        return self.head(x)


class Wide_CNN(nn.Module):
    """两阶段宽卷积：卷积核更大，感受野更早覆盖全脸。"""

    def __init__(self, n_class=6, p_drop=0.4):
        super().__init__()
        self.f = nn.Sequential(
            conv_bn(1, 64, k=5, s=2, p=2),     # 24
            conv_bn(64, 64, k=3),
            nn.MaxPool2d(2),                    # 12
            conv_bn(64, 128, k=3),
            conv_bn(128, 128, k=3),
            nn.MaxPool2d(2),                    # 6
            conv_bn(128, 256, k=3),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(p_drop), nn.Linear(256, n_class))

    def forward(self, x):
        return self.head(self.f(x))


NETS = {"vgg_fer": VGG_FER, "resnet_fer": ResNet_FER, "wide_cnn": Wide_CNN}


def build(net, n_class=6, **kw):
    if net not in NETS:
        raise ValueError(f"未知网络 {net}，可选 {list(NETS)}")
    return NETS[net](n_class=n_class, **kw)


# ---------------------------------------------------------------- 张量级增广
# 随机擦除块的最大边长（像素）；48x48 输入下不宜过大，否则会盖掉整张脸的关键区域
ERASE_MAX = 15


def augment(x, train=True, flip_p=0.5, max_rot=12.0, max_shift=0.08,
            scale=(0.90, 1.10), erase_p=0.25):
    """对整批 (B,1,48,48) 已归一化张量做增广，全部是批量张量操作，CPU 上很快。"""
    if not train:
        return x
    B = x.size(0)
    dev = x.device
    # 水平翻转
    m = (torch.rand(B, device=dev) < flip_p).view(B, 1, 1, 1)
    x = torch.where(m, x.flip(-1), x)
    # 仿射：旋转 + 平移 + 缩放（一次性批量 grid_sample）
    ang = (torch.rand(B, device=dev) * 2 - 1) * (max_rot * 3.14159265 / 180)
    tx = (torch.rand(B, device=dev) * 2 - 1) * max_shift
    ty = (torch.rand(B, device=dev) * 2 - 1) * max_shift
    sc = scale[0] + torch.rand(B, device=dev) * (scale[1] - scale[0])
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    # 亮度/对比度抖动
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mu) * c + mu) * b
    # 随机擦除
    if erase_p > 0:
        n_e = int(B * erase_p)
        if n_e > 0:
            idx = torch.randperm(B, device=dev)[:n_e]
            for i in idx.tolist():
                for _ in range(1):
                    eh = int(torch.randint(4, ERASE_MAX + 1, (1,)).item())
                    ew = int(torch.randint(4, ERASE_MAX + 1, (1,)).item())
                    y0 = int(torch.randint(0, 48 - eh, (1,)).item())
                    x0 = int(torch.randint(0, 48 - ew, (1,)).item())
                    x[i, :, y0:y0 + eh, x0:x0 + ew] = 0.0
    return x


def normalize_np(arr):
    """uint8 (N,48,48) -> float32 (N,1,48,48)，标准化到 mean0 std1（约 mean .5 / std .25）。"""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    t = t.div_(255.0).sub_(0.5).div_(0.25)
    return t.unsqueeze(1)


import numpy as np  # noqa: E402  (放在末尾避免遮蔽上面的用法)
