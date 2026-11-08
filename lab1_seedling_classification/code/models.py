# -*- coding: utf-8 -*-
"""模型库与特征抽取：用 ImageNet 预训练骨干做迁移学习。

两种用法：
  1) 冻结骨干 + 线性分类头（小样本最稳，见 train_linear.py）
  2) 微调（见 finetune.py）

说明：沙箱禁止进程间命名管道，DataLoader 必须 num_workers=0（单进程加载）。
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import common as C

# 骨干注册表：name -> (构造器, torchvision 权重枚举类名, weights 成员名, 特征维度)
# 全部使用 ImageNet-1K 预训练权重。枚举类名必须精确（torchvision 命名不规则）。
BACKBONES = {
    "resnet18":           ("resnet18",           "ResNet18_Weights",           "IMAGENET1K_V1", 512),
    "resnet34":           ("resnet34",           "ResNet34_Weights",           "IMAGENET1K_V1", 512),
    "resnet50":           ("resnet50",           "ResNet50_Weights",           "IMAGENET1K_V2", 2048),
    "efficientnet_b0":    ("efficientnet_b0",    "EfficientNet_B0_Weights",    "IMAGENET1K_V1", 1280),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "IMAGENET1K_V2", 960),
    "densenet121":        ("densenet121",        "DenseNet121_Weights",        "IMAGENET1K_V1", 1024),
    "convnext_tiny":      ("convnext_tiny",      "ConvNeXt_Tiny_Weights",      "IMAGENET1K_V1", 768),
    "vgg16_bn":           ("vgg16_bn",           "VGG16_BN_Weights",           "IMAGENET1K_V1", 4096),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def weights_enum(name):
    """返回该骨干的 torchvision 权重枚举类。"""
    return getattr(tvm, BACKBONES[name][1])


def build_backbone(name, pretrained=True):
    """返回 (backbone, feat_dim)，backbone 输出已展平的特征向量。"""
    ctor_name, wcls, wname, dim = BACKBONES[name]
    ctor = getattr(tvm, ctor_name)
    w = getattr(weights_enum(name), wname) if pretrained else None
    model = ctor(weights=w)
    if ctor_name.startswith("resnet"):
        model.fc = nn.Identity()
    elif ctor_name.startswith(("efficientnet", "mobilenet", "densenet")):
        model.classifier = nn.Identity()
    elif ctor_name.startswith("convnext"):
        model.classifier = nn.Sequential(model.classifier[0], model.classifier[1], nn.Flatten(1))
    elif ctor_name.startswith("vgg"):
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Flatten(1))
    else:
        raise ValueError(name)
    return model, dim


def mean_std(name):
    """按骨干的官方归一化参数返回 (mean, std)。"""
    try:
        meta = getattr(weights_enum(name), BACKBONES[name][2]).transforms()
        return list(meta.mean), list(meta.std)
    except Exception:
        return IMAGENET_MEAN, IMAGENET_STD


def preprocess_for(name, size=224):
    """按骨干官方预处理（缩放-中心裁剪-归一化）。"""
    mean, std = mean_std(name)
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(mean, std)])


def normalize_only(name="resnet18"):
    mean, std = mean_std(name)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class ViewsDataset(Dataset):
    """每张图产出 4 个视图：整图缩放 / 中心裁剪，各含水平翻转。"""

    def __init__(self, paths, size=224, crop_frac=0.875):
        self.paths = paths
        self.size = size
        self.crop = crop_frac
        self.nz = normalize_only()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = C.load_rgb(self.paths[i])
        S = self.size
        w, h = im.size
        v1 = im.resize((S, S), Image.BICUBIC)
        cw, ch = int(w * self.crop), int(h * self.crop)
        l, t = (w - cw) // 2, (h - ch) // 2
        v2 = im.crop((l, t, l + cw, t + ch)).resize((S, S), Image.BICUBIC)
        out = []
        for v in (v1, v2):
            out.append(self.nz(v))
            out.append(self.nz(v.transpose(Image.FLIP_LEFT_RIGHT)))
        return torch.stack(out)  # (4, 3, S, S)


@torch.no_grad()
def extract_features(backbone_name, paths, size=224, batch=16, verbose=True):
    """返回 (N, 2*D)：整图视图与中心裁剪视图的特征拼接（翻转已平均）。"""
    dev = torch.device("cpu")
    model, dim = build_backbone(backbone_name)
    model.eval().to(dev)
    ds = ViewsDataset(paths, size=size)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    feats, n = [], 0
    for xb in dl:
        B, K, c, h, w = xb.shape
        f = model(xb.reshape(B * K, c, h, w).to(dev)).reshape(B, K, -1).float().numpy()
        feats.append(np.concatenate([(f[:, 0] + f[:, 1]) / 2, (f[:, 2] + f[:, 3]) / 2], axis=1))
        n += B
        if verbose and n % 200 < B:
            print(f"      {n}/{len(paths)}", flush=True)
    return np.concatenate(feats, axis=0), dim
