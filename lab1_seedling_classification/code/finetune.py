# -*- coding: utf-8 -*-
"""迁移学习微调：ImageNet 预训练骨干 + 5 折分层交叉验证 + TTA 推断。

单折运行（便于并行）:
  python finetune.py --backbone resnet18 --fold 0 --epochs 40 --tag ft_r18
一次跑全部折:
  python finetune.py --backbone resnet18 --all-folds --epochs 40 --tag ft_r18

输出（result/）:
  oof_<tag>_fold<k>.npy    该折验证集概率
  oofidx_<tag>_fold<k>.npy 该折验证集在训练集中的索引
  oof_<tag>.npy            训练集 OOF 概率（跑完所有折后为完整矩阵）
  test_<tag>_fold<k>.npy   该折对测试集的概率（含 TTA）
  test_<tag>.npy           各折平均后的测试集概率
  model_<tag>_fold<k>.pt   该折最优权重
  cv_<tag>.json            交叉验证记录

注意：沙箱禁止命名管道，DataLoader 只能 num_workers=0。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import common as C
import models as M


# ------------------------------------------------------------------ 数据
class TrainDS(Dataset):
    """训练集：强增广（随机裁剪/旋转/翻转/色彩抖动/随机擦除）。"""

    def __init__(self, paths, labels, size=224, train=True):
        self.paths, self.labels, self.size, self.train = paths, labels, size, train
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),          # 俯拍幼苗图上下翻转语义安全
                T.RandomRotation(30, fill=(124, 116, 104)),
                T.ColorJitter(0.30, 0.30, 0.30, 0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ])
        else:
            self.tf = M.preprocess_for("resnet18", size=size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = C.load_rgb(self.paths[i])
        x = self.tf(im)
        return x, (self.labels[i] if self.labels is not None else -1)


def tta_batches(im, size=224):
    """一张图的 TTA 视图：{整图, 中心裁剪} × {原图, 水平翻转} × {1.0x, 1.15x 尺度}。"""
    nz = M.normalize_only()
    views = []
    for S in (size, int(size * 1.15)):
        base = im.resize((S, S), Image.BICUBIC)
        cw, ch = int(S * 0.875), int(S * 0.875)
        l, t = (S - cw) // 2, (S - ch) // 2
        center = base.crop((l, t, l + cw, t + ch)).resize((size, size), Image.BICUBIC)
        fh = base.transpose(Image.FLIP_LEFT_RIGHT)
        fhc = fh.crop((l, t, l + cw, t + ch)).resize((size, size), Image.BICUBIC)
        for v in (base.resize((size, size), Image.BICUBIC), center,
                  fh.resize((size, size), Image.BICUBIC), fhc):
            views.append(nz(v))
    return views


def build_model(backbone, num_classes=C.NUM_CLASSES):
    """构造带 N 类头的微调模型。"""
    model, dim = M.build_backbone(backbone)
    ctor_name = M.BACKBONES[backbone][0]
    head = nn.Linear(dim, num_classes)
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, 0, 0.01)
    if ctor_name.startswith("resnet"):
        model.fc = head
    elif ctor_name.startswith(("efficientnet", "mobilenet", "densenet")):
        model.classifier = head
    elif ctor_name.startswith("convnext"):
        model.classifier[2] = head
    elif ctor_name.startswith("vgg"):
        model.classifier[-1] = head
    else:
        raise ValueError(backbone)
    return model


@torch.no_grad()
def predict_probs(model, paths, size=224, batch=16, use_tta=True):
    """返回 (N,5) 概率；use_tta=True 时对 TTA 视图取平均。"""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(paths), batch):
        ims = [C.load_rgb(p) for p in paths[i:i + batch]]
        if use_tta:
            K = len(tta_batches(ims[0], size))
            x = torch.stack([v for im in ims for v in tta_batches(im, size)])
            p = F.softmax(model(x.to(dev)), dim=1).reshape(len(ims), K, -1).mean(1)
        else:
            nz = M.normalize_only()
            x = torch.stack([nz(im.resize((size, size), Image.BICUBIC)) for im in ims])
            p = F.softmax(model(x.to(dev)), dim=1)
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


def train_one_fold(backbone, tr_paths, tr_y, va_paths, va_y, te_paths, size, epochs,
                   lr, wd, batch, seed, label_smooth=0.1):
    """训练单折，返回 (最优记录, 验证集概率, 测试集概率)。"""
    C.seed_everything(seed)
    dev = torch.device("cpu")
    model = build_model(backbone).to(dev)
    ds = TrainDS(tr_paths, tr_y, size=size, train=True)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

    head_prefix = ("fc", "classifier")
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        (head_params if n.startswith(head_prefix) else base_params).append(p)
    opt = torch.optim.AdamW([{"params": base_params, "lr": lr * 0.1},
                             {"params": head_params, "lr": lr}], weight_decay=wd)
    steps = max(1, len(dl)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr * 0.1, lr], total_steps=steps,
        pct_start=0.25, div_factor=10, final_div_factor=100)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    best = {"acc": -1.0, "state": None, "epoch": -1}
    for ep in range(epochs):
        model.train()
        t0, tot, correct, loss_sum = time.time(), 0, 0, 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            # 前 60% epoch 以 50% 概率启用 mixup：小样本下显著抑制过拟合
            if ep < epochs * 0.6 and np.random.rand() < 0.5:
                lam = float(np.random.beta(0.2, 0.2))
                idx = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx]
                out = model(xb)
                loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[idx])
            else:
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            loss_sum += float(loss) * xb.size(0)
            correct += int((out.argmax(1) == yb).sum())
            tot += xb.size(0)
        va_p = predict_probs(model, va_paths, size=size, use_tta=False)
        acc = float((va_p.argmax(1) == np.array(va_y)).mean())
        print(f"    ep{ep+1:3d}/{epochs}  loss={loss_sum/tot:.4f}  train={correct/tot:.4f}  "
              f"val={acc:.4f}  {time.time()-t0:.0f}s", flush=True)
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": ep + 1,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    print(f"    >> 最优 val={best['acc']:.4f} @ep{best['epoch']}", flush=True)
    va_probs = predict_probs(model, va_paths, size=size, use_tta=True)
    te_probs = predict_probs(model, te_paths, size=size, use_tta=True)
    return best, va_probs, te_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="每进程 torch 线程数，0=沿用环境变量")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    tag = a.tag or f"ft_{a.backbone}"
    size, bs = a.size, a.batch
    if a.backbone in ("resnet50", "convnext_tiny", "vgg16_bn"):
        bs = min(bs, 24)

    train = C.list_train()
    tr_all = [p for p, _ in train]
    y = np.array([l for _, l in train])
    te_paths = C.list_test()
    splits = C.fold_indices(y, a.folds, a.seed)
    fold_ids = list(range(a.folds)) if a.all_folds else ([a.fold] if a.fold >= 0 else [0])

    oof = np.zeros((len(y), C.NUM_CLASSES), dtype=np.float32)
    te_acc, per_fold = [], {}
    for k in fold_ids:
        tri, vai = splits[k]
        print(f"\n{'='*70}\n[{tag}] fold {k}  训练 {len(tri)} / 验证 {len(vai)}\n{'='*70}", flush=True)
        best, va_p, te_p = train_one_fold(
            a.backbone, [tr_all[i] for i in tri], y[tri],
            [tr_all[i] for i in vai], y[vai], te_paths,
            size, a.epochs, a.lr, a.wd, bs, a.seed + k)
        oof[vai] = va_p
        te_acc.append(te_p)
        per_fold[str(k)] = {"val_acc": best["acc"], "best_epoch": best["epoch"]}
        np.save(os.path.join(C.OUT, f"oof_{tag}_fold{k}.npy"), va_p.astype(np.float32))
        np.save(os.path.join(C.OUT, f"test_{tag}_fold{k}.npy"), te_p.astype(np.float32))
        np.save(os.path.join(C.OUT, f"oofidx_{tag}_fold{k}.npy"), vai)
        torch.save(best["state"], os.path.join(C.OUT, f"model_{tag}_fold{k}.pt"))
        np.save(os.path.join(C.OUT, f"oof_{tag}.npy"), oof.astype(np.float32))
        C.save_json({"tag": tag, "backbone": a.backbone, "epochs": a.epochs, "lr": a.lr,
                     "size": size, "batch": bs, "seed": a.seed, "done_folds": fold_ids,
                     "per_fold": per_fold}, f"cv_{tag}.json")

    if a.all_folds:
        oof_acc = float((oof.argmax(1) == y).mean())
        np.save(os.path.join(C.OUT, f"test_{tag}.npy"), np.mean(te_acc, axis=0).astype(np.float32))
        res = {"tag": tag, "backbone": a.backbone, "epochs": a.epochs, "lr": a.lr, "size": size,
               "batch": bs, "folds": a.folds, "seed": a.seed, "oof_acc": oof_acc,
               "per_fold": per_fold,
               "fold_val_mean": float(np.mean([v["val_acc"] for v in per_fold.values()]))}
        C.save_json(res, f"cv_{tag}.json")
        print(f"\n[{tag}] OOF 准确率 = {oof_acc:.4f}", flush=True)
    else:
        k = fold_ids[0]
        vai = splits[k][1]
        acc = float((oof[vai].argmax(1) == y[vai]).mean())
        print(f"\n[{tag}] fold {k} 验证准确率 = {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
