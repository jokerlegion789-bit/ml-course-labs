# -*- coding: utf-8 -*-
"""task4 眼底病灶分割：小型 U-Net + 4 折交叉验证 + 官方 RLE 编码。

关键约束（来自数据实测）：
  - 图像 565x584 RGB，标签同为 565x584 灰度，前景占比 8.63% ± (6.03~11.59)
  - 提交采用 512x512 行优先、1-indexed 的 RLE（见官方 segmentation_to_csv.py）
  - 官方脚本流程：convert('L') -> resize((512,512), BILINEAR) -> 值>0 视为前景
    本脚本严格复刻该流程，保证编码与平台一致。

训练：128x128 输入（提速），输出同尺寸掩膜；推理时再按官方方式放大到 512。
验证：4 折交叉验证（20 张 -> 每折 5 张），报告 OOF Dice。
"""
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

D = r"D:\deepseek\task4\segmentation"
OUT = r"D:\deepseek\task4\result"
os.makedirs(OUT, exist_ok=True)
S = 128
SEED = 42
THREADS = 20


# ------------------------------------------------------------------ 数据
def load_pairs():
    names = sorted(os.listdir(os.path.join(D, "train", "image")),
                   key=lambda x: int(os.path.splitext(x)[0]))
    X, Y = [], []
    for f in names:
        im = Image.open(os.path.join(D, "train", "image", f)).convert("RGB").resize((S, S), Image.BILINEAR)
        lb = Image.open(os.path.join(D, "train", "label", f)).convert("L").resize((S, S), Image.NEAREST)
        X.append(np.asarray(im, dtype=np.float32) / 255.0)
        Y.append((np.asarray(lb, dtype=np.float32) > 127).astype(np.float32))
    return np.stack(X), np.stack(Y), names


def load_test():
    names = sorted(os.listdir(os.path.join(D, "test", "image")),
                   key=lambda x: int(os.path.splitext(x)[0]))
    X, raw = [], []
    for f in names:
        im = Image.open(os.path.join(D, "test", "image", f)).convert("RGB")
        raw.append(im.resize((512, 512), Image.BILINEAR))
        X.append(np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0)
    return np.stack(X), names, raw


# ------------------------------------------------------------------ 模型
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    """早下采样的小型 U-Net：128 -> 64 -> 32 -> 16 -> 8，再对称上采样回 128。"""

    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(cbr(3, 16), cbr(16, 16))        # 128
        self.e2 = nn.Sequential(cbr(16, 32), cbr(32, 32))       # 64
        self.e3 = nn.Sequential(cbr(32, 64), cbr(64, 64))       # 32
        self.e4 = nn.Sequential(cbr(64, 128), cbr(128, 128))    # 16
        self.bot = nn.Sequential(cbr(128, 256), cbr(256, 256))  # 8
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c4 = cbr(256 + 128, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c3 = cbr(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c2 = cbr(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1 = cbr(32 + 16, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        s4 = self.e4(F.max_pool2d(s3, 2))
        b = self.bot(F.max_pool2d(s4, 2))
        d4 = self.c4(torch.cat([self.up4(b), s4], 1))
        d3 = self.c3(torch.cat([self.up3(d4), s3], 1))
        d2 = self.c2(torch.cat([self.up2(d3), s2], 1))
        d1 = self.c1(torch.cat([self.up1(d2), s1], 1))
        return self.out(d1)          # (B,1,128,128) logits


# ------------------------------------------------------------------ 增广
def augment(xb, yb, rng):
    B = xb.size(0)
    dev = xb.device
    hf = torch.rand(B, device=dev) < 0.5
    if bool(hf.any()):
        xb[hf] = xb[hf].flip(-1); yb[hf] = yb[hf].flip(-1)
    vf = torch.rand(B, device=dev) < 0.5
    if bool(vf.any()):
        xb[vf] = xb[vf].flip(-2); yb[vf] = yb[vf].flip(-2)
    ang = (torch.rand(B, device=dev) * 2 - 1) * 0.35
    sc = 0.85 + torch.rand(B, device=dev) * 0.30
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.08
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.08
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = cos; th[:, 0, 1] = -sin; th[:, 0, 2] = tx
    th[:, 1, 0] = sin; th[:, 1, 1] = cos; th[:, 1, 2] = ty
    gx = F.affine_grid(th, list(xb.shape), align_corners=False)
    xb = F.grid_sample(xb, gx, align_corners=False, padding_mode="reflection")
    gy = F.affine_grid(th, list(yb.shape), align_corners=False)
    yb = F.grid_sample(yb, gy, align_corners=False, padding_mode="zeros")
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    mu = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = ((xb - mu) * c + mu) * b
    return xb, yb


def dice_loss(p, t, eps=1e-6):
    p = torch.sigmoid(p)
    inter = (p * t).sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps)).mean()


@torch.no_grad()
def predict(model, X, bs=4):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32)
        if xb.shape[-1] == 3:                      # (B,H,W,C) -> (B,C,H,W)
            xb = xb.permute(0, 3, 1, 2).contiguous()
        p = torch.sigmoid(model(xb))
        out.append(p.cpu().numpy()[:, 0])
    return np.concatenate(out, 0)


def dice_np(pred_bool, gt_bool, eps=1e-6):
    inter = (pred_bool & gt_bool).sum()
    return float((2 * inter + eps) / (pred_bool.sum() + gt_bool.sum() + eps))


def train_one(X, Y, epochs=250, seed=0, lr=3e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    m = SmallUNet()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(X) // 4) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    Yt = torch.from_numpy(Y).unsqueeze(1)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt) - 3, 4):
            idx = perm[i:i + 4]
            xb = Xt[idx].clone(); yb = Yt[idx].clone()
            xb, yb = augment(xb, yb, rng)
            p = m(xb)
            loss = bce(p, yb) + dice_loss(p, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
    return m


# ------------------------------------------------------------------ 官方 RLE
def rle_from_mask(mask512):
    """严格复刻 segmentation_to_csv.py：二值 -> 展平 -> 1-indexed RLE。"""
    flat = mask512.reshape(-1)
    dots = np.where(flat > 0)[0]
    out = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            out.extend((int(b) + 1, 0))
        out[-1] += 1
        prev = b
    return ' '.join(str(v) for v in out)


def main():
    t0 = time.time()
    torch.set_num_threads(THREADS)
    X, Y, names = load_pairs()
    print(f"训练数据 {X.shape} 掩膜 {Y.shape}  前景占比 {Y.mean()*100:.2f}%", flush=True)
    n = len(X)
    rng = np.random.default_rng(123)
    folds = np.array_split(rng.permutation(n), 4)

    # ---- 4 折交叉验证 ----
    oof_dice, oof_pred = [], np.zeros((n, S, S), np.float32)
    for k, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), va)
        tk = time.time()
        m = train_one(X[tr], Y[tr], epochs=250, seed=k)
        p = predict(m, X[va])
        oof_pred[va] = p
        d = [dice_np(p[i] > 0.5, Y[va][i] > 0.5) for i in range(len(va))]
        oof_dice.extend(d)
        print(f"  折{k+1}: Dice = {np.mean(d):.4f}  ({time.time()-tk:.0f}s)", flush=True)
    oof_bool = oof_pred > 0.5
    print(f"\n>>> OOF Dice = {np.mean(oof_dice):.4f} ± {np.std(oof_dice):.4f}", flush=True)
    print(f"    OOF 预测前景占比 = {oof_bool.mean()*100:.2f}%  (真值 {Y.mean()*100:.2f}%)",
          flush=True)

    # 阈值敏感性分析
    print("\n阈值敏感性：", flush=True)
    best_th, best_d = 0.5, -1
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        dd = [dice_np(oof_pred[i] > th, Y[i] > 0.5) for i in range(n)]
        md = float(np.mean(dd))
        print(f"   th={th:.1f}  Dice={md:.4f}  前景={ (oof_pred>th).mean()*100:.2f}%", flush=True)
        if md > best_d:
            best_d, best_th = md, th
    print(f"   -> OOF 最优阈值 {best_th} (Dice {best_d:.4f})", flush=True)

    # ---- 全量训练 + 测试预测 ----
    print("\n全量训练（20 张）...", flush=True)
    Xte, te_names, raw = load_test()
    tfold = [train_one(X, Y, epochs=250, seed=100 + s) for s in range(3)]
    for si, mm in enumerate(tfold):
        torch.save(mm.state_dict(), os.path.join(OUT, f"unet_full_s{100+si}.pt"))
    ps = np.mean([predict(m, Xte) for m in tfold], axis=0)
    print(f"   3 个种子集成完成 ({time.time()-t0:.0f}s)", flush=True)

    # 按官方流程放大到 512 后编码
    os.makedirs(os.path.join(OUT, "pred_mask"), exist_ok=True)
    rows = []
    for i, nm in enumerate(te_names):
        m512 = np.asarray(
            Image.fromarray(((ps[i] > best_th) * 255).astype(np.uint8)).resize(
                (512, 512), Image.NEAREST))
        rows.append((os.path.splitext(nm)[0], rle_from_mask(m512)))
    dst = r"D:\deepseek\task4\submission_task4_unet.csv"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for i, s in rows:
            w.writerow([i, s])
    print(f"\n>>> 提交文件: {dst}", flush=True)

    # 与已有提交的占比对比
    def ratio(s):
        v = np.array([int(x) for x in s.split()]) if s else np.zeros(0, int)
        return (v[1::2].sum() if len(v) else 0) / 262144 * 100
    print(f"    我的提交平均前景占比 = {np.mean([ratio(s) for _, s in rows]):.2f}%", flush=True)
    print(f"    真值训练集平均 = {Y.mean()*100:.2f}%", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
