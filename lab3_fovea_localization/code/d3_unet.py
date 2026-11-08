# -*- coding: utf-8 -*-
"""task3 fovea 定位 —— U-Net 热图回归 + 5 折×多种子集成 + TTA + 解剖先验。

修正了上一版两个致命问题：
  1) 仿射增强的坐标映射方向。affine_grid 给的是 输入 = sc*输出 + t，
     故特征在输出中的位置是 输出 = (输入 - t)/sc。旧版写成 输入*sc+t，等于给标签加噪。
  2) 热图分辨率 8x8 过粗 -> 64x64，并用 max 邻域加权质心做亚像素解码。
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

D = r"D:\deepseek\task3\detection"
OUT = r"D:\deepseek\task3\result"
os.makedirs(OUT, exist_ok=True)
S = 128
HG = 64
SIGMA = 1.6
THREADS = 20
MX, SX = 0.4935, 0.11
MY, SY = 0.5290, 0.060
BS = 16


# ------------------------------------------------------------------ 数据
def load_all():
    gt = {}
    with open(os.path.join(D, "fovea_localization_train_GT.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gt[int(r["data"])] = (float(r["Fovea_X"]), float(r["Fovea_Y"]))
    X, Y, Sz = [], [], []
    for i in range(1, 81):
        im = Image.open(os.path.join(D, "train", f"{i:04d}.jpg")).convert("L")
        W, H = im.size
        X.append(np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0)
        gx, gy = gt[i]
        Y.append((gx / W, gy / H)); Sz.append((W, H))
    names = sorted(os.listdir(os.path.join(D, "test")))
    Xe, Se = [], []
    for f in names:
        im = Image.open(os.path.join(D, "test", f)).convert("L")
        W, H = im.size
        Xe.append(np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0)
        Se.append((W, H))
    return (np.stack(X)[:, None], np.array(Y, np.float32), np.array(Sz, np.float32),
            np.stack(Xe)[:, None], np.array(Se, np.float32), names)


def prior_t(size=HG):
    ys = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    YY = ys.view(1, size, 1); XX = ys.view(1, 1, size)
    return torch.exp(-0.5 * ((XX - MX) / SX) ** 2 - 0.5 * ((YY - MY) / SY) ** 2)


# ------------------------------------------------------------------ 模型
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(cbr(1, 16), cbr(16, 16))      # 128
        self.e2 = nn.Sequential(cbr(16, 32), cbr(32, 32))     # 64
        self.e3 = nn.Sequential(cbr(32, 64), cbr(64, 64))     # 32
        self.e4 = nn.Sequential(cbr(64, 128), cbr(128, 128))  # 16
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")   # 16 -> 32
        self.c3 = cbr(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")   # 32 -> 64
        self.c2 = cbr(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)                        # 64x64 heatmap

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        s4 = self.e4(F.max_pool2d(s3, 2))
        d3 = self.c3(torch.cat([self.up3(s4), s3], 1))
        d2 = self.c2(torch.cat([self.up2(d3), s2], 1))
        return self.out(d2)


def gauss_target(y, size=HG, sigma=SIGMA):
    B = y.shape[0]
    a = (torch.arange(size, dtype=torch.float32, device=y.device) + 0.5) / size
    YY = a.view(1, size, 1); XX = a.view(1, 1, size)
    cx = y[:, 0].view(B, 1, 1); cy = y[:, 1].view(B, 1, 1)
    d2 = (XX - cx) ** 2 + (YY - cy) ** 2
    return torch.exp(-d2 / (2 * (sigma / size) ** 2)).unsqueeze(1)


def decode(hm, prior=None, win=3):
    """hm: Tensor (B,1,H,W) 概率热图 -> (B,2) 归一化坐标（亚像素质心）。"""
    if prior is not None:
        hm = hm * prior.reshape(1, 1, prior.shape[-2], prior.shape[-1])
    B, _, H, W = hm.shape
    h = hm[:, 0]
    flat = h.reshape(B, -1)
    idx = flat.argmax(1)
    iy = idx // W; ix = idx % W
    out = torch.zeros(B, 2)
    hh = h.clamp(min=1e-8)
    for b in range(B):
        y0 = int(max(0, iy[b] - win)); y1 = int(min(H, iy[b] + win + 1))
        x0 = int(max(0, ix[b] - win)); x1 = int(min(W, ix[b] + win + 1))
        w = hh[b, y0:y1, x0:x1]
        s = float(w.sum())
        yy = torch.arange(y0, y1, dtype=torch.float32).view(-1, 1) + 0.5
        xx = torch.arange(x0, x1, dtype=torch.float32).view(1, -1) + 0.5
        out[b, 0] = float((w * xx).sum()) / s / W
        out[b, 1] = float((w * yy).sum()) / s / H
    return out


def augment(xb, yb, rng):
    B = xb.size(0); dev = xb.device
    hf = torch.rand(B, device=dev) < 0.5
    if bool(hf.any()):
        xb[hf] = xb[hf].flip(-1); yb[hf, 0] = 1.0 - yb[hf, 0]
    vf = torch.rand(B, device=dev) < 0.5
    if bool(vf.any()):
        xb[vf] = xb[vf].flip(-2); yb[vf, 1] = 1.0 - yb[vf, 1]
    sc = 0.88 + torch.rand(B, device=dev) * 0.24
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.05
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.035
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = sc; theta[:, 1, 1] = sc
    theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(xb.shape), align_corners=False)
    xb = F.grid_sample(xb, grid, align_corners=False, padding_mode="border")
    yb = torch.stack([(yb[:, 0] - tx) / sc, (yb[:, 1] - ty) / sc], 1)
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.12
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.12
    mu = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = ((xb - mu) * c + mu) * b
    return xb, yb


@torch.no_grad()
def predict_hm(model, X, bs=16):
    """返回 (B,1,HG,HG) 概率热图，含 3 视图 TTA。"""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32)
        a = torch.sigmoid(model(xb))
        b = torch.sigmoid(model(xb.flip(-1))).flip(-1)
        c = torch.sigmoid(model(xb.flip(-2))).flip(-2)
        out.append(((a + b + c) / 3).numpy())
    return np.concatenate(out, 0)


def err_px(pn, sz, Yg):
    px = pn[:, 0] * sz[:, 0]; py = pn[:, 1] * sz[:, 1]
    gx = Yg[:, 0] * sz[:, 0]; gy = Yg[:, 1] * sz[:, 1]
    return np.hypot(px - gx, py - gy)


def train_one(X, Y, Xv, Yv, Sv, epochs=600, seed=0, lr=3e-3, tag=""):
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(X) // BS) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
    prior = prior_t()
    best = (1e9, None, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt) - BS + 1, BS):
            idx = perm[i:i + BS]
            xb = Xt[idx].clone(); yb = Yt[idx].clone()
            xb, yb = augment(xb, yb, rng)
            hm = model(xb)
            loss = F.mse_loss(torch.sigmoid(hm), gauss_target(yb))
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 30 == 0 or ep == epochs - 1:
            hv = torch.from_numpy(predict_hm(model, Xv))
            pv = decode(hv, prior).numpy()
            m = float(err_px(pv, Sv, Yv).mean())
            if m < best[0]:
                best = (m, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    print(f"      [{tag}] best val {best[0]:.2f}px @ep{best[2]}", flush=True)
    return model, best[0]


if __name__ == "__main__":
    t0 = time.time()
    X, Y, Sz, Xte, Ste, te_names = load_all()
    print(f"数据 train{X.shape} test{Xte.shape}  ({time.time()-t0:.0f}s)", flush=True)
    n = len(X)
    rng = np.random.default_rng(123)
    folds = np.array_split(rng.permutation(n), 5)
    va = folds[0]; tr = np.setdiff1d(np.arange(n), va)
    _, m = train_one(X[tr], Y[tr], X[va], Y[va], Sz[va], epochs=400, seed=0, tag="折1-s0")
    print(f"第 1 折 400 轮验证 = {m:.2f} px，总耗时 {time.time()-t0:.0f}s", flush=True)
