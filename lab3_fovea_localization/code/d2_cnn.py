# -*- coding: utf-8 -*-
"""task3 fovea 定位：小样本 CNN 热图回归（soft-argmax）。

思路：把图像缩到 128×128 灰度，CNN 输出 8×8 热图，对热图做 softmax 后取
期望坐标作为 (nx, ny)。相比直接回归，热图 + 期望对手部位置的空间先验建模更好，
且对 80 张小样本更稳。

增强：水平翻转、垂直翻转、随机平移与缩放（坐标同步变换）。
评估：固定留出 20 张验证，报告原始像素欧氏距离；最终用全部 80 张训练。
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
CACHE = r"D:\deepseek\task3\cache"
OUT = r"D:\deepseek\task3\result"
os.makedirs(OUT, exist_ok=True)
S = 128
G = 8
SEED = 42


def load_data():
    gt = {}
    with open(os.path.join(D, "fovea_localization_train_GT.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gt[int(r["data"])] = (float(r["Fovea_X"]), float(r["Fovea_Y"]))
    X, Y, sizes = [], [], []
    for i in range(1, 81):
        im = Image.open(os.path.join(D, "train", f"{i:04d}.jpg")).convert("L")
        W, H = im.size
        a = np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0
        X.append(a)
        gx, gy = gt[i]
        Y.append((gx / W, gy / H))
        sizes.append((W, H))
    X = np.stack(X)[:, None]           # (80,1,S,S)
    Y = np.array(Y, dtype=np.float32)
    return X, Y, np.array(sizes, np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(1, 24), nn.MaxPool2d(2),        # 64
            blk(24, 48), blk(48, 48), nn.MaxPool2d(2),   # 32
            blk(48, 96), blk(96, 96), nn.MaxPool2d(2),   # 16
            blk(96, 128), nn.MaxPool2d(2),      # 8
            nn.Conv2d(128, 1, 1),               # 8x8 heatmap
        )
        yy, xx = np.mgrid[0:G, 0:G]
        gx = (xx.reshape(-1) + 0.5) / G
        gy = (yy.reshape(-1) + 0.5) / G
        self.register_buffer("gx", torch.tensor(gx, dtype=torch.float32))
        self.register_buffer("gy", torch.tensor(gy, dtype=torch.float32))

    def forward(self, x):
        h = self.f(x).flatten(1)                 # (B, G*G)
        p = F.softmax(h * 10.0, dim=1)
        nx = (p * self.gx).sum(1)                # (B,)
        ny = (p * self.gy).sum(1)
        return torch.stack([nx, ny], 1), h, p


def augment(xb, yb, rng):
    """xb: (B,1,S,S) tensor, yb: (B,2) 归一化坐标。几何增强 + 坐标同步。"""
    B = xb.size(0)
    dev = xb.device
    # 水平/垂直翻转
    hf = torch.rand(B, device=dev) < 0.5
    vf = torch.rand(B, device=dev) < 0.5
    idx_h = hf.nonzero().flatten()
    if len(idx_h):
        xb[idx_h] = xb[idx_h].flip(-1)
        yb[idx_h, 0] = 1.0 - yb[idx_h, 0]
    idx_v = vf.nonzero().flatten()
    if len(idx_v):
        xb[idx_v] = xb[idx_v].flip(-2)
        yb[idx_v, 1] = 1.0 - yb[idx_v, 1]
    # 随机仿射：缩放 + 平移
    sc = 0.85 + torch.rand(B, device=dev) * 0.30       # 0.85~1.15
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.06
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.04
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = sc
    theta[:, 1, 1] = sc
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(xb.shape), align_corners=False)
    xb = F.grid_sample(xb, grid, align_corners=False, padding_mode="border")
    yb = yb * sc.unsqueeze(1) + torch.stack([tx, ty], 1)   # 坐标同步变换
    # 亮度/对比度
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    mu = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = ((xb - mu) * c + mu) * b
    return xb, yb


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32) if not torch.is_tensor(X) else X
    out = []
    for i in range(0, len(Xt), bs):
        p, _, _ = model(Xt[i:i + bs])
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


def dist_px(pred_n, ysizes, Y):
    px = pred_n[:, 0] * ysizes[:, 0]
    py = pred_n[:, 1] * ysizes[:, 1]
    gx = Y[:, 0] * ysizes[:, 0]
    gy = Y[:, 1] * ysizes[:, 1]
    return np.hypot(px - gx, py - gy)


def train(X, Y, Xv=None, Yv=None, Sv=None, Yv_n=None, epochs=500, seed=SEED, threads=12,
          verbose=True):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cpu")
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = max(1, len(X) // 16) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps,
                                                pct_start=0.2)
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt) - 15, 16):
            idx = perm[i:i + 16]
            xb = Xt[idx].clone()
            yb = Yt[idx].clone()
            xb, yb = augment(xb, yb, rng)
            p, _, _ = model(xb)
            loss = F.mse_loss(p, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if Xv is not None and (ep + 1) % 25 == 0:
            pv = predict(model, Xv)
            d = dist_px(pv, Sv, Yv_n)
            m = float(d.mean())
            if m < best[0]:
                best = (m, {k: v.clone() for k, v in model.state_dict().items()})
            if verbose and (ep + 1) % 100 == 0:
                print(f"      ep{ep+1:4d} loss={float(loss):.6f} val={m:7.2f}px", flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0]


def main():
    t0 = time.time()
    X, Y, sz = load_data()
    print(f"数据 {X.shape}, 耗时 {time.time()-t0:.0f}s")
    # 固定留出 20 张（按索引均匀抽样，保证覆盖不同尺寸）
    idx = np.arange(len(X))
    rng = np.random.default_rng(0)
    val = np.sort(rng.choice(idx, 20, replace=False))
    tr = np.setdiff1d(idx, val)
    print(f"训练 {len(tr)} / 验证 {len(val)}")
    model, valm = train(X[tr], Y[tr], Xv=X[val], Yv=Y[val], Sv=sz[val], Yv_n=Y[val],
                        epochs=500)
    print(f"留出验证平均距离 = {valm:.2f} px   (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
