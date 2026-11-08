# -*- coding: utf-8 -*-
"""task3 fovea 定位 —— 两阶段：粗定位 + 原分辨率裁剪精修。

为什么这么做：整图缩到 128x128 后，3000px 图上的黄斑只剩 5~13 像素，
定位精度天然受限（单阶段验证 37px）。两阶段先粗定位，再在原分辨率上
裁 320x320 窗口、缩放 2.5 倍送入精修模型，精度可大幅提升。

诚实评估协议：
  - 粗模型在训练折上训练，在验证折上给出粗预测（37px 量级）
  - 精修模型用「以真值为中心 + 与粗误差同量级的抖动」裁出的窗口训练
  - 验证时用**粗模型的预测**为裁剪中心，喂给精修模型，再换算回原图坐标
  这样验证误差反映的是真实使用场景，不是真值泄漏。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_unet import (D, OUT, S, HG, BS, THREADS, UNet, augment, decode, err_px,
                     gauss_target, load_all, predict_hm, prior_t, train_one)

CROP = 224           # 原分辨率裁剪边长
CS = 128             # 精修模型输入边长
CW = 0.55            # 精修先验宽度（归一化到裁剪窗口）


def crop_patch(im_gray_arr, cx, cy, crop=CROP):
    """在原图上以 (cx,cy) 为中心裁 crop×crop（越界用边缘填充），返回缩放后的 128x128。"""
    H, W = im_gray_arr.shape
    x0 = int(round(cx - crop / 2)); y0 = int(round(cy - crop / 2))
    x1 = x0 + crop; y1 = y0 + crop
    px0 = max(0, -x0); py0 = max(0, -y0)
    px1 = max(0, x1 - W); py1 = max(0, y1 - H)
    sub = im_gray_arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if px0 or py0 or px1 or py1:
        sub = np.pad(sub, ((py0, py1), (px0, px1)), mode="edge")
    im = Image.fromarray((sub * 255).astype(np.uint8)).resize((CS, CS), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_full_gray(i, split="train"):
    p = os.path.join(D, split, f"{i:04d}.jpg")
    im = Image.open(p).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0, im.size


def make_refine_data(ids, gt_xy, jitter=40.0, reps=8, seed=0):
    """以真值为中心 + 抖动，构造精修训练集。返回 (Xpatch, Yrel, sizes)。"""
    rng = np.random.default_rng(seed)
    X, Y, Sz = [], [], []
    for i in ids:
        arr, (W, H) = load_full_gray(i)
        gx, gy = gt_xy[i]
        for _ in range(reps):
            cx = gx + rng.normal(0, jitter)
            cy = gy + rng.normal(0, jitter)
            cx = float(np.clip(cx, 0, W - 1)); cy = float(np.clip(cy, 0, H - 1))
            patch = crop_patch(arr, cx, cy)
            # 目标：真值相对裁剪窗口中心的偏移，归一化到 [0,1]
            nx = 0.5 + (gx - cx) / CROP
            ny = 0.5 + (gy - cy) / CROP
            if not (0.02 <= nx <= 0.98 and 0.02 <= ny <= 0.98):
                continue
            X.append(patch); Y.append((nx, ny)); Sz.append((CROP, CROP))
    return (np.stack(X)[:, None], np.array(Y, np.float32), np.array(Sz, np.float32))


def refine_prior(size=HG, w=CW):
    a = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    YY = a.view(1, size, 1); XX = a.view(1, 1, size)
    s = (0.5 - w / 2)
    p = torch.exp(-0.5 * (((XX - 0.5) / (w / 2)) ** 2 + ((YY - 0.5) / (w / 2)) ** 2))
    return p


def main():
    t0 = time.time()
    X, Y, Sz, Xte, Ste, te_names = load_all()
    n = len(X)
    gt_xy = {i + 1: (Y[i, 0] * Sz[i, 0], Y[i, 1] * Sz[i, 1]) for i in range(n)}
    rng = np.random.default_rng(123)
    folds = np.array_split(rng.permutation(n), 5)

    print("=" * 70, flush=True)
    print("阶段1：粗定位（整图 128px，单折验证）", flush=True)
    va = folds[0]; tr = np.setdiff1d(np.arange(n), va)
    model_c, m_c = train_one(X[tr], Y[tr], X[va], Y[va], Sz[va], epochs=350, seed=0,
                             tag="粗-折1")
    hv = torch.from_numpy(predict_hm(model_c, X[va]))
    pc = decode(hv, prior_t()).numpy()
    hor = float(err_px(pc, Sz[va], Y[va]).mean())
    print(f"  粗定位验证 = {hor:.2f} px   ({time.time()-t0:.0f}s)", flush=True)

    print("=" * 70, flush=True)
    print("阶段2：精修（原分辨率 320 裁剪 -> 128）", flush=True)
    ids_tr = [int(i) + 1 for i in tr]
    Xr, Yr, Sr = make_refine_data(ids_tr, gt_xy, jitter=45.0, reps=10, seed=1)
    print(f"  精修训练样本 {Xr.shape}", flush=True)
    va_idx = [int(i) for i in va]
    Xrv = np.stack([crop_patch(load_full_gray(i + 1)[0], pc[k, 0] * Sz[va][k, 0],
                              pc[k, 1] * Sz[va][k, 1]) for k, i in enumerate(va_idx)])[:, None]
    Yrv = np.array([(0.5 + (gt_xy[i + 1][0] - pc[k, 0] * Sz[va][k, 0]) / CROP,
                     0.5 + (gt_xy[i + 1][1] - pc[k, 1] * Sz[va][k, 1]) / CROP)
                    for k, i in enumerate(va_idx)], dtype=np.float32)
    Srv = np.full((len(va_idx), 2), CROP, dtype=np.float32)
    model_r, m_r = train_one(Xr, Yr, Xrv, Yrv, Srv, epochs=350, seed=2, tag="精-折1")

    # 用精修结果换算回原图坐标，统计真实误差
    hr = torch.from_numpy(predict_hm(model_r, Xrv))
    pr = decode(hr, refine_prior()).numpy()
    gx = np.array([gt_xy[i + 1][0] for i in va_idx])
    gy = np.array([gt_xy[i + 1][1] for i in va_idx])
    px = pc[:, 0] * Sz[va][:, 0] + (pr[:, 0] - 0.5) * CROP
    py = pc[:, 1] * Sz[va][:, 1] + (pr[:, 1] - 0.5) * CROP
    d = np.hypot(px - gx, py - gy)
    print("\n" + "=" * 70, flush=True)
    print(f"两阶段端到端验证：平均 {d.mean():.2f} px  中位 {np.median(d):.2f}  "
          f"<10px 占比 {(d<10).mean()*100:.1f}%", flush=True)
    print(f"（粗定位单阶段为 {hor:.2f} px，精修后提升 {hor-d.mean():.2f} px）", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, "stage_eval.npy"),
            np.array([hor, float(d.mean()), float(np.median(d))]))


if __name__ == "__main__":
    main()
