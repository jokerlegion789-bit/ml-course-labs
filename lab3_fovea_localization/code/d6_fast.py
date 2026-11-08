# -*- coding: utf-8 -*-
"""task3 快速版：折1 评估 + 全量训练 + 生成提交（约 12 分钟）。

相比 d5 的改动：
  - 裁剪窗 320 -> 224（粗误差 33px，±112 覆盖 3σ，且等效分辨率更高）
  - 精修样本重复数 10 -> 6，轮次 350 -> 130（精修任务简单，收敛快）
  - 粗模型轮次 350 -> 260
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_unet import OUT, decode, err_px, load_all, predict_hm, prior_t, train_one
from d4_twostage import CROP, crop_patch, load_full_gray, make_refine_data, refine_prior

COARSE_EP = 260
REFINE_EP = 130
JITTER = 38.0
REPS = 6


def coarse_predict(model, X):
    return decode(torch.from_numpy(predict_hm(model, X)), prior_t()).numpy()


def refine_predict(model, patches):
    return decode(torch.from_numpy(predict_hm(model, patches)), refine_prior()).numpy()


def build_patches(base_xy, sizes, ids, split):
    return np.stack([crop_patch(load_full_gray(i, split)[0], base_xy[k, 0], base_xy[k, 1])
                     for k, i in enumerate(ids)])[:, None]


def main():
    t0 = time.time()
    X, Y, Sz, Xte, Ste, te_names = load_all()
    n = len(X)
    gt_xy = {i + 1: (Y[i, 0] * Sz[i, 0], Y[i, 1] * Sz[i, 1]) for i in range(n)}
    rng = np.random.default_rng(123)
    folds = np.array_split(rng.permutation(n), 5)

    # ---------- A 折1 评估 ----------
    va = folds[0]; tr = np.setdiff1d(np.arange(n), va)
    print("[A] 折1 粗模型", flush=True)
    mc, _ = train_one(X[tr], Y[tr], X[va], Y[va], Sz[va], epochs=COARSE_EP, seed=0, tag="粗1")
    pc = coarse_predict(mc, X[va])
    hor = float(err_px(pc, Sz[va], Y[va]).mean())
    print(f"    粗 {hor:.2f}px ({time.time()-t0:.0f}s)", flush=True)

    print("[A] 折1 精修模型", flush=True)
    Xr, Yr, _ = make_refine_data([int(i) + 1 for i in tr], gt_xy, jitter=JITTER,
                                 reps=REPS, seed=1)
    va_ids = [int(i) + 1 for i in va]
    base_xy = np.stack([pc[:, 0] * Sz[va][:, 0], pc[:, 1] * Sz[va][:, 1]], 1)
    Xrv = build_patches(base_xy, None, va_ids, "train")
    gx = np.array([gt_xy[i][0] for i in va_ids]); gy = np.array([gt_xy[i][1] for i in va_ids])
    Yrv = np.stack([0.5 + (gx - base_xy[:, 0]) / CROP,
                    0.5 + (gy - base_xy[:, 1]) / CROP], 1).astype(np.float32)
    Srv = np.full((len(va_ids), 2), CROP, dtype=np.float32)
    mr, _ = train_one(Xr, Yr, Xrv, Yrv, Srv, epochs=REFINE_EP, seed=2, tag="精1")
    pr = refine_predict(mr, Xrv)
    px = base_xy[:, 0] + (pr[:, 0] - 0.5) * CROP
    py = base_xy[:, 1] + (pr[:, 1] - 0.5) * CROP
    d = np.hypot(px - gx, py - gy)
    print(f"\n>>> 折1 两阶段：平均 {d.mean():.2f}px 中位 {np.median(d):.2f} "
          f"<10px {(d<10).mean()*100:.0f}%  ({time.time()-t0:.0f}s)\n", flush=True)

    # ---------- B 全量训练 ----------
    print("[B] 全量训练", flush=True)
    mc2, _ = train_one(X, Y, X[va], Y[va], Sz[va], epochs=COARSE_EP, seed=10, tag="粗全")
    Xr2, Yr2, _ = make_refine_data(list(range(1, n + 1)), gt_xy, jitter=JITTER,
                                   reps=REPS, seed=11)
    mr2, _ = train_one(Xr2, Yr2, Xrv, Yrv, Srv, epochs=REFINE_EP, seed=12, tag="精全")
    print(f"    完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---------- C 预测并写提交 ----------
    print("[C] 测试集预测", flush=True)
    pct = coarse_predict(mc2, Xte)
    base_t = np.stack([pct[:, 0] * Ste[:, 0], pct[:, 1] * Ste[:, 1]], 1)
    test_ids = [int(n.split(".")[0]) for n in te_names]     # 0081..0100
    Xtp = build_patches(base_t, None, test_ids, "test")
    prt = refine_predict(mr2, Xtp)
    px = base_t[:, 0] + (prt[:, 0] - 0.5) * CROP
    py = base_t[:, 1] + (prt[:, 1] - 0.5) * CROP
    px = np.clip(px, 0, Ste[:, 0] - 1); py = np.clip(py, 0, Ste[:, 1] - 1)

    sample = os.path.join(r"D:\deepseek\task3\detection", "sample_submission.csv")
    with open(sample, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header, ids = rows[0], [r[0] for r in rows[1:]]
    pred = {}
    for k in range(len(te_names)):
        num = int(te_names[k].split(".")[0])
        pred[f"{num}_Fovea_X"] = float(px[k])
        pred[f"{num}_Fovea_Y"] = float(py[k])
    out = r"D:\deepseek\task3\submission_task3_twostage.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for i in ids:
            w.writerow([i, f"{pred[i]:.6f}"])
    print(f"\n>>> 提交: {out}", flush=True)
    print(f">>> 折1 诚实估计 {d.mean():.2f}px  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
