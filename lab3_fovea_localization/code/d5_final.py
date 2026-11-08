# -*- coding: utf-8 -*-
"""task3 完整流程：折1 诚实评估 -> 全量训练 -> 生成提交。

流程：
  A. 折 1：粗模型（20 张验证）+ 精修模型 -> 端到端误差（诚实估计）
  B. 用全部 80 张训练粗模型 + 精修模型
  C. 对 20 张测试图做两阶段预测 + 翻转 TTA -> 写 submission CSV
"""
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_unet import (D, OUT, HG, THREADS, decode, err_px, load_all, predict_hm,
                     prior_t, train_one)
from d4_twostage import (CROP, crop_patch, load_full_gray, make_refine_data,
                         refine_prior)


def coarse_predict(model, X):
    hv = torch.from_numpy(predict_hm(model, X))
    return decode(hv, prior_t()).numpy()


def refine_predict(model, patches):
    hr = torch.from_numpy(predict_hm(model, patches))
    return decode(hr, refine_prior()).numpy()


def main():
    t0 = time.time()
    X, Y, Sz, Xte, Ste, te_names = load_all()
    n = len(X)
    gt_xy = {i + 1: (Y[i, 0] * Sz[i, 0], Y[i, 1] * Sz[i, 1]) for i in range(n)}
    rng = np.random.default_rng(123)
    folds = np.array_split(rng.permutation(n), 5)

    # ---------- A. 折 1 诚实评估 ----------
    print("[A] 折1：训练粗模型", flush=True)
    va = folds[0]; tr = np.setdiff1d(np.arange(n), va)
    mc, hor = train_one(X[tr], Y[tr], X[va], Y[va], Sz[va], epochs=350, seed=0, tag="粗-折1")
    pc = coarse_predict(mc, X[va])
    hor = float(err_px(pc, Sz[va], Y[va]).mean())
    print(f"    粗定位验证 {hor:.2f} px  ({time.time()-t0:.0f}s)", flush=True)

    print("[A] 折1：训练精修模型", flush=True)
    ids_tr = [int(i) + 1 for i in tr]
    Xr, Yr, _ = make_refine_data(ids_tr, gt_xy, jitter=45.0, reps=10, seed=1)
    va_idx = [int(i) for i in va]
    Xrv = np.stack([crop_patch(load_full_gray(i + 1)[0], pc[k, 0] * Sz[va][k, 0],
                              pc[k, 1] * Sz[va][k, 1]) for k, i in enumerate(va_idx)])[:, None]
    gx = np.array([gt_xy[i + 1][0] for i in va_idx])
    gy = np.array([gt_xy[i + 1][1] for i in va_idx])
    Yrv = np.array([(0.5 + (gx[k] - pc[k, 0] * Sz[va][k, 0]) / CROP,
                     0.5 + (gy[k] - pc[k, 1] * Sz[va][k, 1]) / CROP)
                    for k in range(len(va_idx))], dtype=np.float32)
    Srv = np.full((len(va_idx), 2), CROP, dtype=np.float32)
    mr, _ = train_one(Xr, Yr, Xrv, Yrv, Srv, epochs=350, seed=2, tag="精-折1")
    pr = refine_predict(mr, Xrv)
    px = pc[:, 0] * Sz[va][:, 0] + (pr[:, 0] - 0.5) * CROP
    py = pc[:, 1] * Sz[va][:, 1] + (pr[:, 1] - 0.5) * CROP
    d = np.hypot(px - gx, py - gy)
    print(f"\n>>> 折1 两阶段端到端：平均 {d.mean():.2f} px  中位 {np.median(d):.2f}  "
          f"<10px {(d<10).mean()*100:.0f}%  ({time.time()-t0:.0f}s)\n", flush=True)

    # ---------- B. 全量训练 ----------
    print("[B] 全量训练（80 张）", flush=True)
    mc2, _ = train_one(X, Y, X[va], Y[va], Sz[va], epochs=350, seed=10, tag="粗-全量")
    ids_all = list(range(1, n + 1))
    Xr2, Yr2, _ = make_refine_data(ids_all, gt_xy, jitter=45.0, reps=10, seed=11)
    mr2, _ = train_one(Xr2, Yr2, Xrv, Yrv, Srv, epochs=350, seed=12, tag="精-全量")
    print(f"    全量训练完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---------- C. 测试集预测 ----------
    print("[C] 测试集两阶段预测", flush=True)
    pct = coarse_predict(mc2, Xte)
    patches = np.stack([crop_patch(load_full_gray(k + 1, "test")[0],
                                   pct[k, 0] * Ste[k, 0], pct[k, 1] * Ste[k, 1])
                        for k in range(len(te_names))])[:, None]
    prt = refine_predict(mr2, patches)
    px = pct[:, 0] * Ste[:, 0] + (prt[:, 0] - 0.5) * CROP
    py = pct[:, 1] * Ste[:, 1] + (prt[:, 1] - 0.5) * CROP
    px = np.clip(px, 0, Ste[:, 0] - 1); py = np.clip(py, 0, Ste[:, 1] - 1)

    ids, _ = None, None
    with open(os.path.join(D, "sample_submission.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ids = [r[0] for r in rows[1:]]
    assert len(ids) == 2 * len(te_names), (len(ids), len(te_names))
    pred = {}
    for k, nm in enumerate(te_names):
        num = nm.split(".")[0]
        pred[f"{int(num)}_Fovea_X"] = float(px[k])
        pred[f"{int(num)}_Fovea_Y"] = float(py[k])
    out = os.path.join(r"D:\deepseek\task3", "submission_task3_twostage.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in ids:
            w.writerow([i, f"{pred[i]:.6f}"])
    print(f"\n>>> 提交文件: {out}", flush=True)
    print(f">>> 折1 诚实估计 = {d.mean():.2f} px（目标 <10）", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
