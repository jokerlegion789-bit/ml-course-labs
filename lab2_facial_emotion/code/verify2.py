# -*- coding: utf-8 -*-
"""独立核验 task2 提交文件：用保存的权重重新推理，逐张核对 ID↔预测。

关键点：核验必须使用**与提交完全相同的集成成员、权重与 TTA 配置**，
否则弱模型的分歧会被误报成配对错误。成员与权重取自 ensemble2_report.json。
由于权重本身是从这些模型算出来的，本核验独立验证的是「图片 -> 预测 -> CSV 行」
这一条链路（即按文件名配对是否正确），这正是 task1 曾经出错的地方。

用法: python verify2.py [--n 12] [--encoding fer_order]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

import common2 as C
from retest_tta import load_model, predict10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoding", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    enc = C.ENCODINGS[a.encoding or C.ENCODING]
    inv = {v: k for k, v in enc.items()}

    sub = os.path.join(C.ROOT, "submission.csv")
    with open(sub, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    csv_pred = {r[0]: r[1] for r in rows}

    rep = C.load_json("ensemble2_report.json", {}) or {}
    members = rep.get("ensemble") or rep.get("final", {}).get("members")
    weights = rep.get("weights") or rep.get("final", {}).get("weights")
    if not members:
        print("!! 找不到集成成员信息（ensemble2_report.json）"); sys.exit(1)
    print(f"提交所用集成: {[(m, round(float(w), 3)) for m, w in zip(members, weights)]}\n")

    X, y, Xt, te_names = C.build_cache()
    name2i = {nm: i for i, nm in enumerate(te_names)}
    if set(csv_pred) != set(name2i):
        print("!! 提交 ID 集合与测试目录文件名集合不一致"); sys.exit(1)

    rng = np.random.default_rng(a.seed)
    names = sorted(rng.choice(sorted(name2i), size=min(a.n, len(name2i)), replace=False))
    idx = [name2i[n] for n in names]
    Xs = Xt[idx]

    P = None
    for m, w in zip(members, weights):
        if w <= 0:
            continue
        model = load_model(m)
        p = predict10(model, Xs)
        P = w * p if P is None else P + w * p
        print(f"  [{m}] 重新推理完成 (权重 {w:.3f})", flush=True)

    ind = [C.CLASSES[i] for i in P.argmax(1)]
    ok = 0
    print(f"\n{'文件名':<18}{'CSV':<6}{'CSV→类别':<18}{'独立推理':<18}{'一致'}")
    for i, n in enumerate(names):
        csv_cls = inv.get(int(csv_pred[n]), "?")
        good = csv_cls == ind[i]
        ok += good
        print(f"{n:<18}{csv_pred[n]:<6}{csv_cls:<18}{ind[i]:<18}{'OK' if good else '*** 不一致 ***'}")
    print(f"\n一致率 {ok}/{len(names)} = {ok/len(names):.4f}")
    if ok == len(names):
        print("结论：提交文件的 ID↔预测配对完全正确（与提交同权重的独立重算逐张吻合）。")
    else:
        print("结论：存在错配或推理口径不一致，需排查！")
        sys.exit(2)


if __name__ == "__main__":
    main()
