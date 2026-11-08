# -*- coding: utf-8 -*-
"""独立验证提交文件：用保存的折模型重新推理，逐张核对 ID↔预测的配对。

为什么需要这个脚本：提交模板的行顺序与 sorted(os.listdir()) 不同，
一旦把「按文件名排序的第 k 张图」的预测写到「模板第 k 行」，就会出现
ID 与预测的静默错配（成绩掉到随机水平）。本脚本不信任任何中间文件，
直接从 test/<文件名>.png 重新前向推理，再与 CSV 中同名 ID 的预测比对。

用法: python verify_submission.py [--n 40] [--seed 7]
"""
import argparse
import os
import sys

import numpy as np
import torch

import common as C
from finetune import build_model, predict_probs

TAGS = [("ft_r18", "resnet18"), ("ft_r50", "resnet50"), ("ft_effb0", "efficientnet_b0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽检张数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

    # 读提交文件
    import csv
    with open(os.path.join(r"D:\deepseek\2026task1", "submission-for-task2.csv"),
              encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    csv_pred = {r[0]: r[1] for r in rows}

    te_paths = C.list_test()                       # 按文件名排序
    name2path = {os.path.basename(p): p for p in te_paths}
    assert set(name2path) == set(csv_pred), "提交 ID 与测试目录文件名集合不一致"

    rng = np.random.default_rng(a.seed)
    names = sorted(rng.choice(sorted(name2path), size=min(a.n, len(name2path)), replace=False))
    print(f"独立抽检 {len(names)} 张测试图，覆盖 {len(TAGS)} 个模型族 × {a.folds} 折 × 8 视图 TTA\n")

    tag_probs = {}
    for tag, backbone in TAGS:
        acc = np.zeros((len(names), C.NUM_CLASSES), dtype=np.float64)
        used = 0
        for k in range(a.folds):
            p = os.path.join(C.OUT, f"model_{tag}_fold{k}.pt")
            if not os.path.exists(p):
                print(f"  [缺] {p}")
                continue
            model = build_model(backbone)
            model.load_state_dict(torch.load(p, map_location="cpu"))
            model.eval()
            acc += predict_probs(model, [name2path[n] for n in names], size=224, use_tta=True)
            used += 1
            print(f"  [{tag}] 折{k} 推理完成", flush=True)
        if used:
            tag_probs[tag] = acc / used
    if not tag_probs:
        print("没有可用模型"); sys.exit(1)

    # 与提交所用完全一致的口径：各 tag 等权平均（mean_top3）
    P = sum(tag_probs.values()) / len(tag_probs)
    ind_pred = [C.CLASSES[i] for i in P.argmax(1)]

    agree = sum(1 for n, p in zip(names, ind_pred) if csv_pred[n] == p)
    print(f"\n{'文件名':<18}{'CSV 中的预测':<20}{'独立推理结果':<20}{'一致'}")
    for i, n in enumerate(names):
        ok = "OK" if csv_pred[n] == ind_pred[i] else "*** 不一致 ***"
        print(f"{n:<18}{csv_pred[n]:<20}{ind_pred[i]:<20}{ok}")
    print(f"\n一致率: {agree}/{len(names)} = {agree/len(names):.4f}")
    if agree == len(names):
        print("结论：提交文件的 ID↔预测配对**完全正确**，逐张核对通过。")
    else:
        print("结论：存在错配，提交文件不可用，必须排查！！")
        sys.exit(2)


if __name__ == "__main__":
    main()
