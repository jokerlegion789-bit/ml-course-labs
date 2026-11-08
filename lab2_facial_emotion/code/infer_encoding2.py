# -*- coding: utf-8 -*-
"""精确比较几种「自然」编号方案在实测 0.417 下的期望分数。

关键洞察：我们提交的是字母序 0..5（Angry0 Fear1 Happy2 Neutral3 Sad4 Surprise5）。
若平台用的是「FER 原始顺序压缩到 0..5」（Angry0 Fear1 Happy2 Sad3 Surprise4 Neutral5），
则 Angry/Fear/Happy 三个类完全对上，而 Neutral/Sad/Surprise 三者相互错位 ——
这正好产生「一部分对、一部分错」的部分得分，与实测 0.417 的现象吻合。
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\deepseek\2026task2\code")
import common2 as C
from train2 import stratified_holdout

OBS = 0.417
FER_TEST = {"Angry": 980, "Fear": 1024, "Happy": 1774, "Sad": 1247,
            "Surprise": 831, "Neutral": 1211}
ALPHA = {c: i for i, c in enumerate(C.CLASSES)}

CANDIDATES = {
    "① 字母序（我提交的）Angry0 Fear1 Happy2 Neutral3 Sad4 Surprise5": ALPHA,
    "② FER顺序压缩 Angry0 Fear1 Happy2 Sad3 Surprise4 Neutral5":
        {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5},
    "③ 原始FER编码 Angry0 Fear2 Happy3 Sad4 Surprise5 Neutral6":
        {"Angry": 0, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6},
    "④ 字母序1..6": {c: i + 1 for i, c in enumerate(C.CLASSES)},
}


def main():
    y_all = np.load(os.path.join(C.CACHE, "train_label.npy"))
    _, hoi = stratified_holdout(y_all)
    y = y_all[hoi]
    root = r"D:\deepseek\2026task2\result"
    Ps = [np.load(os.path.join(root, f)) for f in sorted(os.listdir(root))
          if f.startswith("hold_") and f.endswith(".npy")]
    P = np.mean(Ps, axis=0)
    pred = P.argmax(1)
    cm = np.zeros((6, 6), float)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    cond = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    idx = {c: i for i, c in enumerate(C.CLASSES)}

    # 两种先验口径做稳健性检查
    tot = sum(FER_TEST.values())
    prior_fer = np.array([FER_TEST[c] / tot for c in C.CLASSES])
    prior_emp = np.bincount(pred, minlength=6) / len(pred)

    print(f"留出集准确率 = {(pred == y).mean():.4f}   实测线上 = {OBS}\n")
    print(f"{'编号方案':<58}{'FER先验':>10}{'实测分布先验':>14}")
    for name, M in CANDIDATES.items():
        row = []
        for pi in (prior_fer, prior_emp):
            a_inv = {v: c for c, v in ALPHA.items()}
            s = 0.0
            for c in C.CLASSES:
                pc = a_inv.get(M[c])
                s += pi[idx[c]] * (cond[idx[c], idx[pc]] if pc else 0.0)
            row.append(s)
        print(f"{name:<58}{row[0]:>10.4f}{row[1]:>14.4f}")

    # 逐类诊断：若平台是方案②，我们提交的哪些数字会被判对
    M2 = CANDIDATES["② FER顺序压缩 Angry0 Fear1 Happy2 Sad3 Surprise4 Neutral5"]
    a_inv = {v: c for c, v in ALPHA.items()}
    print("\n若平台为方案②，我们提交的数字分别被解释成哪个类别：")
    for c in C.CLASSES:
        num = ALPHA[c]
        print(f"  我们预测 {c:<9} -> 提交数字 {num} -> 平台解释为 {a_inv[num]:<9} "
              f"{'（正确）' if a_inv[num] == c else '（错位）'}")


if __name__ == "__main__":
    main()
