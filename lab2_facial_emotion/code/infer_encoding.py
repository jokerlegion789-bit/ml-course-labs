# -*- coding: utf-8 -*-
"""反推平台使用的 Emotion 数字编号。

已知：留出集上模型准确率约 0.685，但线上实测只有 0.417。
若只是「纯编号错位」，分数会掉到约 0.17（随机）；0.417 说明只有一部分类别对上了。

做法：枚举所有「类别 -> 数字」的单射编号 M，用留出集混淆矩阵与 FER2013 官方
测试集类别比例，计算该编号下应得的期望准确率，找出与实测 0.417 最接近的编号。

score(M) = Σ_T π(T) · P( P = A⁻¹(M(T)) | T )
其中 A 是我们提交时用的编号（字母序 0..5），A⁻¹ 表示把数字换回类别。
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\deepseek\2026task2\code")
import common2 as C
from train2 import stratified_holdout

OBSERVED = 0.417

# FER2013 官方测试集类别数（已去掉 Disgust）
FER_TEST = {"Angry": 980, "Fear": 1024, "Happy": 1774, "Sad": 1247,
            "Surprise": 831, "Neutral": 1211}
# 原始 FER2013 的数字编码
FER_CODE = {"Angry": 0, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
ALPHA = {c: i for i, c in enumerate(C.CLASSES)}


def main():
    y_all = np.load(os.path.join(C.CACHE, "train_label.npy"))
    _, hoi = stratified_holdout(y_all)
    y = y_all[hoi]

    # 用留出集上的集成概率估计 P(P|T)
    Ps = []
    for f in sorted(os.listdir(C.OUT)):
        if f.startswith("hold_") and f.endswith(".npy"):
            Ps.append(np.load(os.path.join(C.OUT, f)))
    P = np.mean(Ps, axis=0)
    pred = P.argmax(1)
    cm = np.zeros((6, 6), float)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    row = cm.sum(1, keepdims=True)
    cond = cm / np.maximum(row, 1)          # cond[T, P] = P(预测P | 真实T)
    print(f"留出集整体准确率 = {(pred == y).mean():.4f}（{len(Ps)} 个模型集成）\n")

    tot = sum(FER_TEST.values())
    idx = {c: i for i, c in enumerate(C.CLASSES)}
    pi = np.array([FER_TEST[c] / tot for c in C.CLASSES])

    def score(M):
        """M: {类别名: 提交数字}，返回期望准确率。"""
        a_inv = {v: c for c, v in ALPHA.items()}   # 数字 -> 我们提交的类别名
        s = 0.0
        for c in C.CLASSES:
            num = M[c]
            t = idx[c]
            p_cls = a_inv.get(num)                 # 数字对应我们预测的哪个类别
            s += pi[t] * (cond[t, idx[p_cls]] if p_cls else 0.0)
        return s

    print(f"实测分数 = {OBSERVED}")
    print(f"{'编号方案':<52}{'期望准确率':>12}{'与实测之差':>12}")
    cands = {
        "字母序 0..5 (=我提交的)": ALPHA,
        "原始 FER 编码 {0,2,3,4,5,6}": FER_CODE,
        "字母序 1..6": {c: i + 1 for i, c in enumerate(C.CLASSES)},
        "FER 编码 +1 {1,3,4,5,6,7}": {c: v + 1 for c, v in FER_CODE.items()},
    }
    for name, M in cands.items():
        s = score(M)
        print(f"{name:<52}{s:>12.4f}{s-OBSERVED:>+12.4f}")

    # 枚举全部单射编号（类别 -> {0..6} 中 6 个不同数字）
    best = []
    for combo in itertools.permutations(range(7), 6):
        M = {c: combo[i] for i, c in enumerate(C.CLASSES)}
        s = score(M)
        best.append((abs(s - OBSERVED), s, M))
    best.sort(key=lambda t: t[0])
    print(f"\n最接近实测 {OBSERVED} 的 12 种编号：")
    print(f"{'编号（Angry/Fear/Happy/Neutral/Sad/Surprise）':<44}{'期望':>9}{'差值':>9}")
    for d, s, M in best[:12]:
        nums = "/".join(str(M[c]) for c in C.CLASSES)
        print(f"{nums:<44}{s:>9.4f}{s-OBSERVED:>+9.4f}")


if __name__ == "__main__":
    main()
