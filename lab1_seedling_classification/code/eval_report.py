# -*- coding: utf-8 -*-
"""评估诊断：OOF 混淆矩阵、各类精确率/召回率/F1、主要混淆对、置信度分析。

复用 ensemble.py 的成员定义与最终权重，保证与提交结果口径一致。
用法: python eval_report.py
输出: reports/fig5_混淆矩阵.png, result/eval_report.json
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import common as C
from ensemble import collect_members

for _f in ["Microsoft YaHei", "SimHei", "Noto Sans SC"]:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def main():
    members, y, te_paths = collect_members(224)
    rep = C.load_json("ensemble_report.json", None)
    if rep and rep.get("final"):
        names = rep["final"]["members"]
        w = rep["final"]["weights"]
    else:
        from ensemble import member_acc
        names = sorted(members, key=lambda k: -member_acc(members[k], y))[:4]
        w = [1.0 / len(names)] * len(names)
    print(f"\n评估成员（与提交一致）: {list(zip(names, np.round(w, 3)))}")

    from ensemble import pick, common_mask
    P = pick(members, names, w, "oof")
    mask = common_mask(members, names, y)
    print(f"公共覆盖样本数: {int(mask.sum())} / {len(y)}")
    pred = P.argmax(1)
    acc = float((pred[mask] == y[mask]).mean())
    cm = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=int)
    for t, p in zip(y[mask], pred[mask]):
        cm[t, p] += 1

    per = {}
    print(f"\n集成 OOF 准确率 = {acc:.4f}")
    print(f"{'类别':<20}{'精确率':>8}{'召回率':>8}{'F1':>8}{'样本数':>7}")
    for i, c in enumerate(C.CLASSES):
        tp = cm[i, i]
        prec = tp / max(1, cm[:, i].sum())
        rec = tp / max(1, cm[i, :].sum())
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        per[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                  "support": int(cm[i, :].sum())}
        print(f"{c:<20}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{cm[i,:].sum():>7d}")

    pairs = []
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": C.CLASSES[i], "pred": C.CLASSES[j],
                              "n": int(cm[i, j])})
    pairs.sort(key=lambda d: -d["n"])
    print("\n主要混淆对（真实→预测）:")
    for d in pairs[:6]:
        print(f"  {d['true']:<20} → {d['pred']:<20} {d['n']} 次")

    conf = P.max(1)
    correct = pred == y
    bins = [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
    print("\n置信度分层准确率（OOF，仅公共覆盖样本）:")
    calib = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi) & mask
        a = float(correct[m].mean()) if m.any() else None
        calib.append({"range": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()), "acc": a})
        print(f"  [{lo:.2f},{hi:.2f})  n={m.sum():3d}  准确率={('%.3f'%a) if a is not None else 'n/a'}")

    # 测试集预测分布与两种先验假设对比
    fin_p = np.load(os.path.join(C.OUT, "final_test_proba.npy")) if \
        os.path.exists(os.path.join(C.OUT, "final_test_proba.npy")) else None
    dist_hyp = {}
    if fin_p is not None:
        tp = fin_p.argmax(1)
        obs = {C.CLASSES[i]: int((tp == i).sum()) for i in range(C.NUM_CLASSES)}
        n = len(tp)
        # 假设A：测试集与训练集同样均衡
        hyp_bal = {c: n / C.NUM_CLASSES for c in C.CLASSES}
        # 假设B：测试集比例与原 Kaggle 幼苗数据集（12类）中这5类的相对频数一致
        orig = {"Black-grass": 263, "Common wheat": 221, "Loose Silky-bent": 654,
                "Scentless Mayweed": 516, "Sugar beet": 385}
        s = sum(orig.values())
        hyp_prop = {c: n * orig[c] / s for c in C.CLASSES}
        print("\n测试集预测分布与先验假设对比:")
        print(f"  {'类别':<20}{'实测':>7}{'均衡假设':>10}{'原数据比例假设':>16}")
        for c in C.CLASSES:
            print(f"  {c:<20}{obs[c]:>7d}{hyp_bal[c]:>10.1f}{hyp_prop[c]:>16.1f}")
        dist_hyp = {"observed": obs,
                    "hypothesis_balanced": {k: round(v, 2) for k, v in hyp_bal.items()},
                    "hypothesis_original_ratio": {k: round(v, 2) for k, v in hyp_prop.items()},
                    "note": "原数据比例来自 Kaggle Plant Seedlings 数据集公开的各类样本数，"
                            "仅作为分布参照，未用于任何预测校正。"}

    # 混淆矩阵图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, M, title, fmt in ((axes[0], cm, "图5(a) OOF 混淆矩阵（样本数）", "d"),
                              (axes[1], cm / np.maximum(1, cm.sum(1, keepdims=True)),
                               "图5(b) OOF 混淆矩阵（按真实类归一化）", ".2f")):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() if fmt == "d" else 1.0)
        ax.set_xticks(range(C.NUM_CLASSES)); ax.set_yticks(range(C.NUM_CLASSES))
        ax.set_xticklabels([c.replace(" ", "\n") for c in C.CLASSES], fontsize=8.5)
        ax.set_yticklabels(C.CLASSES, fontsize=8.5)
        ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
        ax.set_title(title, fontsize=11)
        for i in range(C.NUM_CLASSES):
            for j in range(C.NUM_CLASSES):
                v = M[i, j]
                txt = f"{int(v)}" if fmt == "d" else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if v > (M.max() * 0.55 if fmt == "d" else 0.55) else "black")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"最终集成在训练集上的 OOF 表现（准确率 {acc:.4f}）", fontsize=12,
                 fontweight="bold")
    p = os.path.join(C.REPORTS, "fig5_混淆矩阵.png")
    fig.savefig(p); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig)
    print(f"\n  [图] {p}")

    C.save_json({"oof_acc": acc, "confusion": cm.tolist(), "classes": C.CLASSES,
                 "per_class": per, "top_confusions": pairs[:10],
                 "confidence_calibration": calib,
                 "members": names, "weights": [float(x) for x in w],
                 "test_distribution_analysis": dist_hyp}, "eval_report.json")


if __name__ == "__main__":
    main()
