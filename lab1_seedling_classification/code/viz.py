# -*- coding: utf-8 -*-
"""实验报告配图：数据概览、模型对比、混淆矩阵、集成增益、预测分布。

用法: python viz.py
输出: reports/fig_*.png (+300dpi)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import common as C

# 中文字体：优先微软雅黑
for _f in ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Source Han Sans SC"]:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

PALETTE = ["#2E5A88", "#C0392B", "#27875F", "#B7791F", "#6B4C9A", "#1F7A8C", "#8C4A2F"]


def save(fig, name):
    p = os.path.join(C.REPORTS, name)
    fig.savefig(p)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  [图] {p}", flush=True)


def fig_dataset():
    """图 1：类别分布 + 样例图。"""
    from PIL import Image
    train = C.list_train()
    counts = [sum(1 for _, l in train if l == i) for i in range(C.NUM_CLASSES)]
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15], hspace=0.5, wspace=0.28)
    ax = fig.add_subplot(gs[0, :3])
    bars = ax.bar(range(C.NUM_CLASSES), counts, color=PALETTE[:C.NUM_CLASSES], width=0.62,
                  edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(C.NUM_CLASSES))
    ax.set_xticklabels([c.replace(" ", "\n") for c in C.CLASSES], fontsize=9)
    ax.set_ylabel("训练样本数")
    ax.set_title(f"图1(a) 训练集类别分布（共 {len(train)} 张，各类均衡）", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.22)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.03, str(c),
                ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="x", visible=False)

    ax2 = fig.add_subplot(gs[0, 3:])
    te = C.list_test()
    sizes = [C.load_rgb(p).size[0] for p in te[:200]]
    ax2.hist(sizes, bins=28, color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax2.set_xlabel("图像边长 / px")
    ax2.set_ylabel("张数")
    ax2.set_title(f"图1(b) 测试集图像尺寸分布（抽 200 张，n={len(te)}）", fontsize=11)
    ax2.grid(axis="x", visible=False)

    rng = np.random.default_rng(C.SEED)
    for i, cname in enumerate(C.CLASSES):
        paths = [p for p, l in train if l == i]
        pick = paths[rng.integers(0, len(paths))]
        axi = fig.add_subplot(gs[1, i + 1 if i < 5 else 5])
        axi = fig.add_subplot(gs[1, i]) if i < 6 else axi
        axi.imshow(C.load_rgb(pick).resize((150, 150)))
        axi.set_title(cname, fontsize=9)
        axi.axis("off")
    fig.suptitle("植物幼苗图像分类数据集概览", fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig1_数据集概览.png")


def fig_models():
    """图 2：各骨干 × 各分类头 的 CV 准确率热力图。"""
    s = C.load_json("linear_summary.json", {}) or {}
    s = {k: v for k, v in s.items() if k != "__concat__" and "heads" in v}
    if not s:
        print("  [跳过] 无 linear_summary.json"); return
    heads = list(next(iter(s.values()))["heads"].keys())
    backs = list(s.keys())
    Mx = np.array([[s[b]["heads"][h]["mean"] for h in heads] for b in backs])
    fig, ax = plt.subplots(figsize=(1.15 * len(heads) + 3.4, 0.62 * len(backs) + 2.4))
    im = ax.imshow(Mx, cmap="RdYlGn", vmin=0.55, vmax=max(0.86, Mx.max() + 0.01), aspect="auto")
    ax.set_xticks(range(len(heads))); ax.set_xticklabels(heads, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(backs)))
    ax.set_yticklabels([b.replace("_", " ") for b in backs], fontsize=9)
    for i in range(len(backs)):
        for j in range(len(heads)):
            v = Mx[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="black" if v < 0.80 else "white",
                    fontweight="bold" if v == Mx[i].max() else "normal")
    ax.set_title("图2 冻结特征 + 分类头：5 折交叉验证准确率矩阵", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label="CV 准确率")
    ax.grid(False)
    save(fig, "fig2_模型CV热力图.png")


def fig_ensemble():
    """图 3：成员 OOF 准确率 + 集成权重。"""
    r = C.load_json("ensemble_report.json", None)
    if not r:
        print("  [跳过] 无 ensemble_report.json"); return
    mem = r["members"]
    names = sorted(mem, key=lambda k: mem[k]["oof_acc"])
    vals = [mem[k]["oof_acc"] for k in names]
    fin = r.get("final", {})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    cols = [PALETTE[1] if k in fin.get("members", []) else "#9AA5B1" for k in names]
    ax.barh(range(len(names)), vals, color=cols, height=0.66, edgecolor="white")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("OOF 准确率")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("图3(a) 各成员单独 OOF 准确率（红色为入选集成）", fontsize=11)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    if fin.get("members"):
        w = fin["weights"]
        ax.pie(w, labels=fin["members"], autopct="%1.0f%%",
               colors=PALETTE[:len(w)], textprops={"fontsize": 9},
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title(f"图3(b) 最终集成权重（OOF={fin['oof_acc']:.4f}）", fontsize=11)
    save(fig, "fig3_集成权重.png")


def fig_confusion():
    """图 4：最终集成在训练集上的 OOF 混淆矩阵。"""
    r = C.load_json("ensemble_report.json", None)
    if not r:
        print("  [跳过] 无 ensemble_report.json"); return
    # 用与 ensemble.py 相同的成员与权重复算 OOF 混淆矩阵
    print("  [提示] 混淆矩阵由 eval_report.py 生成（需要 OOF 概率）")


def fig_training():
    """图 5：从训练日志解析每折 val 曲线。"""
    import re
    logs = [f for f in os.listdir(C.OUT) if f.startswith("log_ft_") and f.endswith(".txt")]
    if not logs:
        print("  [跳过] 无微调日志"); return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, lg in enumerate(sorted(logs)):
        eps, accs = [], []
        with open(os.path.join(C.OUT, lg), encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r"ep\s*(\d+)/\d+.*?val=([0-9.]+)", line)
                if m:
                    eps.append(int(m.group(1))); accs.append(float(m.group(2)))
        if eps:
            ax.plot(eps, accs, color=PALETTE[i % len(PALETTE)], lw=1.8,
                    label=lg.replace("log_", "").replace(".txt", ""))
            k = int(np.argmax(accs))
            ax.scatter([eps[k]], [accs[k]], s=42, color=PALETTE[i % len(PALETTE)], zorder=5)
            ax.annotate(f"{accs[k]:.3f}", (eps[k], accs[k]), textcoords="offset points",
                        xytext=(4, 5), fontsize=8.5)
    ax.set_xlabel("训练轮次 epoch"); ax.set_ylabel("验证集准确率")
    ax.set_title("图4 微调过程验证准确率曲线（每折最优已标注）", fontsize=11)
    ax.legend(fontsize=8.5, ncol=2)
    save(fig, "fig4_微调曲线.png")


def fig_prior():
    """图 6：测试集预测分布 vs 先验假设 + 深度检验结论。"""
    pt = C.load_json("prior_test.json", None)
    ps = C.load_json("prior_shift.json", None)
    if not pt:
        print("  [跳过] 无 prior_test.json"); return
    obs = np.array(pt["test_pred_counts"], float)
    n = obs.sum()
    oofm = np.array(pt["oof_marginal"], float) * n
    hyp = pt.get("hypotheses", {})
    ratio = hyp.get("原数据集比例", {}).get("expected_counts", {})
    ratio = np.array([ratio.get(c, 0) for c in C.CLASSES], float) if ratio else None
    if ratio is None or ratio.sum() == 0:
        orig = np.array([263, 221, 654, 516, 385], float)
        ratio = orig / orig.sum() * n

    x = np.arange(C.NUM_CLASSES)
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.bar(x - 1.5 * w, oofm, w, label="训练集同分布时的期望预测数", color="#9AA5B1")
    ax.bar(x - 0.5 * w, obs, w, label="测试集实际预测数", color=PALETTE[0])
    ax.bar(x + 0.5 * w, ratio, w, label="原数据集比例假设下的期望", color=PALETTE[1])
    if ps and ps.get("estimated_prior_test"):
        em = np.array(ps["estimated_prior_test"], float) * n
        ax.bar(x + 1.5 * w, em, w, label="EM 估计先验下的期望", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in C.CLASSES], fontsize=9)
    ax.set_ylabel("样本数")
    ax.set_title(f"图6 测试集预测分布与各先验假设对比\n"
                 f"卡方检验 χ²={pt['chi2_stat']:.2f}, df={pt['dof']}, "
                 f"p={pt['p_value']:.2e}：测试集先验显著不同于训练集",
                 fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_先验分布对比.png")


def fig_confidence():
    """图 7：置信度分布与分层准确率（结构风险视角）。"""
    ev = C.load_json("eval_report.json", None)
    if not ev:
        print("  [跳过] 无 eval_report.json"); return
    calib = ev.get("confidence_calibration", [])
    if not calib:
        print("  [跳过] 置信度分层为空"); return
    labels = [c["range"] for c in calib]
    ns = [c["n"] for c in calib]
    accs = [c["acc"] if c["acc"] is not None else np.nan for c in calib]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.bar(range(len(ns)), ns, color=PALETTE[0], width=0.62, edgecolor="white")
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(labels, rotation=20, fontsize=8.5)
    ax.set_ylabel("样本数"); ax.set_title("图7(a) OOF 置信度分布", fontsize=11)
    for i, v in enumerate(ns):
        ax.text(i, v + max(ns) * 0.02, str(v), ha="center", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    xs = [i for i, a in enumerate(accs) if not np.isnan(a)]
    ys = [accs[i] for i in xs]
    ax.plot(xs, ys, "o-", color=PALETTE[1], lw=2, ms=8)
    ax.axhline(ev["oof_acc"], ls="--", color="#666", lw=1.2,
               label=f"整体 OOF={ev['oof_acc']:.4f}")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("该置信度区间内的准确率")
    ax.set_title("图7(b) 置信度分层准确率（单调上升 = 概率可信）", fontsize=11)
    for i, v in zip(xs, ys):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.legend(fontsize=9)
    save(fig, "fig7_置信度分析.png")


def main():
    print("生成报告配图…", flush=True)
    for fn in (fig_dataset, fig_models, fig_ensemble, fig_training, fig_prior, fig_confidence):
        try:
            fn()
        except Exception as e:
            print(f"  [失败] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
