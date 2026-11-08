# -*- coding: utf-8 -*-
"""集成选择的嵌套验证：在 OOF 的一半上选权重，在另一半上评估。

动机：任何在同一份 OOF 上搜索权重的方法都会对这份数据产生乐观偏差。
本脚本用重复随机半样本划分（分层）做嵌套验证，客观比较：

  best_single    只取在 A 半上最优的单体
  mean_topK      A 半上最优的 K 个成员等权平均
  bagged_greedy  在 A 半上做 bagged 贪心选择的权重
  all_mean       全部成员等权平均

在 B 半（选择时未见过的样本）上评估，从而给出无偏的方法对比。
这决定了最终提交应采用哪种集成策略。

用法: python validate_ensemble.py [--reps 100]
输出: result/ensemble_validation.json
"""
import argparse
import os

import numpy as np

import common as C
from ensemble import bagged_greedy_weights, collect_members, greedy_weights


def acc(P, y):
    return float((P.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    members, y, _ = collect_members(224)
    # 只用完整覆盖的成员
    names = [k for k in members if members[k]["mask"].all()]
    print(f"参与验证的完整成员（{len(names)}）: {names}")
    if len(names) < 2:
        print("成员不足，跳过"); return
    O = {k: members[k]["oof"] for k in names}

    rng = np.random.default_rng(C.SEED)
    idx_by_cls = [np.where(y == c)[0] for c in range(C.NUM_CLASSES)]

    methods = ["best_single", "mean_top3", "mean_top5", "bagged_greedy", "all_mean"]
    res = {m: [] for m in methods}
    picks = {m: {} for m in methods}

    for r in range(a.reps):
        A, B = [], []
        for pool in idx_by_cls:
            p = rng.permutation(pool)
            h = len(p) // 2
            A.extend(p[:h]); B.extend(p[h:])
        A = np.array(A); B = np.array(B)

        def sub(k, ix):
            return O[k][ix]

        # 在 A 上排名
        accA = {k: acc(sub(k, A), y[A]) for k in names}
        ranked = sorted(names, key=lambda k: -accA[k])

        cand = {}
        cand["best_single"] = ([ranked[0]], np.array([1.0]))
        for K in (3, 5):
            sel = ranked[:K]
            cand[f"mean_top{K}"] = (sel, np.ones(len(sel)) / len(sel))
        cand["all_mean"] = (names, np.ones(len(names)) / len(names))

        # bagged 贪心：只在 A 半上选择（用 A 半构造临时成员字典）
        sub_members = {k: {"oof": O[k], "mask": np.zeros(len(y), bool), "test": None,
                           "desc": ""} for k in names}
        for k in names:
            sub_members[k]["mask"][A] = True
        w_b, _ = bagged_greedy_weights(sub_members, y, names, n_bags=20, frac=0.6, iters=40)
        cand["bagged_greedy"] = (names, w_b)

        for m, (nm, w) in cand.items():
            P = sum(wi * sub(k, B) for wi, k in zip(w, nm))
            res[m].append(acc(P, y[B]))
            for k, wi in zip(nm, w):
                if wi > 0:
                    picks[m][k] = picks[m].get(k, 0) + 1

    print(f"\n嵌套验证结果（{a.reps} 次分层半样本划分，评估在未参与选择的 B 半上）")
    print(f"{'方法':<18}{'B 半准确率均值':>14}{'标准差':>10}{'最差':>9}{'最好':>9}")
    out = {}
    base = np.mean(res["best_single"])
    for m in methods:
        v = np.array(res[m])
        out[m] = {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max()),
                  "gain_vs_best_single": float(v.mean() - base)}
        print(f"{m:<18}{v.mean():>14.4f}{v.std():>10.4f}{v.min():>9.4f}{v.max():>9.4f}"
              f"   ({v.mean()-base:+.4f})")
    # 配对比较：bagged 与 best_single 的逐次差值
    d = np.array(res["bagged_greedy"]) - np.array(res["best_single"])
    wins = int((d > 1e-9).sum()); losses = int((d < -1e-9).sum()); ties = a.reps - wins - losses
    print(f"\nbagged_greedy vs best_single 配对比较: 胜 {wins} / 负 {losses} / 平 {ties}")
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(d[d != 0]) if (d != 0).any() else (0.0, 1.0)
        print(f"Wilcoxon 符号秩检验: p = {p:.4f}  "
              f"({'显著' if p < 0.05 else '不显著'}，双尾)")
    except Exception as e:
        stat, p = None, None
        print(f"检验失败: {e}")

    out["paired"] = {"wins": wins, "losses": losses, "ties": ties,
                     "wilcoxon_stat": None if stat is None else float(stat),
                     "p_value": None if p is None else float(p),
                     "mean_diff": float(d.mean())}
    out["member_pick_frequency"] = {m: picks[m] for m in methods}
    C.save_json(out, "ensemble_validation.json")

    # 结论建议
    bg = out["bagged_greedy"]["mean"] - out["best_single"]["mean"]
    print(f"\n结论：bagged 贪心平均比最优单体高 {bg:+.4f}")
    if bg > 0 and (p is None or p < 0.20):
        print("  -> 支持使用 bagged 贪心集成")
    elif bg > 0:
        print("  -> 集成略优但统计上不显著，二者皆可，建议仍用集成以降低方差")
    else:
        print("  -> 嵌套验证不支持集成，建议改用最优单体")


if __name__ == "__main__":
    main()
