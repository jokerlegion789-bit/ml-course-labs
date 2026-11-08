# -*- coding: utf-8 -*-
"""先验校正强度的仿真标定。

问题：既然测试集先验显著不同于训练集，是否应该做先验校正？校正多少？

方法：用 OOF 概率构造“已知真值”的合成偏移验证集——
按目标先验 π_t 对 500 个 OOF 样本做加权重采样（有放回），
得到类别比例服从 π_t 的合成集，其真实标签已知，于是可以客观测量
不同校正强度 α 下的准确率：

    校正后验  w ∝ p_model(y|x) · (π_t(y) / π_s(y))^α,   α ∈ [0, 1]

α=0 表示不校正，α=1 表示完全按估计先验校正。对多个 π_t（含均衡、含 EM 估计值）
重复实验，统计各 α 的平均表现，据此选择稳健的 α。

输出: result/prior_sim.json
"""
import os

import numpy as np

import common as C
from ensemble import collect_members

PI_S = np.ones(C.NUM_CLASSES) / C.NUM_CLASSES
ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def resample_to_prior(y, pi_t, n, rng):
    """按目标先验有放回重采样索引。"""
    idx = []
    for c in range(C.NUM_CLASSES):
        pool = np.where(y == c)[0]
        k = int(round(pi_t[c] * n))
        idx.append(rng.choice(pool, size=k, replace=True))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def corrected_pred(P, pi_t, alpha):
    if alpha == 0:
        return P.argmax(1)
    w = P * (pi_t / PI_S)[None, :] ** alpha
    return w.argmax(1)


def main():
    members, y, _ = collect_members(224)
    rep = C.load_json("ensemble_report.json", None)
    if rep and rep.get("final"):
        names, w = rep["final"]["members"], rep["final"]["weights"]
    else:
        from ensemble import member_acc
        names = sorted(members, key=lambda k: -member_acc(members[k], y))[:4]
        w = [1.0 / len(names)] * len(names)
    from ensemble import pick, common_mask
    mask = common_mask(members, names, y)
    P = pick(members, names, w, "oof")[mask].astype(np.float64)
    y = y[mask]
    print(f"仿真使用公共覆盖样本 {len(y)} 个")

    ps = C.load_json("prior_shift.json", {}) or {}
    pi_em = np.array(ps.get("estimated_prior_test", PI_S), dtype=float)

    scenarios = {
        "均衡(无偏移)": PI_S,
        "EM估计先验": pi_em,
        "EM先验折半到均衡": (pi_em + PI_S) / 2,
        "原数据集比例": np.array([263, 221, 654, 516, 385], float) / 2039,
    }
    N, REPS, rng = 1000, 30, np.random.default_rng(C.SEED)

    print(f"{'场景':<20}" + "".join(f"  α={a:<4}" for a in ALPHAS))
    table, overall = {}, {a: [] for a in ALPHAS}
    for sname, pi_t in scenarios.items():
        accs = {a: [] for a in ALPHAS}
        for _ in range(REPS):
            idx = resample_to_prior(y, pi_t, N, rng)
            Ps, ys = P[idx], y[idx]
            for a in ALPHAS:
                accs[a].append(float((corrected_pred(Ps, pi_t, a) == ys).mean()))
        table[sname] = {a: float(np.mean(accs[a])) for a in ALPHAS}
        for a in ALPHAS:
            overall[a].append(table[sname][a])
        print(f"{sname:<20}" + "".join(f"  {table[sname][a]:.4f}" for a in ALPHAS))

    print(f"\n{'平均':<20}" + "".join(f"  {np.mean(overall[a]):.4f}" for a in ALPHAS))
    # 最坏情形（保守准则）
    print(f"{'最坏情形':<18}" + "".join(f"  {np.min(overall[a]):.4f}" for a in ALPHAS))

    best_avg = max(ALPHAS, key=lambda a: np.mean(overall[a]))
    best_worst = max(ALPHAS, key=lambda a: np.min(overall[a]))
    print(f"\n平均最优 α = {best_avg}   最坏情形最优 α = {best_worst}")

    # 仅在“如果测试集其实是均衡的”这一最坏情况下损失可接受的 α 中取最大值
    alpha_safe = 0.0
    for a in ALPHAS:
        loss_if_balanced = table["均衡(无偏移)"][a] - table["均衡(无偏移)"][0.0]
        if loss_if_balanced >= -0.005:
            alpha_safe = a
    print(f"在“测试集其实是均衡的”时损失不超过 0.5 个百分点的最大 α = {alpha_safe}")

    C.save_json({"alpha_grid": ALPHAS, "scenarios": table,
                 "mean_over_scenarios": {a: float(np.mean(overall[a])) for a in ALPHAS},
                 "worst_case": {a: float(np.min(overall[a])) for a in ALPHAS},
                 "estimated_prior_test": pi_em.tolist(),
                 "best_avg_alpha": best_avg, "best_worst_alpha": best_worst,
                 "alpha_safe": alpha_safe,
                 "oof_baseline_acc": float((P.argmax(1) == y).mean())},
                "prior_sim.json")


if __name__ == "__main__":
    main()
