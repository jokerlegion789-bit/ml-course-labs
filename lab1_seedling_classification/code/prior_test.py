# -*- coding: utf-8 -*-
"""先验偏移的严格检验：比较 OOF 预测边缘分布 与 测试集预测边缘分布。

原理：若测试集与训练集同分布（均衡），则模型在测试集上的预测边缘分布
应与其在 OOF 上的预测边缘分布一致（而非与真实先验一致，因为模型本身有偏）。
用卡方检验比较两者，若显著不同，则存在先验偏移（或协变量偏移）。

输出: result/prior_test.json
"""
import json
import os

import numpy as np

import common as C


def chi2(obs, exp):
    """卡方统计量与自由度（不做连续性校正）。"""
    obs = np.asarray(obs, float)
    exp = np.asarray(exp, float) * obs.sum() / exp.sum()
    stat = float(((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    return stat, len(obs) - 1


def chi2_sf(x, k):
    """卡方分布上尾概率（用 gamma 不完全函数）。"""
    from scipy.stats import chi2 as _c
    return float(_c.sf(x, k))


def main():
    ev = C.load_json("eval_report.json", None)
    if not ev:
        print("缺少 eval_report.json，先运行 eval_report.py"); return
    cm = np.array(ev["confusion"], dtype=float)          # 行=真实, 列=预测
    oof_pred = cm.sum(0)                                  # OOF 预测边缘（500 张）
    oof_marg = oof_pred / oof_pred.sum()

    P_te = np.load(os.path.join(C.OUT, "final_test_proba.npy"))
    te_pred = np.bincount(P_te.argmax(1), minlength=C.NUM_CLASSES).astype(float)
    te_marg = te_pred / te_pred.sum()
    n_te = int(te_pred.sum())

    print("=" * 74)
    print("先验偏移检验：OOF 预测边缘分布  vs  测试集预测边缘分布")
    print("=" * 74)
    print(f"{'类别':<20}{'OOF预测数/500':>14}{'OOF占比':>9}{'测试预测数/378':>15}{'测试占比':>9}{'差值':>8}")
    for i, c in enumerate(C.CLASSES):
        print(f"{c:<20}{oof_pred[i]:>14.0f}{oof_marg[i]:>9.3f}{te_pred[i]:>15.0f}"
              f"{te_marg[i]:>9.3f}{te_marg[i]-oof_marg[i]:>+8.3f}")

    exp = oof_marg * n_te
    stat, k = chi2(te_pred, exp)
    p = chi2_sf(stat, k)
    print(f"\n卡方统计量 = {stat:.2f}, 自由度 = {k}, p 值 = {p:.4f}")
    if p < 0.01:
        verdict = "存在高度显著的分布差异：测试集类别先验很可能不同于训练集"
    elif p < 0.05:
        verdict = "存在显著差异：测试集先验可能不同于训练集"
    else:
        verdict = "无显著差异：没有证据表明测试集先验不同于训练集"
    print(f"结论：{verdict}")

    # 与两种具体假设对比
    orig = {"Black-grass": 263, "Common wheat": 221, "Loose Silky-bent": 654,
            "Scentless Mayweed": 516, "Sugar beet": 385}
    s = sum(orig.values())
    hyp_ratio = np.array([orig[c] for c in C.CLASSES], float) / s
    # “从原数据中每类扣掉 100 张训练图后”的剩余比例
    rem = np.array([orig[c] - 100 for c in C.CLASSES], float)
    hyp_rem = rem / rem.sum()

    print(f"\n{'假设':<34}{'卡方':>9}{'p 值':>10}{'结论':>10}")
    res = {}
    for name, h in [("均衡（与训练集相同）", oof_marg),
                    ("原数据集比例", hyp_ratio),
                    ("原数据集扣除训练后的剩余比例", hyp_rem)]:
        e = h * n_te
        st, _ = chi2(te_pred, e)
        pp = chi2_sf(st, k)
        res[name] = {"chi2": st, "p": pp,
                     "expected_counts": {c: float(h[i] * n_te) for i, c in enumerate(C.CLASSES)}}
        print(f"{name:<34}{st:>9.2f}{pp:>10.4f}{'不拒绝' if pp>0.05 else '拒绝':>10}")

    out = {"oof_pred_counts": oof_pred.tolist(), "oof_marginal": oof_marg.tolist(),
           "test_pred_counts": te_pred.tolist(), "test_marginal": te_marg.tolist(),
           "chi2_stat": stat, "dof": k, "p_value": p,
           "vs_uniform_prior": {"chi2": res["均衡（与训练集相同）"]["chi2"],
                                "p": res["均衡（与训练集相同）"]["p"]},
           "hypotheses": res, "verdict": verdict}
    C.save_json(out, "prior_test.json")


if __name__ == "__main__":
    main()
