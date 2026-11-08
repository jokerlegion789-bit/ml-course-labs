# -*- coding: utf-8 -*-
"""类别先验偏移分析与校正（Saerens–Latinne–Decaestecker EM）。

背景：训练集 5 类各 100 张（完全均衡），但模型在测试集上的预测分布明显偏离均衡
（如 Common wheat 仅占 10.3%），而该类的 OOF 精确率/召回率均在 0.89/0.92 以上，
说明模型本身没有系统性少判该类 —— 差异更可能来自测试集类别先验与训练集不同。

方法：不预设任何先验，用 EM 从测试集的软标签中直接估计目标先验 π_t：
    E 步: w_i(y) ∝ p_model(y|x_i) · π_t(y) / π_s(y)
    M 步: π_t(y) = (1/N) Σ_i w_i(y)
迭代至收敛后，用 w 作为校正后的后验取 argmax。

自检：在 OOF（先验确为均衡）上运行同一估计器，若估计值接近 0.2 且校正不改变 OOF 准确率，
则说明估计器无偏，可用于测试集。

用法:
  python prior_shift.py --check        # 只做 OOF 自检，不改提交
  python prior_shift.py --apply        # 自检通过后输出校正版提交
"""
import argparse
import os
import shutil

import numpy as np

import common as C

PRIOR_S_TRAIN = np.ones(C.NUM_CLASSES) / C.NUM_CLASSES  # 训练集先验：均衡


def em_prior(P, pi_s=PRIOR_S_TRAIN, iters=200, tol=1e-8):
    """从软标签估计目标先验，返回 (估计先验, 校正后后验)。"""
    pi = np.ones(C.NUM_CLASSES) / C.NUM_CLASSES
    ratio = pi / pi_s
    W = None
    for _ in range(iters):
        W = P * ratio[None, :]
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        new = W.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
        ratio = pi / pi_s
    W = P * ratio[None, :]
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    return pi, W


def show(pi, tag):
    print(f"  {tag}: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(C.CLASSES, pi)))


def check_on_oof():
    """在 OOF 概率上自检估计器（OOF 先验应为均衡）。"""
    import ensemble as E
    members, y, _ = E.collect_members(224)
    rep = C.load_json("ensemble_report.json", None)
    if rep and rep.get("final"):
        names, w = rep["final"]["members"], rep["final"]["weights"]
    else:
        names = sorted(members, key=lambda k: -E.member_acc(members[k], y))[:4]
        w = [1.0 / len(names)] * len(names)
    P = E.pick(members, names, w, "oof")
    mask = E.common_mask(members, names, y)
    Pp, yp = P[mask], y[mask]
    acc0 = float((Pp.argmax(1) == yp).mean())
    pi, W = em_prior(Pp)
    acc1 = float((W.argmax(1) == yp).mean())
    print("\n【OOF 自检】训练集先验确为均衡，估计器应还原出接近 0.2 的先验：")
    show(pi, "EM 估计先验")
    print(f"  校正前 OOF 准确率 = {acc0:.4f}   校正后 = {acc1:.4f}   "
          f"差异 = {acc1-acc0:+.4f}")
    dev = float(np.abs(pi - PRIOR_S_TRAIN).max())
    ok = dev < 0.06 and acc1 >= acc0 - 0.004
    print(f"  最大偏差 {dev:.3f}   ->  估计器{'可用' if ok else '不可靠'}")
    return ok, {"oof_acc_before": acc0, "oof_acc_after": acc1,
                "estimated_prior_on_oof": pi.tolist(), "max_dev_from_uniform": dev,
                "usable": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--alpha", type=float, default=None,
                    help="校正强度 α∈[0,1]；默认读取 prior_sim.json 的标定结果")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.alpha is None:
        sim = C.load_json("prior_sim.json", {}) or {}
        # 采用最大最小（worst-case）最优的 α：排名制评分下优先控制下行风险，
        # 该 α 在各先验假设场景中的最差表现最好。
        a.alpha = float(sim.get("best_worst_alpha", sim.get("best_avg_alpha", 0.0)))
        print(f"（α 取自 prior_sim.json 的最大最小标定结果：{a.alpha}；"
              f"平均最优为 {sim.get('best_avg_alpha')}）")

    print("=" * 70)
    ok, chk = check_on_oof()
    print("=" * 70)

    P_te = np.load(os.path.join(C.OUT, "final_test_proba.npy"))
    pi_te, W_te = em_prior(P_te)
    print("\n【测试集】EM 估计的类别先验：")
    show(pi_te, "EM 估计先验")

    # 阻尼校正：w ∝ p · (π_t/π_s)^α；α 由 prior_sim.py 在 OOF 合成偏移集上标定
    alpha = float(a.alpha)
    W_applied = P_te * (pi_te / PRIOR_S_TRAIN)[None, :] ** alpha
    W_applied = W_applied / np.maximum(W_applied.sum(1, keepdims=True), 1e-12)
    print(f"  校正强度 α = {alpha}（α=0 不校正，α=1 完全校正）")

    obs = P_te.argmax(1)
    new = W_applied.argmax(1)
    print("\n  类别            校正前预测数   校正后预测数   均衡期望(75.6)")
    for i, c in enumerate(C.CLASSES):
        print(f"  {c:<20}{(obs==i).sum():>10d}{(new==i).sum():>13d}{378/C.NUM_CLASSES:>15.1f}")
    changed = int((obs != new).sum())
    print(f"\n  校正改变预测的样本数: {changed} / {len(obs)}  ({100*changed/len(obs):.1f}%)")

    rep = C.load_json("eval_report.json", {}) or {}
    dist = rep.get("test_distribution_analysis", {})
    if dist:
        print("\n  与两种先验假设对比（比例）：")
        print(f"  {'类别':<20}{'EM估计':>9}{'均衡假设':>10}{'原数据比例':>11}")
        for i, c in enumerate(C.CLASSES):
            b = dist.get("hypothesis_balanced", {}).get(c, 0) / 378
            r = dist.get("hypothesis_original_ratio", {}).get(c, 0) / 378
            print(f"  {c:<20}{pi_te[i]:>9.3f}{b:>10.3f}{r:>11.3f}")

    out = {"oof_check": chk, "estimated_prior_test": pi_te.tolist(), "alpha": alpha,
           "changed": changed, "n": int(len(obs)),
           "counts_before": {C.CLASSES[i]: int((obs == i).sum()) for i in range(C.NUM_CLASSES)},
           "counts_after": {C.CLASSES[i]: int((new == i).sum()) for i in range(C.NUM_CLASSES)}}

    if a.apply:
        # 应用门槛：α>0 且仿真显示在最坏情形下也不劣于不校正
        sim = C.load_json("prior_sim.json", {}) or {}
        worst = sim.get("worst_case", {})
        gain_worst = None
        if worst:
            k0 = str(min(worst, key=lambda x: abs(float(x)))) if 0.0 not in [float(x) for x in worst] else None
            try:
                base = float(worst["0.0"]) if "0.0" in worst else float(list(worst.values())[0])
            except Exception:
                base = None
            key = str(alpha)
            if key in worst and base is not None:
                gain_worst = float(worst[key]) - base
        gate = alpha > 0 and (gain_worst is None or gain_worst >= -1e-9)
        out["apply_gate"] = {"alpha": alpha, "worst_case_gain": gain_worst, "passed": bool(gate)}
        if not gate:
            print(f"\n[拒绝应用] 门槛未通过（α={alpha}, 最坏情形增益={gain_worst}），不修改提交文件。")
            C.save_json(out, "prior_shift.json")
            return
        ids, _ = C.read_submission_template()
        # 必须按文件名映射（模板行顺序 != sorted 文件名顺序）
        te_names = [os.path.basename(p) for p in C.list_test()]
        if set(te_names) != set(ids):
            raise ValueError("测试目录文件名集合与提交模板 ID 集合不一致")
        pred = {nm: C.CLASSES[int(new[j])] for j, nm in enumerate(te_names)}
        dst = a.out or os.path.join(r"D:\deepseek\2026task1", "submission-for-task2.csv")
        bak = dst.replace(".csv", "_无先验校正.csv")
        if os.path.exists(dst) and not os.path.exists(bak):
            shutil.copy2(dst, bak)
            print(f"\n  已备份未校正提交 -> {bak}")
        C.write_submission(pred, dst)
        C.write_submission(pred, os.path.join(C.OUT, "submission-for-task2.csv"))
        np.save(os.path.join(C.OUT, "final_test_proba_corrected.npy"), W_applied.astype(np.float32))
        out["applied"] = True
    else:
        print("\n（未加 --apply，本次不修改提交文件）")
    C.save_json(out, "prior_shift.json")


if __name__ == "__main__":
    main()
