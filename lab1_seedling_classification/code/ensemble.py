# -*- coding: utf-8 -*-
"""模型集成与提交文件生成。

流程：
  1) 对每个已缓存特征（含多骨干拼接）用统一切分算 OOF 概率与测试集概率
  2) 收集微调模型（finetune.py）产出的 OOF / 测试概率
  3) 在 OOF 上做权重搜索（坐标上升），选出最优集成
  4) 按模板原始顺序写出 submission-for-task2.csv

覆盖率处理：若某成员只跑完部分折，其未覆盖样本的 OOF 不是真实预测。
本脚本为每个成员记录 coverage mask，成员排名与集成评估**只在所有选中成员
都覆盖的样本子集上进行**，避免用填充值把成绩算错。

用法: python ensemble.py [--method auto|best_single|mean|weighted] [--top 6]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold

import common as C
import models as M
from train_linear import build, candidate_heads

warnings.filterwarnings("ignore")

LINEAR_BACKBONES = ["resnet18", "resnet50", "efficientnet_b0", "densenet121"]
LINEAR_HEAD = "svc_lin_C1"


def best_head_for(summary, key, fallback=LINEAR_HEAD):
    d = (summary or {}).get(key)
    if isinstance(d, dict) and d.get("best_head"):
        return d["best_head"]
    return fallback


def oof_and_test(Xtr, y, Xte, head=LINEAR_HEAD, folds=5):
    """用与微调一致的折划分产出 OOF 概率与全量重训后的测试集概率。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)
    oof = np.zeros((len(y), C.NUM_CLASSES), dtype=np.float32)
    mask = np.zeros(len(y), dtype=bool)
    need_scale = candidate_heads()[head][1]
    for tri, vai in skf.split(Xtr, y):
        pipe = build(head, need_scale)
        with joblib.parallel_backend("sequential"):
            pipe.fit(Xtr[tri], y[tri])
        if hasattr(pipe, "predict_proba"):
            oof[vai] = pipe.predict_proba(Xtr[vai])
        else:
            d = pipe.decision_function(Xtr[vai])
            oof[vai] = np.exp(d) / np.exp(d).sum(1, keepdims=True)
        mask[vai] = True
    pipe = build(head, need_scale)
    with joblib.parallel_backend("sequential"):
        pipe.fit(Xtr, y)
    if hasattr(pipe, "predict_proba"):
        te = pipe.predict_proba(Xte)
    else:
        d = pipe.decision_function(Xte)
        te = np.exp(d) / np.exp(d).sum(1, keepdims=True)
    return oof.astype(np.float32), te.astype(np.float32), mask


def collect_members(size=224):
    """返回 {成员名: dict(oof, test, desc, mask)}、标签 y、测试路径。"""
    train = C.list_train()
    tr_paths = [p for p, _ in train]
    y = np.array([l for _, l in train])
    te_paths = C.list_test()
    lin_summary = C.load_json("linear_summary.json", {}) or {}
    members = {}

    feats_tr, feats_te = {}, {}
    for b in LINEAR_BACKBONES:
        p = os.path.join(C.CACHE, f"feat_{b}_train_{size}.npz")
        if not os.path.exists(p):
            print(f"  [跳过] 无 {b} 训练特征缓存", flush=True)
            continue
        feats_tr[b] = np.load(p)["X"]
        feats_te[b] = np.load(os.path.join(C.CACHE, f"feat_{b}_test_{size}.npz"))["X"]

    for b, Xtr in feats_tr.items():
        head = best_head_for(lin_summary, b)
        oof, te, mask = oof_and_test(Xtr, y, feats_te[b], head=head)
        members[f"lin_{b}"] = {"oof": oof, "test": te, "mask": mask,
                               "desc": f"冻结特征 {b} + {head}"}
        print(f"  [成员] lin_{b:18s} OOF={(oof.argmax(1)==y).mean():.4f}  ({head})", flush=True)

    if len(feats_tr) > 1:
        def zs(X):
            return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
        Xtr = np.concatenate([zs(feats_tr[b]) for b in feats_tr], 1)
        Xte = np.concatenate([zs(feats_te[b]) for b in feats_tr], 1)
        head = best_head_for(lin_summary, "__concat__")
        oof, te, mask = oof_and_test(Xtr, y, Xte, head=head)
        members["lin_concat"] = {"oof": oof, "test": te, "mask": mask,
                                 "desc": f"多骨干特征拼接({'+'.join(feats_tr)}) + {head}"}
        print(f"  [成员] lin_concat          OOF={(oof.argmax(1)==y).mean():.4f}  ({head})", flush=True)

    for f in sorted(os.listdir(C.OUT)):
        if f.startswith("oof_ft_") and f.endswith(".npy") and "_fold" not in f:
            tag = f[4:-4]
            te_p = os.path.join(C.OUT, f"test_{tag}.npy")
            if not os.path.exists(te_p):
                continue
            oof = np.load(os.path.join(C.OUT, f)).astype(np.float32)
            te = np.load(te_p).astype(np.float32)
            s = oof.sum(1)
            mask = s > 1e-6                      # 未被任何折覆盖的行（和为 0）
            n_cov = int(mask.sum())
            acc = float((oof[mask].argmax(1) == y[mask]).mean()) if n_cov else 0.0
            members[tag] = {"oof": oof, "test": te, "mask": mask,
                            "desc": f"微调 {tag}（覆盖 {n_cov}/{len(y)} 折样本）"}
            print(f"  [成员] {tag:23s} OOF={acc:.4f}  (覆盖 {n_cov}/{len(y)})", flush=True)
    return members, y, te_paths


def member_acc(m, y):
    return float((m["oof"][m["mask"]].argmax(1) == y[m["mask"]]).mean())


def pick(members, names, w, key="oof"):
    """按权重合并成员概率。key ∈ {'oof','test'}。"""
    return sum(wi * members[k][key] for wi, k in zip(w, names))


def common_mask(members, names, y):
    """所选成员的公共覆盖掩码。"""
    mask = np.ones(len(y), dtype=bool)
    for k in names:
        mask &= members[k]["mask"]
    return mask


def ens_acc(members, names, w, y):
    """在所选成员的公共覆盖子集上评估集成 OOF 准确率。"""
    mask = np.ones(len(y), dtype=bool)
    for k in names:
        mask &= members[k]["mask"]
    P = sum(wi * members[k]["oof"] for wi, k in zip(w, names))
    return float((P[mask].argmax(1) == y[mask]).mean()), int(mask.sum())


def search_weights(members, y, names, iters=80, step=0.1):
    """坐标上升搜索连续权重（作为对照方案，可能陷入局部最优）。"""
    n = len(names)
    w = np.ones(n) / n
    best, _ = ens_acc(members, names, w, y)
    for _ in range(iters):
        improved = False
        for i in range(n):
            for d in (step, -step):
                cand = w.copy()
                cand[i] = max(0.0, cand[i] + d)
                if cand.sum() <= 0:
                    continue
                cand /= cand.sum()
                a, _ = ens_acc(members, names, cand, y)
                if a > best + 1e-9:
                    best, w, improved = a, cand, True
        if not improved:
            step /= 2
            if step < 0.01:
                break
    return w, best


def greedy_weights(members, y, names, iters=60):
    """Caruana 贪心集成选择（有放回）：每步加入使集成准确率提升最多的成员。

    从空集开始，第一步必然选中最优单体，因此结果单调不劣于任何单体；
    产出的权重为整数计数归一化，比连续坐标上升稳健得多。
    """
    mask = common_mask(members, names, y)
    O = {k: members[k]["oof"][mask] for k in names}
    ys = y[mask]
    counts = {k: 0 for k in names}
    cur = np.zeros_like(O[names[0]])
    best_acc = -1.0
    hist = []
    for t in range(iters):
        bk, ba = None, best_acc
        for k in names:
            cand = (cur * t + O[k]) / (t + 1)
            a = float((cand.argmax(1) == ys).mean())
            if a > ba + 1e-12:
                ba, bk = a, k
        if bk is None:
            break
        cur = (cur * t + O[bk]) / (t + 1)
        counts[bk] += 1
        best_acc = ba
        hist.append({"step": t + 1, "member": bk, "oof_acc": ba})
    tot = max(1, sum(counts.values()))
    w = np.array([counts[k] / tot for k in names])
    return w, best_acc, hist


def bagged_greedy_weights(members, y, names, n_bags=20, frac=0.6, iters=40, seed=C.SEED):
    """Caruana 的 bagged 版本：在随机成员子集上做贪心，权重取平均，降低方差。"""
    rng = np.random.default_rng(seed)
    W = np.zeros(len(names))
    k = max(2, int(round(frac * len(names))))
    for _ in range(n_bags):
        sub = list(rng.choice(names, size=min(k, len(names)), replace=False))
        w_sub, _, _ = greedy_weights(members, y, sub, iters=iters)
        for nm, wv in zip(sub, w_sub):
            W[names.index(nm)] += wv
    W = W / max(W.sum(), 1e-9)
    acc, _ = ens_acc(members, names, W, y)
    return W, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default=None)
    ap.add_argument("--method", default="auto",
                    choices=["auto", "best_single", "mean", "weighted", "greedy", "bagged",
                             "mean_top3", "mean_top5"])
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    members, y, te_paths = collect_members(a.size)
    if not members:
        print("没有任何可用成员"); sys.exit(1)

    ranked = sorted(members, key=lambda k: -member_acc(members[k], y))
    print("\n成员 OOF 排名（各自覆盖子集内）:")
    for k in ranked:
        print(f"  {k:24s} {member_acc(members[k],y):.4f}   {members[k]['desc']}")

    report = {"members": {k: {"oof_acc": member_acc(members[k], y),
                              "covered": int(members[k]["mask"].sum()),
                              "desc": members[k]["desc"]} for k in members}}

    top = ranked[:a.top]
    cands = {"best_single": ([ranked[0]], np.array([1.0]))}
    cands["mean"] = (top, np.ones(len(top)) / len(top))
    for K in (3, 5):
        sel = ranked[:min(K, len(ranked))]
        cands[f"mean_top{K}"] = (sel, np.ones(len(sel)) / len(sel))
    w_g, _, hist = greedy_weights(members, y, top)
    cands["greedy"] = (top, w_g)
    w_b, _ = bagged_greedy_weights(members, y, top)
    cands["bagged"] = (top, w_b)
    w_w, _ = search_weights(members, y, top)
    cands["weighted"] = (top, w_w)
    report["greedy_trace"] = hist

    if a.method != "auto":
        names, w = cands[a.method]
        report["chosen"] = a.method
    else:
        # 方法选择以 validate_ensemble.py 的嵌套验证结论为准（避免在同一份 OOF 上选方法）
        val = C.load_json("ensemble_validation.json", None)
        rec = None
        if val:
            keys = [k for k in ("best_single", "mean_top3", "mean_top5", "bagged",
                                "bagged_greedy", "mean", "greedy", "weighted", "all_mean")
                    if k in val]
            if keys:
                rec = max(keys, key=lambda k: val[k]["mean"])
                print(f"  嵌套验证推荐的集成方式: {rec} "
                      f"(B 半准确率 {val[rec]['mean']:.4f})")
                report["validation"] = {k: val[k] for k in keys}
                report["validation_paired"] = val.get("paired")
                report["validation_choice"] = rec
        # 映射验证脚本里的方法名到本脚本候选名
        alias = {"bagged_greedy": "bagged", "bagged": "bagged", "mean_top3": "mean_top3",
                 "mean_top5": "mean_top5", "best_single": "best_single"}
        rec = alias.get(rec, rec)
        for cn, (nm, ww) in cands.items():
            acc, ncov = ens_acc(members, nm, ww, y)
            report[f"cand_{cn}"] = {"members": nm, "weights": [float(x) for x in ww],
                                    "oof_acc": acc, "n_eval": ncov}
            print(f"  候选 {cn:12s} OOF={acc:.4f}  (评估样本 {ncov})")
        if rec in cands and val is not None:
            names, w = cands[rec]
            acc, _ = ens_acc(members, names, w, y)
            report["chosen"] = rec + "(嵌套验证选定)"
            print(f"  >> 选定方案: {rec}（嵌套验证选定）  OOF={acc:.4f}")
        else:
            scored = []
            for cn, (nm, ww) in cands.items():
                acc, ncov = ens_acc(members, nm, ww, y)
                scored.append((cn, acc, nm, ww))
            bag = [t for t in scored if t[0] == "bagged"][0]
            best = max(scored, key=lambda t: t[1])
            cn, acc, names, w = bag if bag[1] >= best[1] - 0.004 else best
            report["chosen"] = cn
            print(f"  >> 选定方案: {cn}  OOF={acc:.4f}")

    P_te = sum(wi * members[k]["test"] for wi, k in zip(w, names))
    oof_acc, n_eval = ens_acc(members, names, w, y)
    print(f"\n最终集成: {[(k, round(float(x),3)) for k, x in zip(names, w)]}")
    print(f"集成 OOF 准确率 = {oof_acc:.4f}（评估样本 {n_eval}）")

    report["final"] = {"members": names, "weights": [float(x) for x in w],
                       "oof_acc": oof_acc, "n_eval": n_eval, "n_test": int(P_te.shape[0])}
    idx = P_te.argmax(1)
    dist = {C.CLASSES[i]: int((idx == i).sum()) for i in range(C.NUM_CLASSES)}
    report["pred_distribution"] = dist
    print("测试集预测分布:", dist)
    conf = P_te.max(1)
    report["confidence"] = {"mean": float(conf.mean()), "min": float(conf.min()),
                            "p05": float(np.percentile(conf, 5)),
                            "below_0.5": int((conf < 0.5).sum())}
    print("置信度: 均值 %.3f  最小 %.3f  5%%分位 %.3f  <0.5 的样本 %d 张"
          % (conf.mean(), conf.min(), np.percentile(conf, 5), (conf < 0.5).sum()))
    C.save_json(report, "ensemble_report.json")
    np.save(os.path.join(C.OUT, "final_test_proba.npy"), P_te.astype(np.float32))

    # 关键：预测的每一行按【文件名】对应，绝不能按位置对应。
    # 提交模板的行顺序与 sorted(os.listdir()) 完全不同（实测 378 行里只有 2 行同位置），
    # 按位置赋值会把几乎所有图片的预测错配到别的 ID 上，成绩会掉到随机水平。
    ids, _ = C.read_submission_template()
    te_names = [os.path.basename(p) for p in te_paths]
    if len(te_names) != len(idx):
        raise ValueError(f"测试文件数 {len(te_names)} 与预测行数 {len(idx)} 不一致")
    if set(te_names) != set(ids):
        raise ValueError("测试目录文件名集合与提交模板 ID 集合不一致，请核对数据源")
    pred = {nm: C.CLASSES[int(idx[j])] for j, nm in enumerate(te_names)}
    out = a.out or os.path.join(r"D:\deepseek\2026task1", "submission-for-task2.csv")
    C.write_submission(pred, out)
    C.write_submission(pred, os.path.join(C.OUT, "submission-for-task2.csv"))


if __name__ == "__main__":
    main()
