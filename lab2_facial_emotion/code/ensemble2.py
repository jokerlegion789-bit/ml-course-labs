# -*- coding: utf-8 -*-
"""task2 集成与提交生成。

所有模型共用同一套分层留出集，因此 hold_<tag>.npy 可以直接平均比较。
集成选择：在留出集上比较「单模型最优 / 全部等权 / 贪心挑选」，
用嵌套验证（半样本 A/B）避免在同一份留出集上选权重造成乐观偏差。

提交必须【按文件名】配对（模板行顺序 != sorted 文件名顺序）。

用法:
  python ensemble2.py [--encoding alpha|fer] [--prefix v2_]
"""
import argparse
import csv
import os
import sys

import numpy as np

import common2 as C
from train2 import stratified_holdout


def load_members(prefix="v2_"):
    """返回 {tag: (hold_probs, test_probs)}。"""
    tags = []
    for f in sorted(os.listdir(C.OUT)):
        if f.startswith("test_") and f.endswith(".npy"):
            tag = f[5:-4]
            if tag.startswith(prefix) and os.path.exists(os.path.join(C.OUT, f"hold_{tag}.npy")):
                tags.append(tag)
    m = {}
    for t in tags:
        m[t] = (np.load(os.path.join(C.OUT, f"hold_{t}.npy")),
                np.load(os.path.join(C.OUT, f"test_{t}.npy")))
    return m


def acc(P, y):
    return float((P.argmax(1) == y).mean())


def nested_validate(members, y, reps=200, seed=C.SEED):
    """半样本 A/B 嵌套验证：A 上选方法，B 上评估。"""
    tags = list(members)
    rng = np.random.default_rng(seed)
    by_cls = [np.where(y == c)[0] for c in range(C.NUM_CLASSES)]
    res = {k: [] for k in ("best_single", "mean_top3", "mean_all", "greedy")}

    def greedy(names, ix):
        cur = None; cnt = {t: 0 for t in names}; tot = 0; best = -1
        for _ in range(12):
            pick, pa = None, best
            for t in names:
                cand = members[t][0][ix] if cur is None else (cur * tot + members[t][0][ix]) / (tot + 1)
                a = acc(cand, y[ix])
                if a > pa + 1e-12:
                    pa, pick = a, t
            if pick is None:
                break
            cur = members[pick][0][ix] if cur is None else (cur * tot + members[pick][0][ix]) / (tot + 1)
            cnt[pick] += 1; tot += 1; best = pa
        w = np.array([cnt[t] / max(1, tot) for t in names])
        return names, w

    for _ in range(reps):
        A, B = [], []
        for pool in by_cls:
            p = rng.permutation(pool); h = len(p) // 2
            A.extend(p[:h]); B.extend(p[h:])
        A = np.array(A); B = np.array(B)
        accA = {t: acc(members[t][0][A], y[A]) for t in tags}
        ranked = sorted(tags, key=lambda t: -accA[t])
        cands = {
            "best_single": ([ranked[0]], np.array([1.0])),
            "mean_all": (tags, np.ones(len(tags)) / len(tags)),
        }
        top3 = ranked[:min(3, len(tags))]
        cands["mean_top3"] = (top3, np.ones(len(top3)) / len(top3))
        cands["greedy"] = greedy(tags, A)
        for k, (nm, w) in cands.items():
            P = sum(wi * members[t][0][B] for wi, t in zip(w, nm))
            res[k].append(acc(P, y[B]))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v))} for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default=None, choices=list(C.ENCODINGS))
    ap.add_argument("--prefix", default="v2_")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    members = load_members(a.prefix)
    if not members:
        print("没有可用的模型结果"); sys.exit(1)
    _, hoi = stratified_holdout(np.array([l for _, l in C.list_train()]))
    # 需要训练标签顺序：直接从缓存取
    y_all = np.load(os.path.join(C.CACHE, "train_label.npy"))
    y = y_all[hoi]

    print(f"可用模型 {len(members)} 个，留出集 {len(y)} 张\n")
    print(f"{'模型':<18}{'留出集准确率':>12}")
    for t in sorted(members, key=lambda k: -acc(members[k][0], y)):
        print(f"{t:<18}{acc(members[t][0], y):>12.4f}")

    val = nested_validate(members, y, a.reps)
    print(f"\n嵌套验证（{a.reps} 次半样本划分，B 半评估）:")
    print(f"{'方法':<16}{'平均':>9}{'标准差':>9}{'最差':>9}")
    for k, v in sorted(val.items(), key=lambda x: -x[1]["mean"]):
        print(f"{k:<16}{v['mean']:>9.4f}{v['std']:>9.4f}{v['min']:>9.4f}")
    pick = max(val, key=lambda k: val[k]["mean"])
    print(f"  >> 嵌套验证选定: {pick}")

    tags = sorted(members, key=lambda k: -acc(members[k][0], y))
    if pick == "best_single":
        names, w = [tags[0]], np.array([1.0])
    elif pick == "mean_top3":
        sel = tags[:3]; names, w = sel, np.ones(3) / 3
    elif pick == "greedy":
        # 在全留出集上重跑贪心
        names = tags
        cur = None; cnt = {t: 0 for t in names}; tot = 0; best = -1
        for _ in range(12):
            pk, pa = None, best
            for t in names:
                cand = members[t][0] if cur is None else (cur * tot + members[t][0]) / (tot + 1)
                if acc(cand, y) > pa + 1e-12:
                    pa, pk = acc(cand, y), t
            if pk is None:
                break
            cur = members[pk][0] if cur is None else (cur * tot + members[pk][0]) / (tot + 1)
            cnt[pk] += 1; tot += 1; best = pa
        names = [t for t in names if cnt[t] > 0]
        w = np.array([cnt[t] / tot for t in names])
    else:
        names, w = tags, np.ones(len(tags)) / len(tags)

    P_hold = sum(wi * members[t][0] for wi, t in zip(w, names))
    P_test = sum(wi * members[t][1] for wi, t in zip(w, names))
    print(f"\n集成成员: {[(t, round(float(x), 3)) for t, x in zip(names, w)]}")
    print(f"集成留出集准确率 = {acc(P_hold, y):.4f}")

    idx = P_test.argmax(1)
    dist = {C.CLASSES[i]: int((idx == i).sum()) for i in range(C.NUM_CLASSES)}
    print("测试集预测分布:", dist)

    # 按文件名配对（关键！）
    te_names = [os.path.basename(p) for p in C.list_test()]
    _, tpl_ids = C.read_template()
    if len(te_names) != len(idx):
        raise ValueError("测试文件数与预测行数不一致")
    if set(te_names) != set(tpl_ids):
        raise ValueError("测试文件名集合与模板 ID 集合不一致")
    pred = {nm: C.CLASSES[int(idx[j])] for j, nm in enumerate(te_names)}

    enc = a.encoding or C.ENCODING
    out = a.out or os.path.join(C.ROOT, "submission.csv")
    C.write_submission(pred, out, encoding=enc)

    C.save_json({"members": {t: acc(members[t][0], y) for t in members},
                 "nested_validation": val, "chosen": pick,
                 "ensemble": names, "weights": [float(x) for x in w],
                 "holdout_acc": acc(P_hold, y), "encoding": enc,
                 "pred_distribution": dist,
                 "confidence_mean": float(P_test.max(1).mean()),
                 "n_test": int(len(idx))}, "ensemble2_report.json")
    np.save(os.path.join(C.OUT, "final_test_proba2.npy"), P_test.astype(np.float32))


if __name__ == "__main__":
    main()
