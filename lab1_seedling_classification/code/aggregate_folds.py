# -*- coding: utf-8 -*-
"""把分折训练产物聚合成整份 OOF / 测试集概率与 CV 记录。

背景：finetune.py 单折运行时只写 `oof_<tag>_fold<k>.npy`、`oofidx_<tag>_fold<k>.npy`
与 `test_<tag>_fold<k>.npy`；`--all-folds` 才会汇总。本脚本在折并行跑完后
统一汇总，产物与 `--all-folds` 完全一致：

  oof_<tag>.npy    训练集 OOF 概率（缺失折填均匀分布，并在记录中标注）
  test_<tag>.npy   各折测试集概率的平均
  cv_<tag>.json    OOF 准确率、各折验证准确率、覆盖折列表

用法: python aggregate_folds.py [tag ...]（缺省聚合 result/ 下所有 ft_* 标签）
"""
import os
import sys
import json

import numpy as np

import common as C


def aggregate(tag):
    y = np.array([l for _, l in C.list_train()])
    n, K = len(y), C.NUM_CLASSES
    oof = np.zeros((n, K), dtype=np.float32)
    covered = np.zeros(n, dtype=bool)
    tres, folds = [], []
    per_fold = {}
    for k in range(5):
        idxp = os.path.join(C.OUT, f"oofidx_{tag}_fold{k}.npy")
        oofp = os.path.join(C.OUT, f"oof_{tag}_fold{k}.npy")
        tep = os.path.join(C.OUT, f"test_{tag}_fold{k}.npy")
        if not (os.path.exists(idxp) and os.path.exists(oofp) and os.path.exists(tep)):
            print(f"  [缺] {tag} fold{k} 产物不全，跳过")
            continue
        idx = np.load(idxp)
        oof[idx] = np.load(oofp)
        covered[idx] = True
        folds.append(k)
        tres.append(np.load(tep))
    if not folds:
        print(f"  [跳过] {tag} 没有任何完整折")
        return None
    # 未覆盖样本保持全零（不填均匀分布）：下游用 mask = (行和 > 0) 识别真实覆盖，
    # 填充任何常数都会让未覆盖样本被误当成“已预测”。
    np.save(os.path.join(C.OUT, f"oof_{tag}.npy"), oof)
    te = np.mean(tres, axis=0).astype(np.float32)
    np.save(os.path.join(C.OUT, f"test_{tag}.npy"), te)

    acc_cov = float((oof[covered].argmax(1) == y[covered]).mean())
    acc_all = float((oof.argmax(1) == y).mean())
    for k in folds:
        idxp = os.path.join(C.OUT, f"oofidx_{tag}_fold{k}.npy")
        idx = np.load(idxp)
        per_fold[str(k)] = {"val_acc": float((oof[idx].argmax(1) == y[idx]).mean())}
    # 从分折记录里回收骨干等信息
    backbone = None
    for k in folds:
        fj = os.path.join(C.OUT, f"cv_{tag}_fold{k}.json")
        if os.path.exists(fj):
            try:
                with open(fj, encoding="utf-8") as f:
                    backbone = json.load(f).get("backbone", backbone)
            except Exception:
                pass
            break
    rec = {"tag": tag, "backbone": backbone or tag, "folds": folds,
           "n_folds": len(folds), "n_covered": int(covered.sum()),
           "fold_val_mean": float(np.mean([v["val_acc"] for v in per_fold.values()])),
           "oof_acc": acc_cov, "oof_acc_including_uncovered_as_uniform": acc_all,
           "per_fold": per_fold, "note": "由 aggregate_folds.py 汇总分折产物"}
    C.save_json(rec, f"cv_{tag}.json")
    print(f"  [聚合] {tag}: 折 {folds}  覆盖 {covered.sum()}/{n}  "
          f"各折验证均值={rec['fold_val_mean']:.4f}  OOF={acc_cov:.4f}（仅覆盖样本）")
    return rec


def main():
    args = sys.argv[1:]
    if not args:
        tags = sorted({f[4:-4] for f in os.listdir(C.OUT)
                       if f.startswith("oof_ft_") and f.endswith(".npy") and "_fold" not in f})
        # 也纳入只有分折产物的标签
        extra = sorted({f.split("_fold")[0][4:] for f in os.listdir(C.OUT)
                        if f.startswith("oof_ft_") and "_fold" in f and f.endswith(".npy")})
        tags = sorted(set(tags) | set(extra))
    else:
        tags = args
    print(f"待聚合标签: {tags}")
    for t in tags:
        aggregate(t)


if __name__ == "__main__":
    main()
