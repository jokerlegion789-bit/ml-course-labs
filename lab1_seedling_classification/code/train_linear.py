# -*- coding: utf-8 -*-
"""冻结特征 + 线性/核分类头：5 折分层交叉验证，给出可信的 CV 准确率。

用法:
  python train_linear.py <backbone> [<backbone> ...] [--size 224]

对每个骨干：
  - 抽取训练集与测试集特征（缓存到 cache/feat_*.npz）
  - 训练集上做 5 折分层 CV，横向对比多个分类头
  - 用最优头在全量训练集重训，预测测试集，概率存 result/prob_<backbone>.npy
"""
import os
import sys
import time
import warnings

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

import common as C
import models as M

warnings.filterwarnings("ignore")
# 沙箱禁止进程间命名管道，joblib 的多进程/线程池均不可用 -> 强制串行后端
os.environ.setdefault("JOBLIB_BACKEND", "sequential")


def candidate_heads():
    """分类头候选：覆盖线性 / RBF 核 / 树模型三类。返回 {名称: (估计器, 是否标准化)}。

    注意：n_jobs 一律为 1 或 None，避免 joblib 创建线程池（沙箱下会因命名管道失败）。
    """
    return {
        "logreg_C0.1": (LogisticRegression(C=0.1, max_iter=3000), True),
        "logreg_C1": (LogisticRegression(C=1.0, max_iter=3000), True),
        "logreg_C10": (LogisticRegression(C=10.0, max_iter=3000), True),
        "logreg_C100": (LogisticRegression(C=100.0, max_iter=5000), True),
        "svc_rbf_C1": (SVC(C=1.0, kernel="rbf", gamma="scale", probability=True), True),
        "svc_rbf_C10": (SVC(C=10.0, kernel="rbf", gamma="scale", probability=True), True),
        "svc_lin_C0.1": (LinearSVC(C=0.1, max_iter=5000), True),
        "svc_lin_C1": (LinearSVC(C=1.0, max_iter=5000), True),
        "rf500": (RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=C.SEED), False),
    }


def build(head_name, need_scale):
    est = candidate_heads()[head_name][0]
    return make_pipeline(StandardScaler(), Normalizer(), est) if need_scale \
        else make_pipeline(Normalizer(), est)


def get_features(name, split, paths, size=224, force=False, batch=16):
    p = os.path.join(C.CACHE, f"feat_{name}_{split}_{size}.npz")
    if os.path.exists(p) and not force:
        d = np.load(p)
        if d["X"].shape[0] == len(paths):
            print(f"    [缓存] {os.path.basename(p)} {d['X'].shape}", flush=True)
            return d["X"]
    print(f"    [抽取] {name} / {split}  ({len(paths)} 张, {size}px)", flush=True)
    t0 = time.time()
    X, _ = M.extract_features(name, paths, size=size, batch=batch)
    np.savez_compressed(p, X=X.astype(np.float32))
    print(f"    [完成] {X.shape}  用时 {time.time()-t0:.0f}s", flush=True)
    return X


def eval_heads(Xtr, y, folds=5, label=""):
    """在给定特征上横向对比全部候选头，返回 (成绩字典, 最优头名)。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)
    scores = {}
    for hname, (_, need_scale) in candidate_heads().items():
        pipe = build(hname, need_scale)
        t0 = time.time()
        with joblib.parallel_backend("sequential"):
            acc = cross_val_score(pipe, Xtr, y, cv=skf, scoring="accuracy", n_jobs=1)
        scores[hname] = {"mean": float(acc.mean()), "std": float(acc.std()),
                         "folds": [float(a) for a in acc], "sec": round(time.time() - t0, 1)}
        print(f"  {label}{hname:16s} CV={acc.mean():.4f} ± {acc.std():.4f}   ({time.time()-t0:.0f}s)",
              flush=True)
    best = max(scores, key=lambda k: scores[k]["mean"])
    print(f"  >> {label}最优头: {best}  CV={scores[best]['mean']:.4f}", flush=True)
    return scores, best


def fit_predict_proba(Xtr, y, Xte, head):
    """用指定头在全量训练集上重训，输出测试集概率。"""
    need_scale = candidate_heads()[head][1]
    pipe = build(head, need_scale)
    pipe.fit(Xtr, y)
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(Xte)
    d = pipe.decision_function(Xte)
    return np.exp(d) / np.exp(d).sum(axis=1, keepdims=True)


def main(backbones, size=224, folds=5):
    train = C.list_train()
    tr_paths = [p for p, _ in train]
    y = np.array([l for _, l in train])
    te_paths = C.list_test()
    print(f"训练 {len(tr_paths)} 张 / 测试 {len(te_paths)} 张 / {C.NUM_CLASSES} 类", flush=True)

    summary = C.load_json("linear_summary.json", {}) or {}
    feats_tr, feats_te = {}, {}
    for name in backbones:
        print(f"\n{'='*66}\n骨干: {name}\n{'='*66}", flush=True)
        Xtr = get_features(name, "train", tr_paths, size=size)
        Xte = get_features(name, "test", te_paths, size=size)
        feats_tr[name], feats_te[name] = Xtr, Xte
        scores, best = eval_heads(Xtr, y, folds, label="")
        proba = fit_predict_proba(Xtr, y, Xte, best)
        np.save(os.path.join(C.OUT, f"prob_{name}.npy"), proba.astype(np.float32))
        print(f"  [预测] prob_{name}.npy {proba.shape}（测试集概率）", flush=True)
        summary[name] = {"feat_dim": int(Xtr.shape[1]), "best_head": best,
                         "best_cv": scores[best]["mean"], "heads": scores, "size": size}
        C.save_json(summary, "linear_summary.json")

    # 多骨干特征拼接后的融合模型：小样本下常优于概率平均
    if len(backbones) > 1:
        print(f"\n{'='*66}\n特征拼接融合: {' + '.join(backbones)}\n{'='*66}", flush=True)
        def zs(X):
            return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
        Xtr = np.concatenate([zs(feats_tr[n]) for n in backbones], axis=1)
        Xte = np.concatenate([zs(feats_te[n]) for n in backbones], axis=1)
        print(f"  拼接维度 {Xtr.shape[1]}", flush=True)
        scores, best = eval_heads(Xtr, y, folds, label="[concat] ")
        proba = fit_predict_proba(Xtr, y, Xte, best)
        np.save(os.path.join(C.OUT, "prob_concat.npy"), proba.astype(np.float32))
        print(f"  [预测] prob_concat.npy {proba.shape}", flush=True)
        summary["__concat__"] = {"members": backbones, "feat_dim": int(Xtr.shape[1]),
                                 "best_head": best, "best_cv": scores[best]["mean"],
                                 "heads": scores, "size": size}
        C.save_json(summary, "linear_summary.json")
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    sz = 224
    if "--size" in args:
        i = args.index("--size")
        sz = int(args[i + 1])
        del args[i:i + 2]
    main(args or ["resnet18"], size=sz)
