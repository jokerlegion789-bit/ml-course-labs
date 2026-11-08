# -*- coding: utf-8 -*-
"""更强 TTA：把测试/留出推理从「原图+翻转」2 视图升级为 10 视图（翻转 × 微小平移）。

先在留出集上验证是否真的更好，只有更好才覆盖 test_<tag>.npy，
避免为了不确定的增益破坏已有结果。

用法: python retest_tta.py [--tags v2_vgg_s42 ...] [--apply]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import common2 as C
import nets as N
from train2 import stratified_holdout


def tta10(x, shift=2):
    """x: (B,1,48,48) 已归一化。返回 10 个视图的列表。"""
    views = [x, x.flip(-1)]
    for dy, dx in ((shift, 0), (-shift, 0), (0, shift), (0, -shift)):
        views.append(torch.roll(x, shifts=(dy, dx), dims=(2, 3)))
        views.append(torch.roll(x.flip(-1), shifts=(dy, dx), dims=(2, 3)))
    return views


@torch.no_grad()
def predict10(model, X, bs=512, shift=2):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        xb = N.normalize_np(X[i:i + bs]).to(dev)
        p = None
        for v in tta10(xb, shift):
            q = F.softmax(model(v), 1)
            p = q if p is None else p + q
        out.append((p / 10.0).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def predict2(model, X, bs=512):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        xb = N.normalize_np(X[i:i + bs]).to(dev)
        p = (F.softmax(model(xb), 1) + F.softmax(model(xb.flip(-1)), 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


def load_model(tag):
    meta = C.load_json(f"meta_{tag}.json", {}) or {}
    net = meta.get("net", "vgg_fer")
    kw = {"p_drop": meta.get("drop", 0.4)}
    if net == "vgg_fer":
        kw["width"] = meta.get("width", 1.0)
    m = N.build(net, C.NUM_CLASSES, **kw)
    m.load_state_dict(torch.load(os.path.join(C.OUT, f"model_{tag}.pt"), map_location="cpu"))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--apply", action="store_true", help="留出集变好时才覆盖 test_*.npy")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    X, y, Xt, te_names = C.build_cache()
    y_all = np.load(os.path.join(C.CACHE, "train_label.npy"))
    _, hoi = stratified_holdout(y_all)
    yh = y_all[hoi]

    tags = a.tags or [f[6:-3] for f in sorted(os.listdir(C.OUT))
                      if f.startswith("model_v") and f.endswith(".pt")]
    print(f"对 {len(tags)} 个模型比较 TTA 2 视图 vs 10 视图\n")
    print(f"{'模型':<18}{'2视图':>9}{'10视图':>9}{'增益':>9}{'决定':>8}")
    summary = {}
    for t in tags:
        m = load_model(t)
        p2 = predict2(m, X[hoi])
        p10 = predict10(m, X[hoi])
        a2 = float((p2.argmax(1) == yh).mean())
        a10 = float((p10.argmax(1) == yh).mean())
        use10 = a10 > a2
        summary[t] = {"tta2": a2, "tta10": a10, "gain": a10 - a2, "use10": use10}
        print(f"{t:<18}{a2:>9.4f}{a10:>9.4f}{a10-a2:>+9.4f}{'10视图' if use10 else '保持2视图':>8}",
              flush=True)
        if a.apply and use10:
            p_te = predict10(m, Xt)
            np.save(os.path.join(C.OUT, f"test_{t}.npy"), p_te.astype(np.float32))
            hp = predict10(m, X[hoi])
            np.save(os.path.join(C.OUT, f"hold_{t}.npy"), hp.astype(np.float32))
    g = np.mean([v["gain"] for v in summary.values()]) if summary else 0.0
    print(f"\n平均增益 {g:+.4f}")
    C.save_json(summary, "tta_compare.json")


if __name__ == "__main__":
    main()
