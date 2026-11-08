# -*- coding: utf-8 -*-
"""task2 训练：FER 6 类 CNN。内存数据 + 张量级增广 + OneCycle + mixup + 标签平滑 + SWA + TTA。

所有模型共用同一套【分层 10% 留出集】（种子固定），因此各模型的留出概率
可以直接拼接比较与集成，性能估计不会因划分不同而失真。

用法:
  python train2.py --net vgg_fer --epochs 60 --tag v2_vgg_s42
输出（result/）:
  model_<tag>.pt      最优权重
  test_<tag>.npy      测试集概率 (7065,6)，含 TTA
  hold_<tag>.npy      留出集概率 (2828,6)，含 TTA
  meta_<tag>.json     训练记录
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common2 as C
import nets as N


def stratified_holdout(y, frac=0.10, seed=C.SEED):
    rng = np.random.default_rng(seed)
    ho = []
    for c in range(C.NUM_CLASSES):
        pool = np.where(y == c)[0]
        p = rng.permutation(pool)
        ho.append(p[:int(round(len(p) * frac))])
    ho = np.concatenate(ho)
    mask = np.ones(len(y), bool)
    mask[ho] = False
    return np.where(mask)[0], np.sort(ho)


@torch.no_grad()
def predict(model, X, bs=512, tta=True):
    """X: uint8 (N,48,48)；返回 (N,6) 概率。TTA = 原图 + 水平翻转。"""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        xb = N.normalize_np(X[i:i + bs]).to(dev)
        p = F.softmax(model(xb), 1)
        if tta:
            p = (p + F.softmax(model(xb.flip(-1)), 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="vgg_fer")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--swa-frac", type=float, default=0.25)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--erase", type=float, default=0.25, help="随机擦除比例")
    ap.add_argument("--erase-max", type=int, default=15, help="擦除块最大边长(px)")
    ap.add_argument("--rot", type=float, default=12.0, help="最大旋转角(度)")
    ap.add_argument("--shift", type=float, default=0.08, help="最大平移比例")
    ap.add_argument("--scale-lo", type=float, default=0.90)
    ap.add_argument("--scale-hi", type=float, default=1.10)
    ap.add_argument("--ls", type=float, default=0.1, help="标签平滑")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    C.seed_everything(a.seed)
    dev = torch.device("cpu")

    X, y, Xt, te_names = C.build_cache()
    tri, hoi = stratified_holdout(y)
    print(f"[{a.tag}] {a.net} width={a.width} 训练 {len(tri)} / 留出 {len(hoi)} / "
          f"测试 {len(Xt)}   epochs={a.epochs} batch={a.batch} lr={a.lr} seed={a.seed}",
          flush=True)

    kw = {"p_drop": a.drop}
    if a.net == "vgg_fer":
        kw["width"] = a.width
    model = N.build(a.net, C.NUM_CLASSES, **kw).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"    参数量 {n_par/1e3:.1f}K", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = (len(tri) // a.batch) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.2, div_factor=10,
                                                final_div_factor=50)
    crit = nn.CrossEntropyLoss(label_smoothing=a.ls)
    aug_kw = dict(max_rot=a.rot, max_shift=a.shift, scale=(a.scale_lo, a.scale_hi),
                  erase_p=a.erase)
    if a.erase > 0:
        import nets as _n
        _n.ERASE_MAX = a.erase_max
    swa_start = int(a.epochs * (1 - a.swa_frac))
    swa, swa_n = None, 0
    best = {"acc": -1.0, "state": None, "ep": -1}
    hist = []

    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tri)
        t0, tot, cor, lsum = time.time(), 0, 0, 0.0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            xb = N.normalize_np(X[idx]).to(dev)
            yb = torch.from_numpy(y[idx]).to(dev)
            xb = N.augment(xb, train=True, **aug_kw)
            if a.mixup > 0 and ep < a.epochs * 0.6 and np.random.rand() < a.mixup:
                lam = float(np.random.beta(0.2, 0.2))
                pi = torch.randperm(xb.size(0), device=dev)
                xb = lam * xb + (1 - lam) * xb[pi]
                out = model(xb)
                loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[pi])
            else:
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            lsum += float(loss) * xb.size(0)
            cor += int((out.argmax(1) == yb).sum())
            tot += xb.size(0)
        # 留出集评估（不带 TTA，快）
        hp = predict(model, X[hoi], tta=False)
        hacc = float((hp.argmax(1) == y[hoi]).mean())
        hist.append({"ep": ep + 1, "loss": lsum / tot, "train": cor / tot,
                     "holdout": hacc, "sec": round(time.time() - t0, 1)})
        print(f"    ep{ep+1:3d}/{a.epochs} loss={lsum/tot:.4f} train={cor/tot:.4f} "
              f"holdout={hacc:.4f} {time.time()-t0:.0f}s", flush=True)
        if hacc > best["acc"]:
            best = {"acc": hacc, "ep": ep + 1,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if ep + 1 > swa_start:
            cur = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
            if swa is None:
                swa = cur
            else:
                for k in swa:
                    if swa[k].dtype.is_floating_point:
                        swa[k].mul_(swa_n / (swa_n + 1)).add_(cur[k], alpha=1.0 / (swa_n + 1))
                    else:
                        swa[k] = cur[k]
            swa_n += 1

    res = {"tag": a.tag, "net": a.net, "width": a.width, "epochs": a.epochs, "lr": a.lr,
           "batch": a.batch, "drop": a.drop, "seed": a.seed, "params": int(n_par),
           "best_holdout": best["acc"], "best_ep": best["ep"], "history": hist}

    # 候选权重：最优留出权重 与 SWA 权重，取留出集上更好者
    cands = {"best": best["state"]}
    if swa is not None:
        cands["swa"] = swa
    scored = []
    for name, st in cands.items():
        model.load_state_dict(st)
        hp = predict(model, X[hoi], tta=True)
        acc = float((hp.argmax(1) == y[hoi]).mean())
        scored.append((name, acc, st, hp))
        print(f"    候选权重 {name}: 留出集(含TTA) = {acc:.4f}", flush=True)
    scored.sort(key=lambda t: -t[1])
    pick, acc_pick, st, hp = scored[0]
    model.load_state_dict(st)
    torch.save(st, os.path.join(C.OUT, f"model_{a.tag}.pt"))
    np.save(os.path.join(C.OUT, f"hold_{a.tag}.npy"), hp.astype(np.float32))

    p_te = predict(model, Xt, tta=True)
    np.save(os.path.join(C.OUT, f"test_{a.tag}.npy"), p_te.astype(np.float32))
    res.update({"picked": pick, "holdout_acc_tta": acc_pick,
                "candidates": {n: v for n, v, _, _ in
                               [(x[0], x[1], None, None) for x in scored]}})
    C.save_json(res, f"meta_{a.tag}.json")
    print(f"[{a.tag}] 完成。采用 {pick} 权重，留出集准确率(含TTA) = {acc_pick:.4f}", flush=True)


if __name__ == "__main__":
    main()
