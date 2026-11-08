# -*- coding: utf-8 -*-
"""全量高分辨率微调：用全部 500 张训练图（不再留出验证折），并输出带 TTA 的测试集概率。

为什么要这个脚本：
  1) 交叉验证阶段每折只用了 400 张，全量训练可多利用 25% 的数据；
  2) 5 折 × 5 模型的 CV 组合太贵，赶时间时改为「一次性全量训练 + 多模型集成」；
  3) 分辨率从 224 提到 288/320：Black-grass 与 Loose Silk-bent 的区分依赖幼苗细节纹理，
     这通常是细粒度任务上最有效的单一杠杆。

做法：OneCycle + mixup + 标签平滑 + 强增广，末期做权重平均（SWA），
推理时用 8 视图 TTA（双尺度 × 整图/中心裁剪 × 原图/水平翻转），
结果存 result/testfull_<tag>.npy，供 ensemble_full.py 集成。

用法:
  python finetune_full.py --backbone resnet18 --size 320 --epochs 32 --tag full_r18_320
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common as C
import models as M
from finetune import TrainDS, build_model, predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--swa-frac", type=float, default=0.25, help="最后多少比例轮次做权重平均")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    C.seed_everything(a.seed)
    dev = torch.device("cpu")

    train = C.list_train()
    tr_paths = [p for p, _ in train]
    y = np.array([l for _, l in train])
    te_paths = C.list_test()
    print(f"[{a.tag}] 全量训练 {len(tr_paths)} 张 @ {a.size}px, {a.epochs} 轮, "
          f"batch={a.batch}, lr={a.lr}, seed={a.seed}, threads={a.threads}", flush=True)

    model = build_model(a.backbone).to(dev)
    ds = TrainDS(tr_paths, y, size=a.size, train=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)

    head_prefix = ("fc", "classifier")
    head_p, base_p = [], []
    for n, p in model.named_parameters():
        (head_p if n.startswith(head_prefix) else base_p).append(p)
    opt = torch.optim.AdamW([{"params": base_p, "lr": a.lr * 0.1},
                             {"params": head_p, "lr": a.lr}], weight_decay=a.wd)
    steps = max(1, len(dl)) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr * 0.1, a.lr], total_steps=steps,
        pct_start=0.25, div_factor=10, final_div_factor=100)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    swa_start = int(a.epochs * (1 - a.swa_frac))
    swa_state, swa_n = None, 0
    for ep in range(a.epochs):
        model.train()
        t0, tot, correct, lsum = time.time(), 0, 0, 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            if ep < a.epochs * 0.6 and np.random.rand() < 0.5:
                lam = float(np.random.beta(0.2, 0.2))
                idx = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx]
                out = model(xb)
                loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[idx])
            else:
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            lsum += float(loss) * xb.size(0)
            correct += int((out.argmax(1) == yb).sum())
            tot += xb.size(0)
        if ep + 1 > swa_start:
            cur = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cur
            else:
                for k in swa_state:
                    if swa_state[k].dtype.is_floating_point:
                        swa_state[k].mul_(swa_n / (swa_n + 1)).add_(cur[k], alpha=1.0 / (swa_n + 1))
                    else:
                        swa_state[k] = cur[k]
            swa_n += 1
        print(f"    ep{ep+1:3d}/{a.epochs} loss={lsum/tot:.4f} train={correct/tot:.4f} "
              f"{time.time()-t0:.0f}s" + ("  [SWA]" if ep + 1 > swa_start else ""), flush=True)

    # 用 SWA 权重（若可用）做最终推理
    if swa_state is not None:
        model.load_state_dict(swa_state)
        print(f"    使用 SWA 权重（平均了最后 {swa_n} 轮）", flush=True)
    torch.save(model.state_dict(), os.path.join(C.OUT, f"model_full_{a.tag}.pt"))

    p_te = predict_probs(model, te_paths, size=a.size, use_tta=True)
    np.save(os.path.join(C.OUT, f"testfull_{a.tag}.npy"), p_te.astype(np.float32))
    # 训练集预测（用于伪标签轮次的置信度诊断）
    p_tr = predict_probs(model, tr_paths, size=a.size, use_tta=False)
    np.save(os.path.join(C.OUT, f"trainfull_{a.tag}.npy"), p_tr.astype(np.float32))
    tr_acc = float((p_tr.argmax(1) == y).mean())
    print(f"[{a.tag}] 完成。训练集拟合准确率={tr_acc:.4f}  测试概率已存 testfull_{a.tag}.npy",
          flush=True)
    C.save_json({"tag": a.tag, "backbone": a.backbone, "size": a.size, "epochs": a.epochs,
                 "lr": a.lr, "batch": a.batch, "seed": a.seed, "swa_epochs": swa_n,
                 "train_acc": tr_acc, "n_train": len(tr_paths)},
                f"full_{a.tag}.json")


if __name__ == "__main__":
    main()
