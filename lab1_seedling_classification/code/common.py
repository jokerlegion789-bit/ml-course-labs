# -*- coding: utf-8 -*-
"""任务一 公共配置：路径、类别、随机种子、图像读取、提交文件读写。

类别名严格取自训练集目录名，提交时必须原样使用（含空格与连字符）。
"""
import csv
import hashlib
import json
import os
import random
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- 路径
ROOT = r"D:\deepseek\2026task1"
DATA = os.path.join(ROOT, "data", "dataset-for-task2", "dataset-for-task2")
TRAIN_DIR = os.path.join(DATA, "train")
TEST_DIR = os.path.join(DATA, "test")
CSV_TEMPLATE = os.path.join(ROOT, "submission-for-task2.csv")
OUT = os.path.join(ROOT, "result")
CACHE = os.path.join(ROOT, "cache")
REPORTS = os.path.join(ROOT, "reports")
for _d in (OUT, CACHE, REPORTS):
    os.makedirs(_d, exist_ok=True)

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
NUM_CLASSES = len(CLASSES)
SEED = 42


# ---------------------------------------------------------------- 复现
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------------------------------------------- 数据读取
def list_train():
    """返回 [(path, label_index), ...]，按类目录顺序稳定排序。"""
    items = []
    for ci, cname in enumerate(CLASSES):
        d = os.path.join(TRAIN_DIR, cname)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".png"):
                items.append((os.path.join(d, fn), ci))
    return items


def list_test():
    return [os.path.join(TEST_DIR, fn) for fn in sorted(os.listdir(TEST_DIR))
            if fn.lower().endswith(".png")]


def load_rgb(path):
    """统一读为 RGB。RGBA 先合成到白底，避免透明通道污染特征。"""
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def file_md5(path, blk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(blk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fold_indices(y, n_splits=5, seed=SEED):
    """全局统一的折划分：所有模型共用同一套索引，保证 OOF 可拼接、可融合。"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------- 结果落盘
def save_json(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  [落盘] {p}", flush=True)
    return p


def load_json(name, default=None):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_submission_template():
    """读模板 CSV，保持原始行顺序。返回 (ids, header)。"""
    with open(CSV_TEMPLATE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ids = [r[0] for r in rows[1:] if r]
    return ids, header


def write_submission(pred_by_id, out_path):
    """按模板原始顺序写出提交文件。pred_by_id: {ID: 类别名}。"""
    ids, header = read_submission_template()
    missing = [i for i in ids if i not in pred_by_id]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个预测，例如 {missing[:3]}")
    extra = set(pred_by_id) - set(ids)
    if extra:
        print(f"  [警告] 有 {len(extra)} 个多余 ID 被忽略", flush=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in ids:
            w.writerow([i, pred_by_id[i]])
    print(f"  [提交文件] {out_path}  行数={len(ids)}", flush=True)
    return out_path
