# -*- coding: utf-8 -*-
"""task2（人脸表情分类 / FER2013 6 类）公共配置与数据加载。

数据：fer_data/fer_data/train/<Emotion>/*.jpg（48x48 灰度），test/*.jpg
提交：submission.csv，两列 ID,Emotion —— ID 是【带 .jpg 的文件名】，
      Emotion 是【数字】，不是类别名！

重要：模板行顺序 != sorted(测试文件名)，写提交时必须按文件名配对，
      绝不能按位置配对（task1 踩过这个坑，成绩掉到随机水平）。
"""
import os
import csv
import numpy as np
from PIL import Image

ROOT = r"D:\deepseek\2026task2"
DATA = os.path.join(ROOT, "fer_data", "fer_data")
TRAIN_DIR = os.path.join(DATA, "train")
TEST_DIR = os.path.join(DATA, "test")
SUB_TEMPLATE = os.path.join(ROOT, "submission.csv")
OUT = os.path.join(ROOT, "result")
CACHE = os.path.join(ROOT, "cache")
REPORTS = os.path.join(ROOT, "reports")
for _d in (OUT, CACHE, REPORTS):
    os.makedirs(_d, exist_ok=True)

# 类别按目录名（字母序）——这是唯一无歧义的信息源
CLASSES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 48
SEED = 42

# 提交中 Emotion 的数字编码 —— 由线上实测分数反推确定（见 infer_encoding2.py）：
#   用字母序提交实测只有 0.417，而按「FER 原始顺序压缩」的期望分 0.399~0.409 与实测吻合，
#   其余候选（字母序 0.666 / 原始 FER 跳号编码 0.291 / 1..6 编码 0.051）均相差 12 分以上。
#   即：平台按 FER 原始情感顺序（Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral）
#   去掉 Disgust 后重新编号 0..5。
ENCODING = "fer_order"
ENCODINGS = {
    # 平台实际使用的编号（默认）
    "fer_order": {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5},
    # 目录名字母序（本地实验用，非平台编号）
    "alpha": {c: i for i, c in enumerate(CLASSES)},
    # 原始 FER2013 跳号编码（保留 Disgust 的位置）
    "fer": {"Angry": 0, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6},
}


def seed_everything(seed=SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_train():
    """返回 [(path, label), ...]，标签按 CLASSES 顺序。"""
    items = []
    for ci, c in enumerate(CLASSES):
        d = os.path.join(TRAIN_DIR, c)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, fn), ci))
    return items


def list_test():
    return [os.path.join(TEST_DIR, f) for f in sorted(os.listdir(TEST_DIR))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def build_cache(force=False):
    """把全部图片解码进内存数组（uint8），彻底消除训练时的 JPEG 解码瓶颈。

    返回 (Xtr, ytr, Xte, te_names)。
    """
    ptr = os.path.join(CACHE, "train_uint8.npy")
    ltr = os.path.join(CACHE, "train_label.npy")
    pte = os.path.join(CACHE, "test_uint8.npy")
    if os.path.exists(ptr) and os.path.exists(ltr) and os.path.exists(pte) and not force:
        return (np.load(ptr), np.load(ltr), np.load(pte),
                [os.path.basename(p) for p in list_test()])

    items = list_train()
    X = np.zeros((len(items), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = np.zeros(len(items), dtype=np.int64)
    for i, (p, lab) in enumerate(items):
        X[i] = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        y[i] = lab
        if (i + 1) % 5000 == 0:
            print(f"    训练集缓存 {i+1}/{len(items)}", flush=True)
    te_paths = list_test()
    Xt = np.zeros((len(te_paths), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, p in enumerate(te_paths):
        Xt[i] = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    np.save(ptr, X)
    np.save(ltr, y)
    np.save(pte, Xt)
    print(f"    缓存完成: train {X.shape}, test {Xt.shape}", flush=True)
    return X, y, Xt, [os.path.basename(p) for p in te_paths]


def read_template():
    """读提交模板，保持原始行顺序。"""
    with open(SUB_TEMPLATE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], [r[0] for r in rows[1:] if r]


def save_json(obj, name):
    import json
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  [落盘] {p}", flush=True)
    return p


def load_json(name, default=None):
    import json
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_submission(pred_by_name, out_path, encoding=None):
    """按文件名写出提交。pred_by_name: {文件名: 类别名}。"""
    enc = ENCODINGS[encoding or ENCODING]
    header, ids = read_template()
    missing = [i for i in ids if i not in pred_by_name]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个预测，例如 {missing[:3]}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in ids:
            w.writerow([i, enc[pred_by_name[i]]])
    print(f"  [提交文件] {out_path}  行数={len(ids)}  编码={encoding or ENCODING}", flush=True)
    return out_path
