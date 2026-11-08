# -*- coding: utf-8 -*-
"""task3 黄斑中心凹定位：经典图像处理检测器 + 训练集误差评估。

解剖先验（由训练集统计得到）：
  - fovea 归一化位置 y 非常稳定（0.529±0.022），x 变化较大（0.494±0.055）
  - fovea 在眼底图中表现为一处**较暗、血管稀疏**的区域（黄斑区）
  - 在绿通道上对比度最好（血管与黄斑都最明显）

据此构造候选检测器并逐一评估，用训练集 80 张的平均欧氏距离（原始像素）打分。
"""
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

D = r"D:\deepseek\task3\detection"
CACHE = r"D:\deepseek\task3\cache"
os.makedirs(CACHE, exist_ok=True)
SW = 512          # 统一缩放到宽 512 处理


def load_gt():
    gt = {}
    with open(os.path.join(D, "fovea_localization_train_GT.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gt[int(r["data"])] = (float(r["Fovea_X"]), float(r["Fovea_Y"]))
    return gt


def prep(path, idx):
    """返回 (小图数组 float32 [0,1] 绿通道, 原尺寸 W, H)。带缓存。"""
    cp = os.path.join(CACHE, f"{idx:04d}.npy")
    im = Image.open(path)
    Wo, Ho = im.size
    if os.path.exists(cp):
        g = np.load(cp)
        return g, Wo, Ho
    w = SW
    h = int(round(Ho * SW / Wo))
    im2 = im.resize((w, h), Image.BILINEAR)
    a = np.asarray(im2, dtype=np.float32)
    g = a[:, :, 1] / 255.0
    np.save(cp, g.astype(np.float16))
    return g, Wo, Ho


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)


def detectors(g):
    """返回 {名称: (x_norm, y_norm)}，坐标为相对图像宽高的比例。"""
    H, W = g.shape
    out = {}
    # 视网膜区域掩膜：亮区 + 最大连通域
    bright = g > np.percentile(g, 40)
    bright = ndi.binary_closing(bright, disk(4))
    lab, n = ndi.label(bright)
    if n > 1:
        sz = ndi.sum(bright, lab, range(1, n + 1))
        retina = lab == (1 + int(np.argmax(sz)))
    else:
        retina = bright
    retina = ndi.binary_fill_holes(retina)
    retina = ndi.binary_erosion(retina, disk(6))

    # 搜索区域约束（来自训练集统计，放宽 3 倍标准差）
    yy, xx = np.mgrid[0:H, 0:W]
    ny, nx = yy / H, xx / W
    roi = retina & (ny > 0.40) & (ny < 0.66) & (nx > 0.28) & (nx < 0.72)
    if roi.sum() < 50:
        roi = (ny > 0.40) & (ny < 0.66) & (nx > 0.28) & (nx < 0.72)

    def argmin_loc(score, mask):
        s = np.where(mask, score, np.inf)
        i = int(np.argmin(s))
        return (i % W) / W, (i // W) / H

    def argmax_loc(score, mask):
        s = np.where(mask, score, -np.inf)
        i = int(np.argmax(s))
        return (i % W) / W, (i // W) / H

    # V1 最暗点（轻度模糊）
    gb = ndi.gaussian_filter(g, 3)
    out["V1_最暗点"] = argmin_loc(gb, roi)

    # V2 去背景梯度后的最暗点（减去大尺度亮度）
    bg = ndi.gaussian_filter(g, 40)
    dk = bg - gb                     # 越大表示相对越暗
    out["V2_去背景最暗"] = argmax_loc(dk, roi)

    # V3 再抑制血管：黑顶帽突出细暗结构，从暗度里减掉
    bh = ndi.grey_closing(g, footprint=disk(6)) - g
    dk3 = dk - 1.0 * ndi.gaussian_filter(bh, 6)
    out["V3_去背景去血管"] = argmax_loc(dk3, roi)

    # V4 fovea = 最暗 + 血管密度最低
    vess = ndi.gaussian_filter(bh, 10)
    dk4 = norm01(dk) - 0.8 * norm01(vess)
    out["V4_暗度+低血管"] = argmax_loc(dk4, roi)

    # V5 用原图上的暗度（不缩到更小），重模糊
    gb2 = ndi.gaussian_filter(g, 8)
    bg2 = ndi.gaussian_filter(g, 60)
    out["V5_大尺度暗度"] = argmax_loc(bg2 - gb2, roi)
    return out


def main():
    gt = load_gt()
    names = None
    res = {}
    rows = []
    for i in range(1, 81):
        g, Wo, Ho = prep(os.path.join(D, "train", f"{i:04d}.jpg"), i)
        ds = detectors(g)
        if names is None:
            names = list(ds)
            res = {k: [] for k in names}
        gx, gy = gt[i]
        rows.append((i, gx, gy))
        for k, (nx, ny) in ds.items():
            px, py = nx * Wo, ny * Ho
            res[k].append(np.hypot(px - gx, py - gy))
    print(f"{'检测器':<20}{'平均距离':>10}{'中位数':>9}{'90分位':>9}{'<10px占比':>10}")
    for k in names:
        d = np.array(res[k])
        print(f"{k:<20}{d.mean():>10.2f}{np.median(d):>9.2f}"
              f"{np.percentile(d,90):>9.2f}{(d<10).mean()*100:>9.1f}%")


if __name__ == "__main__":
    main()
