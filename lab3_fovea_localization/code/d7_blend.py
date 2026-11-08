# -*- coding: utf-8 -*-
"""在 v12 基础上做「最优收缩修正」，生成新提交。

思路：
  两个预测器 A(=v12, 已知误差 σA≈11px) 与 B(我的两阶段模型, 误差待估)。
  若两者误差近似独立，则 (1-w)A + wB 的期望误差为
        sqrt((1-w)^2 σA^2 + w^2 σB^2)
  最优权重 w* = σA^2 / (σA^2 + σB^2)，最优误差 sqrt(σA^2 σB^2 /(σA^2+σB^2))。
  σB 无法直接测（无测试集真值），但可用两者差异 d=B-A 反推：
        E[d^2] = σA^2 + σB^2  =>  σB^2 = E[d^2] - σA^2
  再用鲁棒规则：差异远超中位数的样本判为 B 的离群点，权重置 0（保持 A 不变）。
"""
import csv
import numpy as np

V12 = r"D:\deepseek\submission_v12_mass_kl_s43.csv"
MINE = r"D:\deepseek\task3\submission_task3_twostage.csv"
OUT = r"D:\deepseek\task3\submission_task3_blend.csv"
SIGMA_V12 = 11.0          # 已知线上误差


def load(p):
    d = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["ImageID"]] = float(r["value"])
    return d


def main():
    A = load(V12); B = load(MINE)
    assert set(A) == set(B), "ID 集合不一致"
    ids = sorted(A)
    a = np.array([A[i] for i in ids])
    b = np.array([B[i] for i in ids])
    d = b - a
    n = len(ids) // 2

    # 按图统计每张图的二维距离
    dist = np.hypot(d[0::2], d[1::2])
    e2 = float(np.mean(dist ** 2))
    sA2 = SIGMA_V12 ** 2
    sB2 = max(1.0, e2 - sA2)
    sB = np.sqrt(sB2)
    w = sA2 / (sA2 + sB2)
    opt = np.sqrt(sA2 * sB2 / (sA2 + sB2))
    print(f"样本 {n} 张图")
    print(f"两者差异 |B-A|: 平均 {dist.mean():.1f}px  中位 {np.median(dist):.1f}  "
          f"最大 {dist.max():.1f}")
    print(f"由差异反推 σB ≈ {sB:.1f}px  (已知 σA = {SIGMA_V12}px)")
    print(f"最优收缩权重 w* = {w:.3f}")
    print(f"理论上混合后期望误差 ≈ {opt:.2f}px  (纯 v12 为 {SIGMA_V12:.1f}px)")

    # 鲁棒规则：差异超过 3 倍中位数的图判为 B 的离群点，保持 A
    thr = max(30.0, 3.0 * float(np.median(dist)))
    w_img = np.where(dist > thr, 0.0, w)
    print(f"离群判据阈值 {thr:.1f}px，被判为离群并保持 v12 的图: "
          f"{int((dist > thr).sum())} 张 {[81+k for k in np.where(dist > thr)[0]]}")

    # 写提交（按 sample_submission 的原始顺序）
    with open(r"D:\deepseek\task3\detection\sample_submission.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    order = [r[0] for r in rows[1:]]
    pred = {}
    for k, i in enumerate(ids):
        img = int(i.split("_")[0])
        j = (img - 81)
        pred[i] = (1 - w_img[j]) * A[i] + w_img[j] * B[i]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for i in order:
            wr.writerow([i, f"{pred[i]:.6f}"])
    print(f"\n已生成: {OUT}")
    moved = np.abs(np.array([pred[i] - A[i] for i in ids]))
    print(f"相对 v12 的改动量: 平均 {moved.mean():.2f}px  最大 {moved.max():.2f}px")


if __name__ == "__main__":
    main()
