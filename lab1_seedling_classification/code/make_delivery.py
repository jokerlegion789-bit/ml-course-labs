# -*- coding: utf-8 -*-
"""整理最终交付包：复制必要文件到一个目录并打包为 zip。

包含：提交 CSV、全部源码、报告（md/docx）、配图、实验记录 JSON 与训练日志。
排除：模型权重（model_*.pt，几十 MB/个）、特征缓存（cache/）、原始数据集。

用法: python make_delivery.py
输出: D:\\deepseek\\2026task1\\提交包\\ 与 D:\\deepseek\\2026task1\\提交包.zip
"""
import os
import shutil
import zipfile

import common as C

ROOT = r"D:\deepseek\2026task1"
DEST = os.path.join(ROOT, "提交包")
ZIP = os.path.join(ROOT, "提交包.zip")

SKIP_EXT = {".pt", ".pth"}
SKIP_DIRS = {"cache", "提交包", "data", "__pycache__"}


def copy_tree(src, dst, skip_root_files=()):
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src)
        target = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target, exist_ok=True)
        for f in files:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            if rel == "." and f in skip_root_files:
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(target, f))
            n += 1
    return n


def main():
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST)

    # 1) 提交文件
    for f in ("submission-for-task2.csv", "README.md"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(DEST, f))
    for f in os.listdir(ROOT):
        if f.endswith(".csv") and "无先验校正" in f:
            shutil.copy2(os.path.join(ROOT, f), os.path.join(DEST, f))

    # 2) 代码 / 报告 / 实验记录
    n1 = copy_tree(os.path.join(ROOT, "code"), os.path.join(DEST, "code"))
    n2 = copy_tree(os.path.join(ROOT, "reports"), os.path.join(DEST, "reports"))
    n3 = copy_tree(os.path.join(ROOT, "result"), os.path.join(DEST, "result"))
    print(f"  代码 {n1} 个文件 / 报告 {n2} 个 / 实验记录 {n3} 个")

    # 3) 打包
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(DEST):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, ROOT))
    size = os.path.getsize(ZIP) / 1e6
    print(f"  [交付包] {ZIP}  ({size:.1f} MB)")

    # 4) 自检清单
    ids, _ = C.read_submission_template()
    import csv
    with open(os.path.join(DEST, "submission-for-task2.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    valid = set(C.CLASSES)
    ok_rows = len(rows) - 1 == len(ids)
    ok_cat = all(r[1] in valid for r in rows[1:])
    ok_ids = all(r[0] == i for r, i in zip(rows[1:], ids))
    print(f"  自检: 行数={len(rows)-1}/{len(ids)} {'OK' if ok_rows else 'FAIL'} | "
          f"类别合法={'OK' if ok_cat else 'FAIL'} | ID 顺序={'OK' if ok_ids else 'FAIL'}")


if __name__ == "__main__":
    main()
