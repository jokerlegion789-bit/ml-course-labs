# -*- coding: utf-8 -*-
"""启动 task2 的并行集成训练（每个模型一个进程，共用统一留出划分）。"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"D:\Python314\python.exe"
LOG = r"D:\deepseek\2026task2\result"

JOBS = [
    ("v2_vgg_s42",    ["--net", "vgg_fer",    "--seed", "42",   "--width", "1.0"]),
    ("v2_vgg_s123",   ["--net", "vgg_fer",    "--seed", "123",  "--width", "1.0"]),
    ("v2_vggw_s7",    ["--net", "vgg_fer",    "--seed", "7",    "--width", "1.3"]),
    ("v2_res_s42",    ["--net", "resnet_fer", "--seed", "42"]),
    ("v2_res_s123",   ["--net", "resnet_fer", "--seed", "123"]),
    ("v2_wide_s42",   ["--net", "wide_cnn",   "--seed", "42"]),
]

EPOCHS = "80"
THREADS = "4"
BATCH = "256"


def main():
    os.makedirs(LOG, exist_ok=True)
    procs = []
    for tag, extra in JOBS:
        cmd = [PY, os.path.join(HERE, "train2.py"),
               "--epochs", EPOCHS, "--batch", BATCH, "--threads", THREADS, "--tag", tag] + extra
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = r"D:\deepseek\_myscipy\lib;D:\deepseek\pylibs"
        env["MPLCONFIGDIR"] = r"D:\deepseek\.mplconfig"
        env["OMP_NUM_THREADS"] = THREADS
        env["MKL_NUM_THREADS"] = THREADS
        env["PYTHONWARNINGS"] = "ignore"
        f = open(os.path.join(LOG, f"log_{tag}.txt"), "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append((tag, p, f))
        print(f"  启动 {tag}  pid={p.pid}", flush=True)
    print(f"\n共 {len(procs)} 个训练进程并行运行，日志在 {LOG}\\log_*.txt")


if __name__ == "__main__":
    main()
