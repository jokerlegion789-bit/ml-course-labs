# -*- coding: utf-8 -*-
"""等当前 v2/v3 全部跑完后，自动启动主力轮次（更大模型 + 最优增广 + 更长训练）。

增广对比实验（v3，同架构同种子、各 50 轮）最终结论：
     A 无mixup+擦除0.25+强仿射(rot12/shift0.08)  0.6749  <- 最优
     B 无mixup+擦除0.10+强仿射                    0.6678
     D mixup0.2+温和仿射                          0.6558
     C 无mixup+温和仿射                           0.6487  <- 最差
因此主力轮次采用方案 A：**去掉 mixup**，保留较强仿射与随机擦除。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"D:\Python314\python.exe"
OUT = r"D:\deepseek\2026task2\result"

WAIT_TAGS = ["v2_vgg_s42", "v2_vgg_s123", "v2_vggw_s7", "v2_res_s42", "v2_res_s123",
             "v2_wide_s42", "v3_A_nomix_e25", "v3_B_nomix_e10", "v3_C_mild",
             "v3_D_mix20_mild"]

# 方案 A（实验最优）：无 mixup + 擦除 0.25 + 强仿射
AUG_A = ["--mixup", "0", "--erase", "0.25", "--rot", "12", "--shift", "0.08",
         "--scale-lo", "0.90", "--scale-hi", "1.10", "--ls", "0.1"]
JOBS = [
    ("v4_w15_s11", ["--net", "vgg_fer", "--width", "1.5", "--seed", "11", "--epochs", "120"]),
    ("v4_w15_s22", ["--net", "vgg_fer", "--width", "1.5", "--seed", "22", "--epochs", "120"]),
    ("v4_w20_s33", ["--net", "vgg_fer", "--width", "2.0", "--seed", "33", "--epochs", "100"]),
    ("v4_res_s44", ["--net", "resnet_fer", "--seed", "44", "--epochs", "120"]),
    ("v4_wide_s55", ["--net", "wide_cnn", "--seed", "55", "--epochs", "110"]),
    ("v4_vgg_s77", ["--net", "vgg_fer", "--width", "1.0", "--seed", "77", "--epochs", "120"]),
]
THREADS = "6"
BATCH = "256"


def have(tag):
    return os.path.exists(os.path.join(OUT, f"test_{tag}.npy"))


def main():
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        left = [t for t in WAIT_TAGS if not have(t)]
        if not left:
            break
        time.sleep(60)
    print(f"等待结束，仍有 {[t for t in WAIT_TAGS if not have(t)]} 未完成，开始启动主力轮次",
          flush=True)

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = r"D:\deepseek\_myscipy\lib;D:\deepseek\pylibs"
    env["MPLCONFIGDIR"] = r"D:\deepseek\.mplconfig"
    env["OMP_NUM_THREADS"] = THREADS
    env["MKL_NUM_THREADS"] = THREADS
    env["PYTHONWARNINGS"] = "ignore"

    for tag, extra in JOBS:
        cmd = [PY, os.path.join(HERE, "train2.py"), "--batch", BATCH, "--threads", THREADS,
               "--tag", tag] + extra + AUG_A
        f = open(os.path.join(OUT, f"log_{tag}.txt"), "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        print(f"  启动 {tag}  pid={p.pid}", flush=True)
        time.sleep(5)
    print("主力轮次已全部启动", flush=True)


if __name__ == "__main__":
    main()
