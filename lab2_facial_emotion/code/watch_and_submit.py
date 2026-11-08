# -*- coding: utf-8 -*-
"""值守脚本：等待训练完成 -> 自动集成 -> 写提交 -> 独立核验。

阶段：
  1) 等 vgg 两个模型出结果（最快的一对）-> 立刻生成第一版 submission.csv
  2) 等全部模型出结果（或超时）-> 重新集成生成最终 submission.csv
  3) 用保存的权重重新推理，逐张核对 ID↔预测配对

用法: python watch_and_submit.py [--max-wait-min 240]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"D:\Python314\python.exe"
OUT = r"D:\deepseek\2026task2\result"
ROOT = r"D:\deepseek\2026task2"


def env():
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    e["PYTHONPATH"] = r"D:\deepseek\_myscipy\lib;D:\deepseek\pylibs"
    e["MPLCONFIGDIR"] = r"D:\deepseek\.mplconfig"
    e["OMP_NUM_THREADS"] = "8"
    e["MKL_NUM_THREADS"] = "8"
    e["PYTHONWARNINGS"] = "ignore"
    return e


def run(script, args=(), timeout=1800):
    cmd = [PY, os.path.join(HERE, script)] + list(args)
    print(f"\n>>> {' '.join([script] + list(args))}", flush=True)
    r = subprocess.run(cmd, env=env(), cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    print(r.stdout or "", flush=True)
    if r.returncode != 0:
        print("--- stderr ---\n", (r.stderr or "")[-3000:], flush=True)
    return r.returncode


def have(tag):
    return (os.path.exists(os.path.join(OUT, f"test_{tag}.npy"))
            and os.path.exists(os.path.join(OUT, f"hold_{tag}.npy")))


def wait_for(tags, label, max_wait_s):
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        done = [t for t in tags if have(t)]
        if len(done) == len(tags):
            print(f"[{label}] 全部就绪: {done}  ({time.time()-t0:.0f}s)", flush=True)
            return True
        time.sleep(30)
        el = time.time() - t0
        if int(el) % 300 < 30:
            print(f"[{label}] 等待中 {len(done)}/{len(tags)} 已就绪 {done}  ({el/60:.1f} min)",
                  flush=True)
    print(f"[{label}] 超时，已就绪 {[t for t in tags if have(t)]}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wait-min", type=float, default=260)
    a = ap.parse_args()
    MAXW = a.max_wait_min * 60

    fast = ["v2_vgg_s42", "v2_vgg_s123"]
    allt = ["v2_vgg_s42", "v2_vgg_s123", "v2_vggw_s7", "v2_res_s42", "v2_res_s123", "v2_wide_s42"]

    # 阶段 1：最快的两个模型一出就交第一版
    wait_for(fast, "快速组", MAXW)
    if any(have(t) for t in fast):
        print("\n===== 阶段 1：生成第一版提交 =====", flush=True)
        run("ensemble2.py", ["--prefix", "v2_"])
        run("verify2.py", ["--n", "10"])

    # 阶段 2：全部模型到齐后重新集成
    wait_for(allt, "全部模型", MAXW)
    print("\n===== 阶段 2：生成最终提交 =====", flush=True)
    run("ensemble2.py", ["--prefix", "v2_"])
    run("verify2.py", ["--n", "12"])
    print("\n===== 值守结束 =====", flush=True)


if __name__ == "__main__":
    main()
