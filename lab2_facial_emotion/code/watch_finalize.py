# -*- coding: utf-8 -*-
"""值守：等主力轮次全部完成后，自动跑完整条收尾流水线。

与 watch_round4.py 的区别：本脚本用 subprocess.run **同步**执行收尾脚本，
并在子任务全部结束前不退出，因此不会出现「父进程先退出、子进程被一起回收」的问题。

流程：等 v4_* 全部产出 test_*.npy -> retest_tta(10视图, --apply) -> ensemble2 -> verify2
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"D:\Python314\python.exe"
OUT = r"D:\deepseek\2026task2\result"

WAIT_TAGS = ["v4_w15_s11", "v4_w15_s22", "v4_res_s44", "v4_res_s88", "v4_w20_s33"]
MAX_WAIT_H = 5.0


def env():
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    e["PYTHONPATH"] = r"D:\deepseek\_myscipy\lib;D:\deepseek\pylibs"
    e["MPLCONFIGDIR"] = r"D:\deepseek\.mplconfig"
    e["OMP_NUM_THREADS"] = "6"
    e["MKL_NUM_THREADS"] = "6"
    e["PYTHONWARNINGS"] = "ignore"
    return e


def run(script, args=()):
    print(f"\n>>> {script} {' '.join(args)}", flush=True)
    r = subprocess.run([PY, os.path.join(HERE, script)] + list(args),
                       env=env(), cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = r.stdout or ""
    print(out[-6000:], flush=True)
    if r.returncode != 0:
        print(f"[warn] {script} 返回 {r.returncode}\n{(r.stderr or '')[-2000:]}", flush=True)
    return r.returncode


def main():
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_H * 3600:
        done = [t for t in WAIT_TAGS if os.path.exists(os.path.join(OUT, f"test_{t}.npy"))]
        if len(done) == len(WAIT_TAGS):
            print(f"主力轮次全部完成（{len(done)}/{len(WAIT_TAGS)}），用时 {(time.time()-t0)/60:.0f} 分钟",
                  flush=True)
            break
        print(f"  等待中 {len(done)}/{len(WAIT_TAGS)}: {done}  ({(time.time()-t0)/60:.0f} min)",
              flush=True)
        time.sleep(120)

    print("\n===== 收尾：10 视图 TTA 升级 =====", flush=True)
    run("retest_tta.py", ["--apply", "--threads", "6"])
    print("\n===== 收尾：集成 =====", flush=True)
    run("ensemble2.py", ["--prefix", "v", "--encoding", "fer_order"])
    print("\n===== 收尾：独立核验 =====", flush=True)
    run("verify2.py", ["--n", "12", "--encoding", "fer_order"])
    print("\n===== 值守结束 =====", flush=True)


if __name__ == "__main__":
    main()
