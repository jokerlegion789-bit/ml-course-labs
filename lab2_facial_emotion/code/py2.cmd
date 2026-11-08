@echo off
REM Unified runner for Task-2 experiments: py2.cmd <script> [args...]
set PYTHONUTF8=1
set PYTHONPATH=D:\deepseek\_myscipy\lib;D:\deepseek\pylibs
set MPLCONFIGDIR=D:\deepseek\.mplconfig
set TORCH_HOME=D:\deepseek\.torchcache
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set PYTHONWARNINGS=ignore
D:\Python314\python.exe %*
exit /b %ERRORLEVEL%
