@echo off
REM Unified runner for Task-1 experiments: py.cmd <script> [args...]
REM ASCII only on purpose: cmd.exe mis-parses non-ASCII batch comments.
set PYTHONUTF8=1
set PYTHONPATH=D:\deepseek\_myscipy\lib;D:\deepseek\pylibs
set MPLCONFIGDIR=D:\deepseek\.mplconfig
set TORCH_HOME=D:\deepseek\.torchcache
set HF_HOME=D:\deepseek\.torchcache\hf
set OMP_NUM_THREADS=24
set MKL_NUM_THREADS=24
set PYTHONWARNINGS=ignore
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
D:\Python314\python.exe %*
exit /b %ERRORLEVEL%
