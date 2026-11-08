@echo off
REM Final pipeline for task-2 submission:
REM   1) upgrade every model's test/holdout probs to 10-view TTA (only if holdout improves)
REM   2) re-select and re-fit the ensemble over ALL available models
REM   3) independently re-verify the ID <-> prediction pairing
setlocal
set HERE=%~dp0
call "%HERE%py2.cmd" "%HERE%retest_tta.py" --apply --threads 6
call "%HERE%py2.cmd" "%HERE%ensemble2.py" --prefix v --encoding fer_order
call "%HERE%py2.cmd" "%HERE%verify2.py" --n 12 --encoding fer_order
echo.
echo ==== DONE ====
exit /b 0
