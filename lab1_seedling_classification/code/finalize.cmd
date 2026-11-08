@echo off
REM Full final pipeline: aggregate folds -> ensemble -> evaluate -> prior analysis
REM -> damped prior correction -> figures -> report (md + docx)
REM Run as:  finalize.cmd
setlocal
set HERE=%~dp0
call "%HERE%py.cmd" "%HERE%aggregate_folds.py"
if errorlevel 1 echo [warn] aggregate_folds returned non-zero
call "%HERE%py.cmd" "%HERE%validate_ensemble.py" --reps 100
if errorlevel 1 echo [warn] validate_ensemble returned non-zero
call "%HERE%py.cmd" "%HERE%ensemble.py"
if errorlevel 1 goto :fail
call "%HERE%py.cmd" "%HERE%eval_report.py"
call "%HERE%py.cmd" "%HERE%prior_test.py"
call "%HERE%py.cmd" "%HERE%prior_sim.py"
call "%HERE%py.cmd" "%HERE%prior_shift.py" --apply
call "%HERE%py.cmd" "%HERE%viz.py"
call "%HERE%py.cmd" "%HERE%make_report.py"
call "%HERE%py.cmd" "%HERE%make_docx.py"
call "%HERE%py.cmd" "%HERE%make_delivery.py"
echo.
echo ==== DONE ====
exit /b 0
:fail
echo [error] ensemble.py failed
exit /b 1
