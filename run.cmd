@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Bundled Python is missing. Restore the .venv directory from this repository. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
exit /b %errorlevel%
