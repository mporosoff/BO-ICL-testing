@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run run_boicl_local.bat once to set up local dependencies.
    pause
    exit /b 1
)
echo Starting the SYNTHETIC offline demonstration. No model API calls.
".venv\Scripts\python.exe" -m boicl.local_app --demo --port 8766
pause
