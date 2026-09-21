@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo.
echo BO-ICL local runner
echo ====================
echo.

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo Created local .env from .env.example
    ) else (
        type nul > ".env"
        echo Created local .env
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 was not available. Trying the default Python launcher...
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo Could not create .venv. Install Python 3.11 or newer and rerun this file.
        pause
        exit /b 1
    )
)

echo Checking local Python packages...
if "%~1"=="--setup" goto :install_packages
".venv\Scripts\python.exe" -c "import boicl, scipy, pandas, openpyxl, dotenv, torch, botorch, gpytorch, sklearn" >nul 2>&1
if not errorlevel 1 goto :start_app
:install_packages
".venv\Scripts\python.exe" -m pip install -e ".[gpr]" -r dev-requirements.txt
if errorlevel 1 goto :pip_failed

:start_app
echo API keys are entered only inside the browser app and saved to the local .env file.
echo The synthesis-parameter GP does not need any API key or embeddings.

echo.
echo Starting browser app...
".venv\Scripts\python.exe" -m boicl.local_app
echo.
pause
exit /b 0

:pip_failed
echo.
echo Dependency setup failed. Check your internet connection and Python version, then rerun this file.
pause
exit /b 1
