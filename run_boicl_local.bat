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

echo Checking Python packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :pip_failed

".venv\Scripts\python.exe" -m pip install -e ".[gpr]" -r dev-requirements.txt
if errorlevel 1 goto :pip_failed

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$envPath = '.env';" ^
  "$content = if (Test-Path $envPath) { Get-Content -Raw $envPath } else { '' };" ^
  "if ($content -notmatch '(?m)^OPENAI_API_KEY=.+') {" ^
  "  $secure = Read-Host 'Paste OPENAI_API_KEY (hidden; press Enter to skip)' -AsSecureString;" ^
  "  $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure);" ^
  "  try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) };" ^
  "  if ($plain) {" ^
  "    if ($content -match '(?m)^OPENAI_API_KEY=.*$') { $content = [regex]::Replace($content, '(?m)^OPENAI_API_KEY=.*$', ('OPENAI_API_KEY=' + $plain)) }" ^
  "    else { $content = $content.TrimEnd() + \"`r`nOPENAI_API_KEY=$plain`r`n\" };" ^
  "    Set-Content -Path $envPath -Value $content -NoNewline;" ^
  "    Write-Host 'Saved OPENAI_API_KEY to local .env';" ^
  "  }" ^
  "}"

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
