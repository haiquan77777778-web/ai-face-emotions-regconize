@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Cai dat moi truong (chi can lam 1 lan, can Internet, mat khoang 10-15 phut) ===

if exist venv\Scripts\python.exe goto have_venv
for %%P in ("py -3.13" "py -V:Astral/CPython3.13.2" "py -3.12") do (
  if not exist venv\Scripts\python.exe (
    %%~P -m venv venv >nul 2>&1
  )
)
:have_venv
if not exist venv\Scripts\python.exe (
  echo.
  echo KHONG tao duoc moi truong. May can cai Python 3.12 hoac 3.13 ^(64-bit^) tu python.org
  echo roi chay lai file nay. Luu y: Python 3.14/3.15 CHUA dung duoc voi TensorFlow.
  pause
  exit /b 1
)

call venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
pip install -r requirements_full.txt
if errorlevel 1 (
  echo.
  echo Cai thu vien bi loi. Chup man hinh thong bao o tren va gui cho Claude.
  pause
  exit /b 1
)
echo.
echo === Kiem tra he thong ===
python check_sync.py
echo.
echo Xong. Chay ung dung bang file run_app.bat
pause
