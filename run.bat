@echo off
cd /d "%~dp0"
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8765"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
