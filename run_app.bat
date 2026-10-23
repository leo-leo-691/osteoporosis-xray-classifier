@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    OsteoKnee AI Research Web App Launcher (Backend + Frontend)
echo ======================================================================
echo.

set ROOT_DIR=%~dp0

:: Check Python virtual environment
if exist "%ROOT_DIR%.venv\Scripts\activate.bat" (
    echo [1/2] Found Python virtual environment at .venv
) else (
    echo [WARNING] No .venv found in root directory. Falling back to system python.
)

:: Start Backend (FastAPI on Port 8000)
echo [2/2] Starting FastAPI backend on http://localhost:8000 ...
if exist "%ROOT_DIR%.venv\Scripts\activate.bat" (
    start "OsteoKnee AI - FastAPI Backend" cmd /k "cd /d "%ROOT_DIR%backend" && call "%ROOT_DIR%.venv\Scripts\activate.bat" && uvicorn app.main:app --reload --port 8000"
) else (
    start "OsteoKnee AI - FastAPI Backend" cmd /k "cd /d "%ROOT_DIR%backend" && uvicorn app.main:app --reload --port 8000"
)

:: Start Frontend (Vite on Port 5173)
echo Starting Vite frontend on http://localhost:5173 ...
start "OsteoKnee AI - Vite Frontend" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev"

echo.
echo ======================================================================
echo Services started in separate terminal windows:
echo   - Backend API: http://localhost:8000  (Health: http://localhost:8000/api/health)
echo   - Frontend UI: http://localhost:5173
echo.
echo Press any key to close this launcher window (services keep running)...
pause >nul
