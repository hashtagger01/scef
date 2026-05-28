@echo off
title 3D Hologram Gesture Controller Launcher
echo =======================================================
echo     Launching 3D Hologram Gesture Controller...
echo =======================================================
cd /d "%~dp0"
..\.venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo Error: Launcher failed to execute main.py.
    echo Please make sure your webcam is connected and the virtual environment exists.
    echo.
    pause
)
