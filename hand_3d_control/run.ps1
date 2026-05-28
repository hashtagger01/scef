# PowerShell Launcher for 3D Hologram Gesture Controller
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "    Launching 3D Hologram Gesture Controller...        " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

& "..\.venv\Scripts\python.exe" "main.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error: Failed to execute main.py using virtual environment." -ForegroundColor Red
    Write-Host "Please ensure your webcam is connected." -ForegroundColor Red
    Write-Host ""
    Read-Host -Prompt "Press Enter to exit"
}
