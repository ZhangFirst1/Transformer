@echo off
chcp 65001 >nul
REM Transformer Training and Ablation Experiment Script (Windows)

set PYTHONHASHSEED=42

echo ==========================================
echo Transformer Training Script
echo ==========================================

if "%1"=="train" (
    echo Running standard training...
    python src/train.py --mode train
) else if "%1"=="ablation" (
    echo Running ablation experiments...
    python src/train.py --mode ablation
) else (
    echo Usage: scripts\run.bat [train^|ablation]
    exit /b 1
)

if %ERRORLEVEL% EQU 0 (
    echo ==========================================
    echo Training completed!
    echo Results saved in results\ directory
    echo ==========================================
) else (
    echo ==========================================
    echo Training failed with error code %ERRORLEVEL%
    echo ==========================================
)

