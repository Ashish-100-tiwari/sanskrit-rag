@echo off
REM Installation script for Sanskrit RAG System
REM This ensures setuptools is installed first to avoid build errors

echo Installing Sanskrit RAG System Dependencies...
echo.

REM Upgrade pip first
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install build tools first
echo [2/4] Installing build tools (setuptools, wheel)...
pip install setuptools>=65.0.0 wheel>=0.38.0

REM Install other dependencies
echo [3/4] Installing main dependencies...
pip install -r requirements.txt

REM Install llama-cpp-python separately (may need special handling)
echo [4/4] Installing llama-cpp-python (CPU version)...
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download a GGUF model to models/ directory
echo 2. Update .env file with MODEL_PATH
echo 3. Run: python code/main.py
echo.

pause
