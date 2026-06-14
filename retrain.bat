@echo off
echo ========================================
echo  Glass Classifier - Retrain Pipeline
echo ========================================
cd /d "%~dp0"

echo.
echo [1/3] Preprocessing new audio files...
python src/preprocess.py
if %errorlevel% neq 0 goto :error

echo.
echo [2/3] Augmenting data...
python src/augment.py --glass-per-file 95 --scream-per-file 118 --normal-per-file 12 --glass-standalone-per-file 20 --scream-standalone-per-file 25 --enable-pitch-time
if %errorlevel% neq 0 goto :error

echo.
echo [3/3] Training model...
python src/train.py
if %errorlevel% neq 0 goto :error

echo.
echo ========================================
echo  Done! model/glass_classifier.h5 saved.
echo ========================================
goto :end

:error
echo.
echo [ERROR] Check the error message above.

:end
pause
