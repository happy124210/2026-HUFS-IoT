@echo off
echo ========================================
echo  Glass Classifier - Retrain Pipeline
echo ========================================
cd /d "%~dp0"

echo.
echo [1/4] Preprocessing new audio files...
python src/preprocess.py
if %errorlevel% neq 0 goto :error

echo.
echo [2/4] Building capped augmentation data...
python src/augment.py --glass-per-file 10 --scream-per-file 10 --normal-per-file 4 --glass-standalone-per-file 5 --scream-standalone-per-file 5 --enable-pitch-time
if %errorlevel% neq 0 goto :error

echo.
echo [3/4] Training and evaluating safe candidates...
python src/train.py --promote --results-path test_results\training\latest.json
if %errorlevel% neq 0 goto :error

echo.
echo [4/4] Synchronizing TFLite with the selected H5 model...
python src/convert_tflite.py
if %errorlevel% neq 0 goto :error

echo.
echo ========================================
echo  Done! A candidate is promoted only when every quality gate passes.
echo  See test_results\training\latest.json for the decision.
echo ========================================
goto :end

:error
echo.
echo [ERROR] Check the error message above.

:end
pause
