import os
import subprocess
import sys

BASE_DIR = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT'
SRC_DIR  = os.path.join(BASE_DIR, 'src')

def run(script, desc):
    print(f"\n{'='*50}")
    print(f"▶ {desc}")
    print('='*50)
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, script)],
        cwd=SRC_DIR
    )
    if result.returncode != 0:
        print(f"❌ {desc} 실패! 중단합니다.")
        sys.exit(1)
    print(f"✅ {desc} 완료!")

run('download_audioset.py', 'scream 데이터 다운로드')
run('preprocess.py', '전처리 (glass + normal + scream)')
run('train.py', '모델 학습')
run('convert_tflite.py', 'TFLite 변환')

print("\n전체 파이프라인 완료!")
print(f"모델 위치: {BASE_DIR}\\model\\glass_classifier.tflite")
