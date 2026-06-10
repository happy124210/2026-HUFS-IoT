from picamera2 import Picamera2
from datetime import datetime
import time
import os

# 사진 저장 폴더
PHOTO_DIR = os.path.join(os.path.dirname(__file__), '..', 'photos')
os.makedirs(PHOTO_DIR, exist_ok=True)

def take_photo():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{timestamp}.jpg"
    filepath = os.path.join(PHOTO_DIR, filename)

    try:
        picam = Picamera2()
        picam.start()
        time.sleep(1)  # 워밍업
        picam.capture_file(filepath)
        picam.stop()
        print(f"📸 사진 저장: {filepath}")
        return filepath
    except Exception as e:
        print(f"카메라 오류: {e}")
        return None

if __name__ == '__main__':
    print("카메라 테스트 중...")
    path = take_photo()
    if path:
        print(f"✅ 촬영 성공: {path}")
    else:
        print("❌ 촬영 실패")
