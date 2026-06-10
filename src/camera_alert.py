from picamera2 import Picamera2
from datetime import datetime
import time
import os

PHOTO_DIR = os.path.join(os.path.dirname(__file__), '..', 'photos')
os.makedirs(PHOTO_DIR, exist_ok=True)

_picam = None

def get_camera():
    global _picam
    if _picam is None:
        _picam = Picamera2()
        _picam.start()
        time.sleep(1)
        print("📷 카메라 초기화 완료")
    return _picam

def take_photo():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{timestamp}.jpg"
    filepath = os.path.join(PHOTO_DIR, filename)
    try:
        picam = get_camera()
        picam.capture_file(filepath)
        print(f"📸 사진 저장: {filepath}")
        return filepath
    except Exception as e:
        print(f"카메라 오류: {e}")
        return None

def close_camera():
    global _picam
    if _picam is not None:
        _picam.stop()
        _picam = None

if __name__ == '__main__':
    print("카메라 테스트 중...")
    path = take_photo()
    if path:
        print(f"✅ 촬영 성공: {path}")
    close_camera()
