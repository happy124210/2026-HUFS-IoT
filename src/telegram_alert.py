import requests
from datetime import datetime
from dotenv import load_dotenv
import os

# .env 파일에서 토큰 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID   = os.getenv('CHAT_ID')

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {'chat_id': CHAT_ID, 'text': text}
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.ok
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")
        return False

def send_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            r = requests.post(url,
                data={'chat_id': CHAT_ID, 'caption': caption},
                files={'photo': photo},
                timeout=10)
        return r.ok
    except Exception as e:
        print(f"사진 전송 실패: {e}")
        return False

def send_alert(label, confidence, photo_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icon = '🔨' if label == 'glass' else '😱'
    message = (
        f"{icon} 위협 감지!\n"
        f"시간: {timestamp}\n"
        f"종류: {label}\n"
        f"신뢰도: {confidence*100:.1f}%"
    )
    if photo_path:
        send_photo(photo_path, caption=message)
    else:
        send_message(message)
    print(f"✅ 텔레그램 전송 완료: {label} {confidence*100:.1f}%")

if __name__ == '__main__':
    print("텔레그램 연결 테스트 중...")
    send_message("✅ IoT 보안 시스템 연결 테스트 성공!")
    print("폰에서 메시지 확인해보세요!")
