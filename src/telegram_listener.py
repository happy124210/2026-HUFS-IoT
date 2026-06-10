import requests
import time
import threading
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

_last_update_id = None
_running = False
_stop_callback = None
_reset_callback = None

def set_callbacks(stop_cb, reset_cb):
    global _stop_callback, _reset_callback
    _stop_callback = stop_cb
    _reset_callback = reset_cb

def get_updates():
    global _last_update_id
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {'timeout': 5}
    if _last_update_id:
        params['offset'] = _last_update_id + 1
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json().get('result', [])
    except:
        return []

def listen_loop():
    global _last_update_id, _running
    print("📱 텔레그램 명령 대기 중... ('stop' 전송시 알람 종료)")
    while _running:
        updates = get_updates()
        for update in updates:
            _last_update_id = update['update_id']
            msg = update.get('message', {})
            text = msg.get('text', '').lower().strip()
            chat_id = str(msg.get('chat', {}).get('id', ''))

            if chat_id == CHAT_ID and text == 'stop':
                print("\n📱 텔레그램에서 'stop' 수신 → 알람 종료!")
                # 부저/LED 끄기
                if _stop_callback:
                    _stop_callback()
                # alert_active 리셋 → 다시 감지 가능
                if _reset_callback:
                    _reset_callback()
                # 확인 메시지 전송
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    data={'chat_id': CHAT_ID, 'text': '✅ 알람 종료. 계속 모니터링 중...'},
                    timeout=5
                )
                print("🟢 다시 모니터링 시작!")
        time.sleep(1)

def start_listener(stop_cb, reset_cb):
    global _running
    set_callbacks(stop_cb, reset_cb)
    _running = True
    t = threading.Thread(target=listen_loop, daemon=True)
    t.start()

def stop_listener():
    global _running
    _running = False
