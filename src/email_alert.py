import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

# ── 설정 ──────────────────────────────
SENDER_EMAIL    = "jangtaehyun16@gmail.com"
SENDER_PASSWORD = "gkms lqku qurf ehkx"
RECEIVER_EMAIL  = "jangtaehyun16@gmail.com"

# ── 로그 기록 ─────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'detection_log.txt')

def write_log(label, confidence, photo_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icon = '🔨' if label == 'glass' else '😱'
    log_line = f"{timestamp} | {icon} {label} | 신뢰도: {confidence*100:.1f}%"
    if photo_path:
        log_line += f" | 사진: {photo_path}"
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")
    print(f"📝 로그 기록: {log_line}")

# ── 이메일 전송 ───────────────────────
def send_email(label, confidence, photo_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icon = '🔨' if label == 'glass' else '😱'

    msg = MIMEMultipart()
    msg['From']    = SENDER_EMAIL
    msg['To']      = RECEIVER_EMAIL
    msg['Subject'] = f"[IoT 보안 알림] {icon} 위협 감지 - {label}"

    body = f"""
IoT 보안 시스템 위협 감지 알림

시간: {timestamp}
종류: {label}
신뢰도: {confidence*100:.1f}%

HUFS IoT Security System
    """
    msg.attach(MIMEText(body, 'plain'))

    # 사진 첨부
    if photo_path and os.path.exists(photo_path):
        with open(photo_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment',
                          filename=os.path.basename(photo_path))
            msg.attach(img)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ 이메일 전송 완료 → {RECEIVER_EMAIL}")
        return True
    except Exception as e:
        print(f"❌ 이메일 전송 실패: {e}")
        return False

# ── 통합 알림 (이메일 + 로그) ──────────
def send_alert(label, confidence, photo_path=None):
    write_log(label, confidence, photo_path)
    send_email(label, confidence, photo_path)

# ── 테스트 ─────────────────────────────
if __name__ == '__main__':
    print("이메일 + 로그 테스트 중...")
    send_alert('glass', 0.99)
    print("이메일 확인해보세요!")
