import argparse
import os
import queue
import sys
import tempfile
import time
import threading
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
SAMPLE_RATE = 16000
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.0
CLASSES = ['glass', 'normal', 'scream']
THRESHOLDS = {
    'glass': 0.88,
    'scream': 0.85,
}
MIN_EVENT_FRAMES = 2

alert_active = False
alert_lock = threading.Lock()

def reset_alert():
    global alert_active
    with alert_lock:
        alert_active = False
    print("🟢 알람 리셋 → 다시 감지 시작!")

def import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print('sounddevice 패키지가 필요합니다.')
        sys.exit(1)

def load_models():
    os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
    import tensorflow as tf
    import tensorflow_hub as hub
    print('YAMNet 로드 중...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    print('분류기 로드 중...')
    classifier = tf.keras.models.load_model(MODEL_PATH)
    return yamnet, classifier

def resample_to_model_rate(audio, input_sample_rate):
    from scipy import signal
    if input_sample_rate == SAMPLE_RATE:
        return audio
    target_len = int(round(len(audio) * SAMPLE_RATE / input_sample_rate))
    return signal.resample(audio, target_len).astype(np.float32)

def frame_predictions(yamnet, classifier, audio):
    _, embeddings, _ = yamnet(audio.astype(np.float32))
    embeddings = embeddings.numpy()
    probs = classifier.predict(embeddings, verbose=0)
    return probs

def decide(probs):
    max_probs = probs.max(axis=0)
    event_counts = {
        cls: int(np.sum(probs[:, CLASSES.index(cls)] >= THRESHOLDS[cls]))
        for cls in THRESHOLDS
    }
    triggered = [
        cls for cls in THRESHOLDS
        if max_probs[CLASSES.index(cls)] >= THRESHOLDS[cls]
        and event_counts[cls] >= MIN_EVENT_FRAMES
    ]
    if triggered:
        final = max(triggered, key=lambda cls: max_probs[CLASSES.index(cls)])
    else:
        final = 'normal'
    return final, max_probs, event_counts

def handle_threat(label, confidence):
    global alert_active

    icon = '🔨' if label == 'glass' else '😱'
    print(f"\n{'='*50}")
    print(f"{icon} 위협 감지! 클래스: {label} 신뢰도: {confidence*100:.1f}%")
    print(f"{'='*50}")

    # 1. LED + 부저 즉시 울리기
    try:
        from gpio_alert import start_alert
        with alert_lock:
            alert_active = True
        start_alert()
    except Exception as e:
        print(f"GPIO 오류: {e}")

    # 2. 카메라 촬영 (1번만)
    photo_path = None
    try:
        from camera_alert import take_photo
        photo_path = take_photo()
    except Exception as e:
        print(f"카메라 오류: {e}")

    # 3. 텔레그램 + 이메일 + 로그 (별도 스레드)
    def send_notifications():
        try:
            from telegram_alert import send_alert as tg_alert
            tg_alert(label, confidence, photo_path)
        except Exception as e:
            print(f"텔레그램 오류: {e}")
        try:
            from email_alert import send_alert as email_alert
            email_alert(label, confidence, photo_path)
        except Exception as e:
            print(f"이메일 오류: {e}")

    t = threading.Thread(target=send_notifications, daemon=True)
    t.start()

def run(device=1):
    global alert_active

    # 텔레그램 리스너 시작
    from telegram_listener import start_listener
    from gpio_alert import stop_alert
    start_listener(stop_alert, reset_alert)

    sd = import_sounddevice()
    yamnet, classifier = load_models()

    device_info = sd.query_devices(device)
    input_sample_rate = int(device_info['default_samplerate'])
    hop_len = int(HOP_SECONDS * input_sample_rate)
    window_len = int(WINDOW_SECONDS * input_sample_rate)
    audio_buffer = np.zeros(window_len, dtype=np.float32)
    blocks = queue.Queue()
    last_event_time = 0.0
    started_at = time.time()

    def callback(indata, frames, callback_time, status):
        if status:
            print(status, file=sys.stderr)
        blocks.put(indata[:, 0].copy())

    print(f"\n✅ 시스템 시작!")
    print(f"   입력: {input_sample_rate}Hz → 모델: {SAMPLE_RATE}Hz")
    print(f"   임계값: glass={THRESHOLDS['glass']*100}% scream={THRESHOLDS['scream']*100}%")
    print(f"   Ctrl+C로 종료")
    print("-"*50)

    with sd.InputStream(
        samplerate=input_sample_rate,
        channels=1,
        dtype='float32',
        blocksize=hop_len,
        device=device,
        callback=callback,
    ):
        while True:
            block = blocks.get()
            if len(block) != hop_len:
                block = block[:hop_len] if len(block) > hop_len else np.pad(block, (0, hop_len - len(block)))

            audio_buffer[:-hop_len] = audio_buffer[hop_len:]
            audio_buffer[-hop_len:] = block

            elapsed = time.time() - started_at
            if elapsed < WINDOW_SECONDS:
                print(f'\r[워밍업] {elapsed:.1f}s / {WINDOW_SECONDS}s', end='', flush=True)
                continue

            model_audio = resample_to_model_rate(audio_buffer, input_sample_rate)
            probs = frame_predictions(yamnet, classifier, model_audio)
            final, max_probs, event_counts = decide(probs)

            now = time.time()
            with alert_lock:
                is_active = alert_active
            can_alert = now - last_event_time >= 5.0 and not is_active

            if final != 'normal' and can_alert:
                last_event_time = now
                confidence = max_probs[CLASSES.index(final)]
                t = threading.Thread(target=handle_threat, args=(final, confidence), daemon=True)
                t.start()
            else:
                if not is_active:
                    print(f"\r🟢 모니터링 중... "
                          f"glass={max_probs[0]*100:.1f}% "
                          f"normal={max_probs[1]*100:.1f}% "
                          f"scream={max_probs[2]*100:.1f}%", end='', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    try:
        run(device=args.device)
    except KeyboardInterrupt:
        try:
            from gpio_alert import stop_alert, cleanup
            stop_alert()
            cleanup()
        except:
            pass
        print("\n\n🔴 시스템 종료.")
