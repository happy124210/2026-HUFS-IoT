import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from scipy import signal
import soundfile as sf
from datetime import datetime
import threading
import queue
import time
import os

# ── 설정 ──────────────────────────────
MODEL_PATH   = '../model/glass_classifier.h5'
INPUT_RATE   = 48000   # 마이크 실제 샘플레이트
TARGET_RATE  = 16000   # 모델 입력 샘플레이트
DURATION     = 3.0
THRESHOLD    = 0.5
CLASSES      = ['glass', 'normal', 'scream']
DEVICE_ID    = 1

# ── 모델 로드 ─────────────────────────
print("YAMNet 로드 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
print("분류기 로드 중...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 모델 로드 완료!")

# ── 오디오 큐 ─────────────────────────
audio_queue = queue.Queue()
buffer = np.zeros(int(INPUT_RATE * DURATION), dtype=np.float32)

# ── 48kHz → 16kHz 리샘플링 ────────────
def resample(audio, orig_sr, target_sr):
    n_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, n_samples).astype(np.float32)

# ── 전처리 ────────────────────────────
def process_audio(audio):
    # 48kHz → 16kHz 변환
    audio = resample(audio, INPUT_RATE, TARGET_RATE)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    return audio.astype(np.float32)

# ── 예측 ──────────────────────────────
def predict(audio):
    _, embeddings, _ = yamnet(audio)
    embeddings = embeddings.numpy()
    all_probs = []
    for emb in embeddings:
        emb_in = np.expand_dims(emb, axis=0)
        probs = model.predict(emb_in, verbose=0)[0]
        all_probs.append(probs)
    all_probs = np.array(all_probs)
    threat_probs = all_probs[:, [0, 2]]
    max_idx = np.unravel_index(np.argmax(threat_probs), threat_probs.shape)
    frame_idx, _ = max_idx
    return all_probs[frame_idx]

# ── 위협 감지 시 액션 ──────────────────
def on_threat_detected(label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icon = '🔨' if label == 'glass' else '😱'
    print(f"\n{'='*50}")
    print(f"{icon} 위협 감지! [{timestamp}]")
    print(f"   클래스: {label}")
    print(f"   신뢰도: {confidence*100:.1f}%")
    print(f"{'='*50}\n")
    os.makedirs('../logs', exist_ok=True)
    with open('../logs/detection_log.txt', 'a') as f:
        f.write(f"{timestamp} | {label} | {confidence*100:.1f}%\n")

# ── 마이크 콜백 ───────────────────────
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# ── 분석 루프 ─────────────────────────
def analysis_loop():
    global buffer
    print("\n🎤 실시간 감지 시작! (Ctrl+C로 종료)")
    print(f"   감지 클래스: {CLASSES}")
    print(f"   임계값: {THRESHOLD*100}%")
    print("-"*50)
    last_detection_time = 0
    COOLDOWN = 3.0
    while True:
        try:
            chunk = audio_queue.get(timeout=1.0)
            chunk = chunk.flatten().astype(np.float32)
            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk
            audio = process_audio(buffer.copy())
            probs = predict(audio)
            now = time.time()
            if probs[0] > THRESHOLD and (now - last_detection_time) > COOLDOWN:
                on_threat_detected('glass', probs[0])
                last_detection_time = now
            elif probs[2] > THRESHOLD and (now - last_detection_time) > COOLDOWN:
                on_threat_detected('scream', probs[2])
                last_detection_time = now
            else:
                print(f"\r🟢 모니터링 중... "
                      f"glass={probs[0]*100:.1f}% "
                      f"normal={probs[1]*100:.1f}% "
                      f"scream={probs[2]*100:.1f}%", end='', flush=True)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

# ── 메인 ──────────────────────────────
if __name__ == '__main__':
    analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
    analysis_thread.start()
    with sd.InputStream(
        device=DEVICE_ID,
        samplerate=INPUT_RATE,
        channels=1,
        blocksize=int(INPUT_RATE * 0.5),
        dtype=np.float32,
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n🔴 감지 종료.")
