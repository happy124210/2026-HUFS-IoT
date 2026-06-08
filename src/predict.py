import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
from scipy import signal

# ── 설정 ──────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
SAMPLE_RATE = 16000
DURATION = 3.0
CLASSES = ['glass', 'normal', 'scream']
THRESHOLDS = {
    'glass': 0.75,
    'scream': 0.70,
}
MIN_EVENT_FRAMES = 2

# ── wav 로딩 + 리샘플링 (librosa 없이) ─
def load_wav(path, target_sr=SAMPLE_RATE):
    audio, sr = sf.read(path)
    # 스테레오면 모노로
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # float32로
    audio = audio.astype(np.float32)
    # 샘플레이트 다르면 리샘플
    if sr != target_sr:
        n_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, n_samples).astype(np.float32)
    return audio

# ── RMS 계산 (librosa.feature.rms 대체) ─
def frame_rms(audio, frame_length, hop_length):
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    rms_list = []
    for i in range(n_frames):
        frame = audio[i*hop_length : i*hop_length + frame_length]
        rms_list.append(np.sqrt(np.mean(frame**2)))
    return np.array(rms_list)

# ── preprocess.py와 동일한 전처리 ─────
def process_audio(path):
    audio = load_wav(path)
    target_len = int(SAMPLE_RATE * DURATION)

    if len(audio) >= target_len:
        rms_frames = frame_rms(audio, target_len, target_len // 2)
        if len(rms_frames) > 0:
            best_frame = np.argmax(rms_frames)
            start = min(best_frame * (target_len // 2), len(audio) - target_len)
            audio = audio[start:start + target_len]
        else:
            audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    return audio.astype(np.float32)

# ── 모델 로드 ─────────────────────────
print("YAMNet 로드 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

print("분류기 로드 중...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── 예측 ──────────────────────────────
def predict(wav_path):
    print(f"\n파일: {wav_path}")
    audio = process_audio(wav_path)

    # YAMNet으로 embedding 추출 (보통 6~7개 프레임)
    _, embeddings, _ = yamnet(audio)
    embeddings = embeddings.numpy()  # shape: (N, 1024)

    # ★ 각 프레임마다 개별 예측 → 그중 최대 glass 확률 채택
    print(f"\n🔍 프레임별 예측 ({len(embeddings)}개):")
    all_probs = []
    for i, emb in enumerate(embeddings):
        emb_in = np.expand_dims(emb, axis=0)
        probs = model.predict(emb_in, verbose=0)[0]
        if len(probs) != len(CLASSES):
            raise RuntimeError(
                f"모델 출력 차원({len(probs)})과 CLASSES({len(CLASSES)})가 맞지 않습니다."
            )
        all_probs.append(probs)
        scores = ' '.join(
            f"{cls}={probs[idx]*100:5.1f}%"
            for idx, cls in enumerate(CLASSES)
        )
        print(f"  프레임 {i}: {scores}")

    all_probs = np.array(all_probs)
    
    # 평균 결과 (원래 방식)
    mean_probs = all_probs.mean(axis=0)
    
    # 이벤트성 소리는 프레임별 최대 확률을 사용
    max_probs = all_probs.max(axis=0)
    max_indices = all_probs.argmax(axis=0)

    mean_scores = '  '.join(
        f"{cls}: {mean_probs[idx]*100:5.1f}%"
        for idx, cls in enumerate(CLASSES)
    )
    max_scores = '  '.join(
        f"{cls}: {max_probs[idx]*100:5.1f}% (프레임 {max_indices[idx]})"
        for idx, cls in enumerate(CLASSES)
    )
    print(f"\n📊 [평균] {mean_scores}")
    print(f"📊 [최대] {max_scores}")
    
    event_scores = {
        cls: max_probs[CLASSES.index(cls)]
        for cls in THRESHOLDS
    }
    event_counts = {
        cls: int(np.sum(all_probs[:, CLASSES.index(cls)] >= THRESHOLDS[cls]))
        for cls in THRESHOLDS
    }
    triggered = [
        cls for cls, score in event_scores.items()
        if score >= THRESHOLDS[cls] and event_counts[cls] >= MIN_EVENT_FRAMES
    ]
    if triggered:
        final = max(triggered, key=lambda cls: event_scores[cls])
    else:
        final = 'normal'

    count_text = '  '.join(f"{cls}: {event_counts[cls]}프레임" for cls in THRESHOLDS)
    print(f"📊 [감지 프레임] {count_text}")
    print(f"\n🎯 결론: {final}")
    return final
# ── 실행 ──────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python predict.py <wav파일경로>")
        sys.exit(1)
    predict(sys.argv[1])
