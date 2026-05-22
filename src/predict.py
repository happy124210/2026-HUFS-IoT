import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
from scipy import signal

# ── 설정 ──────────────────────────────
MODEL_PATH = '../model/glass_classifier.h5'
SAMPLE_RATE = 16000
DURATION = 3.0
CLASSES = ['glass', 'normal']

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
        all_probs.append(probs)
        print(f"  프레임 {i}: glass={probs[0]*100:5.1f}% normal={probs[1]*100:5.1f}%")

    all_probs = np.array(all_probs)
    
    # 평균 결과 (원래 방식)
    mean_probs = all_probs.mean(axis=0)
    
    # 최대 glass 확률 (개선 방식)
    max_glass_idx = np.argmax(all_probs[:, 0])
    max_probs = all_probs[max_glass_idx]

    print(f"\n📊 [평균] glass: {mean_probs[0]*100:5.1f}%  normal: {mean_probs[1]*100:5.1f}%")
    print(f"📊 [최대] glass: {max_probs[0]*100:5.1f}%  normal: {max_probs[1]*100:5.1f}%  (프레임 {max_glass_idx})")
    
    # 최대값 기준으로 판정 (임계값 50%)
    final = 'glass' if max_probs[0] > 0.5 else 'normal'
    print(f"\n🎯 결론: {final}")
    return final
# ── 실행 ──────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python predict.py <wav파일경로>")
        sys.exit(1)
    predict(sys.argv[1])
