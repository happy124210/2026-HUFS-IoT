import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from audio_pipeline import SAMPLE_RATE, load_audio, loudest_window
from detection_policy import CLASSES, MIN_CONSECUTIVE_FRAMES, THRESHOLDS, decide_deployment

# ── 설정 ──────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
DURATION = 3.0

# ── preprocess.py와 동일한 전처리 ─────
def process_audio(path):
    audio = load_audio(path)
    target_len = int(SAMPLE_RATE * DURATION)
    return loudest_window(audio, target_len)

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
    yamnet_scores, embeddings, _ = yamnet(audio)
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
    
    final, _, _, consecutive_runs = decide_deployment(all_probs)
    count_text = '  '.join(
        f"{cls}: 연속 {consecutive_runs[cls]} / 필요 {MIN_CONSECUTIVE_FRAMES[cls]}"
        for cls in THRESHOLDS
    )
    print(f"📊 [연속 감지 프레임] {count_text}")
    print(f"\n🎯 결론: {final}")
    return final
# ── 실행 ──────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python predict.py <wav파일경로>")
        sys.exit(1)
    predict(sys.argv[1])
