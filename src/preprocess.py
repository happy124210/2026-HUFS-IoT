import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

# ── 설정 ──────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'data_clean')
SAMPLE_RATE = 16000
DURATION    = 3.0
MIN_RMS     = 0.01
CLASSES     = ['glass', 'normal', 'scream']

def process_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    try:
        if path.lower().endswith('.wav'):
            audio, source_sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if source_sr != sr:
                n_samples = int(len(audio) * sr / source_sr)
                audio = signal.resample(audio, n_samples).astype(np.float32)
        else:
            audio, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        return None, f"로드 실패: {e}"

    # 너무 조용한 파일 제거
    rms = np.sqrt(np.mean(audio**2))
    if rms < MIN_RMS:
        return None, "너무 조용함"

    # 길이 3초로 통일
    target_len = int(sr * duration)
    if len(audio) > target_len:
        hop_length = target_len // 2
        rms_frames = []
        for start in range(0, len(audio) - target_len + 1, hop_length):
            frame = audio[start:start + target_len]
            rms_frames.append(np.sqrt(np.mean(frame ** 2)))
        best_frame = int(np.argmax(rms_frames)) if rms_frames else 0
        start = min(best_frame * hop_length, len(audio) - target_len)
        audio = audio[start:start + target_len]
    elif len(audio) == target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # 볼륨 정규화
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio, "OK"

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess audio files into 3-second clean wavs.')
    parser.add_argument(
        '--class-name',
        action='append',
        choices=CLASSES,
        help='처리할 클래스. 여러 번 지정 가능. 생략하면 전체 클래스를 처리합니다.',
    )
    parser.add_argument('--verbose-skip', action='store_true', help='이미 있는 출력 파일도 로그로 표시합니다.')
    parser.add_argument('--prefix', help='파일명이 이 prefix로 시작하는 입력만 처리합니다.')
    return parser.parse_args()


# ── 실행 ──────────────────────────────
args = parse_args()
target_classes = args.class_name or CLASSES
total_ok, total_skip = 0, 0

for cls in target_classes:
    in_dir  = os.path.join(DATA_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    exts  = ('.wav', '.mp3', '.flac', '.m4a', '.webm', '.ogg')
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(exts)]
    if args.prefix:
        files = [f for f in files if f.startswith(args.prefix)]
    print(f"\n[{cls}] {len(files)}개 처리 중...")

    for fname in files:
        out_name = os.path.splitext(fname)[0] + '_clean.wav'
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            if args.verbose_skip:
                print(f"  - skip existing: {fname}")
            continue

        audio, status = process_audio(os.path.join(in_dir, fname))
        if audio is None:
            print(f"  ✗ skip: {fname} ({status})")
            total_skip += 1
        else:
            sf.write(out_path, audio, SAMPLE_RATE)
            print(f"  ✓ {fname}")
            total_ok += 1

print(f"\n완료! 성공 {total_ok}개 / 스킵 {total_skip}개")
print(f"정제된 데이터: {OUTPUT_DIR}")
