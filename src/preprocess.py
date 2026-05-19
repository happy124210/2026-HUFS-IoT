import os
import numpy as np
import librosa
import soundfile as sf

# ── 설정 ──────────────────────────────
DATA_DIR    = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT\\data'
OUTPUT_DIR  = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT\\data_clean'
SAMPLE_RATE = 16000
DURATION    = 3.0
MIN_RMS     = 0.01
CLASSES     = ['glass', 'normal']

def process_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        return None, f"로드 실패: {e}"

    # 너무 조용한 파일 제거
    rms = np.sqrt(np.mean(audio**2))
    if rms < MIN_RMS:
        return None, "너무 조용함"

    # 길이 3초로 통일
    target_len = int(sr * duration)
    if len(audio) >= target_len:
        rms_frames = librosa.feature.rms(
            y=audio, frame_length=target_len, hop_length=target_len//2)[0]
        best_frame = np.argmax(rms_frames)
        start = min(best_frame * (target_len//2), len(audio) - target_len)
        audio = audio[start:start + target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # 볼륨 정규화
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio, "OK"

# ── 실행 ──────────────────────────────
total_ok, total_skip = 0, 0

for cls in CLASSES:
    in_dir  = os.path.join(DATA_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    exts  = ('.wav', '.mp3', '.flac', '.m4a', '.webm', '.ogg')
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(exts)]
    print(f"\n[{cls}] {len(files)}개 처리 중...")

    for fname in files:
        audio, status = process_audio(os.path.join(in_dir, fname))
        if audio is None:
            print(f"  ✗ skip: {fname} ({status})")
            total_skip += 1
        else:
            out_name = os.path.splitext(fname)[0] + '_clean.wav'
            sf.write(os.path.join(out_dir, out_name), audio, SAMPLE_RATE)
            print(f"  ✓ {fname}")
            total_ok += 1

print(f"\n완료! 성공 {total_ok}개 / 스킵 {total_skip}개")
print(f"정제된 데이터: {OUTPUT_DIR}")