import argparse
import os
import random

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


BASE_DIR = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT'
INPUT_DIR = os.path.join(BASE_DIR, 'data_clean')
AUGMENTED_DIR = os.path.join(BASE_DIR, 'data_augmented')
MIXED_DIR = os.path.join(BASE_DIR, 'data_mixed')

SAMPLE_RATE = 16000
DURATION = 3.0
TARGET_LEN = int(SAMPLE_RATE * DURATION)
CLASSES = ['glass', 'normal', 'scream']


def list_audio_files(folder):
    exts = ('.wav', '.mp3', '.flac', '.m4a', '.webm', '.ogg')
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(exts)
    ]


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = signal.resample(audio, n_samples).astype(np.float32)
    return fit_length(audio.astype(np.float32))


def fit_length(audio):
    if len(audio) >= TARGET_LEN:
        return audio[:TARGET_LEN].astype(np.float32)
    return np.pad(audio, (0, TARGET_LEN - len(audio))).astype(np.float32)


def normalize_peak(audio, peak=0.9):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * peak
    return audio.astype(np.float32)


def apply_gain_db(audio, gain_db):
    return (audio * (10 ** (gain_db / 20.0))).astype(np.float32)


def add_noise(audio, noise_db_range=(-45, -25)):
    noise_db = random.uniform(*noise_db_range)
    noise_rms = 10 ** (noise_db / 20.0)
    noise = np.random.normal(0.0, noise_rms, len(audio)).astype(np.float32)
    return (audio + noise).astype(np.float32)


def maybe_filter(audio):
    roll = random.random()
    if roll < 0.25:
        sos = signal.butter(4, random.uniform(2800, 6000), 'lowpass', fs=SAMPLE_RATE, output='sos')
        return signal.sosfilt(sos, audio).astype(np.float32)
    if roll < 0.45:
        sos = signal.butter(2, random.uniform(80, 250), 'highpass', fs=SAMPLE_RATE, output='sos')
        return signal.sosfilt(sos, audio).astype(np.float32)
    return audio


def maybe_clip(audio):
    if random.random() < 0.2:
        limit = random.uniform(0.55, 0.9)
        return np.clip(audio, -limit, limit).astype(np.float32)
    return audio


def transform_event(audio, cls, enable_pitch_time=False):
    if not enable_pitch_time:
        return audio.astype(np.float32)

    if cls == 'scream':
        if random.random() < 0.75:
            audio = librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=random.uniform(-2.0, 2.0))
        if random.random() < 0.65:
            audio = librosa.effects.time_stretch(y=audio, rate=random.uniform(0.9, 1.1))
    elif cls == 'glass':
        if random.random() < 0.35:
            audio = librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=random.uniform(-0.7, 0.7))
        if random.random() < 0.35:
            audio = librosa.effects.time_stretch(y=audio, rate=random.uniform(0.95, 1.05))
    return audio.astype(np.float32)


def active_region(audio, cls):
    frame_length = int(0.08 * SAMPLE_RATE)
    hop_length = int(0.02 * SAMPLE_RATE)
    if len(audio) < frame_length:
        return audio
    rms = []
    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        rms.append(np.sqrt(np.mean(frame ** 2)))
    rms = np.array(rms)
    if len(rms) == 0 or np.max(rms) <= 0:
        return audio

    threshold = max(np.max(rms) * 0.25, np.percentile(rms, 70))
    active = np.where(rms >= threshold)[0]
    if len(active) == 0:
        return audio

    pad = int(0.12 * SAMPLE_RATE)
    start = max(0, active[0] * hop_length - pad)
    end = min(len(audio), active[-1] * hop_length + frame_length + pad)

    min_len = int((0.15 if cls == 'glass' else 0.45) * SAMPLE_RATE)
    max_len = int((1.25 if cls == 'glass' else 2.6) * SAMPLE_RATE)
    if end - start < min_len:
        center = (start + end) // 2
        start = max(0, center - min_len // 2)
        end = min(len(audio), start + min_len)
    if end - start > max_len:
        center = (start + end) // 2
        start = max(0, center - max_len // 2)
        end = min(len(audio), start + max_len)

    return audio[start:end].astype(np.float32)


def match_snr(event, background, snr_db):
    event_rms = np.sqrt(np.mean(event ** 2)) + 1e-8
    bg_rms = np.sqrt(np.mean(background ** 2)) + 1e-8
    target_event_rms = bg_rms * (10 ** (snr_db / 20.0))
    return (event * (target_event_rms / event_rms)).astype(np.float32)


def random_background(normal_files):
    if not normal_files:
        return np.zeros(TARGET_LEN, dtype=np.float32)
    bg = load_audio(random.choice(normal_files))
    bg = apply_gain_db(bg, random.uniform(-12.0, 3.0))
    return bg


def mix_event_with_background(event_path, cls, normal_files, enable_pitch_time=False):
    background = random_background(normal_files)
    event = load_audio(event_path)
    event = active_region(event, cls)
    event = transform_event(event, cls, enable_pitch_time=enable_pitch_time)
    event = apply_gain_db(event, random.uniform(-12.0, 6.0))

    max_start = max(0, TARGET_LEN - len(event))
    start = random.randint(0, max_start) if max_start > 0 else 0
    event = event[:TARGET_LEN - start]
    event = match_snr(event, background[start:start + len(event)], random.choice([-3, 0, 3, 6, 10]))

    mixed = background.copy()
    mixed[start:start + len(event)] += event
    mixed = add_noise(mixed)
    mixed = maybe_filter(mixed)
    mixed = maybe_clip(mixed)
    return normalize_peak(mixed)


def augment_normal(path):
    audio = load_audio(path)
    audio = apply_gain_db(audio, random.uniform(-10.0, 6.0))
    audio = add_noise(audio, noise_db_range=(-48, -28))
    audio = maybe_filter(audio)
    audio = maybe_clip(audio)
    return normalize_peak(audio)


def write_audio(folder, source_path, index, audio):
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(folder, f'{stem}_aug_{index:03d}.wav')
    if os.path.exists(out_path):
        return False
    sf.write(out_path, audio, SAMPLE_RATE)
    return True


def build_dataset(glass_per_file, scream_per_file, normal_per_file, seed, enable_pitch_time):
    random.seed(seed)
    np.random.seed(seed)

    files_by_class = {
        cls: list_audio_files(os.path.join(INPUT_DIR, cls))
        for cls in CLASSES
    }
    normal_files = files_by_class['normal']

    print('Input files:')
    for cls in CLASSES:
        print(f'  {cls}: {len(files_by_class[cls])}')

    counts = {'glass': 0, 'normal': 0, 'scream': 0}

    for path in files_by_class['normal']:
        for i in range(normal_per_file):
            written = write_audio(
                os.path.join(AUGMENTED_DIR, 'normal'),
                path,
                i,
                augment_normal(path),
            )
            if written:
                counts['normal'] += 1
        if counts['normal'] and counts['normal'] % 100 == 0:
            print(f"  normal generated: {counts['normal']}")

    for cls, per_file in [('glass', glass_per_file), ('scream', scream_per_file)]:
        for path in files_by_class[cls]:
            for i in range(per_file):
                written = write_audio(
                    os.path.join(MIXED_DIR, cls),
                    path,
                    i,
                    mix_event_with_background(
                        path,
                        cls,
                        normal_files,
                        enable_pitch_time=enable_pitch_time,
                    ),
                )
                if written:
                    counts[cls] += 1
                if counts[cls] and counts[cls] % 100 == 0:
                    print(f"  {cls} generated: {counts[cls]}")

    print('\nGenerated files in this run:')
    for cls in CLASSES:
        print(f'  {cls}: {counts[cls]}')
    print('\nTotal augmented files on disk:')
    for cls, folder in [
        ('glass', os.path.join(MIXED_DIR, 'glass')),
        ('normal', os.path.join(AUGMENTED_DIR, 'normal')),
        ('scream', os.path.join(MIXED_DIR, 'scream')),
    ]:
        print(f'  {cls}: {len(list_audio_files(folder))}')
    print(f'\nAugmented normal: {os.path.join(AUGMENTED_DIR, "normal")}')
    print(f'Mixed events: {MIXED_DIR}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create realtime-detection-oriented augmented audio windows.'
    )
    parser.add_argument('--glass-per-file', type=int, default=5)
    parser.add_argument('--scream-per-file', type=int, default=7)
    parser.add_argument('--normal-per-file', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--enable-pitch-time',
        action='store_true',
        help='Enable slower pitch shift and time stretch transforms.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_dataset(
        glass_per_file=args.glass_per_file,
        scream_per_file=args.scream_per_file,
        normal_per_file=args.normal_per_file,
        seed=args.seed,
        enable_pitch_time=args.enable_pitch_time,
    )
