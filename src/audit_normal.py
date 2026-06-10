import argparse
import os
import tempfile

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy import signal


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
SAMPLE_RATE = 16000
CLASSES = ['glass', 'normal', 'scream']
THRESHOLDS = {
    'glass': 0.75,
    'scream': 0.70,
}


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = signal.resample(audio, n_samples).astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    return audio.astype(np.float32)


def audit_file(yamnet, model, path):
    audio = load_audio(path)
    _, embeddings, _ = yamnet(audio)
    probs = model.predict(embeddings.numpy(), verbose=0)
    max_probs = probs.max(axis=0)
    event_counts = {
        cls: int(np.sum(probs[:, CLASSES.index(cls)] >= threshold))
        for cls, threshold in THRESHOLDS.items()
    }
    return max_probs, event_counts


def parse_args():
    parser = argparse.ArgumentParser(description='Audit normal files for event-like predictions.')
    parser.add_argument(
        '--folder',
        default=os.path.join(BASE_DIR, 'data_clean', 'normal'),
        help='Folder to scan.',
    )
    parser.add_argument('--prefix', default='audioset_casino_normal_')
    parser.add_argument('--min-event-frames', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    files = [
        os.path.join(args.folder, name)
        for name in sorted(os.listdir(args.folder))
        if name.startswith(args.prefix) and name.lower().endswith('.wav')
    ]

    print('YAMNet 로드 중...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    print('분류기 로드 중...')
    model = tf.keras.models.load_model(MODEL_PATH)

    suspicious = []
    print(f'검수 대상: {len(files)}개')
    for path in files:
        max_probs, event_counts = audit_file(yamnet, model, path)
        glass = max_probs[CLASSES.index('glass')]
        normal = max_probs[CLASSES.index('normal')]
        scream = max_probs[CLASSES.index('scream')]
        flagged = any(count >= args.min_event_frames for count in event_counts.values())
        if flagged:
            suspicious.append((path, glass, normal, scream, event_counts))
        print(
            f'{"FLAG" if flagged else "OK  "} '
            f'{os.path.basename(path)} '
            f'glass={glass:.3f} normal={normal:.3f} scream={scream:.3f} '
            f'frames={event_counts}'
        )

    print(f'\n의심 파일: {len(suspicious)}개')
    for path, glass, normal, scream, event_counts in suspicious:
        print(
            f'  {os.path.basename(path)} '
            f'glass={glass:.3f} normal={normal:.3f} scream={scream:.3f} '
            f'frames={event_counts}'
        )


if __name__ == '__main__':
    main()
