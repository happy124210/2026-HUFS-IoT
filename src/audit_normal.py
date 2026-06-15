import argparse
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from audio_pipeline import load_audio
from detection_policy import CLASSES, MIN_CONSECUTIVE_FRAMES, decide


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
def audit_file(yamnet, model, path):
    audio = load_audio(path)
    _, embeddings, _ = yamnet(audio)
    probs = model.predict(embeddings.numpy(), verbose=0)
    max_probs = probs.max(axis=0)
    _, _, _, consecutive_runs = decide(probs)
    return max_probs, consecutive_runs


def parse_args():
    parser = argparse.ArgumentParser(description='Audit normal files for event-like predictions.')
    parser.add_argument(
        '--folder',
        default=os.path.join(BASE_DIR, 'data_clean', 'normal'),
        help='Folder to scan.',
    )
    parser.add_argument('--prefix', default='audioset_casino_normal_')
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
        max_probs, consecutive_runs = audit_file(yamnet, model, path)
        glass = max_probs[CLASSES.index('glass')]
        normal = max_probs[CLASSES.index('normal')]
        scream = max_probs[CLASSES.index('scream')]
        flagged = any(
            consecutive_runs[cls] >= MIN_CONSECUTIVE_FRAMES[cls]
            for cls in MIN_CONSECUTIVE_FRAMES
        )
        if flagged:
            suspicious.append((path, glass, normal, scream, consecutive_runs))
        print(
            f'{"FLAG" if flagged else "OK  "} '
            f'{os.path.basename(path)} '
            f'glass={glass:.3f} normal={normal:.3f} scream={scream:.3f} '
            f'runs={consecutive_runs}'
        )

    print(f'\n의심 파일: {len(suspicious)}개')
    for path, glass, normal, scream, consecutive_runs in suspicious:
        print(
            f'  {os.path.basename(path)} '
            f'glass={glass:.3f} normal={normal:.3f} scream={scream:.3f} '
            f'runs={consecutive_runs}'
        )


if __name__ == '__main__':
    main()
