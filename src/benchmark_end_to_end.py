import argparse
import json
import os
import platform
import statistics
import tempfile
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from audio_pipeline import load_audio
from detection_policy import decide_final_fusion
from train import audio_items_from_splits, collect_sources


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFEST = os.path.join(BASE_DIR, 'test_results', 'training', 'split_manifest.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
OUTPUT_PATH = os.path.join(
    BASE_DIR, 'test_results', 'benchmarks', 'model_processing_current_host_20260621.json'
)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark YAMNet + custom head + final Fusion on the current host.'
    )
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--output', default=OUTPUT_PATH)
    args = parser.parse_args()

    with open(MANIFEST, encoding='utf-8') as handle:
        splits = json.load(handle)['splits']
    sources, clean_paths, _, _ = collect_sources()
    test_items = audio_items_from_splits(splits, sources, clean_paths)['test'][:args.samples]

    os.environ.setdefault(
        'TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot_fresh')
    )
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    classifier = tf.keras.models.load_model(MODEL_PATH)

    # Warm up TensorFlow before recording latency.
    warm_audio = load_audio(test_items[0]['path'])
    warm_scores, warm_embeddings, _ = yamnet(warm_audio)
    warm_probs = classifier.predict(warm_embeddings.numpy(), verbose=0)
    decide_final_fusion(warm_probs, warm_scores.numpy())

    inference_ms = []
    total_ms = []
    for item in test_items:
        total_start = time.perf_counter()
        audio = load_audio(item['path'])
        inference_start = time.perf_counter()
        scores, embeddings, _ = yamnet(audio)
        probs = classifier.predict(embeddings.numpy(), verbose=0)
        decide_final_fusion(probs, scores.numpy())
        inference_ms.append((time.perf_counter() - inference_start) * 1000)
        total_ms.append((time.perf_counter() - total_start) * 1000)

    output = {
        'platform': platform.platform(),
        'audio_duration_seconds': 3.0,
        'sample_count': len(test_items),
        'scope': 'three-second WAV load + YAMNet + H5 classifier + final Fusion',
        'inference_ms': {
            'mean': float(statistics.mean(inference_ms)),
            'median': float(statistics.median(inference_ms)),
            'p95': percentile(inference_ms, 95),
            'max': float(max(inference_ms)),
        },
        'file_load_plus_model_ms': {
            'mean': float(statistics.mean(total_ms)),
            'median': float(statistics.median(total_ms)),
            'p95': percentile(total_ms, 95),
            'max': float(max(total_ms)),
        },
        'limitation': (
            'This is model-processing latency, not complete event-to-alert latency. It excludes '
            'microphone buffering, window and hop alignment, and GPIO/camera/network alert time.'
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
