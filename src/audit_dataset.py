import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from detection_policy import CLASSES
from split_validation import youtube_id


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATASET = os.path.join(BASE_DIR, 'data_clean')
AUDIO_EXTENSIONS = ('.wav', '.flac', '.ogg', '.mp3', '.m4a', '.webm')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audio_stats(path):
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    clipped_fraction = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0
    return {
        'sample_rate': int(sample_rate),
        'duration_seconds': float(len(audio) / sample_rate) if sample_rate else 0.0,
        'rms': rms,
        'peak': peak,
        'clipped_fraction': clipped_fraction,
    }


def percentile_summary(values):
    return {
        'min': float(np.min(values)),
        'p50': float(np.percentile(values, 50)),
        'p95': float(np.percentile(values, 95)),
        'max': float(np.max(values)),
    } if values else {}


def main():
    parser = argparse.ArgumentParser(description='Audit clean audio for quality, duplicates, and source conflicts.')
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET)
    parser.add_argument('--json-output', required=True)
    args = parser.parse_args()

    records = []
    failures = []
    for cls in CLASSES:
        folder = os.path.join(args.dataset_dir, cls)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(AUDIO_EXTENSIONS):
                continue
            path = os.path.join(folder, name)
            relative_path = os.path.relpath(path, args.dataset_dir).replace('\\', '/')
            try:
                stats = audio_stats(path)
                stem = os.path.splitext(name)[0]
                if stem.endswith('_clean'):
                    stem = stem[:-len('_clean')]
                records.append({
                    'class': cls,
                    'path': relative_path,
                    'sha256': file_hash(path),
                    'youtube_id': youtube_id(f'{cls}:{stem}'),
                    **stats,
                })
            except Exception as exc:
                failures.append({'class': cls, 'path': relative_path, 'error': str(exc)})

    hashes = defaultdict(list)
    origins = defaultdict(list)
    for record in records:
        hashes[record['sha256']].append(record)
        if record['youtube_id']:
            origins[record['youtube_id']].append(record)
    duplicate_groups = [
        [item['path'] for item in items]
        for items in hashes.values() if len(items) > 1
    ]
    cross_class_duplicates = [
        [item['path'] for item in items]
        for items in hashes.values()
        if len({item['class'] for item in items}) > 1
    ]
    cross_class_origins = [
        {
            'youtube_id': source_id,
            'classes': sorted({item['class'] for item in items}),
            'paths': [item['path'] for item in items],
        }
        for source_id, items in origins.items()
        if len({item['class'] for item in items}) > 1
    ]
    low_rms = [item['path'] for item in records if item['rms'] < 0.01]
    heavy_clipping = [
        item['path'] for item in records if item['clipped_fraction'] >= 0.01
    ]
    wrong_sample_rate = [item['path'] for item in records if item['sample_rate'] != 16000]
    class_summary = {}
    for cls in CLASSES:
        items = [item for item in records if item['class'] == cls]
        class_summary[cls] = {
            'count': len(items),
            'duration_seconds': percentile_summary([item['duration_seconds'] for item in items]),
            'rms': percentile_summary([item['rms'] for item in items]),
        }
    output = {
        'dataset_dir': os.path.abspath(args.dataset_dir),
        'file_count': len(records),
        'class_summary': class_summary,
        'decode_failures': failures,
        'exact_duplicate_groups': duplicate_groups,
        'cross_class_exact_duplicates': cross_class_duplicates,
        'cross_class_youtube_origins': cross_class_origins,
        'quality_flags': {
            'low_rms_below_0_01': low_rms,
            'clipped_fraction_at_least_1_percent': heavy_clipping,
            'sample_rate_not_16khz': wrong_sample_rate,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
    with open(args.json_output, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        'file_count': len(records),
        'decode_failures': len(failures),
        'exact_duplicate_groups': len(duplicate_groups),
        'cross_class_exact_duplicates': len(cross_class_duplicates),
        'cross_class_youtube_origins': len(cross_class_origins),
        'low_rms': len(low_rms),
        'heavy_clipping': len(heavy_clipping),
        'wrong_sample_rate': len(wrong_sample_rate),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
