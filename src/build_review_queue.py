import argparse
import csv
import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from detection_policy import CLASSES, decide
from train import audio_items_from_splits, collect_sources, full_embeddings


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MANIFEST = os.path.join(BASE_DIR, 'test_results', 'training', 'split_manifest.json')
DEFAULT_MODEL = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')


def main():
    parser = argparse.ArgumentParser(description='Build a human-review queue from difficult training examples.')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit-per-class', type=int, default=100)
    args = parser.parse_args()

    with open(args.manifest, encoding='utf-8') as handle:
        splits = json.load(handle)['splits']
    sources, clean_paths, _, _ = collect_sources()
    items = []
    for group in splits['train']:
        cls = group.split(':', 1)[0]
        items.append({
            'group': group,
            'label': CLASSES.index(cls),
            'path': clean_paths.get(group, sources[group][0]),
        })

    os.environ.setdefault(
        'TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot_fresh')
    )
    print('Loading YAMNet and classifier...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    model = tf.keras.models.load_model(args.model)
    rows = []
    for index, item in enumerate(items, 1):
        probs = model.predict(full_embeddings(yamnet, item['path']), verbose=0)
        predicted = decide(probs)[0]
        true_index = item['label']
        max_probs = probs.max(axis=0)
        true_peak = float(max_probs[true_index])
        strongest_other = float(np.max(np.delete(max_probs, true_index)))
        rows.append({
            'priority_score': strongest_other - true_peak,
            'actual': CLASSES[true_index],
            'predicted': predicted,
            'needs_review': predicted != CLASSES[true_index] or true_peak < 0.60,
            'true_peak': true_peak,
            'glass_peak': float(max_probs[CLASSES.index('glass')]),
            'normal_peak': float(max_probs[CLASSES.index('normal')]),
            'scream_peak': float(max_probs[CLASSES.index('scream')]),
            'group': item['group'],
            'path': os.path.relpath(item['path'], BASE_DIR).replace('\\', '/'),
            'review_action': 'listen_and_mark_keep_relabel_or_remove',
        })
        if index % 100 == 0 or index == len(items):
            print(f'[{index}/{len(items)}]')

    selected = []
    for cls in CLASSES:
        candidates = [row for row in rows if row['actual'] == cls and row['needs_review']]
        selected.extend(sorted(candidates, key=lambda row: row['priority_score'], reverse=True)[:args.limit_per_class])
    selected.sort(key=lambda row: row['priority_score'], reverse=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fields = list(selected[0]) if selected else list(rows[0])
    with open(args.output, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({
        'train_items': len(items),
        'review_items': len(selected),
        'by_class': {cls: sum(row['actual'] == cls for row in selected) for cls in CLASSES},
        'output': os.path.abspath(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
