import argparse
import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from audio_pipeline import load_audio
from detection_policy import CLASSES, decide_fusion
from train import audio_items_from_splits, collect_sources


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFEST = os.path.join(BASE_DIR, 'test_results', 'training', 'split_manifest.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
OUTPUT_PATH = os.path.join(
    BASE_DIR, 'test_results', 'training', 'fusion_policy_selection_20260621.json'
)

# Three domain-motivated candidates replace the previous large grid search.
# They represent recall-oriented, balanced, and precision-oriented behavior.
CANDIDATE_POLICIES = {
    'recall_oriented': {
        'glass': {
            'strong_threshold': 0.90, 'strong_min_frames': 1,
            'fusion_custom_threshold': 0.40, 'fusion_yamnet_threshold': 0.10,
            'fusion_min_frames': 1,
        },
        'scream': {
            'strong_threshold': 0.40, 'strong_min_frames': 3,
            'fusion_custom_threshold': 0.70, 'fusion_yamnet_threshold': 0.05,
            'fusion_min_frames': 1,
        },
    },
    'balanced': {
        'glass': {
            'strong_threshold': 0.95, 'strong_min_frames': 2,
            'fusion_custom_threshold': 0.40, 'fusion_yamnet_threshold': 0.10,
            'fusion_min_frames': 1,
        },
        'scream': {
            'strong_threshold': 0.55, 'strong_min_frames': 3,
            'fusion_custom_threshold': 0.85, 'fusion_yamnet_threshold': 0.02,
            'fusion_min_frames': 1,
        },
    },
    'precision_oriented': {
        'glass': {
            'strong_threshold': 0.97, 'strong_min_frames': 2,
            'fusion_custom_threshold': 0.55, 'fusion_yamnet_threshold': 0.20,
            'fusion_min_frames': 1,
        },
        'scream': {
            'strong_threshold': 0.70, 'strong_min_frames': 3,
            'fusion_custom_threshold': 0.85, 'fusion_yamnet_threshold': 0.05,
            'fusion_min_frames': 1,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select one fixed Fusion candidate using validation only.'
    )
    parser.add_argument('--output', default=OUTPUT_PATH)
    return parser.parse_args()


def metric_summary(y_true, y_pred):
    count = len(CLASSES)
    matrix = np.bincount(
        count * y_true + y_pred, minlength=count * count
    ).reshape(count, count)
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive, true_counts,
        out=np.zeros(count, dtype=float), where=true_counts != 0,
    )
    precision = np.divide(
        true_positive, predicted_counts,
        out=np.zeros(count, dtype=float), where=predicted_counts != 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(count, dtype=float), where=(precision + recall) != 0,
    )
    normal = CLASSES.index('normal')
    normal_mask = y_true == normal
    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'macro_f1': float(f1.mean()),
        'normal_false_alarm_rate': float(np.mean(y_pred[normal_mask] != normal)),
        'recall': {cls: float(recall[index]) for index, cls in enumerate(CLASSES)},
        'confusion_matrix': matrix.tolist(),
    }


def main():
    args = parse_args()
    with open(MANIFEST, encoding='utf-8') as handle:
        splits = json.load(handle)['splits']
    sources, clean_paths, _, _ = collect_sources()
    validation_items = audio_items_from_splits(splits, sources, clean_paths)['validation']

    os.environ.setdefault(
        'TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot_fresh')
    )
    print('Loading YAMNet and retained classifier...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    classifier = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    predictions = {name: [] for name in CANDIDATE_POLICIES}
    for index, item in enumerate(validation_items, 1):
        scores, embeddings, _ = yamnet(load_audio(item['path']))
        probs = classifier.predict(embeddings.numpy(), verbose=0)
        y_true.append(item['label'])
        for name, policy in CANDIDATE_POLICIES.items():
            label = decide_fusion(probs, scores.numpy(), policy=policy)[0]
            predictions[name].append(CLASSES.index(label))
        if index % 25 == 0 or index == len(validation_items):
            print(f'[validation] {index}/{len(validation_items)}')

    y_true = np.asarray(y_true, dtype=np.int64)
    results = {
        name: metric_summary(y_true, np.asarray(values, dtype=np.int64))
        for name, values in predictions.items()
    }
    eligible = [
        name for name, result in results.items()
        if result['recall']['glass'] >= 0.70 and result['recall']['scream'] >= 0.70
    ]
    selected = max(
        eligible,
        key=lambda name: (
            results[name]['macro_f1'],
            -results[name]['normal_false_alarm_rate'],
        ),
    )
    output = {
        'selection_data': 'validation only',
        'candidate_count': len(CANDIDATE_POLICIES),
        'selection_rule': (
            'Require glass and scream recall >= 0.70; then maximize Macro F1 and '
            'use lower normal false-alarm rate as tie-breaker.'
        ),
        'candidates': CANDIDATE_POLICIES,
        'validation_results': results,
        'selected_name': selected,
        'selected_policy': CANDIDATE_POLICIES[selected],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
