import argparse
import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from audio_pipeline import load_audio
from detection_policy import (
    CLASSES,
    YAMNET_EVENT_INDICES,
    YAMNET_SUPPORT_THRESHOLDS,
    decide,
)
from train import audio_items_from_splits, collect_sources


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MANIFEST = os.path.join(BASE_DIR, 'test_results', 'training', 'split_manifest.json')
DEFAULT_MODEL = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')


def parse_model(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Use NAME=PATH for --model.')
    name, path = value.split('=', 1)
    return name, os.path.abspath(path)


def metric_summary(y_true, y_pred):
    recalls = recall_score(
        y_true, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0
    )
    normal = CLASSES.index('normal')
    normal_mask = y_true == normal
    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'macro_f1': float(f1_score(
            y_true, y_pred, labels=range(len(CLASSES)), average='macro', zero_division=0
        )),
        'normal_false_alarm_rate': (
            float(np.mean(y_pred[normal_mask] != normal)) if np.any(normal_mask) else 0.0
        ),
        'recall': {cls: float(recalls[index]) for index, cls in enumerate(CLASSES)},
        'confusion_matrix': confusion_matrix(
            y_true, y_pred, labels=range(len(CLASSES))
        ).tolist(),
    }


def yamnet_only_label(scores):
    supported = []
    for cls, indices in YAMNET_EVENT_INDICES.items():
        peak = float(scores[:, indices].max())
        threshold = YAMNET_SUPPORT_THRESHOLDS[cls]
        if peak >= threshold:
            supported.append((peak / threshold, cls))
    return max(supported)[1] if supported else 'normal'


def main():
    parser = argparse.ArgumentParser(description='Evaluate detector component ablations on a manifest split.')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument(
        '--model', action='append', type=parse_model,
        help='Classifier in NAME=PATH form; repeat to compare models.',
    )
    parser.add_argument('--json-output')
    args = parser.parse_args()

    model_specs = args.model or [('baseline', DEFAULT_MODEL)]
    with open(args.manifest, encoding='utf-8') as handle:
        splits = json.load(handle)['splits']
    sources, clean_paths, _, _ = collect_sources()
    items = audio_items_from_splits(splits, sources, clean_paths)[args.split]

    os.environ.setdefault(
        'TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot_fresh')
    )
    print('Loading YAMNet and classifiers...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    models = {name: tf.keras.models.load_model(path) for name, path in model_specs}
    predictions = {'yamnet_only': []}
    for name in models:
        predictions[f'{name}_custom_policy'] = []
        predictions[f'{name}_yamnet_fusion'] = []

    y_true = []
    sample_records = []
    for index, item in enumerate(items, 1):
        scores, embeddings, _ = yamnet(load_audio(item['path']))
        scores = scores.numpy()
        embeddings = embeddings.numpy()
        y_true.append(item['label'])
        sample_records.append({
            'group': item['group'],
            'path': os.path.relpath(item['path'], BASE_DIR).replace('\\', '/'),
            'actual': CLASSES[item['label']],
        })
        predictions['yamnet_only'].append(CLASSES.index(yamnet_only_label(scores)))
        for name, model in models.items():
            probs = model.predict(embeddings, verbose=0)
            predictions[f'{name}_custom_policy'].append(CLASSES.index(decide(probs)[0]))
            predictions[f'{name}_yamnet_fusion'].append(
                CLASSES.index(decide(probs, yamnet_scores=scores)[0])
            )
        if index % 50 == 0 or index == len(items):
            print(f'[{index}/{len(items)}]')

    y_true = np.asarray(y_true)
    results = {
        name: metric_summary(y_true, np.asarray(y_pred))
        for name, y_pred in predictions.items()
    }
    errors = {
        name: [
            {
                **sample_records[index],
                'predicted': CLASSES[int(predicted)],
            }
            for index, predicted in enumerate(y_pred)
            if int(predicted) != int(y_true[index])
        ]
        for name, y_pred in predictions.items()
    }
    output = {
        'manifest': os.path.abspath(args.manifest),
        'split': args.split,
        'sample_count': len(items),
        'models': {name: path for name, path in model_specs},
        'results': results,
        'errors': errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.json_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, 'w', encoding='utf-8') as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
