import argparse
import itertools
import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from audio_pipeline import SAMPLE_RATE, load_audio, loudest_window
from detection_policy import CLASSES, MIN_CONSECUTIVE_FRAMES, THRESHOLDS, decide


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, 'evaluation_data')
WINDOW_SAMPLES = 3 * SAMPLE_RATE
AUDIO_EXTENSIONS = ('.wav', '.flac', '.ogg')


def collect_files(dataset_dir):
    samples = []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(dataset_dir, cls)
        if not os.path.isdir(folder):
            continue
        samples.extend(
            (os.path.join(folder, name), label)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith(AUDIO_EXTENSIONS)
        )
    return samples


def predict_frames(yamnet, classifier, path):
    audio = loudest_window(load_audio(path), WINDOW_SAMPLES)
    yamnet_scores, embeddings, _ = yamnet(audio.astype(np.float32))
    return classifier.predict(embeddings.numpy(), verbose=0), yamnet_scores.numpy()


def labels_for(all_probs, thresholds):
    return np.array([
        CLASSES.index(decide(probs, thresholds=thresholds)[0])
        for probs, _ in all_probs
    ])


def metric_summary(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    normal_index = CLASSES.index('normal')
    normal_mask = y_true == normal_index
    false_alarm_rate = float(np.mean(y_pred[normal_mask] != normal_index)) if np.any(normal_mask) else 0.0
    return {
        'confusion_matrix': matrix.tolist(),
        'macro_f1': float(f1_score(
            y_true,
            y_pred,
            labels=range(len(CLASSES)),
            average='macro',
            zero_division=0,
        )),
        'normal_false_alarm_rate': false_alarm_rate,
        'accuracy': float(np.mean(y_true == y_pred)),
    }


def print_metrics(y_true, y_pred):
    summary = metric_summary(y_true, y_pred)
    print('\nConfusion matrix (rows=true, columns=predicted)')
    print('          ' + ' '.join(f'{cls:>7}' for cls in CLASSES))
    matrix = np.asarray(summary['confusion_matrix'])
    for cls, row in zip(CLASSES, matrix):
        print(f'{cls:>8}  ' + ' '.join(f'{value:7d}' for value in row))
    print('\n' + classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        zero_division=0,
        digits=3,
    ))

    print(f"Normal false alarm rate: {summary['normal_false_alarm_rate']:.3f}")
    return summary


def threshold_search(y_true, all_probs):
    candidates = [value / 100.0 for value in range(70, 100)]
    normal_index = CLASSES.index('normal')
    normal_mask = y_true == normal_index
    ranked = []
    for glass_threshold, scream_threshold in itertools.product(candidates, repeat=2):
        thresholds = {
            'glass': float(glass_threshold),
            'scream': float(scream_threshold),
        }
        y_pred = labels_for(all_probs, thresholds)
        macro_f1 = f1_score(y_true, y_pred, labels=range(len(CLASSES)), average='macro', zero_division=0)
        false_alarm_rate = float(np.mean(y_pred[normal_mask] != normal_index)) if np.any(normal_mask) else 0.0
        score = macro_f1 - false_alarm_rate
        ranked.append({
            'score': float(score),
            'macro_f1': float(macro_f1),
            'normal_false_alarm_rate': false_alarm_rate,
            'thresholds': thresholds,
        })

    print('\nThreshold candidates (macro F1 - normal false alarm rate)')
    ranked = sorted(ranked, key=lambda result: result['score'], reverse=True)
    for result in ranked[:10]:
        print(
            f"score={result['score']:.3f} macro_f1={result['macro_f1']:.3f} "
            f"false_alarm={result['normal_false_alarm_rate']:.3f} "
            f"glass={result['thresholds']['glass']:.2f} scream={result['thresholds']['scream']:.2f}"
        )
    return ranked


def main():
    parser = argparse.ArgumentParser(description='Evaluate event detection and search thresholds.')
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--search-thresholds', action='store_true')
    parser.add_argument('--json-output', help='Optional path for machine-readable evaluation results.')
    args = parser.parse_args()

    samples = collect_files(args.dataset_dir)
    if not samples:
        raise SystemExit(
            f'No evaluation files found. Add files under {args.dataset_dir}/glass, normal, scream.'
        )

    os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
    print('Loading YAMNet and classifier...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    classifier = tf.keras.models.load_model(args.model_path)

    all_probs = []
    y_true = []
    for index, (path, label) in enumerate(samples, start=1):
        all_probs.append(predict_frames(yamnet, classifier, path))
        y_true.append(label)
        print(f'[{index:4d}/{len(samples)}] {CLASSES[label]:<6} {os.path.basename(path)}')

    y_true = np.array(y_true)
    y_pred = labels_for(all_probs, THRESHOLDS)
    print(f'\nPolicy: thresholds={THRESHOLDS}, consecutive_frames={MIN_CONSECUTIVE_FRAMES}')
    summary = print_metrics(y_true, y_pred)
    ranked = []
    if args.search_thresholds:
        ranked = threshold_search(y_true, all_probs)

    if args.json_output:
        output = {
            'dataset_dir': os.path.abspath(args.dataset_dir),
            'model_path': os.path.abspath(args.model_path),
            'classes': CLASSES,
            'sample_count': len(samples),
            'policy': {
                'thresholds': THRESHOLDS,
                'min_consecutive_frames': MIN_CONSECUTIVE_FRAMES,
            },
            'metrics': summary,
            'predictions': [
                {
                    'path': os.path.relpath(path, args.dataset_dir).replace('\\', '/'),
                    'actual': CLASSES[int(actual)],
                    'predicted': CLASSES[int(predicted)],
                }
                for (path, _), actual, predicted in zip(samples, y_true, y_pred)
            ],
            'threshold_search_top10': ranked[:10],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, 'w', encoding='utf-8') as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f'JSON results: {os.path.abspath(args.json_output)}')


if __name__ == '__main__':
    main()
