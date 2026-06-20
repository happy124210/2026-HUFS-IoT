import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from audio_pipeline import SAMPLE_RATE, load_audio, rms
from detection_policy import CLASSES, MIN_CONSECUTIVE_FRAMES, MIN_MEAN_PROBS, MIN_PROB_MARGINS, decide
from model_selection import passes_deployment_gate, rank_key
from split_validation import assert_disjoint_splits, origin_overlaps, source_unit


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot_fresh'))
DATA_DIRS = [
    os.path.join(BASE_DIR, 'data_clean'),
    os.path.join(BASE_DIR, 'data_augmented'),
    os.path.join(BASE_DIR, 'data_mixed'),
]
MODEL_DIR = os.path.join(BASE_DIR, 'model')
EMBEDDING_CACHE_DIR = os.path.join(MODEL_DIR, 'embedding_cache')
REVIEW_DIR = os.path.join(BASE_DIR, 'evaluation_data', 'review_20260617')
RESULTS_DIR = os.path.join(BASE_DIR, 'test_results', 'training')
APPROVED_GLASS_SOURCES_PATH = os.path.join(
    BASE_DIR, 'data_quality', 'approved_audioset_glass.txt'
)
SEED = 42
EMBEDDING_CACHE_VERSION = 'frame-v1-no-peak-normalization'
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.webm', '.ogg')
# Confirmed microphone false negatives are hard examples collected specifically
# for retraining. Keep them in train instead of letting the random split place
# them in validation/test, where they would not correct the observed failure.
FORCED_TRAIN_PREFIXES = ('review_scream_pred_normal_',)

# Per-source limits include the clean file. They approximately balance the number
# of embedding frames while avoiding the old 85k-file augmentation flood.
MAX_FILES_PER_TRAIN_SOURCE = {'glass': 16, 'normal': 4, 'scream': 9}
MAX_FRAMES_PER_FILE = {'glass': 2, 'normal': 3, 'scream': 4}


def source_group(cls, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    stem = re.sub(r'_aug_\d+$', '', stem)
    stem = re.sub(r'_clean$', '', stem)
    return f'{cls}:{stem}'


def preferred_duplicate_group(groups):
    def rank(group):
        stem = group.split(':', 1)[1]
        has_copy_suffix = ' (1)' in stem
        has_source_provenance = stem[:1].isdigit() or stem.startswith('audioset_')
        return (has_copy_suffix, not has_source_provenance, len(stem), stem)
    return min(groups, key=rank)


def clean_source_aliases():
    """Map byte-identical clean files to one canonical source group."""
    hashes = defaultdict(list)
    clean_dir = os.path.join(BASE_DIR, 'data_clean')
    for cls in CLASSES:
        folder = os.path.join(clean_dir, cls)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(AUDIO_EXTENSIONS):
                continue
            path = os.path.join(folder, name)
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            hashes[digest.hexdigest()].append(source_group(cls, name))
    aliases = {}
    for groups in hashes.values():
        canonical = preferred_duplicate_group(groups)
        for group in groups:
            if group != canonical:
                aliases[group] = canonical
    return aliases


def approved_glass_sources():
    if not os.path.exists(APPROVED_GLASS_SOURCES_PATH):
        return set()
    with open(APPROVED_GLASS_SOURCES_PATH, encoding='utf-8') as handle:
        return {
            line.strip() for line in handle
            if line.strip() and not line.lstrip().startswith('#')
        }


def cache_path_for(path):
    rel_path = os.path.relpath(path, BASE_DIR).replace('\\', '/')
    stat = os.stat(path)
    key = f'{EMBEDDING_CACHE_VERSION}|{rel_path}|{stat.st_size}|{stat.st_mtime_ns}'
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return os.path.join(EMBEDDING_CACHE_DIR, f'{digest}.npy')


def full_embeddings(yamnet, path):
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    cache_path = cache_path_for(path)
    if os.path.exists(cache_path):
        return np.load(cache_path)
    audio = load_audio(path)
    _, embeddings, _ = yamnet(audio)
    embeddings = embeddings.numpy()
    np.save(cache_path, embeddings)
    return embeddings


def embedding_frame_rms(audio, frame_count):
    if frame_count <= 0:
        return np.array([], dtype=np.float32)
    frame_length = min(len(audio), int(0.96 * SAMPLE_RATE))
    starts = [max(0, (len(audio) - frame_length) // 2)] if frame_count == 1 else np.linspace(
        0, max(0, len(audio) - frame_length), frame_count
    ).astype(int)
    return np.array([rms(audio[start:start + frame_length]) for start in starts], dtype=np.float32)


def selected_embeddings(yamnet, path, cls):
    embeddings = full_embeddings(yamnet, path)
    frame_count = len(embeddings)
    limit = MAX_FRAMES_PER_FILE[cls]
    if frame_count <= limit:
        return embeddings
    audio = load_audio(path)
    levels = embedding_frame_rms(audio, frame_count)
    if cls == 'normal':
        indices = np.linspace(0, frame_count - 1, limit).astype(int)
    else:
        relative_floor = max(float(levels.max()) * 0.35, float(np.percentile(levels, 60)))
        active = np.flatnonzero(levels >= relative_floor)
        if len(active) == 0:
            active = np.array([int(np.argmax(levels))])
        indices = np.sort(active[np.argsort(levels[active])[::-1]][:limit])
    return embeddings[indices]


def collect_sources():
    sources = defaultdict(list)
    clean_paths = {}
    aliases = clean_source_aliases()
    approved_glass = approved_glass_sources()
    excluded_unverified = set()
    priority = {'data_clean': 0, 'data_augmented': 1, 'data_mixed': 2}
    for data_dir in DATA_DIRS:
        root_name = os.path.basename(data_dir)
        for cls in CLASSES:
            folder = os.path.join(data_dir, cls)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(AUDIO_EXTENSIONS):
                    continue
                path = os.path.join(folder, name)
                original_group = source_group(cls, name)
                if (
                    original_group.startswith('glass:audioset_glass_event_')
                    and original_group not in approved_glass
                ):
                    excluded_unverified.add(original_group)
                    continue
                group = aliases.get(original_group, original_group)
                if root_name == 'data_clean' and original_group != group:
                    continue
                sources[group].append((priority[root_name], path))
                if root_name == 'data_clean':
                    clean_paths.setdefault(group, path)
    for group in sources:
        sources[group] = [path for _, path in sorted(sources[group], key=lambda item: (item[0], item[1]))]
    return sources, clean_paths, aliases, excluded_unverified


def forced_review_groups():
    groups = set()
    for cls in CLASSES:
        folder = os.path.join(REVIEW_DIR, cls)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(AUDIO_EXTENSIONS):
                groups.add(source_group(cls, name))
    return groups


def split_groups(sources):
    groups = np.array(sorted(sources))
    forced = forced_review_groups() & set(groups)
    forced_train = {
        group for group in groups
        if any(group.split(':', 1)[1].startswith(prefix) for prefix in FORCED_TRAIN_PREFIXES)
    }

    unit_groups = defaultdict(list)
    for group in groups:
        unit_groups[source_unit(group)].append(group)
    unit_labels = {}
    for unit, members in unit_groups.items():
        labels = {CLASSES.index(group.split(':', 1)[0]) for group in members}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels for source {unit}: {members}')
        unit_labels[unit] = labels.pop()

    forced_train_units = {source_unit(group) for group in forced_train}
    forced_test_units = {source_unit(group) for group in forced} - forced_train_units
    remaining = np.array(sorted(
        set(unit_groups) - forced_train_units - forced_test_units
    ))
    remaining_labels = np.array([unit_labels[unit] for unit in remaining])
    train_val, random_test = train_test_split(
        remaining, test_size=0.15, stratify=remaining_labels, random_state=SEED
    )
    train_val_labels = np.array([unit_labels[unit] for unit in train_val])
    train, val = train_test_split(
        train_val, test_size=0.1764705882, stratify=train_val_labels, random_state=SEED
    )

    def expand(units):
        return sorted(
            group
            for unit in units
            for group in unit_groups[unit]
        )

    # A reviewed false negative can either teach the model or evaluate it, never both.
    splits = {
        'train': expand(set(train) | forced_train_units),
        'validation': expand(val),
        'test': expand(set(random_test) | forced_test_units),
    }
    assert_disjoint_splits(splits)
    duplicate_origins = origin_overlaps(splits)
    if duplicate_origins:
        examples = list(duplicate_origins.items())[:10]
        raise ValueError(
            'Data leakage: YouTube source IDs occur in multiple splits. '
            f'Examples: {examples}'
        )
    return splits, {
        'train': sorted(forced_train),
        'test': sorted(forced - forced_train),
    }


def load_split(yamnet, split_groups_map, sources, clean_paths):
    arrays = {}
    audio_items = {}
    for split, groups in split_groups_map.items():
        X, y = [], []
        audio_items[split] = []
        for group in groups:
            cls = group.split(':', 1)[0]
            label = CLASSES.index(cls)
            if split == 'train':
                paths = sources[group][:MAX_FILES_PER_TRAIN_SOURCE[cls]]
            else:
                paths = [clean_paths[group]] if group in clean_paths else sources[group][:1]
                audio_items[split].append({'group': group, 'label': label, 'path': paths[0]})
            for path in paths:
                try:
                    embeddings = selected_embeddings(yamnet, path, cls)
                    X.extend(embeddings)
                    y.extend([label] * len(embeddings))
                except Exception as exc:
                    print(f'skip {path}: {exc}')
        arrays[split] = (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
        counts = {cls: int(np.sum(arrays[split][1] == idx)) for idx, cls in enumerate(CLASSES)}
        print(f'{split}: frames={len(X)}, groups={len(groups)}, counts={counts}')
    return arrays, audio_items


def audio_items_from_splits(split_groups_map, sources, clean_paths):
    audio_items = {}
    for split in ('validation', 'test'):
        audio_items[split] = []
        for group in split_groups_map[split]:
            cls = group.split(':', 1)[0]
            path = clean_paths.get(group, sources[group][0])
            audio_items[split].append({
                'group': group,
                'label': CLASSES.index(cls),
                'path': path,
            })
    return audio_items


def build_model():
    reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def balanced_indices(y):
    rng = np.random.default_rng(SEED)
    class_indices = [np.flatnonzero(y == label) for label in range(len(CLASSES))]
    target = min(map(len, class_indices))
    return np.concatenate([rng.choice(indices, target, replace=False) for indices in class_indices])


def train_candidate(name, arrays, strategy, initial_model_path=None):
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    X_train, y_train = arrays['train']
    X_val, y_val = arrays['validation']
    class_weight = None
    if strategy == 'balanced-frames':
        indices = balanced_indices(y_train)
        X_train, y_train = X_train[indices], y_train[indices]
    else:
        classes = np.arange(len(CLASSES))
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
    if initial_model_path:
        model = tf.keras.models.load_model(initial_model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
    else:
        model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        X_train, y_train, epochs=60, batch_size=64,
        validation_data=(X_val, y_val), class_weight=class_weight,
        callbacks=callbacks, verbose=2,
    )
    output = os.path.join(MODEL_DIR, 'candidates', f'{name}.h5')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_output = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temporary_output.close()
    try:
        model.save(temporary_output.name)
        shutil.copy2(temporary_output.name, output)
    finally:
        if os.path.exists(temporary_output.name):
            os.remove(temporary_output.name)
    return model, output, {
        'strategy': strategy,
        'epochs': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'train_frames': int(len(y_train)),
    }


def audio_probabilities(model, yamnet, items):
    output = []
    for item in items:
        embeddings = full_embeddings(yamnet, item['path'])
        output.append(model.predict(embeddings, verbose=0))
    return output


def policy_metrics(items, all_probs, policy):
    y_true = np.array([item['label'] for item in items])
    y_pred = np.array([
        CLASSES.index(decide(
            probs,
            thresholds=policy['thresholds'],
            min_consecutive_frames=policy['min_consecutive_frames'],
            min_mean_probs=policy['min_mean_probs'],
            min_prob_margins=policy['min_prob_margins'],
        )[0]) for probs in all_probs
    ])
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    recalls = recall_score(y_true, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    normal = CLASSES.index('normal')
    normal_mask = y_true == normal
    false_alarm = float(np.mean(y_pred[normal_mask] != normal)) if np.any(normal_mask) else 0.0
    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=range(len(CLASSES)), average='macro', zero_division=0)),
        'normal_false_alarm_rate': false_alarm,
        'recall': {cls: float(recalls[idx]) for idx, cls in enumerate(CLASSES)},
        'confusion_matrix': matrix.tolist(),
    }


def tune_thresholds(items, all_probs):
    best = None
    candidates = [value / 100 for value in range(40, 100, 5)]
    policy_options = itertools.product(
        candidates,
        candidates,
        (1, 2, 3),
        (0.0, 0.15, 0.25, 0.35),
        (0.0, 0.10, 0.15),
    )
    for glass, scream, scream_frames, scream_mean, scream_margin in policy_options:
        policy = {
            'thresholds': {'glass': glass, 'scream': scream},
            'min_consecutive_frames': {'glass': 1, 'scream': scream_frames},
            'min_mean_probs': {'scream': scream_mean},
            'min_prob_margins': {'scream': scream_margin},
        }
        metrics = policy_metrics(items, all_probs, policy)
        event_floor = min(metrics['recall']['glass'], metrics['recall']['scream'])
        feasible = metrics['normal_false_alarm_rate'] <= 0.05
        score = metrics['macro_f1'] - metrics['normal_false_alarm_rate'] + 0.15 * event_floor
        record = (feasible, score, metrics['macro_f1'], -metrics['normal_false_alarm_rate'], policy, metrics)
        if best is None or record[:4] > best[:4]:
            best = record
    return best[4], best[5]


def select_policy(model, yamnet, validation_items):
    val_probs = audio_probabilities(model, yamnet, validation_items)
    policy, validation = tune_thresholds(validation_items, val_probs)
    return {'policy': policy, 'validation': validation}


def final_test(model, yamnet, test_items, policy):
    test_probs = audio_probabilities(model, yamnet, test_items)
    return policy_metrics(test_items, test_probs, policy)


def main():
    parser = argparse.ArgumentParser(description='Train capped, source-separated model candidates.')
    parser.add_argument('--promote', action='store_true', help='Promote the best candidate only if it passes gates.')
    parser.add_argument('--evaluate-existing', action='store_true', help='Evaluate already trained candidate files.')
    parser.add_argument(
        '--train-candidate', action='append',
        choices=('candidate_a_capped', 'candidate_b_balanced', 'candidate_c_finetuned'),
        help='Train only selected candidates and reuse other candidate files.',
    )
    parser.add_argument('--prepare-only', action='store_true', help='Write a validated split manifest without loading or training models.')
    parser.add_argument('--baseline-path', default=os.path.join(MODEL_DIR, 'glass_classifier.h5'))
    parser.add_argument('--results-path', default=os.path.join(RESULTS_DIR, 'latest.json'))
    args = parser.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    sources, clean_paths, aliases, excluded_unverified = collect_sources()
    splits, forced = split_groups(sources)

    manifest = {
        'seed': SEED,
        'forced_review_groups': forced,
        'max_files_per_train_source': MAX_FILES_PER_TRAIN_SOURCE,
        'exact_duplicate_source_alias_count': len(aliases),
        'unverified_broad_label_glass_source_exclusion_count': len(excluded_unverified),
        'splits': splits,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest_path = os.path.join(RESULTS_DIR, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    if args.prepare_only:
        print(json.dumps({
            'manifest_path': manifest_path,
            'exact_duplicate_source_alias_count': len(aliases),
            'unverified_broad_label_glass_source_exclusion_count': len(excluded_unverified),
            'split_counts': {name: len(groups) for name, groups in splits.items()},
        }, ensure_ascii=False, indent=2))
        return

    print('Loading YAMNet...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    if args.evaluate_existing:
        arrays = None
        audio_items = audio_items_from_splits(splits, sources, clean_paths)
    else:
        arrays, audio_items = load_split(yamnet, splits, sources, clean_paths)

    baseline_model = tf.keras.models.load_model(args.baseline_path)
    baseline_selection = select_policy(baseline_model, yamnet, audio_items['validation'])
    results = {
        'manifest_path': manifest_path,
        'selection_rule': 'Candidate and policy selected using validation only; test is evaluated once after selection.',
        'baseline': baseline_selection,
        'candidates': {},
    }

    specifications = [
        ('candidate_a_capped', 'class-weight', None),
        ('candidate_b_balanced', 'balanced-frames', None),
        ('candidate_c_finetuned', 'class-weight', args.baseline_path),
    ]
    trained = {}
    for name, strategy, initial_model_path in specifications:
        print(f'\nTraining {name} ({strategy})...')
        path = os.path.join(MODEL_DIR, 'candidates', f'{name}.h5')
        reuse_existing = args.evaluate_existing or (
            args.train_candidate and name not in args.train_candidate
        )
        if reuse_existing:
            model = tf.keras.models.load_model(path)
            training = {'strategy': strategy, 'reused_existing_model': True}
        else:
            model, path, training = train_candidate(name, arrays, strategy, initial_model_path)
        selection = select_policy(model, yamnet, audio_items['validation'])
        passed = passes_deployment_gate(
            selection['validation'], baseline_selection['validation']
        )
        results['candidates'][name] = {
            'path': path,
            'training': training,
            'selection': selection,
            'passed_validation_gate': passed,
        }
        trained[name] = model

    ranked = sorted(
        results['candidates'],
        key=lambda name: rank_key(
            results['candidates'][name]['selection']['validation']
        ),
        reverse=True,
    )
    winner = next(
        (name for name in ranked if results['candidates'][name]['passed_validation_gate']),
        None,
    )
    results['winner'] = winner
    results['final_test'] = {
        'baseline': final_test(
            baseline_model,
            yamnet,
            audio_items['test'],
            baseline_selection['policy'],
        )
    }
    if winner:
        winner_record = results['candidates'][winner]
        results['final_test']['winner'] = {
            'name': winner,
            'metrics': final_test(
                trained[winner],
                yamnet,
                audio_items['test'],
                winner_record['selection']['policy'],
            ),
        }
    results['promoted'] = bool(args.promote and winner)
    if args.promote and winner:
        shutil.copy2(results['candidates'][winner]['path'], os.path.join(MODEL_DIR, 'glass_classifier.h5'))

    os.makedirs(os.path.dirname(os.path.abspath(args.results_path)), exist_ok=True)
    with open(args.results_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
