import os
import re
import hashlib
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from audio_pipeline import SAMPLE_RATE, load_audio, rms

# ── 설정 ──────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))
DATA_DIRS = [
    os.path.join(BASE_DIR, 'data_clean'),
    os.path.join(BASE_DIR, 'data_augmented'),
    os.path.join(BASE_DIR, 'data_mixed'),
]
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
EMBEDDING_CACHE_DIR = os.path.join(MODEL_DIR, 'embedding_cache')
CLASSES     = ['glass', 'normal', 'scream']  # 0=유리파손, 1=일반, 2=비명
SEED = 42
EMBEDDING_CACHE_VERSION = 'frame-v1-no-peak-normalization'
MAX_FRAMES_PER_FILE = {
    'glass': 2,
    'normal': 3,
    'scream': 4,
}

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── YAMNet 로드 ────────────────────────
print("YAMNet 로드 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# ── 오디오 → embedding 변환 ────────────
def cache_path_for(path):
    rel_path = os.path.relpath(path, BASE_DIR).replace('\\', '/')
    stat = os.stat(path)
    key = f'{EMBEDDING_CACHE_VERSION}|{rel_path}|{stat.st_size}|{stat.st_mtime_ns}'
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return os.path.join(EMBEDDING_CACHE_DIR, f'{digest}.npy')


def embedding_frame_rms(audio, frame_count):
    if frame_count <= 0:
        return np.array([], dtype=np.float32)
    frame_length = min(len(audio), int(0.96 * SAMPLE_RATE))
    if frame_count == 1:
        starts = [max(0, (len(audio) - frame_length) // 2)]
    else:
        starts = np.linspace(0, max(0, len(audio) - frame_length), frame_count).astype(int)
    return np.array([
        rms(audio[start:start + frame_length])
        for start in starts
    ], dtype=np.float32)


def select_embedding_frames(embeddings, audio, cls):
    frame_count = len(embeddings)
    if frame_count <= MAX_FRAMES_PER_FILE[cls]:
        return embeddings

    levels = embedding_frame_rms(audio, frame_count)
    if cls == 'normal':
        indices = np.linspace(0, frame_count - 1, MAX_FRAMES_PER_FILE[cls]).astype(int)
    else:
        relative_floor = max(float(levels.max()) * 0.35, float(np.percentile(levels, 60)))
        active = np.flatnonzero(levels >= relative_floor)
        if len(active) == 0:
            active = np.array([int(np.argmax(levels))])
        ranked = active[np.argsort(levels[active])[::-1]]
        indices = np.sort(ranked[:MAX_FRAMES_PER_FILE[cls]])
    return embeddings[indices]


def wav_to_embeddings(path, cls):
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    cache_path = cache_path_for(path)
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        audio = load_audio(path)
        return select_embedding_frames(embeddings, audio, cls)

    audio = load_audio(path)
    _, embeddings, _ = yamnet(audio)
    embeddings = embeddings.numpy()
    np.save(cache_path, embeddings)
    return select_embedding_frames(embeddings, audio, cls)

def source_group(cls, fname):
    stem = os.path.splitext(fname)[0]
    stem = re.sub(r'_aug_\d+$', '', stem)
    stem = re.sub(r'_clean$', '', stem)
    return f'{cls}:{stem}'

# ── 데이터 로드 ────────────────────────
print("데이터 로드 중...")
X, y, groups = [], [], []
for label, cls in enumerate(CLASSES):
    class_count = 0
    for data_dir in DATA_DIRS:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue
        files  = [f for f in os.listdir(folder) if f.lower().endswith(('.wav','.mp3','.flac','.m4a','.webm','.ogg'))]
        print(f"  {cls} / {os.path.basename(data_dir)}: {len(files)}개")
        for fname in files:
            try:
                embeddings = wav_to_embeddings(os.path.join(folder, fname), cls)
                group = source_group(cls, fname)
                X.extend(embeddings)
                y.extend([label] * len(embeddings))
                groups.extend([group] * len(embeddings))
                class_count += len(embeddings)
            except Exception as e:
                print(f"  skip: {fname} ({e})")
    print(f"  {cls}: 총 {class_count}개")

X = np.array(X)
y = np.array(y)
groups = np.array(groups)
print(f"총 {len(X)}개 샘플 로드 완료")
if len(X) == 0:
    raise RuntimeError("학습할 오디오 샘플을 찾지 못했습니다.")

# ── 분류기 학습 ────────────────────────
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
    tf.keras.layers.Dense(3, activation='softmax'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

unique_groups = np.array(sorted(set(groups)))
group_labels = np.array([
    y[np.where(groups == group)[0][0]]
    for group in unique_groups
])
train_groups, val_groups = train_test_split(
    unique_groups,
    test_size=0.2,
    stratify=group_labels,
    random_state=42
)
train_mask = np.isin(groups, train_groups)
val_mask = np.isin(groups, val_groups)
X_train, X_val = X[train_mask], X[val_mask]
y_train, y_val = y[train_mask], y[val_mask]

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"class_weight: {class_weight_dict}")

print(f"train: {len(X_train)}개 / val: {len(X_val)}개")
print(f"groups: train={len(train_groups)}개 / val={len(val_groups)}개")
for label, cls in enumerate(CLASSES):
    print(f"  {cls}: train={np.sum(y_train==label)}, val={np.sum(y_val==label)}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    X_train,
    y_train,
    epochs=120,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
best_val_accuracy = max(history.history.get('val_accuracy', [0.0]))
print(f"best_val_accuracy: {best_val_accuracy:.4f}")

# ── 모델 저장 ──────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, 'glass_classifier.h5'))
print("모델 저장 완료!")
