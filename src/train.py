import os
import re
import hashlib
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

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
SAMPLE_RATE = 16000
CLASSES     = ['glass', 'normal', 'scream']  # 0=유리파손, 1=일반, 2=비명
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── YAMNet 로드 ────────────────────────
print("YAMNet 로드 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# ── 오디오 → embedding 변환 ────────────
def load_audio(path):
    if path.lower().endswith('.wav'):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio.astype(np.float32)
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def cache_path_for(path):
    rel_path = os.path.relpath(path, BASE_DIR).replace('\\', '/')
    stat = os.stat(path)
    key = f'{rel_path}|{stat.st_size}|{stat.st_mtime_ns}'
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return os.path.join(EMBEDDING_CACHE_DIR, f'{digest}.npy')


def wav_to_embedding(path):
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    cache_path = cache_path_for(path)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    audio = load_audio(path)
    audio = audio.astype(np.float32)
    _, embeddings, _ = yamnet(audio)
    emb = tf.reduce_mean(embeddings, axis=0).numpy()  # (1024,)
    np.save(cache_path, emb)
    return emb

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
                emb = wav_to_embedding(os.path.join(folder, fname))
                X.append(emb)
                y.append(label)
                groups.append(source_group(cls, fname))
                class_count += 1
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
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))
print(f"class_weight: {class_weight_dict}")

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

print(f"train: {len(X_train)}개 / val: {len(X_val)}개")
print(f"groups: train={len(train_groups)}개 / val={len(val_groups)}개")
for label, cls in enumerate(CLASSES):
    print(f"  {cls}: train={np.sum(y_train==label)}, val={np.sum(y_val==label)}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    X_train,
    y_train,
    epochs=60,
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
