import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ── 설정 ──────────────────────────────
DATA_DIR = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT\\data_clean'
MODEL_DIR  = '../model'
SAMPLE_RATE = 16000
CLASSES     = ['glass', 'normal', 'scream']  # 0=유리파손, 1=일반, 2=비명

# ── YAMNet 로드 ────────────────────────
print("YAMNet 로드 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# ── 오디오 → embedding 변환 ────────────
def wav_to_embedding(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    _, embeddings, _ = yamnet(audio)
    return tf.reduce_mean(embeddings, axis=0).numpy()  # (1024,)

# ── 데이터 로드 ────────────────────────
print("데이터 로드 중...")
X, y = [], []
for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, cls)
    files  = [f for f in os.listdir(folder) if f.endswith(('.wav','.mp3'))]
    print(f"  {cls}: {len(files)}개")
    for fname in files:
        try:
            emb = wav_to_embedding(os.path.join(folder, fname))
            X.append(emb)
            y.append(label)
        except Exception as e:
            print(f"  skip: {fname} ({e})")

X = np.array(X)
y = np.array(y)
print(f"총 {len(X)}개 샘플 로드 완료")

# ── 분류기 학습 ────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1024,)),
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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"train: {len(X_train)}개 / val: {len(X_val)}개")
for label, cls in enumerate(CLASSES):
    print(f"  {cls}: train={np.sum(y_train==label)}, val={np.sum(y_val==label)}")

model.fit(X_train, y_train, epochs=30,
          validation_data=(X_val, y_val),
          class_weight=class_weight_dict, verbose=1)

# ── 모델 저장 ──────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, 'glass_classifier.h5'))
print("모델 저장 완료!")