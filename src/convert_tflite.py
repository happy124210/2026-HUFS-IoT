import tensorflow as tf
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
h5_path     = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
tflite_path = os.path.join(BASE_DIR, 'model', 'glass_classifier.tflite')

model = tf.keras.models.load_model(h5_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"변환 완료! 크기: {len(tflite_model)/1024:.1f} KB")
print(f"저장 위치: {tflite_path}")
