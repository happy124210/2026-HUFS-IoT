import argparse
import os

import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description='Convert a Keras H5 classifier to TFLite.')
    parser.add_argument('--input', default=os.path.join(BASE_DIR, 'model', 'glass_classifier.h5'))
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'model', 'glass_classifier.tflite'))
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(tflite_model)

    print(f"변환 완료! 크기: {len(tflite_model)/1024:.1f} KB")
    print(f"저장 위치: {args.output}")


if __name__ == '__main__':
    main()
