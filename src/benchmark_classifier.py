import argparse
import json
import os
import statistics
import tempfile
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from audio_pipeline import load_audio


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_H5 = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
DEFAULT_TFLITE = os.path.join(BASE_DIR, 'model', 'glass_classifier.tflite')


def tflite_predict(interpreter, embeddings):
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    wanted_shape = np.asarray(embeddings.shape, dtype=np.int32)
    if not np.array_equal(input_info['shape'], wanted_shape):
        interpreter.resize_tensor_input(input_info['index'], wanted_shape, strict=False)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_info['index'], embeddings.astype(input_info['dtype']))
    interpreter.invoke()
    return interpreter.get_tensor(output_info['index'])


def latency_summary(values):
    ordered = sorted(values)
    p95_index = max(0, int(np.ceil(len(ordered) * 0.95)) - 1)
    return {
        'mean_ms': float(statistics.mean(ordered)),
        'median_ms': float(statistics.median(ordered)),
        'p95_ms': float(ordered[p95_index]),
    }


def load_tflite_interpreter(model_content):
    loaders = []
    try:
        from ai_edge_litert.interpreter import Interpreter
        loaders.append(('ai_edge_litert', Interpreter))
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        loaders.append(('tflite_runtime', Interpreter))
    except ImportError:
        pass
    loaders.append(('tensorflow', tf.lite.Interpreter))
    errors = []
    for runtime, interpreter_class in loaders:
        try:
            return runtime, interpreter_class(model_content=model_content), None
        except Exception as exc:
            message = str(exc).split('Invoked with:', 1)[0].strip()
            errors.append(f'{runtime}: {type(exc).__name__}: {message}')
    return None, None, ' | '.join(errors)


def main():
    parser = argparse.ArgumentParser(description='Compare Keras and TFLite classifier outputs and latency.')
    parser.add_argument('audio_path')
    parser.add_argument('--h5', default=DEFAULT_H5)
    parser.add_argument('--tflite', default=DEFAULT_TFLITE)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--json-output')
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit('--iterations must be at least 1')

    os.environ.setdefault(
        'TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot_fresh')
    )
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    _, embeddings, _ = yamnet(load_audio(args.audio_path))
    embeddings = embeddings.numpy().astype(np.float32)
    keras_model = tf.keras.models.load_model(args.h5)
    # Loading from bytes avoids the Windows TFLite wrapper's non-ASCII path bug.
    with open(args.tflite, 'rb') as handle:
        tflite_model = handle.read()
    runtime, interpreter, tflite_error = load_tflite_interpreter(tflite_model)
    if interpreter is not None:
        interpreter.allocate_tensors()

    keras_output = keras_model.predict(embeddings, verbose=0)
    keras_times = []
    tflite_times = []
    tflite_output = None
    if interpreter is not None:
        tflite_output = tflite_predict(interpreter, embeddings)
    for _ in range(args.iterations):
        start = time.perf_counter()
        keras_model.predict(embeddings, verbose=0)
        keras_times.append((time.perf_counter() - start) * 1000)
        if interpreter is not None:
            start = time.perf_counter()
            tflite_predict(interpreter, embeddings)
            tflite_times.append((time.perf_counter() - start) * 1000)

    output = {
        'audio_path': os.path.abspath(args.audio_path),
        'embedding_frames': int(len(embeddings)),
        'iterations': args.iterations,
        'keras_model': os.path.abspath(args.h5),
        'tflite_model': os.path.abspath(args.tflite),
        'keras_latency': latency_summary(keras_times),
        'scope': 'Classifier head only; YAMNet feature extraction is excluded from latency.',
    }
    if interpreter is not None:
        output['tflite_status'] = 'ok'
        output['tflite_runtime'] = runtime
        output['max_absolute_output_error'] = float(
            np.max(np.abs(keras_output - tflite_output))
        )
        output['frame_label_agreement'] = float(np.mean(
            np.argmax(keras_output, axis=1) == np.argmax(tflite_output, axis=1)
        ))
        output['tflite_latency'] = latency_summary(tflite_times)
    else:
        output['tflite_status'] = 'unavailable'
        output['tflite_error'] = tflite_error
        output['recommended_fix'] = (
            'Use a clean environment with one TensorFlow distribution, or install '
            'ai-edge-litert/tflite-runtime on the target Raspberry Pi.'
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.json_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, 'w', encoding='utf-8') as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
