import argparse
import os
import queue
import sys
import tempfile
import time

import numpy as np
from scipy import signal


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
SAMPLE_RATE = 16000
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.0
CLASSES = ['glass', 'normal', 'scream']
THRESHOLDS = {
    'glass': 0.88,
    'scream': 0.85,
}
MIN_EVENT_FRAMES = 2


def import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print('sounddevice 패키지가 필요합니다.')
        print('설치: python -m pip install sounddevice')
        print('라즈베리파이 PortAudio 오류: sudo apt-get install portaudio19-dev')
        sys.exit(1)


def load_models(no_model):
    if no_model:
        return None, None

    os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))

    import tensorflow as tf
    import tensorflow_hub as hub

    print('YAMNet 로드 중...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

    print('분류기 로드 중...')
    classifier = tf.keras.models.load_model(MODEL_PATH)
    return yamnet, classifier


def frame_predictions(yamnet, classifier, audio):
    _, embeddings, _ = yamnet(audio.astype(np.float32))
    embeddings = embeddings.numpy()
    probs = classifier.predict(embeddings, verbose=0)
    return probs


def decide(probs):
    max_probs = probs.max(axis=0)
    mean_probs = probs.mean(axis=0)
    event_counts = {
        cls: int(np.sum(probs[:, CLASSES.index(cls)] >= THRESHOLDS[cls]))
        for cls in THRESHOLDS
    }
    triggered = [
        cls for cls in THRESHOLDS
        if max_probs[CLASSES.index(cls)] >= THRESHOLDS[cls]
        and event_counts[cls] >= MIN_EVENT_FRAMES
    ]
    if triggered:
        final = max(triggered, key=lambda cls: max_probs[CLASSES.index(cls)])
    else:
        final = 'normal'
    return final, mean_probs, max_probs, event_counts


def format_scores(probs):
    return ' '.join(
        f'{cls}={probs[idx] * 100:5.1f}%'
        for idx, cls in enumerate(CLASSES)
    )


def list_devices(sd):
    print(sd.query_devices())


def resolve_input_sample_rate(sd, device, requested_sample_rate):
    if requested_sample_rate:
        return int(requested_sample_rate)

    info = sd.query_devices(device, 'input')
    default_sr = int(info['default_samplerate'])
    return default_sr


def resample_to_model_rate(audio, input_sample_rate):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if input_sample_rate == SAMPLE_RATE:
        return audio

    target_len = int(round(len(audio) * SAMPLE_RATE / input_sample_rate))
    return signal.resample(audio, target_len).astype(np.float32)


def run_loop(args):
    sd = import_sounddevice()
    if args.list_devices:
        list_devices(sd)
        return

    yamnet, classifier = load_models(args.no_model)

    input_sample_rate = resolve_input_sample_rate(sd, args.device, args.input_sample_rate)
    window_len = int(args.window * input_sample_rate)
    hop_len = int(args.hop * input_sample_rate)
    audio_buffer = np.zeros(window_len, dtype=np.float32)
    blocks = queue.Queue()
    last_event_time = 0.0
    started_at = time.time()

    def callback(indata, frames, callback_time, status):
        if status:
            print(status, file=sys.stderr)
        blocks.put(indata[:, 0].copy())

    print(f'입력 장치: {args.device if args.device is not None else "default"}')
    print(
        f'루프: input={input_sample_rate}Hz mono -> model={SAMPLE_RATE}Hz, '
        f'window={args.window:.1f}s, hop={args.hop:.1f}s'
    )
    print('중단: Ctrl+C')

    with sd.InputStream(
        samplerate=input_sample_rate,
        channels=1,
        dtype='float32',
        blocksize=hop_len,
        device=args.device,
        callback=callback,
    ):
        while True:
            block = blocks.get()
            if len(block) != hop_len:
                if len(block) > hop_len:
                    block = block[:hop_len]
                else:
                    block = np.pad(block, (0, hop_len - len(block)))

            audio_buffer[:-hop_len] = audio_buffer[hop_len:]
            audio_buffer[-hop_len:] = block

            rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
            peak = float(np.max(np.abs(audio_buffer)))
            elapsed = time.time() - started_at

            if elapsed < args.window:
                print(f'[warming] rms={rms:.4f} peak={peak:.4f}')
                continue

            if args.no_model:
                print(f'[audio] rms={rms:.4f} peak={peak:.4f}')
                continue

            model_audio = resample_to_model_rate(audio_buffer, input_sample_rate)
            probs = frame_predictions(yamnet, classifier, model_audio)
            final, mean_probs, max_probs, event_counts = decide(probs)

            now = time.time()
            can_alert = now - last_event_time >= args.cooldown
            alert = final != 'normal' and can_alert
            if alert:
                last_event_time = now

            marker = 'ALERT' if alert else '     '
            count_text = ' '.join(f'{cls}_frames={event_counts[cls]}' for cls in THRESHOLDS)
            print(
                f'[{marker}] final={final:<6} rms={rms:.4f} peak={peak:.4f} '
                f'max({format_scores(max_probs)}) mean({format_scores(mean_probs)}) {count_text}'
            )


def parse_args():
    parser = argparse.ArgumentParser(description='Realtime microphone loop for glass/scream detection.')
    parser.add_argument('--list-devices', action='store_true', help='오디오 입력 장치 목록 출력')
    parser.add_argument('--device', default=None, help='sounddevice 입력 장치 ID 또는 이름')
    parser.add_argument('--window', type=float, default=WINDOW_SECONDS, help='판정 윈도우 길이 초')
    parser.add_argument('--hop', type=float, default=HOP_SECONDS, help='판정 간격 초')
    parser.add_argument('--cooldown', type=float, default=3.0, help='이벤트 재알림 최소 간격 초')
    parser.add_argument(
        '--input-sample-rate',
        type=int,
        default=None,
        help='마이크 입력 sample rate. 생략하면 장치 default_samplerate를 사용합니다.',
    )
    parser.add_argument('--no-model', action='store_true', help='모델 없이 마이크 레벨만 출력')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        run_loop(parse_args())
    except KeyboardInterrupt:
        print('\n중단됨')
