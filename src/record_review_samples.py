import argparse
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import soundfile as sf

from audio_pipeline import SAMPLE_RATE, resample_audio
from detection_policy import CLASSES, decide_deployment


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'glass_classifier.h5')
RAW_REVIEW_DIR = os.path.join(BASE_DIR, 'test_results', 'manual_review')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEFAULT_DURATION = 3.0
CHANNELS = 1
LABEL_ALIASES = {
    'g': 'glass',
    'glass': 'glass',
    'n': 'normal',
    'normal': 'normal',
    's': 'scream',
    'scream': 'scream',
}


def import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print('sounddevice 패키지가 필요합니다.')
        print('설치: python -m pip install sounddevice')
        print('라즈베리파이 PortAudio 오류: sudo apt-get install portaudio19-dev')
        sys.exit(1)


def load_models():
    os.environ.setdefault('TFHUB_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'tfhub_cache_hufs_iot'))

    import tensorflow as tf
    import tensorflow_hub as hub

    print('YAMNet 로드 중...')
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

    print('분류기 로드 중...')
    classifier = tf.keras.models.load_model(MODEL_PATH)
    return yamnet, classifier


def resolve_input_sample_rate(sd, device, requested_sample_rate):
    if requested_sample_rate:
        return int(requested_sample_rate)

    info = sd.query_devices(device, 'input')
    return int(info['default_samplerate'])


def normalize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def record_clip(sd, duration, device, input_sample_rate):
    recording = sd.rec(
        int(duration * input_sample_rate),
        samplerate=input_sample_rate,
        channels=CHANNELS,
        dtype='float32',
        device=device,
    )
    sd.wait()

    audio = normalize_audio(recording)
    if input_sample_rate != SAMPLE_RATE:
        audio = resample_audio(audio, input_sample_rate, SAMPLE_RATE)
    return audio


def predict_audio(yamnet, classifier, audio):
    yamnet_scores, embeddings, _ = yamnet(audio.astype(np.float32))
    probs = classifier.predict(embeddings.numpy(), verbose=0)
    final, mean_probs, max_probs, consecutive_runs = decide_deployment(probs, yamnet_scores.numpy())
    return final, mean_probs, max_probs, consecutive_runs


def format_scores(probs):
    return ' '.join(
        f'{cls}={probs[idx] * 100:5.1f}%'
        for idx, cls in enumerate(CLASSES)
    )


def audio_stats(audio):
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return rms, peak


def save_wav(folder, filename, audio):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    sf.write(path, audio, SAMPLE_RATE)
    return path


def ask_actual_label(predicted):
    prompt = (
        f'실제 라벨 입력 '
        f'[Enter=정답({predicted}), g=glass, n=normal, s=scream, r=재녹음, q=종료]: '
    )
    while True:
        value = input(prompt).strip().lower()
        if value == '':
            return predicted
        if value in ('q', 'quit', 'exit'):
            return 'quit'
        if value in ('r', 'retry'):
            return 'retry'
        if value in LABEL_ALIASES:
            return LABEL_ALIASES[value]
        print('입력값을 이해하지 못했습니다. Enter, g, n, s, r, q 중 하나를 입력하세요.')


def reviewed_filename(actual, predicted, index):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'review_{actual}_pred_{predicted}_{timestamp}_{index:03d}.wav'


def run(args):
    sd = import_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return

    yamnet, classifier = load_models()
    input_sample_rate = resolve_input_sample_rate(sd, args.device, args.input_sample_rate)

    print(f'입력 장치: {args.device if args.device is not None else "default"}')
    print(f'녹음 설정: input={input_sample_rate}Hz mono -> save={SAMPLE_RATE}Hz, {args.duration:.1f}s')
    print(f'오판 학습 후보 저장: {DATA_DIR}\\<actual_label>')
    print(f'전체 리뷰 원본 저장: {RAW_REVIEW_DIR}')
    print('중단: q 입력 또는 Ctrl+C')

    saved_mistakes = []
    correct_count = 0
    trial = 1

    while trial <= args.count:
        print(f'\n[{trial}/{args.count}] 준비...')
        time.sleep(args.prepare)
        print('녹음 중...')
        audio = record_clip(sd, args.duration, args.device, input_sample_rate)
        rms, peak = audio_stats(audio)

        predicted, mean_probs, max_probs, consecutive_runs = predict_audio(yamnet, classifier, audio)
        raw_name = reviewed_filename('unknown', predicted, trial)
        raw_path = save_wav(RAW_REVIEW_DIR, raw_name, audio)

        print(f'판정: {predicted}  rms={rms:.4f} peak={peak:.4f}')
        print(f'최대: {format_scores(max_probs)}')
        print(f'평균: {format_scores(mean_probs)}')
        print(f'연속: {consecutive_runs}')
        print(f'원본: {raw_path}')

        actual = ask_actual_label(predicted)
        if actual == 'quit':
            break
        if actual == 'retry':
            print('이번 녹음은 저장 후보에서 제외하고 다시 녹음합니다.')
            continue

        if actual == predicted:
            correct_count += 1
            print('정답 처리: 학습 데이터에는 추가하지 않습니다.')
        else:
            filename = reviewed_filename(actual, predicted, trial)
            data_path = save_wav(os.path.join(DATA_DIR, actual), filename, audio)
            saved_mistakes.append((actual, predicted, data_path))
            print(f'오판 저장: {data_path}')
            print('다음 재학습 때 preprocess.py가 data_clean으로 변환합니다.')

        trial += 1

    print('\n완료')
    print(f'정답 처리: {correct_count}개')
    print(f'오판 저장: {len(saved_mistakes)}개')
    for actual, predicted, path in saved_mistakes:
        print(f'  actual={actual} predicted={predicted} path={path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Record clips, run prediction, and save only misclassified clips for retraining.'
    )
    parser.add_argument('--count', type=int, default=10, help='테스트 녹음 횟수')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION, help='녹음 길이 초')
    parser.add_argument('--prepare', type=float, default=2.0, help='녹음 전 대기 시간 초')
    parser.add_argument('--device', default=None, help='sounddevice 입력 장치 ID 또는 이름')
    parser.add_argument(
        '--input-sample-rate',
        type=int,
        default=None,
        help='마이크 입력 sample rate. 생략하면 장치 default_samplerate를 사용합니다.',
    )
    parser.add_argument('--list-devices', action='store_true', help='오디오 장치 목록 출력')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print('\n중단됨')
