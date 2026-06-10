import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy import signal


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'normal')
SAMPLE_RATE = 16000
DURATION = 3.0
CHANNELS = 1

SCENARIOS = [
    'speech',
    'clap',
    'door_close',
    'knock',
    'desk_hit',
    'book_drop',
    'keys',
    'chair_move',
    'footstep',
    'keyboard',
    'plastic_bag',
    'water_bottle',
    'cough',
    'laugh',
    'ambient',
]


def import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        print('sounddevice 패키지가 필요합니다.')
        print('설치 예시:')
        print('  python -m pip install sounddevice')
        print('라즈베리파이에서 PortAudio 오류가 나면:')
        print('  sudo apt-get install portaudio19-dev')
        sys.exit(1)


def next_index(prefix):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [
        name for name in os.listdir(OUTPUT_DIR)
        if name.startswith(prefix) and name.lower().endswith('.wav')
    ]
    return len(existing)


def normalize_int16(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def resolve_input_sample_rate(sd, device, requested_sample_rate):
    if requested_sample_rate:
        return int(requested_sample_rate)

    info = sd.query_devices(device, 'input')
    return int(info['default_samplerate'])


def resample_to_save_rate(audio, input_sample_rate):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if input_sample_rate == SAMPLE_RATE:
        return audio

    target_len = int(round(len(audio) * SAMPLE_RATE / input_sample_rate))
    return signal.resample(audio, target_len).astype(np.float32)


def record_clip(sd, duration, device, input_sample_rate):
    recording = sd.rec(
        int(duration * input_sample_rate),
        samplerate=input_sample_rate,
        channels=CHANNELS,
        dtype='float32',
        device=device,
    )
    sd.wait()
    audio = normalize_int16(recording)
    return resample_to_save_rate(audio, input_sample_rate)


def save_clip(audio, scenario, index):
    filename = f'fp_normal_{scenario}_{index:03d}.wav'
    path = os.path.join(OUTPUT_DIR, filename)
    sf.write(path, audio, SAMPLE_RATE)
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return path, rms, peak


def list_devices(sd):
    print(sd.query_devices())


def collect(args):
    sd = import_sounddevice()
    if args.list_devices:
        list_devices(sd)
        return

    scenarios = args.scenario or SCENARIOS
    input_sample_rate = resolve_input_sample_rate(sd, args.device, args.input_sample_rate)
    print(f'저장 위치: {OUTPUT_DIR}')
    print(f'녹음 설정: input={input_sample_rate}Hz mono -> save={SAMPLE_RATE}Hz, {args.duration:.1f}s')
    print('중단하려면 Ctrl+C를 누르세요.')

    for scenario in scenarios:
        prefix = f'fp_normal_{scenario}_'
        index = next_index(prefix)
        print(f'\n[{scenario}] {args.count}개 녹음')
        for _ in range(args.count):
            print(f'  준비: {scenario} #{index:03d}')
            time.sleep(args.prepare)
            print('  녹음 중...')
            audio = record_clip(sd, args.duration, args.device, input_sample_rate)
            path, rms, peak = save_clip(audio, scenario, index)
            print(f'  저장: {path}  rms={rms:.4f} peak={peak:.4f}')
            index += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Record false-positive normal clips for realtime sound detection.'
    )
    parser.add_argument(
        '--scenario',
        action='append',
        choices=SCENARIOS,
        help='녹음할 normal 상황. 여러 번 지정 가능. 생략하면 전체 시나리오를 순회합니다.',
    )
    parser.add_argument('--count', type=int, default=3, help='시나리오별 녹음 개수')
    parser.add_argument('--duration', type=float, default=DURATION, help='클립 길이 초')
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
    collect(parse_args())
