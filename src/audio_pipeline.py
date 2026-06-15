import numpy as np
import soundfile as sf
from scipy import signal


SAMPLE_RATE = 16000


def to_mono_float32(audio):
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def resample_audio(audio, source_rate, target_rate=SAMPLE_RATE):
    audio = to_mono_float32(audio)
    if source_rate == target_rate:
        return audio

    divisor = np.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return signal.resample_poly(audio, up, down).astype(np.float32)


def load_audio(path, target_rate=SAMPLE_RATE):
    audio, source_rate = sf.read(path)
    return resample_audio(audio, source_rate, target_rate)


def rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def peak_frame_rms(audio, frame_length, hop_length=None):
    audio = to_mono_float32(audio)
    if audio.size == 0:
        return 0.0
    frame_length = min(int(frame_length), len(audio))
    hop_length = int(hop_length or max(1, frame_length // 2))
    return max(
        rms(audio[start:start + frame_length])
        for start in range(0, max(1, len(audio) - frame_length + 1), hop_length)
    )


def fit_length(audio, length):
    audio = to_mono_float32(audio)
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio))).astype(np.float32)


def loudest_window(audio, length):
    audio = to_mono_float32(audio)
    if len(audio) <= length:
        return fit_length(audio, length)

    hop_length = max(1, length // 2)
    starts = list(range(0, len(audio) - length + 1, hop_length))
    if starts[-1] != len(audio) - length:
        starts.append(len(audio) - length)
    best_start = max(starts, key=lambda start: rms(audio[start:start + length]))
    return audio[best_start:best_start + length].astype(np.float32)
