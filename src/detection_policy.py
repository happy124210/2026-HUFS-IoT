import numpy as np


CLASSES = ['glass', 'normal', 'scream']
THRESHOLDS = {
    'glass': 0.97,
    'scream': 0.92,
}
MIN_CONSECUTIVE_FRAMES = {
    'glass': 1,
    'scream': 2,
}


def longest_consecutive_run(mask):
    longest = 0
    current = 0
    for matched in mask:
        current = current + 1 if matched else 0
        longest = max(longest, current)
    return longest


def decide(probs, thresholds=None, min_consecutive_frames=None):
    thresholds = thresholds or THRESHOLDS
    min_consecutive_frames = min_consecutive_frames or MIN_CONSECUTIVE_FRAMES
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(f'Expected probabilities shaped (frames, {len(CLASSES)}), got {probs.shape}')

    max_probs = probs.max(axis=0)
    mean_probs = probs.mean(axis=0)
    consecutive_runs = {
        cls: longest_consecutive_run(probs[:, CLASSES.index(cls)] >= thresholds[cls])
        for cls in thresholds
    }
    triggered = [
        cls for cls in thresholds
        if consecutive_runs[cls] >= min_consecutive_frames[cls]
    ]
    final = max(triggered, key=lambda cls: max_probs[CLASSES.index(cls)]) if triggered else 'normal'
    return final, mean_probs, max_probs, consecutive_runs
