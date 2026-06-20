import numpy as np


CLASSES = ['glass', 'normal', 'scream']
THRESHOLDS = {
    'glass': 0.95,
    'scream': 0.40,
}
MIN_CONSECUTIVE_FRAMES = {
    'glass': 1,
    'scream': 3,
}
MIN_MEAN_PROBS = {'scream': 0.0}
MIN_PROB_MARGINS = {'scream': 0.0}

# Locked deployment policy selected on validation in
# test_results/training/run_20260620_strict_labels.json. Production inference,
# offline evaluation, and the live demo must use these classifier-only values.
DEPLOYMENT_POLICY = {
    'thresholds': THRESHOLDS,
    'min_consecutive_frames': MIN_CONSECUTIVE_FRAMES,
    'min_mean_probs': MIN_MEAN_PROBS,
    'min_prob_margins': MIN_PROB_MARGINS,
}

# YAMNet AudioSet output indices. These scores are used only as corroborating
# evidence; the project classifier remains the primary detector.
YAMNET_EVENT_INDICES = {
    'scream': (11,),              # Screaming
    'glass': (435, 437),          # Glass, Shatter
}
YAMNET_SUPPORT_THRESHOLDS = {
    'scream': 0.05,
    'glass': 0.10,
}
YAMNET_CUSTOM_FLOORS = {
    # Historical fusion ablations only; not used by the locked deployment path.
    'scream': 0.92,
    'glass': 0.70,
}
YAMNET_MIN_CONSECUTIVE_FRAMES = {
    'scream': 2,
    'glass': 1,
}


def longest_consecutive_run(mask):
    longest = 0
    current = 0
    for matched in mask:
        current = current + 1 if matched else 0
        longest = max(longest, current)
    return longest


def decide(
    probs,
    yamnet_scores=None,
    thresholds=None,
    min_consecutive_frames=None,
    min_mean_probs=None,
    min_prob_margins=None,
):
    thresholds = thresholds or THRESHOLDS
    min_consecutive_frames = min_consecutive_frames or MIN_CONSECUTIVE_FRAMES
    min_mean_probs = min_mean_probs or MIN_MEAN_PROBS
    min_prob_margins = min_prob_margins or MIN_PROB_MARGINS
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(f'Expected probabilities shaped (frames, {len(CLASSES)}), got {probs.shape}')

    max_probs = probs.max(axis=0)
    mean_probs = probs.mean(axis=0)
    frame_margins = {}
    consecutive_runs = {}
    for cls in thresholds:
        cls_index = CLASSES.index(cls)
        other_indices = [idx for idx in range(len(CLASSES)) if idx != cls_index]
        frame_margins[cls] = probs[:, cls_index] - probs[:, other_indices].max(axis=1)
        strict_mask = probs[:, cls_index] >= thresholds[cls]
        if cls in min_prob_margins:
            strict_mask &= frame_margins[cls] >= min_prob_margins[cls]
        consecutive_runs[cls] = longest_consecutive_run(strict_mask)

    yamnet_event_probs = None
    if yamnet_scores is not None:
        yamnet_scores = np.asarray(yamnet_scores)
        if yamnet_scores.ndim != 2:
            raise ValueError(f'Expected 2-D YAMNet scores, got {yamnet_scores.shape}')
        if yamnet_scores.shape[0] != probs.shape[0]:
            raise ValueError(
                f'Classifier/YAMNet frame count mismatch: {probs.shape[0]} != {yamnet_scores.shape[0]}'
            )
        yamnet_event_probs = {
            cls: yamnet_scores[:, indices].max(axis=1)
            for cls, indices in YAMNET_EVENT_INDICES.items()
        }

    triggered = []
    for cls in thresholds:
        cls_index = CLASSES.index(cls)
        mean_ok = mean_probs[cls_index] >= min_mean_probs.get(cls, 0.0)
        strict = consecutive_runs[cls] >= min_consecutive_frames[cls] and mean_ok

        corroborated = False
        if yamnet_event_probs is not None and cls in yamnet_event_probs:
            custom_mask = probs[:, cls_index] >= YAMNET_CUSTOM_FLOORS[cls]
            custom_run = longest_consecutive_run(custom_mask)
            yamnet_support = float(yamnet_event_probs[cls].max()) >= YAMNET_SUPPORT_THRESHOLDS[cls]
            corroborated = (
                custom_run >= YAMNET_MIN_CONSECUTIVE_FRAMES[cls]
                and yamnet_support
                and mean_ok
            )

        if strict or corroborated:
            triggered.append(cls)

    final = max(triggered, key=lambda cls: max_probs[CLASSES.index(cls)]) if triggered else 'normal'
    return final, mean_probs, max_probs, consecutive_runs


def decide_deployment(probs):
    """Apply the locked classifier-only policy used for official evaluation."""
    return decide(probs, **DEPLOYMENT_POLICY)
