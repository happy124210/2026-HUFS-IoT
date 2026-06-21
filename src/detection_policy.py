import numpy as np


CLASSES = ['glass', 'normal', 'scream']

# Final frame-aligned fusion policy selected on the source-separated validation
# set on 2026-06-21. YAMNet scores and embeddings come from one forward pass.
FINAL_FUSION_POLICY = {
    'glass': {
        'strong_threshold': 0.95,
        'strong_min_frames': 2,
        'fusion_custom_threshold': 0.40,
        'fusion_yamnet_threshold': 0.10,
        'fusion_min_frames': 1,
    },
    'scream': {
        'strong_threshold': 0.55,
        'strong_min_frames': 3,
        'fusion_custom_threshold': 0.85,
        'fusion_yamnet_threshold': 0.02,
        'fusion_min_frames': 1,
    },
}

# Compatibility constants used by runtime status displays.
THRESHOLDS = {
    cls: values['strong_threshold'] for cls, values in FINAL_FUSION_POLICY.items()
}
MIN_CONSECUTIVE_FRAMES = {
    cls: values['strong_min_frames'] for cls, values in FINAL_FUSION_POLICY.items()
}
MIN_MEAN_PROBS = {'scream': 0.0}
MIN_PROB_MARGINS = {'scream': 0.0}
DEPLOYMENT_POLICY = FINAL_FUSION_POLICY

# Historical classifier-only policy retained solely to reproduce the June 20
# comparison. It is no longer the final runtime policy.
LEGACY_CLASSIFIER_ONLY_POLICY = {
    'thresholds': {'glass': 0.95, 'scream': 0.40},
    'min_consecutive_frames': {'glass': 1, 'scream': 3},
    'min_mean_probs': {'scream': 0.0},
    'min_prob_margins': {'scream': 0.0},
}

YAMNET_EVENT_INDICES = {
    'scream': (11,),              # Screaming
    'glass': (435, 437),          # Glass, Shatter
}


def longest_consecutive_run(mask):
    longest = 0
    current = 0
    for matched in mask:
        current = current + 1 if matched else 0
        longest = max(longest, current)
    return longest


def _validate_probs(probs):
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(
            f'Expected probabilities shaped (frames, {len(CLASSES)}), got {probs.shape}'
        )
    return probs


def _validate_yamnet_scores(yamnet_scores, frame_count):
    yamnet_scores = np.asarray(yamnet_scores)
    if yamnet_scores.ndim != 2:
        raise ValueError(f'Expected 2-D YAMNet scores, got {yamnet_scores.shape}')
    if yamnet_scores.shape[0] != frame_count:
        raise ValueError(
            f'Classifier/YAMNet frame count mismatch: {frame_count} != {yamnet_scores.shape[0]}'
        )
    return yamnet_scores


def decide(
    probs,
    yamnet_scores=None,
    thresholds=None,
    min_consecutive_frames=None,
    min_mean_probs=None,
    min_prob_margins=None,
):
    """Legacy configurable classifier-only decision used by training utilities."""
    thresholds = thresholds or LEGACY_CLASSIFIER_ONLY_POLICY['thresholds']
    min_consecutive_frames = (
        min_consecutive_frames
        or LEGACY_CLASSIFIER_ONLY_POLICY['min_consecutive_frames']
    )
    min_mean_probs = min_mean_probs or LEGACY_CLASSIFIER_ONLY_POLICY['min_mean_probs']
    min_prob_margins = (
        min_prob_margins or LEGACY_CLASSIFIER_ONLY_POLICY['min_prob_margins']
    )
    probs = _validate_probs(probs)
    max_probs = probs.max(axis=0)
    mean_probs = probs.mean(axis=0)
    consecutive_runs = {}
    triggered = []
    for cls in thresholds:
        cls_index = CLASSES.index(cls)
        other_indices = [index for index in range(len(CLASSES)) if index != cls_index]
        margin = probs[:, cls_index] - probs[:, other_indices].max(axis=1)
        mask = probs[:, cls_index] >= thresholds[cls]
        if cls in min_prob_margins:
            mask &= margin >= min_prob_margins[cls]
        consecutive_runs[cls] = longest_consecutive_run(mask)
        if (
            consecutive_runs[cls] >= min_consecutive_frames[cls]
            and mean_probs[cls_index] >= min_mean_probs.get(cls, 0.0)
        ):
            triggered.append(cls)
    final = max(
        triggered, key=lambda cls: max_probs[CLASSES.index(cls)]
    ) if triggered else 'normal'
    return final, mean_probs, max_probs, consecutive_runs


def decide_fusion(probs, yamnet_scores, policy=None):
    """Apply strong-classifier OR frame-aligned classifier/YAMNet fusion paths."""
    policy = policy or FINAL_FUSION_POLICY
    probs = _validate_probs(probs)
    yamnet_scores = _validate_yamnet_scores(yamnet_scores, probs.shape[0])
    max_probs = probs.max(axis=0)
    mean_probs = probs.mean(axis=0)
    consecutive_runs = {}
    triggered = []

    for cls, values in policy.items():
        cls_index = CLASSES.index(cls)
        strong_mask = probs[:, cls_index] >= values['strong_threshold']
        strong_run = longest_consecutive_run(strong_mask)
        event_scores = yamnet_scores[:, YAMNET_EVENT_INDICES[cls]].max(axis=1)
        aligned_mask = (
            (probs[:, cls_index] >= values['fusion_custom_threshold'])
            & (event_scores >= values['fusion_yamnet_threshold'])
        )
        fusion_run = longest_consecutive_run(aligned_mask)
        consecutive_runs[cls] = strong_run
        if (
            strong_run >= values['strong_min_frames']
            or fusion_run >= values['fusion_min_frames']
        ):
            triggered.append(cls)

    final = max(
        triggered, key=lambda cls: max_probs[CLASSES.index(cls)]
    ) if triggered else 'normal'
    return final, mean_probs, max_probs, consecutive_runs


def decide_deployment(probs, yamnet_scores):
    """Apply the final 2026-06-21 frame-aligned fusion policy."""
    return decide_fusion(probs, yamnet_scores)


def decide_final_fusion(probs, yamnet_scores):
    """Apply the same final policy used by evaluation and the live demo."""
    return decide_fusion(probs, yamnet_scores)


def decide_legacy_classifier_only(probs):
    """Reproduce the historical June 20 classifier-only comparison."""
    return decide(probs, **LEGACY_CLASSIFIER_ONLY_POLICY)
