import re


YOUTUBE_ID_PATTERN = re.compile(r'_([A-Za-z0-9_-]{11})_\d{6}_\d{6}$')


def assert_disjoint_splits(splits):
    """Fail fast when a source group appears in more than one data split."""
    split_names = tuple(splits)
    split_sets = {name: set(splits[name]) for name in split_names}
    overlaps = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            shared = sorted(split_sets[left] & split_sets[right])
            if shared:
                overlaps[f'{left}/{right}'] = shared
    if overlaps:
        details = '; '.join(
            f'{pair}: {groups[:5]}' for pair, groups in overlaps.items()
        )
        raise ValueError(f'Data leakage: source groups overlap across splits ({details})')


def youtube_id(group):
    """Return an AudioSet/YouTube source ID when it is encoded in a group name."""
    stem = group.split(':', 1)[-1]
    match = YOUTUBE_ID_PATTERN.search(stem)
    return match.group(1) if match else None


def source_unit(group):
    """Canonical split unit; renamed clips from one video stay together."""
    source_id = youtube_id(group)
    return f'youtube:{source_id}' if source_id else group


def origin_overlaps(splits):
    """Report YouTube IDs that occur in multiple splits under different filenames."""
    locations = {}
    for split, groups in splits.items():
        for group in groups:
            source_id = youtube_id(group)
            if source_id:
                locations.setdefault(source_id, set()).add(split)
    return {
        source_id: sorted(source_splits)
        for source_id, source_splits in locations.items()
        if len(source_splits) > 1
    }
