import argparse
import csv
import os
import subprocess


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# AudioSet class IDs used by this project.
# Source: https://github.com/audioset/ontology/blob/master/ontology.json
PROFILES = {
    'casino-normal': {
        'class_name': 'normal',
        'prefix': 'audioset_casino_normal',
        'ids': [
            '/m/09x0r',    # Speech
            '/m/01h8n0',   # Conversation
            '/m/07rkbfh',  # Chatter
            '/m/03qtwd',   # Crowd
            '/m/07qfr4h',  # Hubbub, speech noise, speech babble
            '/m/01j3sz',   # Laughter
            '/m/028ght',   # Applause
            '/m/053hz1',   # Cheering
            '/m/0242l',    # Coin dropping
            '/m/03v3yw',   # Keys jangling
            '/m/02fs_r',   # Beep, bleep
            '/m/07qwdck',  # Ping
            '/m/07phxs1',  # Ding
        ],
    },
    'music-normal': {
        'class_name': 'normal',
        'prefix': 'audioset_music_normal',
        'ids': [
            '/m/04rlf',    # Music
            '/m/015lz1',   # Singing
            '/m/0y4f8',    # Vocal music
            '/m/0z9c',     # A capella
            '/m/0l14jd',   # Choir
            '/t/dd00003',  # Male singing
            '/t/dd00004',  # Female singing
            '/t/dd00005',  # Child singing
            '/m/0ggq0m',   # Pop music
            '/m/064t9',    # Hip hop music
            '/m/06by7',    # Rock music
            '/m/0fd3y',    # Ambient music
            '/m/08cyft',   # Electronic dance music
        ],
    },
    'scream': {
        'class_name': 'scream',
        'prefix': 'audioset_scream',
        'ids': [
            '/m/03qc9zr',  # Screaming
        ],
    },
    'glass-impact': {
        'class_name': 'glass',
        'prefix': 'audioset_glass_event',
        'ids': [
            '/m/07pjjrj',  # Smash, crash
            '/m/07pc8lb',  # Breaking
            '/m/07plct2',  # Crushing
        ],
    },
}


def download_segment(ytid, start, end, out_path):
    url = f'https://www.youtube.com/watch?v={ytid}'
    duration = max(0.1, end - start)
    output_template = os.path.splitext(out_path)[0] + '.%(ext)s'
    cmd = [
        'yt-dlp',
        url,
        '--extract-audio',
        '--audio-format',
        'wav',
        '--external-downloader',
        'ffmpeg',
        '--external-downloader-args',
        f'ffmpeg_i:-ss {start} -t {duration}',
        '-o',
        output_template,
        '--quiet',
        '--no-warnings',
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def parse_audioset_csv(csv_path, target_ids, max_count):
    items = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(line for line in f if not line.startswith('#'))
        for row in reader:
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            try:
                start = float(row[1])
                end = float(row[2])
            except ValueError:
                continue
            labels = row[3]
            if any(target_id in labels for target_id in target_ids):
                items.append((ytid, start, end))
            if len(items) >= max_count:
                break
    return items


def collect_candidates(csv_paths, target_ids, max_candidates):
    candidates = []
    seen = set()
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f'  skip missing CSV: {csv_path}')
            continue
        items = parse_audioset_csv(csv_path, target_ids, max_candidates)
        added = 0
        for item in items:
            ytid, start, end = item
            key = (ytid, start, end)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)
            added += 1
            if len(candidates) >= max_candidates:
                break
        print(f'  {os.path.basename(csv_path)}: {len(items)} found, {added} added')
        if len(candidates) >= max_candidates:
            break
    return candidates


def safe_segment_name(prefix, ytid, start, end):
    token = ytid.replace('-', 'm').replace('_', 'u')
    start_cs = int(round(start * 100))
    end_cs = int(round(end * 100))
    return f'{prefix}_{token}_{start_cs:06d}_{end_cs:06d}.wav'


def output_exists(path):
    stem = os.path.splitext(path)[0]
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        return False
    return any(
        name.startswith(os.path.basename(stem) + '.')
        for name in os.listdir(folder)
    )


def download_profile(profile_name, csv_paths, target_count, max_candidates, dry_run):
    profile = PROFILES[profile_name]
    out_dir = os.path.join(DATA_DIR, profile['class_name'])
    prefix = profile['prefix']

    print(f'\n[{profile_name}]')
    print(f'  class: {profile["class_name"]}')
    print(f'  output: {out_dir}')
    print(f'  target new files: {target_count}')

    candidates = collect_candidates(csv_paths, profile['ids'], max_candidates)
    print(f'  candidates: {len(candidates)}')

    if dry_run:
        for ytid, start, end in candidates[:target_count]:
            print(f'  dry-run: {ytid} {start}-{end}')
        return

    os.makedirs(out_dir, exist_ok=True)
    success = 0
    for ytid, start, end in candidates:
        if success >= target_count:
            break
        out_path = os.path.join(out_dir, safe_segment_name(prefix, ytid, start, end))
        if output_exists(out_path):
            continue
        ok = download_segment(ytid, start, end, out_path)
        marker = 'OK' if ok else 'FAIL'
        if ok:
            success += 1
        print(f'  {marker} {os.path.basename(out_path)} {ytid} ({start}-{end}) [{success}/{target_count}]')

    print(f'\n  downloaded: {success}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download AudioSet clips for this project using yt-dlp and local AudioSet CSV files.'
    )
    parser.add_argument(
        '--profile',
        choices=sorted(PROFILES),
        default='casino-normal',
        help='Download profile. casino-normal is best for crowded indoor false-positive training.',
    )
    parser.add_argument(
        '--csv',
        action='append',
        default=[],
        help='AudioSet CSV path. Can be repeated. Defaults to balanced and unbalanced CSVs in project root.',
    )
    parser.add_argument('--target-count', type=int, default=80, help='Number of new clips to download.')
    parser.add_argument('--max-candidates', type=int, default=1500, help='Maximum matching CSV rows to scan/download from.')
    parser.add_argument('--dry-run', action='store_true', help='Only print candidate clips; do not download.')
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = args.csv or [
        os.path.join(BASE_DIR, 'balanced_train_segments.csv'),
        os.path.join(BASE_DIR, 'unbalanced_train_segments.csv'),
    ]

    print('Required tools: yt-dlp and ffmpeg')
    print('CSV files:')
    for csv_path in csv_paths:
        print(f'  {csv_path}')

    download_profile(
        profile_name=args.profile,
        csv_paths=csv_paths,
        target_count=args.target_count,
        max_candidates=args.max_candidates,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
