import os
import subprocess
import pandas as pd
import csv

# ── AudioSet 클래스 ID ──────────────────
# /m/0k4j = Glass (유리 파손)
# /m/07yv9 = Vehicle (일반 소음)
# /m/0dgbq = Silence/ambient
GLASS_ID  = '/m/0k4j'
NORMAL_IDS = ['/m/07yv9', '/m/0dgbq', '/m/09x0r']  # 차량, 배경, 발화

DATA_DIR = '../data'
MAX_PER_CLASS = 50  # 클래스당 최대 개수

def download_segment(ytid, start, end, out_path):
    """YouTube에서 특정 구간만 다운로드"""
    url = f'https://www.youtube.com/watch?v={ytid}'
    duration = end - start
    cmd = [
        'yt-dlp',
        url,
        '--extract-audio',
        '--audio-format', 'wav',
        '--external-downloader', 'ffmpeg',
        '--external-downloader-args',
        f'ffmpeg_i:-ss {start} -t {duration}',
        '-o', out_path,
        '--quiet',
        '--no-warnings',
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

def parse_audioset_csv(csv_path, target_ids, max_count):
    """AudioSet CSV에서 target 클래스 항목만 추출"""
    items = []
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(', ')
            if len(parts) < 4:
                continue
            ytid, start, end, labels = parts[0], float(parts[1]), float(parts[2]), parts[3]
            for tid in target_ids:
                if tid in labels:
                    items.append((ytid, start, end))
                    break
            if len(items) >= max_count:
                break
    return items

def parse_all_scream_items(csv_paths, target_ids):
    """여러 CSV에서 모든 scream 항목을 ytid 중복 제거하여 추출"""
    all_items = []
    seen = set()
    for cp in csv_paths:
        if not os.path.exists(cp):
            print(f"  [건너뜀] CSV 없음: {os.path.basename(cp)}")
            continue
        items = parse_audioset_csv(cp, target_ids, max_count=10**9)
        added = 0
        for it in items:
            if it[0] not in seen:
                all_items.append(it)
                seen.add(it[0])
                added += 1
        print(f"  {os.path.basename(cp)}: {len(items)}개 (신규 {added}개)")
    return all_items

# ── CSV 다운로드 안내 ───────────────────
print("""
AudioSet CSV 파일을 먼저 다운로드해주세요:
  balanced_train_segments.csv:
  http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv

다운로드 후 프로젝트 루트에 저장하세요.
""")

csv_path = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT\\balanced_train_segments.csv'
if not os.path.exists(csv_path):
    print("CSV 파일이 없어요! 위 링크에서 받아주세요.")
    exit()

# ── normal 다운로드 ────────────────────
# print("\n[normal] 파싱 중...")
# normal_items = parse_audioset_csv(csv_path, NORMAL_IDS, MAX_PER_CLASS)
# print(f"  {len(normal_items)}개 찾음")
#
# out_dir = os.path.join(DATA_DIR, 'normal')
# os.makedirs(out_dir, exist_ok=True)
# for i, (ytid, start, end) in enumerate(normal_items):
#     out = os.path.join(out_dir, f'normal_{i:03d}.wav')
#     ok = download_segment(ytid, start, end, out)
#     print(f"  {'✓' if ok else '✗'} {ytid} ({start}~{end}s)")

# ── scream 다운로드 ────────────────────
SCREAM_IDS = ['/m/03qc9zr', '/m/0158x5']
SCREAM_TARGET = 100

unbalanced_csv = 'C:\\윤아\\Workspace\\HUFSWorkspace\\2026-HUFS-IoT\\unbalanced_train_segments.csv'
csv_paths = [csv_path, unbalanced_csv]

print("\n[scream] 파싱 중...")
scream_items = parse_all_scream_items(csv_paths, SCREAM_IDS)
print(f"  중복 제거 후 총 {len(scream_items)}개 후보")

out_dir = os.path.join(DATA_DIR, 'scream')
os.makedirs(out_dir, exist_ok=True)

existing_wavs = [f for f in os.listdir(out_dir) if f.lower().endswith('.wav')]
existing_count = len(existing_wavs)
print(f"  기존 .wav: {existing_count}개 / 목표: {SCREAM_TARGET}개")

if existing_count >= SCREAM_TARGET:
    print("  목표 이미 달성, 다운로드 생략")
else:
    next_idx = 50  # 기존이 000~049 영역이므로 050부터 이어 받음
    while os.path.exists(os.path.join(out_dir, f'scream_{next_idx:03d}.wav')):
        next_idx += 1

    success = 0
    for ytid, start, end in scream_items:
        cur_total = len([f for f in os.listdir(out_dir) if f.lower().endswith('.wav')])
        if cur_total >= SCREAM_TARGET:
            print(f"  목표 {SCREAM_TARGET}개 달성, 중단")
            break
        out = os.path.join(out_dir, f'scream_{next_idx:03d}.wav')
        if os.path.exists(out):
            next_idx += 1
            continue
        ok = download_segment(ytid, start, end, out)
        marker = '✓' if ok else '✗'
        if ok:
            success += 1
        print(f"  {marker} scream_{next_idx:03d} {ytid} ({start}~{end}s) [{cur_total + (1 if ok else 0)}/{SCREAM_TARGET}]")
        next_idx += 1

    print(f"\n  신규 다운로드 성공: {success}개")

print("\n전체 완료!")