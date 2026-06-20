import argparse
import csv
import json
import os
import re


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STRICT_IDS = {
    'glass': {'/m/039jq', '/m/07rn7sz'},
    'scream': {'/m/03qc9zr'},
}
FILE_PATTERN = re.compile(
    r'^audioset_(?:glass_event|scream)_([A-Za-z0-9_-]{11})_(\d{6})_(\d{6})'
)


def encoded_token(ytid):
    return ytid.replace('-', 'm').replace('_', 'u')


def metadata_index(csv_paths):
    index = {}
    for csv_path in csv_paths:
        with open(csv_path, encoding='utf-8') as handle:
            reader = csv.reader(line for line in handle if not line.startswith('#'))
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    start_cs = int(round(float(row[1]) * 100))
                    end_cs = int(round(float(row[2]) * 100))
                except ValueError:
                    continue
                key = (encoded_token(row[0].strip()), start_cs, end_cs)
                index.setdefault(key, set()).update(
                    label.strip().strip('"') for label in row[3].split(',')
                )
    return index


def main():
    parser = argparse.ArgumentParser(description='Validate project AudioSet labels against strict event IDs.')
    parser.add_argument('--csv', action='append', default=[])
    parser.add_argument('--dataset-dir', default=os.path.join(BASE_DIR, 'data_clean'))
    parser.add_argument('--json-output', required=True)
    parser.add_argument('--review-csv', required=True)
    parser.add_argument('--approved-output', help='Optional allowlist for strict-label glass sources.')
    args = parser.parse_args()
    csv_paths = args.csv or [
        os.path.join(BASE_DIR, 'balanced_train_segments.csv'),
        os.path.join(BASE_DIR, 'unbalanced_train_segments.csv'),
    ]
    metadata = metadata_index(csv_paths)
    records = []
    for cls in ('glass', 'scream'):
        folder = os.path.join(args.dataset_dir, cls)
        for name in sorted(os.listdir(folder)):
            match = FILE_PATTERN.match(name)
            if not match:
                continue
            token, start_cs, end_cs = match.groups()
            labels = metadata.get((token, int(start_cs), int(end_cs)))
            strict_matches = sorted(STRICT_IDS[cls] & (labels or set()))
            records.append({
                'class': cls,
                'group': f'{cls}:{os.path.splitext(name)[0].removesuffix("_clean")}',
                'path': os.path.relpath(os.path.join(folder, name), BASE_DIR).replace('\\', '/'),
                'metadata_found': labels is not None,
                'strict_label_match': bool(strict_matches),
                'strict_ids': '|'.join(strict_matches),
                'all_ids': '|'.join(sorted(labels or set())),
                'review_action': (
                    'keep' if strict_matches else 'listen_before_keep_or_quarantine'
                ),
            })
    review = [record for record in records if not record['strict_label_match']]
    os.makedirs(os.path.dirname(os.path.abspath(args.review_csv)), exist_ok=True)
    with open(args.review_csv, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(review)
    if args.approved_output:
        approved = sorted(
            record['group'] for record in records
            if record['class'] == 'glass' and record['strict_label_match']
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.approved_output)), exist_ok=True)
        with open(args.approved_output, 'w', encoding='utf-8') as handle:
            handle.write('# Auto-generated from strict AudioSet Glass/Shatter metadata.\n')
            handle.write('# Broad-label legacy clips still require human review.\n')
            for group in approved:
                handle.write(group + '\n')
    output = {
        'strict_ids': {key: sorted(value) for key, value in STRICT_IDS.items()},
        'records': records,
        'summary': {
            cls: {
                'total': sum(record['class'] == cls for record in records),
                'strict_match': sum(
                    record['class'] == cls and record['strict_label_match'] for record in records
                ),
                'needs_review': sum(
                    record['class'] == cls and not record['strict_label_match'] for record in records
                ),
            }
            for cls in ('glass', 'scream')
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
    with open(args.json_output, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
