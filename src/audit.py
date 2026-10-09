"""Read-only corpus audit; writes evidence, never modifies images or labels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps
import yaml

from prepare import CLASS_MAPS, IMAGE_SUFFIXES, PROJECT_ROOT, class_names_by_id, remap_label_line


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def difference_hash(image):
    pixels = image.convert('L').resize((9, 8)).tobytes()
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return result


def same_box(a, b, tolerance=1e-5):
    pa, pb = a.split(), b.split()
    return len(pa) == len(pb) == 5 and pa[0] == pb[0] and all(abs(float(x)-float(y)) <= tolerance for x, y in zip(pa[1:], pb[1:]))


def audit(processed, output, config, raw=None, manifests=None):
    output.mkdir(parents=True, exist_ok=True)
    classes = {int(k): v for k, v in yaml.safe_load(config.read_text())['classes'].items()}
    final_names = {k: str(k) for k in classes}
    identity_map = {str(k): k for k in classes}
    images = sorted(p for p in processed.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES and p.parent.name == 'images')
    issues, records, source_rows, coverage = [], [], [], []
    source_counts, class_instances, class_images = defaultdict(Counter), Counter(), Counter()
    by_hash, by_pixels = defaultdict(list), defaultdict(list)
    expected_labels = set()
    raw_indexes, manifest_indexes = {}, {}
    for image in images:
        relative = image.relative_to(processed).as_posix()
        source = image.relative_to(processed).parts[0]
        label = image.parent.parent / 'labels' / f'{image.stem}.txt'
        expected_labels.add(label)
        source_counts[source]['images'] += 1
        record = dict(image=relative, source=source, label=label.relative_to(processed).as_posix(), width=0, height=0, sha256='', pixel_sha256='', dhash='', classes='', instances=0, source_split='', source_image='', duplicate_group='', near_group='')
        try:
            with Image.open(image) as im:
                im.load()
                rgb = im.convert('RGB')
                record.update(width=rgb.width, height=rgb.height, sha256=hashlib.sha256(image.read_bytes()).hexdigest(), pixel_sha256=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(), dhash=format(difference_hash(rgb), '016x'))
                record['_thumb'] = rgb.resize((16, 16)).tobytes()
            by_hash[record['sha256']].append(relative)
            by_pixels[record['pixel_sha256']].append(relative)
        except Exception as exc:
            issues.append(dict(kind='unreadable_image', image=relative, detail=str(exc)))
        rows = []
        if not label.is_file():
            issues.append(dict(kind='missing_label', image=relative, detail=str(label)))
        else:
            for line_number, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    if len(line.split()) != 5:
                        raise ValueError('expected exactly five detection values')
                    valid = remap_label_line(line, final_names, identity_map, label, line_number)
                    cid = int(valid.split()[0]); rows.append(valid)
                    class_instances[cid] += 1
                    source_counts[source][f'class_{cid}'] += 1
                except ValueError as exc:
                    issues.append(dict(kind='invalid_box', image=relative, detail=f'{line_number}: {exc}'))
            if not rows:
                issues.append(dict(kind='empty_label', image=relative, detail='No valid annotations; explicit verified negative status required'))
        ids = {int(r.split()[0]) for r in rows}
        class_images.update(ids)
        record.update(classes=';'.join(map(str, sorted(ids))), instances=len(rows))
        source_counts[source]['instances'] += len(rows)
        if manifests:
            if source not in manifest_indexes:
                manifest_file = manifests / f'{source}.csv'
                if manifest_file.exists():
                    with manifest_file.open(newline='') as f:
                        entries = list(csv.DictReader(f))
                    manifest_indexes[source] = {r.get('processed_image') or r.get('filename') or r.get('image'): r for r in entries}
                else:
                    manifest_indexes[source] = {}
                    issues.append(dict(kind='missing_manifest', image=source, detail=str(manifest_file)))
            entry = manifest_indexes[source].get(image.name)
            if entry is None:
                issues.append(dict(kind='unmatched_manifest', image=relative, detail='No retained provenance row'))
            else:
                record.update(source_split=entry.get('source_split', entry.get('original_split', '')), source_image=entry.get('source_image', entry.get('filename', entry.get('image', ''))))
                if entry.get('sha256') and entry['sha256'] != record['sha256']:
                    issues.append(dict(kind='manifest_hash_mismatch', image=relative, detail='Bytes differ from original retained manifest'))
                stated = entry.get('instances_kept', entry.get('instances', ''))
                if stated and int(stated) != len(rows):
                    issues.append(dict(kind='manifest_count_mismatch', image=relative, detail=f'manifest={stated}, actual={len(rows)}'))
        if raw:
            if source not in raw_indexes:
                raw_source = raw / source
                data_file = raw_source / 'data.yaml'
                if data_file.is_file() and source in CLASS_MAPS:
                    names = class_names_by_id(yaml.safe_load(data_file.read_text())['names'])
                    labels = {}
                    for p in raw_source.rglob('*.txt'):
                        if p.parent.name == 'labels':
                            labels[(p.parent.parent.name, p.stem)] = p
                    raw_indexes[source] = (names, labels)
                else:
                    raw_indexes[source] = None
                    issues.append(dict(kind='missing_raw_source', image=source, detail='Cannot verify source annotation preservation'))
            index = raw_indexes[source]
            if index:
                names, labels = index
                original_name = record['source_image'] or image.name
                split = record['source_split']
                if original_name.split('__', 1)[0] in {'train','valid','val','test'} and not split:
                    split, original_name = original_name.split('__', 1)
                if image.name.split('__', 1)[0] in {'train','valid','val','test'} and not split:
                    split, original_name = image.name.split('__', 1)
                stem = Path(original_name).stem
                candidates = [p for (s, st), p in labels.items() if st == stem and (not split or s == split)]
                if not candidates:
                    # The re-uploaded Roboflow archive flattened its original
                    # train/valid folders. Match a unique filename but retain
                    # the earlier manifest's original split in the inventory.
                    candidates = [p for (s, st), p in labels.items() if st == stem]
                if len(candidates) != 1:
                    issues.append(dict(kind='source_label_resolution', image=relative, detail=f'{len(candidates)} matching original labels'))
                else:
                    source_label = candidates[0]
                    source_image = source_label.parent.parent / 'images' / original_name
                    if not source_image.is_file():
                        issues.append(dict(kind='missing_source_image', image=relative, detail=str(source_image)))
                    elif hashlib.sha256(source_image.read_bytes()).hexdigest() != record['sha256']:
                        # Different encodings are acceptable only when every
                        # decoded RGB pixel and dimension is identical.
                        try:
                            with Image.open(source_image) as original:
                                rgb = original.convert('RGB')
                                original_pixels = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                            if original_pixels != record['pixel_sha256']:
                                issues.append(dict(kind='source_image_mismatch', image=relative, detail='Matched filename has different original image pixels'))
                        except Exception as exc:
                            issues.append(dict(kind='unreadable_source_image', image=relative, detail=str(exc)))
                    if not record['source_split']:
                        record['source_split'] = source_label.parent.parent.name
                    record['source_image'] = original_name
                    missing, unmapped = 0, 0
                    for number, line in enumerate(source_label.read_text().splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            converted = remap_label_line(line, names, CLASS_MAPS[source], source_label, number)
                            name = str(names[int(line.split()[0])])
                            if converted is None:
                                unmapped += 1
                                source_rows.append(dict(image=relative, source=source, source_class=name, status='unmapped_source_class', expected_row='', source_line=number))
                            elif not any(same_box(converted, row) for row in rows):
                                missing += 1
                                source_rows.append(dict(image=relative, source=source, source_class=name, status='missing_target_box', expected_row=converted, source_line=number))
                        except ValueError as exc:
                            issues.append(dict(kind='invalid_source_annotation', image=relative, detail=str(exc)))
                    coverage.append(dict(image=relative, source=source, missing_target_boxes=missing, unmapped_source_boxes=unmapped))
        record['_rows'] = rows
        records.append(record)
    for label in sorted(processed.rglob('*.txt')):
        if label.parent.name == 'labels' and label not in expected_labels:
            issues.append(dict(kind='orphan_label', image=label.relative_to(processed).as_posix(), detail='No matching image'))
    if not images:
        issues.append(dict(kind='no_images', image='', detail='Empty corpus cannot pass an audit'))
    exact_groups = [g for g in by_pixels.values() if len(g)>1]
    exact_pairs = []
    record_by_path = {r['image']: r for r in records}
    parents = {r['image']: r['image'] for r in records}
    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]; k = parents[k]
        return k
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a,b)] = min(a,b)
    for number, group in enumerate(exact_groups, 1):
        for path in group:
            record_by_path[path]['duplicate_group'] = f'exact_{number:04d}'
            union(group[0], path)
            exact_pairs.append(dict(group=f'exact_{number:04d}', image=path))
    near_pairs = []
    for i, a in enumerate(records):
        if not a['dhash']:
            continue
        for b in records[i+1:]:
            if not b['dhash'] or a['pixel_sha256'] == b['pixel_sha256']:
                continue
            distance = (int(a['dhash'], 16)^int(b['dhash'], 16)).bit_count()
            if distance > 6 or abs(a['width']/a['height']-b['width']/b['height']) > .05:
                continue
            error = sum(abs(v-w) for v,w in zip(a['_thumb'], b['_thumb']))/(16*16*3)
            if error > 15:
                continue
            union(a['image'], b['image'])
            near_pairs.append(dict(image_a=a['image'], image_b=b['image'], dhash_distance=distance, thumbnail_mean_absolute_error=round(error,4), status='review_candidate_do_not_delete'))
    components = defaultdict(list)
    for path in parents:
        components[find(path)].append(path)
    for number, group in enumerate(sorted(components.values()),1):
        for path in group:
            record_by_path[path]['near_group'] = f'group_{number:04d}'
    missing_count = sum(r['missing_target_boxes'] for r in coverage)
    if missing_count:
        issues.append(dict(kind='missing_target_annotations', image='corpus', detail=f'{missing_count} raw target boxes absent from retained labels'))
    class_rows = [dict(class_id=k, name=name, images=class_images[k], instances=class_instances[k]) for k,name in sorted(classes.items())]
    summary = dict(images=len(images), instances=sum(class_instances.values()), class_counts=class_rows, sources={k:dict(v) for k,v in sorted(source_counts.items())}, exact_duplicate_groups=len(exact_groups), byte_duplicate_groups=sum(len(g)>1 for g in by_hash.values()), near_duplicate_candidate_pairs=len(near_pairs), missing_source_target_boxes=missing_count, missing_classes=[r['name'] for r in class_rows if not r['instances']], issues=issues, structural_pass=not issues, complete_11_class_annotations_verified=False, completeness_note='Source comparison checks preservation, not all visible target objects. Full multi-class visual annotation review remains required; near-duplicate matches are candidates, not confirmed duplicates.')
    fields = ['image','source','label','width','height','sha256','pixel_sha256','dhash','classes','instances','source_split','source_image','duplicate_group','near_group']
    write_csv(output/'inventory.csv', [{k:r[k] for k in fields} for r in records], fields)
    write_csv(output/'class_counts.csv', class_rows, ['class_id','name','images','instances'])
    write_csv(output/'exact_duplicates.csv', exact_pairs, ['group','image'])
    write_csv(output/'near_duplicates.csv', near_pairs, ['image_a','image_b','dhash_distance','thumbnail_mean_absolute_error','status'])
    write_csv(output/'source_annotation_gaps.csv', source_rows, ['image','source','source_class','status','expected_row','source_line'])
    write_csv(output/'annotation_coverage.csv', coverage, ['image','source','missing_target_boxes','unmapped_source_boxes'])
    write_csv(output/'issues.csv', issues, ['kind','image','detail'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['images','instances','exact_duplicate_groups','near_duplicate_candidate_pairs','missing_source_target_boxes','missing_classes','structural_pass']}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=PROJECT_ROOT/'data/processed')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'data/audits/day1')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'config.yaml')
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--manifests', type=Path)
    args = parser.parse_args()
    summary = audit(args.processed, args.output, args.config, args.raw, args.manifests)
    return 0 if summary['structural_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
