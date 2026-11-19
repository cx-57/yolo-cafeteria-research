"""Conservative local provenance and error audit. Never certifies absence of exposure.

Writes a detailed external report; uses frozen outputs without running YOLO/training.
Unmatched local photographs remain unresolved unless provenance is independently checked.
"""
import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps
import yaml
import specialists as s
from evaluate_sorting import iou

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic'}
SHA_PATTERN = re.compile(r'^[0-9a-f]{64}$')


def verify_freeze(freeze, root=s.ROOT):
    for name, digest in freeze['protected_sha256'].items():
        if s.sha(root/name) != digest:
            raise ValueError('Frozen research input changed: '+name)
    for cohort, record in freeze.get('manifests', {}).items():
        if s.sha(record['path']) != record['sha256']:
            raise ValueError('Frozen annotation manifest changed: '+cohort)


def search_label_formats(root):
    """Search metadata formats, without asserting human authorship from field names."""
    counts = Counter(csv=0, xml=0, yaml=0)
    hashes, headers, candidates, errors, classes = [], [], [], [], Counter()
    bins = {'compost', 'recycling', 'landfill'}
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {'venv', '.git', '__pycache__', 'research-finalization-20261101'})
        for name in sorted(names):
            path = Path(directory)/name
            kind = path.suffix.lower().lstrip('.')
            kind = 'yaml' if kind == 'yml' else kind
            if kind not in counts:
                continue
            counts[kind] += 1
            relative = str(path.relative_to(root))
            try:
                raw = path.read_bytes()
                hashes.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest()))
                if kind == 'csv':
                    fields = next(csv.reader(raw.decode('utf-8-sig').splitlines()), [])
                    if any(any(term in field.lower() for term in ['bin', 'disposal', 'human', 'verified', 'reviewer']) for field in fields):
                        headers.append(dict(path=relative, headers=fields))
                elif kind == 'xml':
                    labels = [node.text for node in ET.fromstring(raw).findall('.//object/name')]
                    classes.update(labels)
                    if any(str(label).lower() in bins for label in labels):
                        candidates.append(relative)
                else:
                    value = yaml.safe_load(raw)
                    labels = value.get('names', []) if isinstance(value, dict) else []
                    labels = list(labels.values()) if isinstance(labels, dict) else labels
                    if isinstance(labels, list) and any(str(label).lower() in bins for label in labels):
                        candidates.append(relative)
            except (OSError, ValueError, UnicodeError, ET.ParseError, yaml.YAMLError) as error:
                errors.append(dict(path=relative, error=str(error)))
    return dict(counts=dict(counts), file_hashes=hashes, csv_header_candidates=headers,
                bin_class_candidates=sorted(candidates), xml_object_classes=dict(classes), read_errors=errors,
                method='CSV headers screened for bin/disposal/human/verified/reviewer; XML object/name and YAML names screened for exact compost/recycling/landfill. Format matches are leads, never verified human truth.')


def fingerprint(path):
    result = dict(sha256=s.sha(path))
    try:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            result.update(width=image.width, height=image.height)
            result['pixel_sha256'] = hashlib.sha256(
                f'{image.width},{image.height},RGB:'.encode() + image.tobytes()).hexdigest()
            gray = image.convert('L')
            # Rotation/reflection invariant screening, not proof of duplicate identity.
            variants = [gray] + [gray.transpose(op) for op in (
                Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
                Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
                Image.Transpose.TRANSVERSE)]
            hashes = []
            for variant in variants:
                values = variant.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
                code = 0
                for y in range(8):
                    for x in range(8):
                        code = (code << 1) | (values[y*9+x] > values[y*9+x+1])
                hashes.append(code)
            result['dhashes'] = sorted(set(hashes))
    except (OSError, ValueError) as error:
        result['decode_error'] = str(error)
    return result


def hash_distance(left, right):
    return min(((int(a)^int(b)).bit_count() for a in left for b in right), default=65)


class PerceptualIndex:
    """BK-tree for exact Hamming-radius screening; a match is only a review flag."""
    def __init__(self):
        self.root = None

    def add(self, code, source):
        if self.root is None:
            self.root = (code, source, {})
            return
        node = self.root
        while True:
            distance = (code ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[2]:
                node[2][distance] = (code, source, {})
                return
            node = node[2][distance]

    def find(self, codes, radius=5):
        best = None
        for code in codes:
            pending = [self.root] if self.root else []
            while pending:
                value, source, children = pending.pop()
                distance = (code ^ value).bit_count()
                bound = min(radius, best['distance']) if best else radius
                if distance <= bound:
                    best = dict(distance=distance, sha256=source)
                    bound = distance
                    if distance == 0:
                        return best
                pending.extend(child for edge, child in children.items()
                               if distance-bound <= edge <= distance+bound)
        return best


def overlap_reasons(candidate, reference, radius=5):
    reasons = []
    for key, reason in [('sha256', 'exact_bytes'), ('pixel_sha256', 'exact_decoded_pixels')]:
        if candidate.get(key) and candidate[key] == reference.get(key):
            reasons.append(reason)
    if set(candidate.get('groups', [])) & set(reference.get('groups', [])):
        reasons.append('capture_or_group_overlap')
    if hash_distance(candidate.get('dhashes', []), reference.get('dhashes', [])) <= radius:
        reasons.append('perceptual_near_duplicate_flag')
    return reasons


def independence_status(reasons, complete_provenance=False):
    if set(reasons) - {'decode_unavailable_no_visual_screen'}:
        return 'exposure_or_duplicate'
    if 'decode_unavailable_no_visual_screen' in reasons:
        return 'unresolved_visual_screen'
    return ('eligible_pending_human_provenance_review' if complete_provenance
            else 'unresolved_history')


def error_diagnostics(truth, predictions, cls):
    """Diagnostic hypotheses from object boxes, not new annotations or bin truth."""
    targets = [j for j, row in enumerate(truth) if int(row[4]) == cls]
    unused = set(targets)
    matched_predictions = set()
    false_predictions = []
    for index, row in sorted(enumerate(predictions), key=lambda item: -item[1][4]):
        if int(row[5]) != cls or row[4] < .5:
            continue
        score, j = max([(iou(row[:4], truth[j][:4]), j) for j in sorted(unused)],
                       default=(0, -1), key=lambda pair: (pair[0], -pair[1]))
        if score >= .5:
            unused.remove(j)
            matched_predictions.add(index)
        else:
            false_predictions.append(index)
    fn, fp, details = Counter(), Counter(), []
    for j in sorted(unused):
        same = [(iou(row[:4], truth[j][:4]), row[4]) for row in predictions if int(row[5]) == cls]
        if any(overlap >= .5 and confidence >= .5 for overlap, confidence in same):
            reason = 'duplicate_or_competing_box'
        elif any(overlap >= .5 for overlap, _ in same):
            reason = 'below_automatic_threshold'
        elif any(overlap >= .1 for overlap, _ in same):
            reason = 'localization_candidate'
        else:
            reason = 'no_same_class_overlap'
        fn[reason] += 1
        details.append(dict(kind='fn', target_index=j, reason=reason))
    for index in false_predictions:
        row = predictions[index]
        same = [iou(row[:4], truth[j][:4]) for j in targets]
        if any(overlap >= .5 for overlap in same):
            reason = 'duplicate_or_competing_box'
        elif any(overlap >= .1 for overlap in same):
            reason = 'localization_candidate'
        elif any(iou(row[:4], target[:4]) >= .5 for target in truth if int(target[4]) != cls):
            reason = 'other_class_overlap_or_annotation_ambiguity'
        else:
            reason = 'background_or_missing_annotation'
        fp[reason] += 1
        details.append(dict(kind='fp', prediction_index=index, reason=reason))
    return dict(tp=len(matched_predictions), false_negatives=dict(fn), false_positives=dict(fp), details=details)


def inventory(root):
    files, errors = [], []
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {'venv', '.git', '__pycache__', 'research-finalization-20261101'})
        for name in sorted(names):
            path = Path(directory)/name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    stat = path.stat()
                    files.append(dict(path=str(path), relative=str(path.relative_to(root)),
                                      inode=(stat.st_dev, stat.st_ino), size=stat.st_size))
                except OSError as error:
                    errors.append(dict(path=str(path), error=str(error)))
    return files, errors


def is_training_path(path):
    parts = Path(path).parts
    return ('images' in parts and any(part in {'train', 'training'} for part in parts[parts.index('images')+1:])
            and not any(part in {'raw', 'processed', 'candidates', 'v2_candidates'} for part in parts))


def is_candidate_path(path):
    parts = Path(path).parts
    if len(parts) < 3 or parts[0] != 'data':
        return False
    if parts[1] in {'school-submission-20261027', 'school-submission-20261031'}:
        return len(parts) == 4 and parts[2] in {'images', 'originals'}
    return parts[1] in {'raw', 'processed', 'candidates', 'v2_candidates'}


def metadata_index(root):
    groups, training, evaluated = defaultdict(set), defaultdict(set), defaultdict(set)
    files, errors, bin_candidates = [], [], []
    def visit(value, location, context=(), inherited_evaluation=False):
        if isinstance(value, dict):
            evaluation = inherited_evaluation or ('predictions' in value and 'images' in value)
            name = value.get('name', '')
            digest = value.get('sha256', value.get('image_sha256', ''))
            if (isinstance(digest, str) and SHA_PATTERN.fullmatch(digest) and
                    (isinstance(name, str) and Path(name).suffix.lower() in IMAGE_EXTENSIONS or 'image_path' in value)):
                for key in ['group', 'capture_group']:
                    if isinstance(value.get(key), str) and value[key]:
                        groups[digest].add(value[key])
                if name.startswith('IMG_') and value.get('capture_date'):
                    groups[digest].add('school-day-'+value['capture_date'])
                if value.get('split') == 'train' or any('train' in part or 'fit_records' == part for part in context):
                    training[digest].add(location)
                if evaluation or value.get('inference_run') is True:
                    evaluated[digest].add(location)
            # Candidate format only; human authorship must be verified separately.
            images = value.get('images')
            if isinstance(images, dict) and not ('predictions' in value or 'decisions' in value):
                objects = [obj for rows in images.values() if isinstance(rows, list) for obj in rows if isinstance(obj, dict)]
                if any(obj.get('category') in {'compost', 'recycling', 'landfill'} and
                       'box_xyxy_normalized' in obj and 'detection_confidence' not in obj for obj in objects):
                    bin_candidates.append(location)
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    visit(child, location, context+(key,), evaluation)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child, location, context, inherited_evaluation)
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {'venv', '.git', '__pycache__', 'research-finalization-20261101'})
        for name in sorted(names):
            if not name.endswith('.json'):
                continue
            path = Path(directory)/name
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
                files.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
                visit(value, str(path.relative_to(root)))
            except (OSError, ValueError) as error:
                errors.append(dict(path=str(path.relative_to(root)), error=str(error)))
    return groups, training, evaluated, files, errors, sorted(set(bin_candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=s.ROOT.parent/(s.ROOT.name+'-assets'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output')
    root = args.assets.resolve()
    freeze = s.read(s.ROOT/'configs/research-freeze.json')
    verify_freeze(freeze)
    images, image_errors = inventory(root)
    groups, training, evaluated, metadata, metadata_errors, bin_candidates = metadata_index(root)
    print(f'Indexed {len(images)} readable image paths and {len(metadata)} JSON files', flush=True)
    # Resolve aliases once by inode, then merge re-encoded copies by byte hash.
    inode_paths = {}
    for row in images:
        inode_paths.setdefault(row['inode'], row['path'])
    def hash_file(item):
        inode, path = item
        try:
            return inode, dict(sha256=s.sha(path))
        except OSError as error:
            return inode, dict(error=str(error), path=path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inode_hashes = dict(pool.map(hash_file, inode_paths.items()))
    by_sha = {}
    for row in images:
        hashed = inode_hashes[row['inode']]
        if 'error' in hashed:
            image_errors.append(dict(path=row['path'], error=hashed['error']))
            continue
        digest = hashed['sha256']
        entry = by_sha.setdefault(digest, dict(sha256=digest, paths=[], groups=sorted(groups[digest]),
            training_evidence=sorted(training[digest]), evaluation_evidence=sorted(evaluated[digest])))
        entry['paths'].append(row['relative'])
        if is_training_path(row['relative']):
            entry['training_evidence'].append('training_directory:'+row['relative'])
    print(f'Hashed {len(inode_hashes)} distinct inodes: {len(by_sha)} byte-unique images', flush=True)
    entries = list(by_sha.values())
    def decode(entry):
        # Original HEICs are byte-audited; their converted JPEGs are independently decoded.
        path = root/entry['paths'][0]
        try:
            return entry['sha256'], fingerprint(path)
        except OSError as error:
            return entry['sha256'], dict(decode_error=str(error))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, (digest, feature) in enumerate(pool.map(decode, entries), 1):
            by_sha[digest].update(feature)
            if index % 2000 == 0:
                print(f'Decoded {index}/{len(entries)} unique images', flush=True)
    trained = [r for r in entries if r['training_evidence']]
    trained_pixels = {r.get('pixel_sha256') for r in trained} - {None}
    trained_groups = set().union(*(set(r['groups']) for r in trained))
    evaluated_groups = set().union(*(set(r['groups']) for r in entries if r['evaluation_evidence']))
    # Public candidates plus all supplied school originals/conversions; no generated experiment plots.
    candidates = [r for r in entries if any(is_candidate_path(p) for p in r['paths'])]
    near_reference = PerceptualIndex()
    for row in trained:
        for code in row.get('dhashes', []):
            near_reference.add(code, row['sha256'])
    for index, entry in enumerate(candidates, 1):
        reasons=[]
        if entry['training_evidence']:
            reasons.append('training_directory_or_manifest_membership')
        if entry.get('pixel_sha256') in trained_pixels:
            reasons.append('exact_decoded_pixels_in_training_pool')
        if set(entry['groups']) & trained_groups:
            reasons.append('capture_or_group_overlap_with_training_pool')
        if entry['evaluation_evidence']:
            reasons.append('recorded_development_inference')
        if set(entry['groups']) & evaluated_groups:
            reasons.append('capture_or_group_overlap_with_development_pool')
        if not entry['training_evidence'] and entry.get('dhashes'):
            nearest = near_reference.find(entry['dhashes'])
            entry['training_dhash_match_within_radius5'] = nearest
            if nearest is not None:
                reasons.append('perceptual_near_duplicate_flag_radius5')
        if 'decode_error' in entry:
            reasons.append('decode_unavailable_no_visual_screen')
        entry['audit_reasons'] = reasons
        # No complete lifetime training/model-selection ledger supplied for unmatched public originals.
        entry['independence_status'] = independence_status(reasons)
        if index % 2000 == 0:
            print(f'Screened {index}/{len(candidates)} candidate identities', flush=True)
    supplementary_labels = search_label_formats(root)
    verify_freeze(freeze)
    result = dict(
        freeze_sha256=s.sha(s.ROOT/'configs/research-freeze.json'),
        inventory=dict(readable_paths=len(images), distinct_inodes=len(inode_hashes), byte_unique_images=len(entries),
                       candidate_unique_images=len(candidates), metadata_json_files=len(metadata),
                       trained_pool_unique_images=len(trained)),
        candidate_status_counts=dict(Counter(r['independence_status'] for r in candidates)),
        candidate_reason_counts=dict(Counter(reason for r in candidates for reason in r['audit_reasons'])),
        confirmed_independent_images=0,
        independent_disposal_metrics=None,
        bin_label_format_candidates=bin_candidates,
        supplemental_label_search=supplementary_labels,
        audit_source_sha256=s.sha(Path(__file__)),
        metadata_files=metadata, metadata_read_errors=metadata_errors, image_read_errors=image_errors,
        images=entries,
        scope='Local assets only, venv/cache/git excluded. All image extensions byte-audited; 64-bit dHash screens rotations/reflections at Hamming radius5. Training-directory or manifest membership is potential exposure, not proof of optimizer use. Unmatched is unresolved, never independent by default.',
        limitations=['Incomplete lifetime training and selection provenance, generic-pretraining exposure and unobserved near duplicates cannot be excluded.',
                     'Perceptual similarity flags require visual review; crops, viewpoint changes and temporally adjacent captures can evade hashing.',
                     'Read/decode errors remain explicitly unverified; HEIC bytes are linked to converted JPEGs only through recorded manifests.',
                     'No independently verified human bin truth is implied by object annotations or format candidates.'])
    s.save(args.output, result)
    print(json.dumps({key:result[key] for key in ['inventory','candidate_status_counts','candidate_reason_counts','confirmed_independent_images','bin_label_format_candidates']},indent=2))


if __name__ == '__main__':
    main()
