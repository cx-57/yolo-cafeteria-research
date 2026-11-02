"""Diagnose frozen YOLO predictions on new cafeteria photos, without retraining.

Outputs raw predictions, per-class confidence summaries and local image overlays.
Object-detection scores require COMPLETE human-reviewed seven-class YOLO labels.
This tool cannot certify that the photos are independent.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import specialists as s
from evaluate_sorting import iou

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
DISPLAY_CONFIDENCE = .25
STANDARD_CONFIDENCE = .50
COLORS = ['#286a47', '#3564a0', '#784bba', '#c47831', '#268a95', '#ac4670', '#9a6529']


def photos_under(source):
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    paths = sorted(p for p in source.rglob('*') if p.suffix.lower() in EXTENSIONS) if source.is_dir() else [source]
    if not paths or any(not p.is_file() or p.suffix.lower() not in EXTENSIONS for p in paths):
        raise ValueError('Provide JPEG/PNG/WEBP/BMP photos, not an empty folder or unsupported file')
    return paths


def inspect_predictions(predictions, threshold=DISPLAY_CONFIDENCE):
    """Only prediction counts; these are NOT accuracy or correctness scores."""
    classes = {name: {'detections': 0, 'photos_with_detection': 0} for name in s.NAMES}
    per_photo, conflicts = [], []
    for name, rows in sorted(predictions.items()):
        s.validate_prediction_rows(rows)
        visible = [r for r in rows if r[4] >= threshold]
        standard = [r for r in visible if r[4] >= STANDARD_CONFIDENCE]
        lower = [r for r in visible if r[4] < STANDARD_CONFIDENCE]
        counts = Counter(s.NAMES[int(r[5])] for r in visible)
        for cls, count in counts.items():
            classes[cls]['detections'] += count
            classes[cls]['photos_with_detection'] += 1
        for i, row in enumerate(visible):
            for other in visible[i+1:]:
                if int(row[5]) == int(other[5]):
                    continue
                overlap = iou(row[:4], other[:4])
                if overlap >= .60:
                    conflicts.append({'photo': name, 'classes': [s.NAMES[int(row[5])], s.NAMES[int(other[5])]],
                                      'overlap_iou': round(overlap, 3), 'confidences': [round(row[4], 3), round(other[4], 3)]})
        per_photo.append({
            'photo': name, 'detections_at_025': len(visible), 'detections_at_050': len(standard),
            'uncertain_025_to_050': len(lower), 'classes_at_025': dict(counts),
            'highest_confidence': round(max((r[4] for r in rows), default=0), 4),
        })
    return {
        'images': len(predictions), 'display_threshold': threshold,
        'photos_without_displayed_detections': sum(not v['detections_at_025'] for v in per_photo),
        'detections_at_025': sum(v['detections_at_025'] for v in per_photo),
        'detections_at_050': sum(v['detections_at_050'] for v in per_photo),
        'uncertain_025_to_050': sum(v['uncertain_025_to_050'] for v in per_photo),
        'per_class': classes, 'possible_cross_class_conflicts': conflicts,
        'per_photo': per_photo,
        'interpretation': 'Unlabeled predictions are not accuracy measurements. Overlapping boxes are possible conflicts, not proved errors.',
    }


def read_complete_labels(photos, label_dir):
    """Every visible object from ALL seven target classes must be checked."""
    label_dir = Path(label_dir).expanduser().resolve()
    records = []
    stems = [p.stem for p in photos]
    if len(set(stems)) != len(stems):
        raise ValueError('Duplicate image stems: cannot match flat labels unambiguously')
    for photo in photos:
        label = label_dir / (photo.stem + '.txt')
        if not label.is_file():
            raise ValueError(f'Missing COMPLETE seven-class annotation for {photo.name}: {label}')
        rows = [line.strip() for line in label.read_text().splitlines() if line.strip()]
        records.append({'name': str(photo), 'rows': rows, 'known_classes': list(range(7))})
    s.validate_annotations(records, list(range(7)))
    return records


def draw_overlays(photos, predictions, destination, threshold=DISPLAY_CONFIDENCE):
    destination.mkdir(parents=True, exist_ok=True)
    for index, photo in enumerate(photos):
        with Image.open(photo) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        draw = ImageDraw.Draw(image)
        for row in sorted(predictions[str(photo)], key=lambda value: value[4]):
            x1, y1, x2, y2, conf, cls = row
            if conf < threshold:
                continue
            w, h = image.size
            box = (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
            color = COLORS[int(cls)] if conf >= STANDARD_CONFIDENCE else '#e0a637'
            draw.rectangle(box, outline=color, width=max(2, w//350))
            label = f'{s.NAMES[int(cls)]} {conf:.2f}'
            yy = max(0, box[1]-16)
            draw.text((box[0], yy), label, fill=color, stroke_width=1, stroke_fill='white')
        image.thumbnail((1600, 1600))
        image.save(destination / f'{index:04d}-{photo.stem}.jpg', quality=88)


def run(photo_dir, output, device='mps', labels=None, labels_reviewed=False):
    if labels is not None and not labels_reviewed:
        raise ValueError('--labels requires --labels-reviewed: all seven classes must be exhaustively human reviewed')
    if labels_reviewed and labels is None:
        raise ValueError('--labels-reviewed requires --labels')
    paths = photos_under(photo_dir)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    if any(target == photo or target in photo.parents for photo in paths):
        raise ValueError('Output location must not overwrite source images')
    if labels is not None:
        records = read_complete_labels(paths, labels)
    else:
        records = None
    inputs = [{'name': str(p), 'image_path': str(p), 'sha256': s.sha(p)} for p in paths]
    receipt = s.predict(inputs, device=device)
    summary = inspect_predictions(receipt['predictions'])
    report = {'status': 'DIAGNOSTIC_ONLY_NOT_INDEPENDENT_ACCURACY',
              'source_photo_hashes': [{'name': r['name'], 'sha256': r['sha256']} for r in inputs],
              'predictions': summary,
              'important': 'Images viewed or used for model decisions cannot later be treated as untouched final test images.'}
    if records is not None:
        report['object_metrics'] = {
            'confidence_025': s.recognition_at_threshold(records, receipt['predictions'], confidence=.25),
            'confidence_050': s.recognition_at_threshold(records, receipt['predictions'], confidence=.50),
        }
        report['label_caveat'] = 'User attested full manual object annotation for seven classes; script cannot verify completeness.'
    target.mkdir(parents=True)
    s.save(target/'predictions.json', receipt)
    draw_overlays(paths, receipt['predictions'], target/'overlays')
    s.save(target/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photos', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--labels', type=Path, help='Directory of exhaustively human-reviewed seven-class YOLO labels')
    parser.add_argument('--labels-reviewed', action='store_true', help='Attest that EVERY visible target object was manually checked')
    args = parser.parse_args()
    report = run(args.photos, args.output, args.device, args.labels, args.labels_reviewed)
    info = report['predictions']
    print(json.dumps({'images': info['images'], 'no_detection_photos': info['photos_without_displayed_detections'],
                      'detections_at_025': info['detections_at_025'], 'detections_at_050': info['detections_at_050'],
                      'cross_class_conflicts': len(info['possible_cross_class_conflicts']),
                      'object_metrics_measured': 'object_metrics' in report,
                      'report': str(args.output / 'report.json')}, indent=2))


if __name__ == '__main__':
    main()
