"""Train the eight selected classes; report development AP50 per class."""
import argparse
import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/utensil-focus/grouped-v7-reviewed-negatives'
RUN = ROOT / 'model/remaining-classes'
KEEP = [0, 1, 4, 5, 6, 8, 9, 10]


def verify_prepared():
    """Check the frozen prepared copy without requiring offloaded ancestor copies."""
    from utensil_recovery import audit, require_local, pixels, validate_image_file
    from v2 import digest, dataset_fingerprint
    from prepare import IMAGE_SUFFIXES
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if manifest['fingerprint'] != 'cc30ea3b8886e87caf6252f9ee62dd5cdd3814bbf774072a4d7beb224aa3a1bd':
        raise ValueError('Unexpected prepared dataset revision')
    for split in ('train', 'val'):
        expected = {r['name'] for r in manifest['records'] if r['split'] == split}
        actual = {p.name for p in (DATA / 'images' / split).rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.name for p in (DATA / 'labels' / split).rglob('*.txt')}
        if actual != expected or labels != {str(Path(n).with_suffix('.txt')) for n in expected}:
            raise ValueError('Dataset membership changed')
    for record in manifest['records']:
        image = DATA / 'images' / record['split'] / record['name']
        label = DATA / 'labels' / record['split'] / Path(record['name']).with_suffix('.txt')
        require_local([image, label])
        validate_image_file(image)
        if digest(image) != record['sha'] or pixels(image) != record['pixels'] or label.read_text().splitlines() != record['rows']:
            raise ValueError(f'Prepared file changed: {image}')
    if audit(manifest['records']) != manifest['counts'] or dataset_fingerprint(DATA / 'data.yaml') != manifest['fingerprint']:
        raise ValueError('Dataset counts/fingerprint changed')
    print('Prepared images, labels, boxes and recorded split identities verified.', flush=True)
    print(json.dumps(manifest['image_counts']), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'train', 'evaluate'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume an interrupted last.pt, including optimizer state')
    args = parser.parse_args()
    manifest = verify_prepared()
    import yaml
    names = yaml.safe_load((DATA / 'data.yaml').read_text())['names']
    counts = {split: {names[i]: manifest['counts'][split][names[i]] for i in KEEP}
              for split in ('train', 'val')}
    print(json.dumps(counts, indent=2), flush=True)
    print('Development validation only: related capture setups remain a limitation. '
          'Bowl has only two validation objects; its AP is inconclusive.', flush=True)
    if args.action == 'verify':
        return
    from ultralytics import YOLO
    if args.action == 'train':
        from partial_labels import make_trainer
        base = make_trainer(ROOT / 'data/utensil-focus/coverage-reviewed-v7.json')

        class Trainer(base):
            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
                return super().get_dataloader(dataset_path, 4 if mode == 'val' else batch_size, rank, mode)

        if args.resume:
            import torch
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if checkpoint.get('optimizer') is None or checkpoint.get('epoch', -1) < 0:
                raise ValueError('Checkpoint has no resumable optimizer/epoch state')
            print(f"Resuming at epoch {checkpoint['epoch'] + 2} of {checkpoint['train_args']['epochs']}", flush=True)
            del checkpoint
        model = YOLO(str(args.resume or ROOT / 'model/weights/yolo11n.pt'))
        model.train(trainer=Trainer, data=str(DATA / 'data.yaml'), classes=KEEP,
                    epochs=args.epochs, batch=4, imgsz=640, device='mps', workers=0,
                    optimizer='AdamW', lr0=.0003, warmup_bias_lr=0., patience=0,
                    mosaic=0., mixup=0., cutmix=0., copy_paste=0.,
                    project=str(RUN), name='yolo11n-80epochs', save=True, plots=True,
                    resume=bool(args.resume))
        weights = model.trainer.best
        del model
        gc.collect()
        import torch
        torch.mps.empty_cache()
    else:
        if args.weights is None:
            parser.error('evaluate requires --weights pointing to the actual run checkpoint')
        weights = args.weights
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA / 'data.yaml'), classes=KEEP, imgsz=640,
                        batch=4, device='mps', workers=0, project=str(RUN), name='evaluation')
    result = dict(checkpoint=str(weights), independent_test=False,
                  overall=metrics.results_dict, per_class={})
    for index, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        result['per_class'][names[int(class_id)]] = dict(precision=float(p), recall=float(r),
                                                       AP50=float(ap50), AP50_95=float(ap))
    target = Path(model.validator.save_dir) / 'metrics.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print(f'Report: {target}', flush=True)


if __name__ == '__main__':
    main()
