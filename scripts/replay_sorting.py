"""Replay sorting rules on a frozen YOLO prediction receipt without loading weights.

Accepted input: result of specialists.py predict/evaluate, or disposal.py sort.
This validates receipt/config consistency; original image bytes are NOT verified
unless --verify-images is supplied.
"""
import argparse
import json
from pathlib import Path
import disposal as d
import specialists as s


def replay(payload, policy, config, config_sha, min_conf=0.25,
           expected_images=None):
    receipt = payload.get('prediction_receipt', payload)
    if not isinstance(receipt, dict) or not isinstance(receipt.get('images'), list):
        raise ValueError('Input needs a frozen prediction_receipt with image hashes')
    records = []
    for image in receipt['images']:
        if (not isinstance(image, dict) or not isinstance(image.get('name'), str)
                or not isinstance(image.get('sha256'), str)):
            raise ValueError('Each receipt image requires name and sha256')
        records.append(dict(name=image['name'], sha256=image['sha256'],
                            image_path=image['name']))
    if not records:
        raise ValueError('Cannot replay an empty receipt')
    if expected_images is not None and len(records) != expected_images:
        raise ValueError('Wrong image cohort size: expected %s, found %s' %
                         (expected_images, len(records)))
    verified = s.validate_receipt(receipt, records, config, config_sha)
    decisions = d.classify_predictions(verified['predictions'], policy, min_conf)
    counts = {name: 0 for name in policy['categories']}
    for rows in decisions.values():
        for row in rows:
            counts[row['category']] += 1
    return dict(
        category_counts=counts,
        images=len(decisions),
        detections=sum(counts.values()),
        policy=policy,
        model_config_sha256=config_sha,
        decisions=decisions,
        prediction_receipt=verified,
        provenance_check='Frozen prediction receipt and active config metadata validated; original image pixels and model weight bytes not checked.',
        actual_disposal_accuracy=None,
        limitation='No independent disposal-bin ground truth; this is a replay of object predictions under user-selected experimental rules.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path, help='Frozen predict/evaluate/sort JSON receipt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rules', type=Path, default=d.DEFAULT_RULES)
    parser.add_argument('--min-conf', type=float, default=0.25)
    parser.add_argument('--expected-images', type=int)
    parser.add_argument('--verify-images', action='store_true',
                        help='Also hash original image files, if available locally')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output')
    config = s.read(s.CONFIG)
    payload = s.read(args.receipt)
    policy = d.validate_policy(s.read(args.rules))
    result = replay(payload, policy, config, s.sha(s.CONFIG),
                    args.min_conf, args.expected_images)
    if args.verify_images:
        s.verify_images([dict(name=im['name'], image_path=im['name'],
                              sha256=im['sha256'])
                         for im in result['prediction_receipt']['images']])
        result['provenance_check'] += ' Original image bytes verified.'
    result['policy_sha256'] = s.sha(args.rules)
    result['source_receipt_sha256'] = s.sha(args.receipt)
    s.save(args.output, result)
    print(json.dumps({key: result[key] for key in
                      ('images', 'detections', 'category_counts',
                       'actual_disposal_accuracy', 'provenance_check')}, indent=2))


if __name__ == '__main__':
    main()
