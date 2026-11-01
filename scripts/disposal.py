"""Configurable disposal decisions for individual detections; no material inference."""
import argparse
import json
import math
from pathlib import Path
import specialists as s

DEFAULT_RULES = s.ROOT / 'configs/disposal.json'


def validate_policy(policy):
    if type(policy.get('verified')) is not bool:
        raise ValueError('Policy verified must be an explicit boolean')
    categories = policy.get('categories', [])
    if not isinstance(categories, list) or not categories or not all(isinstance(x,str) for x in categories) or len(set(categories)) != len(categories) or 'review' not in categories:
        raise ValueError('Policy needs unique categories including review')
    threshold = policy.get('automatic_confidence_threshold')
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid automatic confidence threshold')
    rules = policy.get('classes', {})
    if set(rules) != set(s.NAMES):
        raise ValueError('Rules must cover exactly the seven detector classes')
    for rule in rules.values():
        if rule.get('category') not in categories or not isinstance(rule.get('instruction'),str):
            raise ValueError('Invalid category or instruction')
        requirements = rule.get('requires')
        if not isinstance(requirements,list) or not all(isinstance(x,str) and x for x in requirements):
            raise ValueError('Requirements must name explicit boolean facts')
    return policy


def classify(class_name, confidence, policy, context=None):
    validate_policy(policy)
    if type(confidence) not in (float,int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid detector confidence')
    context = context or {}
    if not isinstance(context,dict):
        raise ValueError('Context must contain explicit facts')
    rule = policy['classes'].get(class_name)
    reasons = []
    if not policy['verified']:
        reasons.append('school_rules_unconfirmed')
    if confidence < policy['automatic_confidence_threshold']:
        reasons.append('low_detection_confidence')
    if rule is None:
        reasons.append('unknown_object')
    else:
        if rule['category'] == 'review':
            reasons.append('material_or_local_acceptance_unknown')
        reasons.extend('confirm_' + fact for fact in rule['requires'] if context.get(fact) is not True)
    return dict(object_class=class_name, detection_confidence=confidence,
                category='review' if reasons else rule['category'],
                suggested_category=rule['category'] if rule else 'review',
                needs_review=bool(reasons), reasons=reasons,
                instruction=rule['instruction'] if rule else 'Identify the object before sorting.')


def classify_predictions(predictions, policy, min_conf=.25, contexts=None):
    validate_policy(policy)
    if type(min_conf) not in (int,float) or not math.isfinite(min_conf) or not 0 <= min_conf <= 1:
        raise ValueError('Invalid minimum detection confidence')
    contexts = contexts or {}
    if not isinstance(contexts,dict):
        raise ValueError('Contexts must map image names and detection indices to explicit facts')
    output = {}
    for name, rows in predictions.items():
        s.validate_prediction_rows(rows)
        output[name] = []
        image_contexts = contexts.get(name, {})
        if not isinstance(image_contexts,dict):
            raise ValueError('Image contexts must map detection indices to facts')
        for index,(x1,y1,x2,y2,confidence,cls) in enumerate(rows):
            if confidence < min_conf:
                continue
            class_name = s.NAMES[int(cls)]
            result = classify(class_name,confidence,policy,image_contexts.get(str(index)))
            output[name].append(dict(result,detection_index=index,box_xyxy_normalized=[x1,y1,x2,y2]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,help='Image or directory of images')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rules',type=Path,default=DEFAULT_RULES)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--contexts',type=Path,help='Optional JSON: image absolute path -> detection index string -> boolean facts')
    parser.add_argument('--min-conf',type=float,default=.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = validate_policy(s.read(args.rules))
    if not 0 <= args.min_conf <= 1:
        parser.error('--min-conf must be between zero and one')
    paths = sorted(p for p in args.source.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}) if args.source.is_dir() else [args.source]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No readable input images')
    records = [dict(name=str(p.resolve()),image_path=str(p.resolve()),sha256=s.sha(p)) for p in paths]
    receipt = s.predict(records,args.device)
    receipt = s.validate_receipt(receipt,records,s.read(s.CONFIG),s.sha(s.CONFIG))
    contexts = s.read(args.contexts) if args.contexts else {}
    decisions = classify_predictions(receipt['predictions'],policy,args.min_conf,contexts)
    result = dict(policy=policy,policy_sha256=s.sha(args.rules),min_detection_confidence=args.min_conf,
                  model_config_sha256=receipt['config_sha256'],explicit_contexts=contexts,decisions=decisions,prediction_receipt=receipt,
                  limitation='Rule-based suggestions, not a trained material classifier. Detector confidence is not disposal confidence. School disposal ground truth has not been evaluated.')
    s.save(args.output,result)
    print(json.dumps(dict(images=len(decisions),detections=sum(map(len,decisions.values())),
                          needs_review=sum(r['needs_review'] for rows in decisions.values() for r in rows),output=str(args.output)),indent=2))


if __name__ == '__main__':
    main()
