# Measuring three-bin sorting performance

**Actual three-bin accuracy has not yet been measured.** The repaired 248-photo
set has object labels, but not independent per-object disposal-bin labels.
The existing 35.39% metric only measures correct automatic compost assignments
as a fraction of all object boxes. It is not sorting accuracy.

## How to measure it

1. Collect a new, held-out cafeteria photo session. Do not reuse images that
   influenced training, validation, test evaluations or model selection.
2. Independently annotate each eligible item with a tight normalized XYXY box
   and the correct school disposal bin: compost, recycling or landfill.
   Determine material, contamination, reuse and whether containers are empty
   from evidence and school acceptance rules. Resolve uncertain cases manually.
3. Run the frozen sorting pipeline without changing thresholds or rules.
4. The current user-selected rule assigns detected bottles and drink cartons
   to recycling at confidence >= 0.50 **without checking emptiness**. This
   experimental shortcut is not proof that filled containers are recyclable
   locally. Preserve independent evidence of correct bin eligibility.
5. Preserve photos with zero labeled items, and use identical image path keys.

Run from the repository root:

    ./scripts/run_specialists.sh sort /absolute/path/to/new_photos \
      --device mps --output /absolute/path/to/sort_predictions.json

Create an independently checked labels file like this:

    {
      "images": {
        "/absolute/path/to/new_photos/image01.jpg": [
          {"box_xyxy_normalized": [0.10, 0.20, 0.40, 0.50],
           "category": "compost"},
          {"box_xyxy_normalized": [0.50, 0.20, 0.80, 0.70],
           "category": "recycling"}
        ],
        "/absolute/path/to/new_photos/empty_image.jpg": []
      }
    }

These labels/boxes are ONLY a format example. They are not actual photo
annotations. Ground truth cannot be generated from model predictions.
The image names must match those in the sorting JSON's decisions map exactly.

Then run:

    python3 scripts/evaluate_sorting.py \
      --predictions /absolute/path/to/sort_predictions.json \
      --truth /absolute/path/to/human_bin_labels.json \
      --output /absolute/path/to/sorting_scores.json

The evaluator matches predictions to labeled objects at box IoU >= 0.50,
highest confidence first. Matching does not look at predicted or true bins.

## Interpret the output

- correct_automatic_over_all_labeled_objects: correctly automatically sorted
  / all labeled objects; a coverage-sensitive overall success fraction.
- accuracy_among_matched_automatic_decisions: correct / (correct + wrong bin)
  among matched automatic decisions; excludes reviews, misses and unmatched
  false-positive predictions.
- precision_including_unmatched_auto_predictions: correct / all automatic
  bin assignments, including unmatched false positives and duplicates.
- automatic_coverage_of_objects: fraction of ground-truth objects receiving
  an automatic, matched compost/recycling/landfill decision.
- matched_review, missed_objects, wrong_bin_automatic, and
  unmatched_predictions must be reported alongside accuracy.
  Review is abstention, not a correct prediction.
- confusion_by_true_bin: actual bin vs predicted bin/review/missed.

For a research-paper claim, use independently collected photos with
human-verified disposal labels and document the local sorting policy.
Previously inspected or trained-on images are not an independent benchmark.
Do not claim complete three-bin accuracy until that evaluation exists.

To check implementation:

    python3 -m unittest discover -s tests -p 'test_sorting_evaluation.py' -v
