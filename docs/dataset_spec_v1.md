# Dataset Specification v1.0 — Working Draft

## Goal

Construct a reproducible YOLO dataset for the locked research question:

> To what extent can a real-time YOLO-based computer vision system provide accurate and reliable disposal guidance under realistic school cafeteria conditions?

## Target classes

`food`, `plate`, `paper_boat`, `utensil`, `compartment_container`, `hinged_container`, `napkin`, `milk_carton`, `condiment_packet`, `wrapper`, `bottle`.

## Current public-data picture

### Food is not the bottleneck

The first verified high-domain sources total **7,533 independent cafeteria/institutional images** before any strict local filtering:

- FLIC — 802
- Ferreira School Canteen — 175 base images
- PUST Cafeteria v2 — 720 original images
- CC-FOOD-100 — 3,354
- UNIMIB2015 — 1,080
- UNIMIB2016 — 1,027
- HPI Food Waste — 375

This total intentionally excludes augmented derivatives.

### Non-food objects require strict visual filtering

Generic waste datasets are not counted toward Dataset V1 merely because they contain a matching label. For example:

- TACO has useful source labels for plastic utensils, drink cartons, wrappers, and beverage bottles, but it is a litter dataset rather than a cafeteria dataset.
- Roboflow Garbage Classification exposes relevant labels such as plastic utensil, milk box, napkin, packet, snack package, and bottle, but per-class counts still need to be measured from a local export.
- Waste Segregation contains napkin, junk food wrapper, and plastic bottle labels, but the same strict image-level audit is required.

## Strict-match rule

An image is retained only if the object is visually plausible in the intended NHS deployment environment.

Examples:

- disposable fiber/bagasse plate: candidate
- ceramic restaurant plate: reject
- disposable plastic/compostable fork: candidate
- metal steak knife: reject
- clear cafeteria clamshell: candidate
- household food-storage container: reject
- crumpled cafeteria napkin: candidate
- cloth table napkin: reject

## Immediate deficits

The classes with no verified strict-match public pool yet are:

1. `plate`
2. `paper_boat`
3. `compartment_container`

Likely difficult after filtering:

4. `hinged_container`
5. `condiment_packet`
6. `napkin`
7. `utensil`
8. `milk_carton`

`wrapper` and `bottle` already have stronger public candidates, but their strict counts are still pending.

## Completion criteria for Dataset Specification v1.0

- all candidate source licenses/usage terms checked
- every source downloaded without modifying originals
- source-class counts computed locally
- strict visual audit completed
- strict-match counts entered in `class_inventory.csv`
- deficits calculated against `data_targets.csv`
- NHS-specific photography plan derived from those deficits
- source mappings version-controlled
- duplicate/near-duplicate audit completed before final merge

No model training should begin until the source audit is far enough along to identify the major class deficits.
