# Dataset Filtering Log — 2026-10-07

## Purpose

This file records the manual visual filtering performed on the public-source datasets used for the NHS cafeteria YOLO project. The filtering goal was not simply to keep images whose source labels matched a project class. An image was retained only when the visible object and scene were useful for the actual NHS cafeteria deployment domain.

Date of review: **October 7, 2026**

## NHS compatibility standard

Each source image was visually reviewed. Images were kept only when the annotated object was plausibly representative of an NHS cafeteria item.

General rules used during review:

- Prefer real-world cafeteria or food-service objects over product mockups, retail packaging displays, household objects, or unrelated trash.
- Keep object shapes, materials, sizes, and presentation that are reasonably similar to NHS items.
- Reject examples that share a broad semantic label but are visually incompatible with the NHS target domain.
- Reject mislabeled, ambiguous, unusable, or non-target examples.
- Do not keep images solely to increase dataset size.
- Rejected images are not copied into `data/processed/`.
- Retained images keep only their matching reviewed labels.
- Public-source images are training/validation candidates only; the final NHS natural test set remains separate and untouched.

## Annotation normalization

The project trains YOLO **object detection**.

Some Roboflow YOLOv11 exports contained segmentation polygons rather than five-value YOLO detection rows. For retained examples, polygon annotations were converted to axis-aligned YOLO bounding boxes:

`class x_center y_center width height`

The conversion and validation logic was added to `src/prepare.py`.

All final labels are validated so that:

- every row has exactly 5 values;
- class IDs match the project-wide class map;
- normalized coordinates are within `[0, 1]`;
- width and height are positive;
- every retained image has a matching label file.

## Review results

| Source | Images reviewed | Kept | Scrapped | Project target |
| --- | ---: | ---: | ---: | --- |
| School Lunch v2 | 1,876 | 467 | 1,409 | food; drink_carton |
| Tetra Pak v2 | 1,710 | 661 | 1,049 | drink_carton |
| MacWorX Containers v7 | 420 | 121 | 299 | hinged_container |
| Utensil v1 | 465 | 128 | 337 | utensil |
| Sauce v2 | 59 | 0 | 59 | condiment_packet |
| napkin v1 | 180 | 0 | 180 | napkin |
| Plastic Detector v11 | 115 | 0 | 115 | bottle; wrapper candidate |
| **Total** | **4,825** | **1,377** | **3,448** | |

## Dataset-specific decisions

### School Lunch v2

**467 / 1,876 retained.**

This dataset contained many visually repetitive East-Asian school-lunch scenes. Different food itself was not an automatic rejection criterion, because the project uses a broad `food` class. The stricter question was whether an image still provided useful cafeteria-food visual information without allowing the source domain to dominate the training set.

Retained:
- cafeteria-style food portions;
- foods visually useful for the broad `food` class;
- small milk/drink cartons useful for `drink_carton`.

Rejected:
- excessive repetitive source-domain examples;
- scenes or objects with little transfer value to NHS;
- classes/items not useful to the final project.

The October 7 re-upload was verified against the already reviewed source and did not require changing the final 467-image subset.

### Tetra Pak v2

**661 / 1,710 retained.**

Retained:
- small, single-serve aseptic cartons with geometry reasonably similar to NHS milk/juice cartons;
- usable real-world carton views at multiple angles and conditions.

Rejected:
- large/family-size cartons;
- non-cafeteria package formats;
- synthetic/product mockups when not useful;
- flattened or unrelated packaging;
- visually incompatible carton geometries.

Final project class: `7 = drink_carton`.

### MacWorX Containers v7

**121 / 420 retained.**

Retained:
- clear/fiber takeout-style hinged containers and clamshells that reasonably resemble cafeteria takeout containers.

Rejected:
- produce packages;
- egg containers;
- household/reusable containers;
- retail packaging;
- incompatible foam/container geometries;
- label-only or otherwise nonrepresentative examples.

Final project class: `5 = hinged_container`.

### Utensil v1

**128 / 465 retained.**

Retained:
- disposable black/white plastic forks, spoons, and similar cafeteria utensils.

Rejected:
- metal silverware;
- colored reusable-looking utensils;
- transparent/wooden utensils when not representative of NHS;
- bottles or unrelated objects;
- other visually incompatible examples.

Final project class: `3 = utensil`.

### Sauce v2

**0 / 59 retained.**

Entire source rejected.

The images primarily showed squeeze-bottle sauces. NHS target objects are single-serve cafeteria condiment packets, so these images would teach the wrong object geometry and packaging format.

### napkin v1

**0 / 180 retained.**

Entire source rejected.

The dataset primarily showed green cloth/reusable napkins. NHS uses disposable cafeteria paper napkins, especially natural/brown paper forms, so the source was considered a domain mismatch.

### Plastic Detector v11

**0 / 115 retained.**

Entire source rejected.

The source used broad material-oriented labels and mixed scenes that did not map cleanly to the project's object classes. Keeping them would require guessing whether a plastic region represented `bottle`, `wrapper`, `container`, or another object, which would introduce label noise.

## Current reviewed training pool from this batch

Retained public-source images: **1,377**

Known retained object instances:
- School Lunch: 2,436
- Tetra Pak: 738
- MacWorX: 121
- Utensil: 242
- **Total known instances: 3,537**

These counts describe the reviewed subset from this batch only. Additional approved datasets and NHS-local images may be added later.

## Replacement needed for Plastic Detector

The rejected `Plastic Detector v11` source should be replaced with an **object-specific litter dataset**, not a generic plastic/material classifier.

Preferred replacement criteria:

- object detection annotations;
- explicit classes such as `Clear plastic bottle`, `Other plastic bottle`, and `Other plastic wrapper`;
- enough source images to survive aggressive NHS-specific visual filtering;
- permissive research-use license;
- real discarded-object scenes rather than isolated material samples.

The replacement source must go through the same image-by-image visual audit before any image is added to `data/processed/`.
