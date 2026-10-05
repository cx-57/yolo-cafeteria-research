# Weak-class public-data audit

Priority classes: `paper_boat`, `compartment_container`, `hinged_container`, and `plate`.

## Acceptance rules

### paper_boat
Keep open-top disposable paper/cardboard food boats or shallow food trays that plausibly resemble NHS cafeteria boats. Reject closed lunch boxes, pizza boxes, cartons, rigid cafeteria trays, and generic cardboard packaging.

### compartment_container
Keep disposable food-service trays/containers with visibly separated compartments. Reject generic flat trays, sushi platters without compartments, storage boxes, and single-cavity takeout containers.

### hinged_container
Keep disposable food-service clamshells with an integrated hinge/lid. Prefer clear PET/PP or fiber clamshell geometry similar to NHS stock. Reject jars, storage tubs, detached-lid containers, and unrelated plastic boxes.

### plate
Keep disposable paper/fiber/bagasse cafeteria plates. Reject ceramic, glass, metal, reusable melamine, and restaurant dinnerware.

## Audit procedure

1. Export/download the source without augmentation.
2. Count source-class images and object instances locally.
3. Generate a contact sheet or random sample before accepting the source.
4. Review every candidate image if the source class has fewer than 500 images; otherwise review all candidates or a documented stratified sample before bulk acceptance.
5. Mark each candidate `keep`, `reject`, or `ambiguous`.
6. Write strict counts to `class_inventory.csv`.
7. Do not merge datasets until duplicate detection has run across approved sources.

## Priority order

1. INSPIRE Phase1 512 — `Paper_foodtray`
2. Form Material v1 — food/non-food clamshell PET/PP
3. MacWorX Containers — `Clamshell`
4. RTWFD — `Bio plate`, `Paper plate`, `Paper tray`
5. RecycAI testing — food trays / leftover containers
6. All Day Kitchens — fiber/plastic clamshells
7. Food Waste Management plate — plate class
8. smaller fallback sources

The source label is only a candidate mapping. No strict-match count is valid until image-level review is complete.
