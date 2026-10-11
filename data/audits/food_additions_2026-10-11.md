# Food additions screening (2026-10-11)

* School Lunch v2: 1,876 source images. Of the 1,409 not already imported, selected 130 rice, curry, mixed rice, udon, spicy tofu, fried noodle, soup and manty photos. Original 467 retained filenames were excluded. Current IDs 0=food, 6=drink_carton.
* Food v1: 849 source images; 380 retained — 89 burgers, 82 French fries, 76 pizzas, 46 fried chicken, 38 hot dogs, 24 salads, 15 donuts, 10 noodles; no steak-only photos.
* 510 JPEGs, 510 detection labels, 1,197 normalized valid boxes (1,072 food + 125 carton). No byte-identical duplicates. Photos decoded; images, labels and source hashes verified.
* Food v1 source annotation polygons were converted to YOLO boxes; selected images were visually screened on contact sheets. Multi-class exhaustiveness is NOT fully verified. Stock/product photos are poorer proxies for natural cafeteria scenes.
* All additions training-only. Source download and image-level manifests included in ZIP deliverable. Upload to GitHub with upload_food_additions.sh.
