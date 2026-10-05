# Dataset selection protocol

## Core rule
**Domain match beats raw image count.**

Priority:
1. Real cafeteria/canteen post-consumer scenes
2. Cafeteria-style disposable serviceware and packaging
3. Generic waste images that closely match an NHS target
4. Generic object imagery only as targeted gap-filling

Reject examples:
- ceramic dinner plates for `plate`
- metal restaurant forks for disposable `utensil`
- generic Tupperware for `hinged_container`
- tissue boxes for `napkin`

Track original photos, augmented derivatives, retained strict-match images, object instances, and duplicates separately. Never report augmentations as independent diversity.
