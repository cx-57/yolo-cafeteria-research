# New hinged-container image sourcing — 2026-11-01

## Downloaded outputs (actual GitHub Actions artifacts)

- [Curated **29-photo batch** (21 likely hinged containers; 8 negatives), ~6.9 MB](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/41332845065/artifacts/13093654495).
  Includes `images/`, `manifest.json` with Commons license/creator/source, `index.html`, and `contact-sheet.jpg`.
- [Original **64-photo source pool**, ~15.6 MB](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/41332494158/artifacts/13093033962).
  Includes 18 from foam-food-boxes, 18 from other foam-container sources, 10 from plastic food boxes, 10 from disposable lunch boxes, and eight expanded-polystyrene tray sources.
- These artifacts expire **2026-12-01**. The GitHub Actions workflow can stage a new batch later, but it may not be identical when upstream sources change.

The 64-image pool was downloaded via the Wikimedia Commons API, using per-image rights metadata that advertises CC BY, CC0, or public-domain licenses. The automated collection **excluded the 38 titles from the prior Commons artifact** and checked byte SHA-256 hashes. Source licensing/attribution still require individual verification before reuse/publication. Cross-checking against the original private Mac historical training collection is NOT complete.

The 29-photo selection was based on qualitative review of the new contact sheet (not on their detector scores) and is **provisional**: 21 photographs appear to show open, closed, foam or plastic food boxes/clamshells; eight are candidate hard negatives (trays, food-on-plates, or other similar-looking non-clamshells). They are not human-adjudicated class labels. One photograph can contain multiple instances; original capture groups and near-duplicate scenes have not been independently certified.

### Before any new training

1. Inspect all 21 likely container photos individually to verify they really depict the target **hinged** class. Some closed boxes may obscure the hinge; mark uncertainty rather than assuming.
2. Annotate every visible target clamshell in YOLO box format, and maintain correctly labeled `food` and `drink_carton` objects on these photos as required by the original three-class specialist. On hard-negative photos, don't mark a visible plate as hinged.
3. Check these source images against *all* existing Mac train, validation and test photos for near-duplicate captures—not only SHA matches. Keep these newly reviewed public photographs in a **new training-only partition**, not the locked evaluation cohort.
4. Make a new candidate, keeping the original four specialists frozen; compare against the unchanged validation metrics and use the 12 public hinged-source photos for qualitative diagnostics only.

**No new model training or improved mAP has occurred during this acquisition.**
