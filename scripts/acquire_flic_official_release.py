#!/usr/bin/env python3
"""Download the complete official public FLIC release without altering image bytes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
import csv, hashlib, io, json, os, random, shutil, sys, time, zipfile
import requests
import gdown

MANIFEST = Path("research/flic_official_drive_manifest.json")
ROOT = Path("flic_release_download")
OUTPUT = Path("flic_official_release_802_images_802_masks.zip")
RESULT = Path("results/flic_acquisition_receipt.json")
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def download_one(kind, entry):
    target = ROOT / kind / entry["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.read_bytes()[:8] == PNG_MAGIC:
        return dict(kind=kind, name=entry["name"], status="ok", bytes=target.stat().st_size)
    url = f"https://drive.google.com/uc?export=download&id={entry['id']}"
    error = ""
    for trial in range(4):
        try:
            r = requests.get(url, timeout=45)
            if r.status_code == 200 and r.content.startswith(PNG_MAGIC):
                target.write_bytes(r.content)
                return dict(kind=kind, name=entry["name"], status="ok", bytes=len(r.content))
            error = f"HTTP {r.status_code}, content_type={r.headers.get('Content-Type')}, bytes={len(r.content)}"
            # Google Drive sometimes requires the browser-confirmation flow.
            returned = gdown.download(id=entry["id"], output=str(target), quiet=True, use_cookies=False)
            if returned and target.is_file() and target.read_bytes()[:8] == PNG_MAGIC:
                return dict(kind=kind, name=entry["name"], status="ok", bytes=target.stat().st_size)
            target.unlink(missing_ok=True)
        except Exception as exc:
            error = str(exc)[:350]
        time.sleep(min(6, 1 + trial * 2))
    return dict(kind=kind, name=entry["name"], id=entry["id"], status="failed", error=error)


def main():
    source = json.loads(MANIFEST.read_text())
    tasks = [(kind, rec) for kind, rows in source["groups"].items() for rec in rows]
    assert len(tasks) == 1604, len(tasks)
    # Test publicly accessible origin before opening many transfers.
    checks = [("full_images", source["groups"]["full_images"][0]),
              ("full_masks", source["groups"]["full_masks"][0])]
    start = time.monotonic()
    first = [download_one(*x) for x in checks]
    if any(x["status"] != "ok" for x in first):
        result = dict(state="blocked_public_file_download", source=source["source"],
            test_results=first, expected_files=len(tasks), downloaded=len([x for x in first if x["status"]=="ok"]))
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(result, indent=2))
        print("PUBLIC DOWNLOAD FAILED:", json.dumps(result), flush=True)
        return 1

    done = {(k, e["name"]) for k,e in checks}
    rest = [(k,e) for k,e in tasks if (k,e["name"]) not in done]
    results = first[:]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(download_one, k, e) for k,e in rest]
        for fut in as_completed(futures):
            results.append(fut.result())
            if len(results) % 100 == 0:
                failures = sum(r["status"] == "failed" for r in results)
                print(f"Processed {len(results)}/{len(tasks)} files; failures {failures}", flush=True)

    errors = [r for r in results if r["status"] != "ok"]
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    for kind in source["groups"]:
        counts[kind] = sum(r["kind"]==kind and r["status"]=="ok" for r in results)
    receipt = dict(state="complete" if not errors else "incomplete", source=source["source"],
        license=source["license"], expected_files=len(tasks), downloaded=len(results)-len(errors),
        counts=counts, failures=errors, elapsed_seconds=round(time.monotonic()-start,1))
    RESULT.write_text(json.dumps(receipt, indent=2))
    print("RESULT:", json.dumps({k:v for k,v in receipt.items() if k!="failures"}), flush=True)
    if errors:
        return 2

    # Only package complete acquisition; PNG bytes are copied exactly.
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for kind, entry in tasks:
            local = ROOT/kind/entry["name"]
            z.write(local, arcname=f"{kind}/{entry['name']}")
        z.writestr("SOURCE_LICENSE_README.txt", (
          "FLIC: A Real-World Dataset for Visual Estimation of Food Leftovers in Canteens\n"
          "Authors: Piccoli et al. (Applied Sciences, 2026)\n"
          "Official source: https://drive.google.com/drive/folders/1IGCjr7y-_zsZfHTsL4CyxdkspTp8Uhnt\n"
          "License: CC BY 4.0 (provide attribution when redistributing)\n"
          "Contains 401 paired pre/post images and 401 corresponding pre/post semantic masks.\n"
          "These are source masks, not seven-class YOLO ground-truth boxes.\n"
          "The user already has an annotation checkpoint of 104 images; ignore/reuse those files based on basename.\n"
        ))
        z.writestr("official_manifest.json", MANIFEST.read_text())
        z.writestr("acquisition_receipt.json", json.dumps(receipt, indent=2))
    print("ARCHIVE:", OUTPUT, OUTPUT.stat().st_size, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
