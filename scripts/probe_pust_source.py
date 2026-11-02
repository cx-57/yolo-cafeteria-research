"""Check whether the official CC BY 4.0 PUST cafeteria dataset permits public file access.

No authentication bypass, copying, model runs or photo downloads. A missing
public link means users must obtain the archive from the official dataset page.
"""
import json
from pathlib import Path
import requests

URLS = [
    "https://api.data.mendeley.com/datasets/publics/fn6yhzjz83/files?version=2",
    "https://api.data.mendeley.com/datasets/fn6yhzjz83/files?version=2",
    "https://api.data.mendeley.com/datasets/fn6yhzjz83/snapshot?version=2",
]


def summarize(value):
    if isinstance(value, list):
        return dict(items=[dict(filename=v.get("filename"), id=v.get("id"),
                                size=v.get("size",v.get("content_details",{}).get("size")))
                           for v in value[:20] if isinstance(v, dict)], count=len(value))
    if isinstance(value, dict):
        return dict(keys=sorted(value.keys()),
                    files=summarize(value["files"]) if "files" in value else None)
    return {"type": type(value).__name__}


def run(output):
    report = dict(dataset="PUST Cafeteria Food Image Dataset",version=2,
                  source_url="https://data.mendeley.com/datasets/fn6yhzjz83/2",
                  license="CC BY 4.0",downloaded_photos=0,
                  public_file_urls_verified=False,
                  probes=[])
    for url in URLS:
        try:
            result = requests.get(url,timeout=30,headers={"Accept":"application/json",
                                            "User-Agent":"CafeteriaResearchCandidateAudit/1.0"})
            probe=dict(endpoint=url,status=result.status_code)
            if result.ok:
                value=result.json()
                probe["metadata"]=summarize(value)
                if isinstance(value,list) and value:
                    report["public_file_urls_verified"]=True
            else:
                probe["error"]=result.text[:160]
        except Exception as err:
            probe=dict(endpoint=url,error=str(err)[:200])
        report["probes"].append(probe)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    run("results/pust-public-api-probe.json")
