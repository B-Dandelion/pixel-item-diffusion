"""Collect only the recorded CC0 export item paths, retaining ownership-conflict flags."""

import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_candidates import fetch, save_original


REVISION = "a6ea1655db5c044829d9eea19a232fa6fcac87b0"


def ensure_evidence():
    evidence = Path("data/research/dcss")
    for name in ("README.md", "ARTISTS.md", "TILES_UNDER_UNKNOWN_LICENSE.md"):
        if not (evidence / name).exists():
            body, _ = fetch(f"https://raw.githubusercontent.com/crawl/tiles/{REVISION}/{name}")
            save_original(evidence / name, body)
    path = Path("data/research/dcss_tree.json")
    if not path.exists():
        body, _ = fetch(f"https://api.github.com/repos/crawl/tiles/git/trees/{REVISION}?recursive=1",
                        12 * 1024 * 1024)
        tree = json.loads(body)
        if tree.get("truncated"):
            raise ValueError("Cannot use a truncated repository tree")
        items = [r for r in tree["tree"] if r["type"] == "blob"
                 and r["path"].startswith("releases/Nov-2015/item/") and r["path"].endswith(".png")]
        save_original(path, json.dumps(dict(revision=REVISION, items=items), indent=2).encode())
    tree = json.loads(path.read_text(encoding="utf-8"))
    if tree["revision"] != REVISION:
        raise ValueError("Local evidence revision differs from pinned export")
    return tree


def main():
    started = time.perf_counter()
    tree = ensure_evidence()
    unknown = set(re.findall(r"(?m)^- (.+\.png)\s*$", Path("data/research/dcss/TILES_UNDER_UNKNOWN_LICENSE.md").read_text(encoding="utf-8")))
    def get(item):
        url = f"https://raw.githubusercontent.com/crawl/tiles/{tree['revision']}/{item['path']}"
        result = dict(repository_path=item["path"], url=url, git_blob_sha1=item["sha"],
                      license_conflict=Path(item["path"]).name in unknown)
        try:
            body, _ = fetch(url, 1024 * 1024)
            blob_hash = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if blob_hash != item["sha"]:
                raise ValueError("Git blob hash differs from pinned tree")
            path = Path("data/raw/dcss_cc0") / item["path"]
            save_original(path, body)
            result.update(status="downloaded", path=path.as_posix(), sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:
            result.update(status="error", error=str(exc))
        return result
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(get, tree["items"]))
    report = dict(revision=tree["revision"], elapsed_seconds=time.perf_counter()-started,
                  downloaded=sum(r["status"] == "downloaded" for r in records),
                  flagged_for_license_review=sum(r["license_conflict"] for r in records), records=records)
    output = Path("results/dcss_collection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)


if __name__ == "__main__":
    main()
