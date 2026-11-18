"""Sample public HF images via paged rows; no datasets package or full shard download."""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_candidates import fetch, save_original


def api(endpoint, **params):
    url = "https://datasets-server.huggingface.co/" + endpoint + "?" + urlencode(params)
    body, _ = fetch(url, 12 * 1024 * 1024)
    return json.loads(body)


def sample_source(candidate, count, seed):
    sid, repo = candidate["source_id"], candidate["repo_id"]
    evidence = Path("data/research/hf") / sid
    # Retain the first observed revision and card so a rerun cannot silently change provenance.
    if not (evidence / "info.json").exists():
        body, _ = fetch(f"https://huggingface.co/api/datasets/{repo}", 12 * 1024 * 1024)
        save_original(evidence / "info.json", body)
    info = json.loads((evidence / "info.json").read_text(encoding="utf-8"))
    revision = info["sha"]
    if not (evidence / "card.md").exists():
        body, _ = fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md")
        save_original(evidence / "card.md", body)
    report = dict(source_id=sid, repository=repo, revision=revision, seed=seed, errors=[])
    for endpoint in ("size", "statistics"):
        try:
            params = dict(dataset=repo)
            if endpoint == "statistics":
                params.update(config="default", split="train")
            data = api(endpoint, **params)
            (evidence / f"{endpoint}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            report["errors"].append(f"{endpoint}: {exc}")
    rng = random.Random(seed)
    rows = []
    if sid == "hf_krea16":
        body, _ = fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/metadata.jsonl")
        save_original(evidence / "metadata.jsonl", body)
        metadata = [json.loads(line) for line in body.decode().splitlines() if line]
        for index in sorted(rng.sample(range(len(metadata)), min(count, len(metadata)))):
            row = dict(metadata[index])
            row["image"] = {"src": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/sprites/{index:04d}.png"}
            rows.append(dict(row_idx=index, row=row, sample_method="seeded_random_native_sprite"))
        report["population_rows"] = len(metadata)
    else:
        first = api("rows", dataset=repo, config="default", split="train", offset=0, length=1)
        total = first["num_rows_total"]
        report["population_rows"] = total
        block = 16
        offsets = [rng.randrange(max(1, total - block + 1)) for _ in range((count + block - 1) // block)]
        report["block_offsets"] = offsets
        def get_rows(offset):
            result = api("rows", dataset=repo, config="default", split="train", offset=offset, length=block)
            return [dict(r, sample_method="seeded_random_block") for r in result["rows"]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for group in pool.map(get_rows, offsets):
                rows.extend(group)
        rows = rows[:count]
        if sid == "hf_pixelart172k":
            categories = ["melee_weapons", "ranged_and_magic_weapons", "armour_and_shields",
                          "potions_and_alchemy", "gems_ores_and_materials", "food", "scrolls_books_and_runes", "rpg_items"]
            for category in categories:
                try:
                    result = api("filter", dataset=repo, config="default", split="train",
                                 where=f'"category"=\'{category}\'', offset=0, length=8)
                    rows.extend(dict(r, sample_method="targeted_category") for r in result["rows"])
                except Exception as exc:
                    report["errors"].append(f"filter/{category}: {exc}")
    unique = {r["row_idx"]: r for r in rows}
    def download_row(record):
        index, row = record["row_idx"], record["row"]
        result = {k: v for k, v in row.items() if k != "image"}
        result.update(row_idx=index, source_id=sid, revision=revision, sample_method=record["sample_method"])
        try:
            url = row["image"]["src"]
            if sid != "hf_krea16" and revision not in url:
                raise ValueError("Viewer revision differs from recorded repository revision")
            body, _ = fetch(url, 2 * 1024 * 1024)
            with Image.open(io.BytesIO(body)) as im:
                im.load()
                result.update(width=im.width, height=im.height, mode=im.mode, actual_format=im.format,
                              source_image_url=url, lossless_original_verified=False)
                extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(im.format)
                if extension is None:
                    raise ValueError(f"Unsupported received image format: {im.format}")
            path = Path("data/raw") / sid / "samples" / f"{index:07d}{extension}"
            save_original(path, body)
            result.update(path=path.as_posix(), sha256=hashlib.sha256(body).hexdigest(), status="downloaded")
        except Exception as exc:
            result.update(status="error", error=str(exc))
        return result
    with ThreadPoolExecutor(max_workers=8) as pool:
        samples = list(pool.map(download_row, unique.values()))
    output = Path("data/raw") / sid / "sample_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    report.update(requested=len(unique), downloaded=sum(r["status"] == "downloaded" for r in samples),
                  failed=sum(r["status"] != "downloaded" for r in samples),
                  note="Small review sample only; block/targeted sampling is not an unbiased quality estimate.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--include-research-only", action="store_true",
                        help="Explicitly opt into synthetic research samples; never training data approval")
    args = parser.parse_args()
    if not args.include_research_only:
        parser.error("Synthetic collection is disabled by default. Existing samples remain research only.")
    if not 1 <= args.count <= 512:
        parser.error("Review samples must contain 1..512 rows per source")
    candidates = json.loads(Path("configs/data_candidates.json").read_text(encoding="utf-8"))
    active_ids = {c["source_id"] for c in candidates if c["kind"] == "hf" and c.get("review_status") == "active_review"}
    if args.ids and not set(args.ids).issubset(active_ids):
        parser.error("Requested HF source is excluded by the current dataset policy")
    reports = []
    started = time.perf_counter()
    output = Path("results") / f"hf_sampling_{time.time_ns()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        if candidate["source_id"] not in active_ids or (args.ids and candidate["source_id"] not in args.ids):
            continue
        try:
            report = sample_source(candidate, args.count, args.seed)
        except Exception as exc:
            report = dict(source_id=candidate["source_id"], error=str(exc))
        reports.append(report)
        print(json.dumps(report), flush=True)
        output.write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started, reports=reports), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
