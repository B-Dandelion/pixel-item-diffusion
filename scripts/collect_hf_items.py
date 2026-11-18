"""Collect a bounded, category-selected HF review pool; never create a training split."""

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_candidates import fetch, save_original


class RangeReader(io.RawIOBase):
    """Seekable HTTP reader. Refuse servers that ignore Range, and bound transfer size."""

    def __init__(self, url, max_bytes=256 * 1024 * 1024):
        super().__init__()
        self.url = url
        self.position = 0
        self.transferred = 0
        self.max_bytes = max_bytes
        self.requests = 0
        self.size = None
        self.buffer_start = 0
        self.buffer = b""
        self._range(0, 7)

    def _range(self, start, end):
        length = end - start + 1
        if self.transferred + length > self.max_bytes:
            raise ValueError("HTTP transfer budget exceeded")
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                             "User-Agent": "PixelItemDatasetReview/1.0"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=40) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, end):
                        raise ValueError("Server did not honor the exact byte range")
                    size = int(match[3])
                    if self.size is not None and size != self.size:
                        raise ValueError("Remote size changed during collection")
                    self.size = size
                    body = response.read(length + 1)
                    self.transferred += len(body)
                    self.requests += 1
                    if len(body) != length:
                        raise ValueError("Incomplete range response")
                    return body
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if self.position >= end:
            return b""
        if self.buffer_start <= self.position and end <= self.buffer_start + len(self.buffer):
            body = self.buffer[self.position-self.buffer_start:end-self.buffer_start]
        else:
            body = self._range(self.position, end - 1)
        self.position = end
        return body

    def readinto(self, target):
        body = self.read(len(target))
        target[:len(body)] = body
        return len(body)

    def prefetch_columns(self, group, names):
        chunks = [group.column(i) for i in range(group.num_columns)
                  if group.column(i).path_in_schema.split(".")[0] in names]
        starts = [min(offset for offset in (c.dictionary_page_offset, c.data_page_offset)
                      if offset is not None and offset >= 0) for c in chunks]
        start = min(starts)
        end = max(offset + chunk.total_compressed_size for offset, chunk in zip(starts, chunks))
        self.buffer = self._range(start, end - 1)
        self.buffer_start = start


def select_rows(rows, category_map, caps, seed):
    """Stable random selection within each coarse class; no invented semantic labels."""
    groups = defaultdict(list)
    for row in rows:
        source_category = row.get("category", row.get("group"))
        label = category_map.get(source_category)
        if label is not None:
            groups[label].append(dict(row, item_class=label, label_status="source_label_unverified"))
    selected = []
    for label, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["row_idx"])
        rng = random.Random(f"{seed}:{label}")
        selected.extend(rng.sample(ordered, min(caps.get(label, len(ordered)), len(ordered))))
    return sorted(selected, key=lambda row: row["row_idx"])


def image_facts(body, expected_size):
    with Image.open(io.BytesIO(body)) as image:
        image.load()
        if image.format != "PNG" or image.size != tuple(expected_size) or image.mode != "RGBA":
            raise ValueError(f"Unexpected original: {image.format} {image.size} {image.mode}")
        rgba = image.copy()
        # Zero invisible RGB only for a hash. Original bytes are never rewritten.
        rgba.paste((0, 0, 0, 0), mask=image.getchannel("A").point(lambda a: 255 if a == 0 else 0))
        return dict(width=image.width, height=image.height, mode=image.mode,
                    actual_format=image.format, empty=image.getchannel("A").getbbox() is None,
                    sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                    canonical_rgba_sha256=hashlib.sha256(str(image.size).encode() + rgba.tobytes()).hexdigest())


def store_image(record, body, sid, revision, size, existing):
    result = {k: v for k, v in record.items() if k != "image"}
    result.update(source_id=sid, revision=revision, lossless_original_verified=True,
                  status="downloaded", final_inclusion="pending_visual_label_license_review",
                  split="source_train" if sid == "hf_pixelart172k" else "unsplit_source")
    result.update(image_facts(body, size))
    old = existing.get(record["row_idx"])
    if old and Path(old["path"]).exists() and hashlib.sha256(Path(old["path"]).read_bytes()).hexdigest() == result["sha256"]:
        path = Path(old["path"])
        result["reused_sample"] = True
    else:
        path = Path("data/raw") / sid / "items" / record["item_class"] / f"{record['row_idx']:07d}.png"
        save_original(path, body)
        result["reused_sample"] = False
    result["path"] = path.as_posix()
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_pixelart(policy, preview_only=False):
    import pyarrow.parquet as pq

    sid = "hf_pixelart172k"
    evidence = Path("data/research/hf") / sid
    info = json.loads((evidence / "info.json").read_text(encoding="utf-8"))
    revision, repo = info["sha"], info["id"]
    cache = evidence / f"item_metadata_{revision}.json"
    if cache.exists():
        metadata = json.loads(cache.read_text(encoding="utf-8"))
    else:
        metadata = []
        offset = 0
        for shard in range(4):
            relative = f"data/train-{shard:05d}-of-00004.parquet"
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}?download=true"
            with RangeReader(url) as remote:
                parquet = pq.ParquetFile(remote)
                columns = [name for name in parquet.schema_arrow.names if name != "image"]
                print(json.dumps(dict(stage="metadata", shard=shard, rows=parquet.metadata.num_rows,
                                      groups=parquet.num_row_groups, columns=columns)), flush=True)
                local_offset = 0
                for group in range(parquet.num_row_groups):
                    remote.prefetch_columns(parquet.metadata.row_group(group), columns)
                    for index, row in enumerate(parquet.read_row_group(group, columns=columns, use_threads=False).to_pylist()):
                        row.update(row_idx=offset + local_offset + index, shard=shard,
                                   row_group=group, index_in_group=index, source_parquet_url=url)
                        metadata.append(row)
                    local_offset += parquet.metadata.row_group(group).num_rows
                offset += local_offset
                print(json.dumps(dict(stage="metadata_done", shard=shard, transferred_bytes=remote.transferred)), flush=True)
        write_json(cache, metadata)
    selected = select_rows(metadata, policy["categories"], policy["class_caps"], policy["seed"])
    report = dict(source_id=sid, revision=revision, population=len(metadata), selected=len(selected),
                  classes=dict(Counter(row["item_class"] for row in selected)),
                  source_categories=dict(Counter(row["category"] for row in selected)),
                  metadata_cache=cache.as_posix())
    print(json.dumps(report), flush=True)
    if preview_only:
        return report
    existing_path = Path("data/raw") / sid / "sample_manifest.json"
    existing = {r["row_idx"]: r for r in json.loads(existing_path.read_text(encoding="utf-8"))}
    by_shard = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_shard[row["shard"]][row["row_group"]].append(row)
    def extract_shard(pair):
        shard, groups = pair
        results = []
        with RangeReader(next(iter(groups.values()))[0]["source_parquet_url"]) as remote:
            parquet = pq.ParquetFile(remote)
            for group, records in sorted(groups.items()):
                images = parquet.read_row_group(group, columns=["image"], use_threads=False).to_pylist()
                for record in records:
                    image = images[record["index_in_group"]]["image"]
                    results.append(store_image(record, image["bytes"], sid, revision, [32, 32], existing))
                print(json.dumps(dict(stage="images", shard=shard, group=group, collected=len(results))), flush=True)
            return results, remote.transferred
    results = []
    transferred = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows, size in pool.map(extract_shard, sorted(by_shard.items())):
            results.extend(rows)
            transferred += size
    results.sort(key=lambda row: row["row_idx"])
    write_json(Path("data/raw") / sid / "item_manifest.json", results)
    report.update(summarize(results), image_range_transfer_bytes=transferred)
    return report


def summarize(rows):
    return dict(downloaded=len(rows), original_png_bytes=sum(row["bytes"] for row in rows),
                reused_samples=sum(row["reused_sample"] for row in rows),
                unique_visible=len({row["canonical_rgba_sha256"] for row in rows}),
                empty=sum(row["empty"] for row in rows),
                source_flags=dict(Counter(flag for row in rows for flag in (row.get("flags") or []))))


def collect_krea(policy, preview_only=False):
    sid = "hf_krea16"
    evidence = Path("data/research/hf") / sid
    info = json.loads((evidence / "info.json").read_text(encoding="utf-8"))
    revision, repo = info["sha"], info["id"]
    metadata = [dict(json.loads(line), row_idx=i) for i, line in
                enumerate((evidence / "metadata.jsonl").read_text(encoding="utf-8").splitlines())]
    selected = select_rows(metadata, policy["groups"], {}, policy["seed"])
    report = dict(source_id=sid, revision=revision, population=len(metadata), selected=len(selected),
                  classes=dict(Counter(row["item_class"] for row in selected)))
    print(json.dumps(report), flush=True)
    if preview_only:
        return report
    existing_path = Path("data/raw") / sid / "sample_manifest.json"
    existing = {r["row_idx"]: r for r in json.loads(existing_path.read_text(encoding="utf-8"))}
    def download(record):
        relative = record["sprite"]
        if not re.fullmatch(r"sprites/\d{4}\.png", relative):
            raise ValueError("Unexpected native sprite path")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}"
        old = existing.get(record["row_idx"])
        if old and Path(old["path"]).exists():
            body = Path(old["path"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != old["sha256"]:
                raise ValueError("Existing original sample hash changed")
        else:
            body, _ = fetch(url, 1024 * 1024)
        return store_image(dict(record, source_image_url=url), body, sid, revision, [16, 16], existing)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(download, selected))
    write_json(Path("data/raw") / sid / "item_manifest.json", rows)
    report.update(summarize(rows))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Read metadata only; no image payloads")
    parser.add_argument("--ids", nargs="+", choices=["hf_pixelart172k", "hf_krea16"],
                        default=["hf_pixelart172k", "hf_krea16"])
    args = parser.parse_args()
    policy = json.loads(Path("configs/item_collection.json").read_text(encoding="utf-8"))
    active = {c["source_id"] for c in json.loads(Path("configs/data_candidates.json").read_text(encoding="utf-8"))
              if c.get("review_status") == "active_review"}
    if not set(args.ids).issubset(active):
        parser.error("Source is no longer allowed for review")
    reports = []
    started = time.perf_counter()
    for sid in args.ids:
        collector = collect_pixelart if sid == "hf_pixelart172k" else collect_krea
        reports.append(collector(dict(policy[sid], seed=policy["seed"]), args.preview))
        write_json(Path("results/hf_item_collection_preview.json" if args.preview else "results/hf_item_collection.json"),
                   dict(elapsed_seconds=time.perf_counter()-started, reports=reports,
                        status="review_pool_only_no_training_split"))
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
