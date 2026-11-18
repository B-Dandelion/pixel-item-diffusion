"""Prepare RGB PNGs, a reproducible split and comparison sheets; keep raw data intact."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_images import inspect_images


def normalize_image(image, size=32, background=(24, 24, 24)):
    """Fit the entire image using nearest-neighbor, then center-pad; never crop."""
    rgba = image.convert("RGBA")
    scale = size / max(rgba.size)
    target = (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale)))
    resized = rgba.resize(target, Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (size, size), (*background, 255))
    canvas.alpha_composite(resized, ((size - target[0]) // 2, (size - target[1]) // 2))
    return canvas.convert("RGB")


def write_sheets(rows, raw_dir, output_dir, sheet_dir, seed):
    sheet_dir.mkdir(parents=True, exist_ok=True)
    order = list(rows)
    random.Random(seed).shuffle(order)
    paths = []
    for page, offset in enumerate(range(0, len(order), 80), 1):
        subset = order[offset:offset + 80]
        sheet = Image.new("RGB", (8 * 160, math.ceil(len(subset) / 8) * 102 + 22), (222, 222, 222))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 4), "Each pair: original RGBA on gray | processed RGB (nearest-neighbor preview)", fill="black")
        for i, row in enumerate(subset):
            x, y = (i % 8) * 160, (i // 8) * 102 + 24
            with Image.open(raw_dir / row["relative_path"]) as original:
                rgba = original.convert("RGBA")
                rgba.thumbnail((34, 34), Image.Resampling.NEAREST)
                rgba = rgba.resize((rgba.width * 2, rgba.height * 2), Image.Resampling.NEAREST)
                sheet.paste(rgba, (x + (72 - rgba.width) // 2, y), rgba)
            with Image.open(output_dir / row["training_path"]) as processed:
                sheet.paste(processed.resize((64, 64), Image.Resampling.NEAREST), (x + 80, y))
            draw.text((x + 2, y + 70), Path(row["relative_path"]).stem[:25], fill="black")
            draw.text((x + 2, y + 84), row["split"], fill="black")
        path = sheet_dir / f"before_after_{page:02d}.png"
        sheet.save(path)
        paths.append(path.as_posix())
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--sources", default="docs/sources.csv")
    parser.add_argument("--results", default="results")
    parser.add_argument("--exclude-skills", action="store_true", help="Optional item-only comparison dataset")
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw_dir, output_dir = Path(args.input), Path(args.output)
    if output_dir.resolve() == raw_dir.resolve() or output_dir.resolve().is_relative_to(raw_dir.resolve()):
        raise SystemExit("Output must be outside the raw data directory.")
    if (output_dir / "manifest.csv").exists():
        raise SystemExit("Dataset already exists. Use a new --output directory to preserve it.")
    fraction = config["validation_fraction"]
    if not 0 < fraction < 1:
        raise SystemExit("validation_fraction must be between 0 and 1.")
    with Path(args.sources).open(encoding="utf-8-sig", newline="") as stream:
        sources = {r["source_id"]: r for r in csv.DictReader(stream)}
    rows, audit = inspect_images(raw_dir)
    aliases = {"496_RPG_icons": "oga_rpg496"}
    include_skills = config.get("include_skills", True) and not args.exclude_skills
    selected, seen_original, seen_pixels = [], {}, {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source_id = aliases.get(row["source"], row["source"])
        source = sources.get(source_id, {})
        row.update(source_id=source_id, source_page_url=source.get("source_url", ""),
                   license_claim=source.get("license_claim", "unknown"),
                   license_review_status=source.get("license_review_status", "pending"),
                   original_path=(raw_dir / row["relative_path"]).as_posix(),
                   training_path="", processed_sha256="", split="excluded", exclusion_reason="")
        if row["status"] != "ok":
            row["exclusion_reason"] = "unreadable"
        elif not source:
            row["exclusion_reason"] = "unregistered_source"
        elif row["empty"] or row["uniform"]:
            row["exclusion_reason"] = "empty_or_uniform"
        elif row["category"] == "skill" and not include_skills:
            row["exclusion_reason"] = "skill_filtered"
        elif row["sha256"] in seen_original:
            row["exclusion_reason"] = "exact_duplicate"
        if row["exclusion_reason"]:
            continue
        with Image.open(raw_dir / row["relative_path"]) as original:
            processed = normalize_image(original, config["image_size"], tuple(config["background_rgb"]))
        pixel_hash = hashlib.sha256(processed.tobytes()).hexdigest()
        row["processed_sha256"] = pixel_hash
        if pixel_hash in seen_pixels:
            row["exclusion_reason"] = "identical_processed_pixels"
            continue
        seen_original[row["sha256"]] = row["relative_path"]
        seen_pixels[pixel_hash] = row["relative_path"]
        training_path = Path("images") / f"{row['sha256']}.png"
        (output_dir / training_path).parent.mkdir(parents=True, exist_ok=True)
        processed.save(output_dir / training_path)
        row["training_path"] = training_path.as_posix()
        selected.append(row)
    if len(selected) < 2:
        raise SystemExit("Need at least two unique usable images to split.")
    indices = list(range(len(selected)))
    random.Random(config["seed"]).shuffle(indices)
    n_val = max(1, min(len(selected) - 1, round(len(selected) * fraction)))
    val_indices = set(indices[:n_val])
    for i, row in enumerate(selected):
        row["split"] = "val" if i in val_indices else "train"
    with (output_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(seed=config["seed"], include_skills=include_skills, raw_audit=audit,
                  selected=len(selected), splits=dict(Counter(r["split"] for r in selected)),
                  categories=dict(Counter(r["category"] for r in selected)),
                  excluded=dict(Counter(r["exclusion_reason"] for r in rows if r["exclusion_reason"])),
                  transform="Full-frame aspect fit, nearest-neighbor, center pad, RGB alpha composite",
                  image_size=config["image_size"], background_rgb=config["background_rgb"],
                  manifest_sha256=hashlib.sha256((output_dir / "manifest.csv").read_bytes()).hexdigest())
    report["contact_sheets"] = write_sheets(selected, raw_dir, output_dir, Path(args.results) / "contact_sheets", config["seed"])
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (output_dir / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.results).mkdir(parents=True, exist_ok=True)
    (Path(args.results) / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
