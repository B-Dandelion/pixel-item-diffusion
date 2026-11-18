"""Inspect extracted PNGs only; never extract archives or alter raw assets."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def infer_category(filename):
    """Filename heuristics for reporting only, not class conditioning."""
    name = Path(filename).stem
    for prefix, category in [("S_", "skill"), ("W_", "weapon"), ("P_", "potion"),
                             ("I_C_", "food"), ("Ac_", "accessory"),
                             ("A_", "armor_shield"), ("C_", "armor_shield"),
                             ("E_", "armor_shield"), ("I_", "inventory_material_misc")]:
        if name.startswith(prefix):
            return category
    return "unknown"


def inspect_images(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise ValueError(f"Raw data folder missing: {raw_dir}")
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    rows, first_hash = [], {}
    for path in files:
        if path.suffix.lower() != ".png":
            continue
        relative = path.relative_to(raw_dir)
        row = dict(relative_path=relative.as_posix(),
                   source=relative.parts[0] if len(relative.parts) > 1 else "unclassified",
                   width="", height="", mode="", has_alpha="", alpha_min="", alpha_max="",
                   alpha_bbox="", partial_alpha_pixels="", empty="", uniform="",
                   category=infer_category(path.name), sha256="", duplicate_of="", status="ok")
        try:
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:
                im.load()
                if im.format != "PNG":
                    raise ValueError("File with PNG suffix is not a PNG")
                rgba = im.convert("RGBA")
                alpha = rgba.getchannel("A")
                histogram = alpha.histogram()
                row.update(width=im.width, height=im.height, mode=im.mode,
                           has_alpha=int("A" in im.getbands() or "transparency" in im.info),
                           alpha_min=alpha.getextrema()[0], alpha_max=alpha.getextrema()[1],
                           alpha_bbox=json.dumps(alpha.getbbox()),
                           partial_alpha_pixels=sum(histogram[1:255]),
                           empty=int(alpha.getbbox() is None),
                           uniform=int(all(lo == hi for lo, hi in rgba.getextrema())))
            row["duplicate_of"] = first_hash.get(row["sha256"], "")
            first_hash.setdefault(row["sha256"], relative.as_posix())
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            row["status"] = f"error:{type(exc).__name__}:{exc}"
        rows.append(row)
    valid = [r for r in rows if r["status"] == "ok"]
    summary = dict(
        png_files=len(rows), readable=len(valid), unreadable=len(rows) - len(valid),
        exact_duplicates=sum(bool(r["duplicate_of"]) for r in valid),
        unique_readable=len({r["sha256"] for r in valid}),
        empty_images=sum(r["empty"] for r in valid),
        uniform_images=sum(r["uniform"] for r in valid),
        modes=dict(Counter(r["mode"] for r in valid)),
        sizes=dict(Counter(f"{r['width']}x{r['height']}" for r in valid)),
        images_with_alpha=sum(r["has_alpha"] for r in valid),
        images_with_partial_alpha=sum(r["partial_alpha_pixels"] > 0 for r in valid),
        categories_inferred=dict(Counter(r["category"] for r in valid)),
        sources=dict(Counter(r["source"] for r in valid)),
        ignored_non_png=[p.relative_to(raw_dir).as_posix() for p in files if p.suffix.lower() != ".png"],
        notes=["Archives are not opened or counted.",
               "Categories and uniform-image flags are heuristics, not semantic verification.",
               "SHA-256 does not detect near-duplicate artwork."])
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="results/raw_audit.csv")
    args = parser.parse_args()
    started = time.perf_counter()
    rows, summary = inspect_images(args.input)
    if not rows:
        raise SystemExit("No extracted PNG files found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
