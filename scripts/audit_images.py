"""Audit raw PNG dimensions and exact duplicate hashes without changing source files."""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="results/raw_audit.csv")
    args = parser.parse_args()

    raw_dir = Path(args.input)
    output_file = Path(args.output)
    if not raw_dir.exists():
        raise SystemExit(f"Raw data folder missing: {raw_dir.resolve()}")

    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    source_counts = Counter()
    size_counts = Counter()
    hash_counts = Counter()
    errors = []

    with output_file.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "source", "width", "height", "mode", "has_alpha", "sha256", "status"])
        for file in files:
            relative = file.relative_to(raw_dir)
            source = relative.parts[0] if len(relative.parts) > 1 else "unclassified"
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            try:
                with Image.open(file) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    alpha = "A" in image.getbands() or "transparency" in image.info
                writer.writerow([str(relative), source, width, height, mode, int(alpha), digest, "ok"])
                source_counts[source] += 1
                size_counts[f"{width}x{height}"] += 1
                hash_counts[digest] += 1
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                errors.append(str(relative))
                writer.writerow([str(relative), source, "", "", "", "", digest, f"error:{type(exc).__name__}"])

    print(f"PNG files found: {len(files)}")
    print(f"Successfully inspected: {sum(source_counts.values())}")
    print("Counts by source:", dict(source_counts))
    print("Top 10 sizes:", size_counts.most_common(10))
    print("Exact duplicate files beyond first copy:", sum(n - 1 for n in hash_counts.values() if n > 1))
    print("Unreadable files:", len(errors))
    print(f"Wrote report: {output_file.resolve()}")
    print("NOTE: checksum matching does not detect visually near-identical images.")


if __name__ == "__main__":
    main()
