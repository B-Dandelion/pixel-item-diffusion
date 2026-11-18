"""Audit the selected HF pool and make class contact sheets; no training or split."""

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_candidates import write_csv


def make_sheet(rows, title, output):
    chosen = random.Random(42).sample(rows, min(48, len(rows)))
    sheet = Image.new("RGB", (1024, 28 + ((len(chosen)+7)//8)*112), (205, 205, 205))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title + " | labels unverified | nearest-neighbor preview", fill="black")
    for i, row in enumerate(chosen):
        x, y = i % 8 * 128, i // 8 * 112 + 28
        with Image.open(row["path"]) as image:
            enlarged = image.resize((64, 64), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (x+32, y), enlarged)
        draw.text((x+2, y+66), str(row["row_idx"]) + " " + str(row.get("backend", "native")), fill="black")
        text = str(row.get("text", row.get("description", "")))
        draw.text((x+2, y+78), text[:20], fill="black")
        draw.text((x+2, y+90), text[20:40], fill="black")
        if row.get("flags"):
            draw.text((x+114, y+66), "!", fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return [row["path"] for row in chosen]


def main():
    started = time.perf_counter()
    output = Path("results/item_review")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = []
    sheet_indices = {}
    canonical_seen = {}
    for sid in ["hf_pixelart172k", "hf_krea16"]:
        manifest = Path("data/raw") / sid / "item_manifest.json"
        group = json.loads(manifest.read_text(encoding="utf-8"))
        for row in group:
            body = Path(row["path"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != row["sha256"]:
                raise ValueError(f"Original changed: {row['path']}")
            with Image.open(row["path"]) as image:
                pixels = np.asarray(image)
                visible = pixels[:, :, 3] > 0
                row["partial_alpha_pixels"] = int(((pixels[:, :, 3] > 0) & (pixels[:, :, 3] < 255)).sum())
                row["exact_dark_magenta_pixels"] = int((np.all(pixels[:, :, :3] == [150, 0, 140], axis=-1) & visible).sum())
            key = row["canonical_rgba_sha256"]
            row["duplicate_of"] = canonical_seen.get(key, "")
            canonical_seen.setdefault(key, row["path"])
        classes = defaultdict(list)
        for row in group:
            classes[row["item_class"]].append(row)
        class_summary = {}
        for label, members in sorted(classes.items()):
            sheet = output / sid / f"{label}.png"
            sheet_indices[sheet.as_posix()] = make_sheet(members, f"{sid}/{label}", sheet)
            class_summary[label] = dict(rows=len(members), unique_visible=len({r['canonical_rgba_sha256'] for r in members}),
                                        source_flagged=sum(bool(r.get('flags')) for r in members))
        reports.append(dict(source_id=sid, rows=len(group), classes=class_summary,
                            png_bytes=sum(r["bytes"] for r in group),
                            unique_visible=len({r['canonical_rgba_sha256'] for r in group}),
                            empty=sum(r["empty"] for r in group),
                            source_flagged=sum(bool(r.get("flags")) for r in group),
                            exact_dark_magenta_images=sum(r["exact_dark_magenta_pixels"] > 0 for r in group),
                            partial_alpha_images=sum(r["partial_alpha_pixels"] > 0 for r in group),
                            backends=dict(Counter(r.get("backend", "native") for r in group))))
        rows.extend(group)
    # Read/transform benchmark only: generated arrays are discarded; raw files are unchanged.
    probe_start = time.perf_counter()
    checksum = 0.0
    for row in rows:
        with Image.open(row["path"]) as image:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (24, 24, 24, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB").resize((32, 32), Image.Resampling.NEAREST)
            array = np.asarray(rgb, dtype=np.float32) / 127.5 - 1
            checksum += float(array.mean())
    probe_seconds = time.perf_counter() - probe_start
    write_csv(output/"inventory.csv", rows)
    (output/"contact_sheet_paths.json").write_text(json.dumps(sheet_indices, indent=2), encoding="utf-8")
    nonempty = [r for r in rows if not r["empty"]]
    items = [r for r in nonempty if r["item_class"] != "skill_effect_optional"]
    clean_flags = [r for r in items if not r.get("flags")]
    summary = dict(sources=reports, selected_rows=len(rows),
                   unique_visible=len(canonical_seen),
                   nonempty_item_unique_visible=len({r['canonical_rgba_sha256'] for r in items}),
                   nonempty_item_unique_without_source_flags=len({r['canonical_rgba_sha256'] for r in clean_flags}),
                   total_png_bytes=sum(r["bytes"] for r in rows),
                   read_and_transform_probe=dict(images=len(rows), seconds=round(probe_seconds, 3),
                                                 checksum=checksum, cache="warm_after_audit", output_saved=False,
                                                 operation="decode/temporary_RGB_composite/nearest32/normalize; not final preprocessing"),
                   elapsed_seconds=round(time.perf_counter()-started, 3),
                   status="review_only_no_train_validation_split_no_model_training")
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
