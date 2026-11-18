"""Review inventory, transparent-pixel deduplication and source contact sheets; no training split."""

import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_images import infer_category


SHEETS = [
    ("oga_icons32", "downloads/icons_32x32_7.png", 32, 0),
    ("oga_items32", "downloads/itemiconpack32.png", 32, 0),
    ("oga_27items", "downloads/items_31.png", 32, 0),
    ("kenney_rpg", "extracted/kenney_rpg/Spritesheet/roguelikeSheet_transparent.png", 16, 1),
    ("kenney_1bit", "extracted/kenney_1bit/Tilesheet/monochrome-transparent_packed.png", 16, 0),
]


def split_review_sheets():
    rows = []
    for sid, relative, size, gap in SHEETS:
        source = Path("data/raw") / sid / relative
        if not source.exists():
            continue
        with Image.open(source) as original:
            image = original.convert("RGBA")
            columns, lines = (image.width + gap) // (size + gap), (image.height + gap) // (size + gap)
            for y in range(lines):
                for x in range(columns):
                    # The published 27-icon sheet has three empty framed cells at the end.
                    if sid == "oga_27items" and y * columns + x >= 27:
                        continue
                    left, top = x * (size + gap), y * (size + gap)
                    tile = image.crop((left, top, left + size, top + size))
                    if tile.getchannel("A").getbbox() is None:
                        continue
                    output = Path("data/candidates") / sid / f"cell_{y:03d}_{x:03d}.png"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    tile.save(output)
                    rows.append(dict(source=sid, path=output.as_posix(), parent_path=source.as_posix(),
                                     box=json.dumps([left, top, left+size, top+size]), transformation="grid_crop_only"))
    return rows


def category(path, sid, metadata):
    if sid == "496_RPG_icons":
        return infer_category(path.name)
    if sid.startswith("hf_"):
        return "label:" + str(metadata.get("category", metadata.get("group", metadata.get("subject", "unknown"))))
    if sid == "dcss_cc0":
        part = path.as_posix().split("/item/")[-1].split("/")[0]
        return {"armour":"armor_shield", "amulet":"accessory", "ring":"accessory", "staff":"weapon",
                "rod":"weapon", "wand":"weapon", "book":"book_scroll", "scroll":"book_scroll",
                "gold":"currency"}.get(part, part)
    if sid == "oga_food_cc0":
        return "food"
    if sid in {"oga_gem_cc0", "oga_ingot_cc0", "oga_mineral_cc0"}:
        return "material"
    if sid == "oga_tool_cc0":
        return "tool"
    name = path.stem.lower()
    for label, tokens in [
        ("potion", ["potion", "bottle", "medicine", "antidote"]),
        ("accessory", ["ring", "amulet", "necklace", "bracer"]),
        ("armor_shield", ["armor", "armour", "shield", "helmet", "helm", "coif", "hood", "robe", "shirt", "breastplate", "chainmail", "boots", "gloves", "pants", "cloak", "leather_", "cloth_", "platemail", "cap", "shoes"]),
        ("weapon", ["sword", "axe", "mace", "bow", "arrow", "bolt", "spear", "staff", "wand", "dagger", "knife", "club", "flail", "javelin", "polearm", "scythe", "blowgun", "shuriken", "throwing", "chakram", "sling", "halberd"]),
        ("food", ["apple", "bread", "steak", "egg", "mushroom", "berry"]),
        ("material", ["ore", "ingot", "gem", "crystal", "diamond", "bone", "hide", "silk", "thread", "board", "log", "acorn", "fiber", "herb", "feather"]),
        ("book_scroll", ["book", "scroll", "grimoire", "rotulus", "parchment"]),
        ("tool", ["pick", "shovel", "rod", "torch", "hammer", "key", "compass"]),
        ("skill", ["whoosh", "explosion", "buff", "fire", "magic", "element"]),
    ]:
        if any(name.startswith(t) or ("_" + t) in name for t in tokens):
            return label
    return "unknown_review"


def inspect(path, sid, metadata=None, allow_synthetic_review=False):
    metadata = metadata or {}
    row = dict(source=sid, path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               status="ok", width=0, height=0, mode="", has_alpha=0, empty=0,
               canonical_rgba_sha256="", cropped_rgba_sha256="", category=category(path, sid, metadata),
               role="single_image_candidate", quality_flags="", final_inclusion="pending")
    try:
        with Image.open(path) as image:
            image.load()
            a = np.array(image.convert("RGBA"))
            a[a[:, :, 3] == 0, :3] = 0
            alpha = a[:, :, 3]
            bbox = Image.fromarray(alpha).getbbox()
            row.update(width=image.width, height=image.height, mode=image.mode, actual_format=image.format,
                       extension_format_mismatch=int(path.suffix.lower() == ".png" and image.format != "PNG"),
                       has_alpha=int("A" in image.getbands() or "transparency" in image.info),
                       empty=int(bbox is None), partial_alpha_pixels=int(((alpha > 0) & (alpha < 255)).sum()),
                       uniform=int(np.all(a == a[0, 0])),
                       visible_fraction=float((alpha > 0).mean()))
            row["canonical_rgba_sha256"] = hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
            if bbox:
                l,t,r,b = bbox
                cropped = a[t:b,l:r]
                row["cropped_rgba_sha256"] = hashlib.sha256(str(cropped.shape).encode() + cropped.tobytes()).hexdigest()
            if max(image.size) > 64:
                row["role"] = "sheet_preview_or_large_image"
            if sid == "oga_idylwild_arsenal" and "border" in path.stem:
                row["role"] = "border_variant"
            if sid == "oga_basic_remix" and image.size == (64,64):
                row["role"] = "resolution_variant"
            if sid == "oga_rpg_base":
                row["role"] = "photo_based_reference"
            if sid == "oga_idylwild_aerial" and (image.size == (96,96) or path.stem.startswith("whoosh")):
                row["role"] = "animation_effect"
            if sid == "dcss_cc0" and path.stem.startswith("i-"):
                row["role"] = "identification_symbol"
            if sid.startswith("kenney"):
                row["role"] = "mixed_tiles_require_selection"
            if row["empty"]:
                row["role"] = "empty"
            if sid == "oga_items32" and row["uniform"]:
                row["role"] = "uniform_background_cell"
            if sid.startswith("hf_"):
                row["role"] = "synthetic_sample"
                row["final_inclusion"] = ("pending_synthetic_review_not_training_approved" if allow_synthetic_review
                                          else "research_only_excluded_by_user_policy")
            flags = list(metadata.get("flags") or [])
            if row["extension_format_mismatch"]:
                flags.append("extension_format_mismatch_preserve_received_bytes")
            if sid.startswith("hf_") and image.format != "PNG":
                flags.append("viewer_preview_not_verified_lossless_original")
            if sid == "hf_pixelart172k":
                magenta = (a[:,:,0] > 90) & (a[:,:,2] > 90) & (a[:,:,1] < 50) & (alpha > 0)
                if magenta.any():
                    flags.append("magenta_pixel_heuristic_not_defect_proof")
            if sid.startswith("oga_") and any(s in path.as_posix().lower() for s in ["-ocal", "-glitch", "-wikimedia"]):
                flags.append("vector_photo_or_mixed_provenance_review")
            if sid == "oga_food_cc0" and "bluecarrot16" in path.as_posix():
                flags.append("original_LPC_per_item_license_mapping_pending")
            row["quality_flags"] = ";".join(flags)
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}:{exc}"
    return row


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(sid, rows, output_dir):
    small = [r for r in rows if r["status"] == "ok" and max(r["width"],r["height"]) <=64 and not r["empty"]]
    if not small:
        return
    chosen = random.Random(42).sample(small, min(48, len(small)))
    sheet = Image.new("RGB", (8*128, math.ceil(len(chosen)/8)*92+28), (205,205,205))
    draw = ImageDraw.Draw(sheet)
    draw.text((8,6),sid + " - review sample; native pixels enlarged with nearest-neighbor",fill="black")
    for i,row in enumerate(chosen):
        x,y=(i%8)*128,(i//8)*92+28
        with Image.open(row["path"]) as im:
            rgba=im.convert("RGBA")
            scale=max(1,64//max(rgba.size))
            rgba=rgba.resize((rgba.width*scale,rgba.height*scale),Image.Resampling.NEAREST)
            sheet.paste(rgba,(x+(128-rgba.width)//2,y),rgba)
        draw.text((x+2,y+66),Path(row["path"]).stem[:20],fill="black")
        draw.text((x+2,y+78),row["category"][:22],fill="black")
    sheet.save(output_dir / f"{sid}.png")


def main():
    started=time.perf_counter()
    out=Path("results/candidate_analysis");out.mkdir(parents=True,exist_ok=True)
    cells=split_review_sheets()
    write_csv(out/"sheet_cells.csv",cells)
    rows=[]
    source_policy = json.loads(Path("configs/data_candidates.json").read_text(encoding="utf-8"))
    synthetic_review = {c["source_id"] for c in source_policy
                        if c["kind"] == "hf" and c.get("review_status") == "active_review"}
    for root in sorted(Path("data/raw").iterdir()):
        if not root.is_dir():continue
        metadata={}
        manifest=root/"sample_manifest.json"
        if manifest.exists():
            metadata={r["path"]:r for r in json.loads(manifest.read_text(encoding="utf-8")) if r.get("status")=="downloaded"}
        for path in sorted(root.rglob("*.png")):
            rows.append(inspect(path,root.name,metadata.get(path.as_posix()),root.name in synthetic_review))
    raw_count=len(rows)
    for cell in cells:
        row=inspect(Path(cell["path"]),cell["source"])
        row.update(parent_path=cell["parent_path"],box=cell["box"])
        rows.append(row)
    groups=defaultdict(list)
    for row in rows:groups[row["source"]].append(row)
    summaries=[]
    for sid, group in groups.items():
        raw=[r for r in group if r["path"].startswith("data/raw/")]
        small=[r for r in group if r["role"] in {"single_image_candidate","synthetic_sample","mixed_tiles_require_selection"} and max(r["width"],r["height"])<=64 and not r["empty"]]
        summaries.append(dict(source=sid,raw_pngs=len(raw),derived_grid_cells=len(group)-len(raw),
                              small_review_images=len(small),unique_visible_small=len({r['canonical_rgba_sha256'] for r in small}),
                              raw_exact_duplicates=len(raw)-len({r['sha256'] for r in raw}),
                              read_errors=sum(r['status']!='ok' for r in group),empty=sum(r['empty'] for r in group),
                              actual_formats=dict(Counter(r.get('actual_format','unknown') for r in raw)),
                              sizes=dict(Counter(f"{r['width']}x{r['height']}" for r in raw)),
                              modes=dict(Counter(r['mode'] for r in raw)),
                              roles=dict(Counter(r['role'] for r in group)),
                              categories_inferred=dict(Counter(r['category'] for r in small))))
        contact_sheet(sid,group,out)
    duplicates=[]
    for field in ["sha256","canonical_rgba_sha256","cropped_rgba_sha256"]:
        matching=defaultdict(list)
        for row in rows:
            if row.get(field):matching[row[field]].append(row)
        for key,group in matching.items():
            if len(group)>1:
                duplicates.append(dict(kind=field,hash=key,count=len(group),sources=";".join(sorted({r['source'] for r in group})),paths=json.dumps([r['path'] for r in group])))
    write_csv(out/"inventory.csv",rows);write_csv(out/"duplicate_groups.csv",duplicates)
    # Proposal only: retain separate source families and original paths, without a train/val split.
    core_sources = {"496_RPG_icons", "dcss_cc0", "oga_idylwild_arsenal", "oga_idylwild_aerial",
                    "oga_idylwild_armory", "oga_idylwild_inventory", "oga_idylwild_arcanum"}
    first = {}
    proposal = []
    for row in rows:
        if row["source"] not in core_sources or row["role"] != "single_image_candidate" or row["status"] != "ok":
            continue
        key = row["canonical_rgba_sha256"]
        proposal.append(dict(source=row["source"], path=row["path"], category_inferred=row["category"],
                             duplicate_of=first.get(key, ""), canonical_rgba_sha256=key,
                             decision="proposal_only_labels_and_variants_unreviewed"))
        first.setdefault(key, row["path"])
    write_csv(out/"core_proposal.csv", proposal)
    other_images = []
    for path in sorted(Path("data/raw").rglob("*")):
        if path.suffix.lower() not in {".gif", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        row = dict(path=path.as_posix(), status="ok", role="non_png_reference_not_in_candidate_count")
        try:
            with Image.open(path) as im:
                im.load()
                row.update(format=im.format, width=im.width, height=im.height, frames=getattr(im, "n_frames", 1))
        except Exception as exc:
            row["status"] = f"error:{exc}"
        other_images.append(row)
    write_csv(out/"other_images.csv", other_images)
    summary=dict(raw_pngs=raw_count,derived_grid_cells=len(cells),sources=summaries,
                 raw_actual_formats=dict(Counter(r.get("actual_format", "unknown") for r in rows[:raw_count])),
                 extension_format_mismatches=sum(r.get("extension_format_mismatch", 0) for r in rows[:raw_count]),
                 core_proposal_unique=len(first),
                 core_categories_inferred=dict(Counter(r["category_inferred"] for r in proposal if not r["duplicate_of"])),
                 other_raster_files=len(other_images), other_raster_errors=sum(r["status"]!="ok" for r in other_images),
                 elapsed_seconds=round(time.perf_counter()-started,3),
                 note="raw_pngs counts .png paths, not guaranteed PNG bytes; inspect raw_actual_formats. Review counts are not training counts. Filename labels are heuristic. No split created.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":main()
