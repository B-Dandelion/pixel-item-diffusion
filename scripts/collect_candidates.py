"""Bounded public asset collection for review, never final dataset approval."""

import argparse
import hashlib
import html
import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def fetch(url, limit=50 * 1024 * 1024):
    request = urllib.request.Request(url, headers={"User-Agent": "PixelItemDiffusion-DatasetReview/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        if int(response.headers.get("Content-Length", 0)) > limit:
            raise ValueError(f"Download exceeds {limit} bytes")
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Response exceeds byte limit")
        return body, response.url


def save_original(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"Refusing to overwrite different original: {path}")
    else:
        path.write_bytes(body)


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        if sum(m.file_size for m in members) > 300 * 1024 * 1024:
            raise ValueError("Expanded archive exceeds 300 MiB")
        for member in members:
            if member.is_dir():
                continue
            name = PurePosixPath(member.filename.replace("\\", "/"))
            target = (destination / str(name)).resolve()
            if name.is_absolute() or ".." in name.parts or not target.is_relative_to(destination):
                raise ValueError("Unsafe archive path")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Archive symlinks are unsupported")
            save_original(target, stream.read(member))


def discover(candidate):
    record = dict(candidate)
    try:
        body, final = fetch(candidate["source_url"], 8 * 1024 * 1024)
        destination = Path("data/research/pages") / f"{candidate['source_id']}.html"
        save_original(destination, body)
        content = body.decode("utf-8", errors="replace")
        links = [urllib.parse.urljoin(final, html.unescape(link))
                 for link in re.findall(r'href=[\"\']([^\"\']+)', content)]
        if "opengameart.org" in final:
            downloads = [u for u in links if "/sites/default/files/" in u
                         and urllib.parse.urlparse(u).path.lower().endswith((".zip", ".png", ".7z"))
                         and "/styles/" not in u]
        else:
            downloads = [u for u in links if "/media/pages/assets/" in u or u.lower().endswith(".zip")]
        record.update(status="page_fetched", final_url=final, download_urls=list(dict.fromkeys(downloads)))
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return record


def eligible_urls(candidate, include_deferred=False):
    if candidate.get("review_status", "active_review") != "active_review" and not include_deferred:
        return []
    deferred = set(candidate.get("deferred_downloads", []))
    return [url for url in candidate.get("downloads", []) if include_deferred or url not in deferred]


def select_candidates(candidates, ids=None, include_deferred=False):
    return [c for c in candidates if c.get("kind") in {"oga", "kenney"}
            and (not ids or c["source_id"] in ids)
            and (include_deferred or c.get("review_status", "active_review") == "active_review")]


def download(candidate, include_deferred=False):
    records = []
    if candidate.get("review_status", "active_review") != "active_review" and not include_deferred:
        return records
    urls = eligible_urls(candidate, include_deferred)
    opener = None
    if candidate.get("kind") == "kenney":
        try:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            headers = {"User-Agent": "Mozilla/5.0", "Referer": candidate["source_url"]}
            page = opener.open(urllib.request.Request(candidate["source_url"], headers=headers), timeout=30).read().decode()
            download_id = re.search(r"data-download-id=['\"]([^'\"]+)", page)[1]
            endpoint = "https://kenney.nl/download-link/" + download_id
            response = opener.open(urllib.request.Request(endpoint, headers=headers), timeout=30).read()
            urls = [json.loads(response)["url"]]
        except Exception as exc:
            return [dict(source_id=candidate["source_id"], status="error", error=str(exc))]
    for url in urls:
        record = dict(source_id=candidate["source_id"], source_url=candidate["source_url"], download_url=url)
        try:
            if opener is None:
                body, final = fetch(url)
            else:
                with opener.open(urllib.request.Request(url, headers=headers), timeout=40) as response:
                    body, final = response.read(50 * 1024 * 1024 + 1), response.url
                if len(body) > 50 * 1024 * 1024:
                    raise ValueError("Download exceeds 50 MiB")
            name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
            if opener is not None:
                name = candidate["source_id"] + ".zip"
            path = Path("data/raw") / candidate["source_id"] / "downloads" / name
            save_original(path, body)
            if zipfile.is_zipfile(path):
                extract_zip(path, path.parent.parent / "extracted" / path.stem)
            record.update(status="downloaded", path=path.as_posix(), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest(), final_url=final)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["discover", "download"])
    parser.add_argument("--config", default="configs/data_candidates.json")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--include-deferred", action="store_true", help="Explicitly include archived/deferred candidates and subpacks")
    parser.add_argument("--dry-run", action="store_true", help="Show selected sources/URLs without network or file changes")
    args = parser.parse_args()
    candidates = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected = select_candidates(candidates, args.ids, args.include_deferred)
    if args.ids:
        skipped = sorted(set(args.ids) - {c["source_id"] for c in selected})
        if skipped:
            print(json.dumps(dict(skipped_ids=skipped, reason="Deferred, unknown, or requires a dedicated HF/DCSS collector")))
    if args.dry_run:
        print(json.dumps([dict(source_id=c["source_id"], source_url=c["source_url"],
                               download_urls=eligible_urls(c, args.include_deferred)) for c in selected], indent=2))
        return
    started = time.perf_counter()
    worker = discover if args.mode == "discover" else lambda c: download(c, args.include_deferred)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(worker, selected))
    if args.mode == "download":
        records = [r for group in records for r in group]
    output = Path("results") / f"candidate_{args.mode}_{time.time_ns()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started, records=records), indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
