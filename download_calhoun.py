"""
CyberTA – Calhoun NPS Thesis Downloader
Uses the DSpace 7 REST API to fetch IS department theses,
download PDFs, and extract metadata (advisor, second reader, title, date, author).

Usage:
    python download_calhoun.py --n 20
    python download_calhoun.py --n 50 --out ./data --search "cyber network"

Outputs:
    ./data/<title>.pdf          — thesis PDF
    ./data/thesis_metadata.csv  — metadata for all downloaded theses
"""

import os
import re
import sys
import csv
import time
import argparse
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = "https://calhoun.nps.edu"
REST_API = f"{BASE}/server/api"
IS_SCOPE = "8170278c-29e1-4af7-8cd6-c960e36516f9"
SLEEP    = 1.0

HEADERS = {
    "User-Agent": "CyberTA-Research/1.0 (NPS thesis downloader; educational use)",
    "Accept"    : "application/json",
}

# Dublin Core metadata field mappings
META_FIELDS = {
    "title"        : ["dc.title"],
    "author"       : ["dc.contributor.author"],
    "advisor"      : ["dc.contributor.advisor",
                      "dc.contributor.firstadvisor",
                      "thesis.degree.grantor"],
    "second_reader": ["dc.contributor.secondreader",
                      "dc.contributor.reader",
                      "dc.contributor.committeemember"],
    "date"         : ["dc.date.issued", "dc.date.created"],
    "abstract"     : ["dc.description.abstract"],
    "department"   : ["dc.contributor.department"],
    "degree"       : ["dc.type.degree", "thesis.degree.name"],
    "keywords"     : ["dc.subject"],
}


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r'\s+', " ", name).strip()
    return name[:max_len]


def extract_metadata(item: dict) -> dict:
    """
    Pull structured metadata from a DSpace item dict.
    Returns a flat dict with title, author, advisor, second_reader, etc.
    """
    # DSpace 7 stores metadata as list of {value, language, ...} dicts
    raw_meta = item.get("metadata", {})

    result = {}
    for field_name, dc_keys in META_FIELDS.items():
        values = []
        for dc_key in dc_keys:
            entries = raw_meta.get(dc_key, [])
            for entry in entries:
                v = entry.get("value", "").strip()
                if v and v not in values:
                    values.append(v)
        result[field_name] = " | ".join(values) if values else ""

    return result


def get_full_item(item_uuid: str) -> dict:
    """Fetch full item including metadata from DSpace REST API."""
    url = f"{REST_API}/core/items/{item_uuid}"
    try:
        resp = requests.get(url, headers=HEADERS,
                            params={"embed": "thumbnail,owningCollection"},
                            timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    ⚠️  Could not fetch item metadata: {e}")
        return {}


def get_items(scope: str, n: int, search: str = "") -> list:
    """Fetch up to N item stubs from the IS collection."""
    items = []
    page  = 0
    size  = min(n, 20)

    print(f"🔍 Fetching item list from Calhoun IS dept (up to {n} theses)…")

    while len(items) < n:
        if search:
            url = f"{REST_API}/discover/search/objects"
            params = {
                "scope"  : scope,
                "query"  : search,
                "dsoType": "item",
                "page"   : page,
                "size"   : size,
            }
        else:
            url = f"{REST_API}/discover/browses/dateissued/items"
            params = {
                "scope": scope,
                "page" : page,
                "size" : size,
                "sort" : "dc.date.issued,DESC",
            }

        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ⚠️  API error on page {page}: {e}")
            break

        if search:
            raw   = (data.get("_embedded", {})
                         .get("searchResult", {})
                         .get("_embedded", {})
                         .get("objects", []))
            batch = [obj.get("_embedded", {}).get("indexableObject", {})
                     for obj in raw]
        else:
            batch = data.get("_embedded", {}).get("items", [])

        if not batch:
            break

        for item in batch:
            if len(items) >= n:
                break
            uuid = item.get("uuid") or item.get("id")
            if uuid:
                items.append({"uuid": uuid,
                              "name": item.get("name", "untitled")})

        page_info   = data.get("page", {})
        total_pages = page_info.get("totalPages", 1)
        if page >= total_pages - 1:
            break
        page += 1
        time.sleep(SLEEP)

    print(f"  Found {len(items)} items.\n")
    return items


def get_pdf_url(item_uuid: str):
    """Return (pdf_download_url, original_filename) or (None, None)."""
    url = f"{REST_API}/core/items/{item_uuid}/bundles"
    try:
        resp    = requests.get(url, headers=HEADERS, timeout=20)
        bundles = resp.json().get("_embedded", {}).get("bundles", [])
    except Exception:
        return None, None

    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue
        bs_url = f"{REST_API}/core/bundles/{bundle['uuid']}/bitstreams"
        try:
            bs_resp    = requests.get(bs_url, headers=HEADERS, timeout=20)
            bitstreams = bs_resp.json().get("_embedded", {}).get("bitstreams", [])
        except Exception:
            continue
        for bs in bitstreams:
            fname = bs.get("name", "")
            if fname.lower().endswith(".pdf"):
                link = bs.get("_links", {}).get("content", {}).get("href", "")
                if link:
                    return link, fname
    return None, None


def download_pdf(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=180)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    ❌ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def print_metadata(meta: dict):
    """Pretty-print thesis metadata to terminal."""
    fields = [
        ("Author",        meta.get("author",       "—")),
        ("Advisor",       meta.get("advisor",      "—")),
        ("Second Reader", meta.get("second_reader","—")),
        ("Date",          meta.get("date",         "—")),
        ("Department",    meta.get("department",   "—")),
        ("Degree",        meta.get("degree",       "—")),
        ("Keywords",      meta.get("keywords",     "—")[:80]),
    ]
    for label, value in fields:
        if value and value != "—":
            print(f"    {label:<14}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Download IS theses from Calhoun NPS with metadata"
    )
    parser.add_argument("--n",      type=int, default=20,
                        help="Number of theses to download (default: 20)")
    parser.add_argument("--out",    default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--search", default="",
                        help="Keyword filter (e.g. 'cyber AI network')")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "thesis_metadata.csv"

    print(f"\n{'='*60}")
    print(f"  Calhoun NPS Thesis Downloader")
    print(f"  Target : {args.n} theses  |  Output: {out_dir.resolve()}")
    if args.search:
        print(f"  Filter : '{args.search}'")
    print(f"{'='*60}\n")

    items      = get_items(IS_SCOPE, args.n, args.search)
    downloaded = skipped = failed = 0
    all_meta   = []

    for i, stub in enumerate(items, 1):
        uuid = stub["uuid"]
        print(f"[{i}/{len(items)}] Fetching metadata…")

        # Get full item with metadata
        full_item = get_full_item(uuid)
        time.sleep(SLEEP)

        meta  = extract_metadata(full_item) if full_item else {}
        title = meta.get("title") or stub["name"] or "untitled"
        meta["uuid"] = uuid

        print(f"  📄 {title[:70]}")
        print_metadata(meta)

        # Determine output path
        safe = sanitize_filename(title)
        dest = out_dir / f"{safe}.pdf"

        if dest.exists() and dest.stat().st_size > 1000:
            print(f"    ⏭️  Already exists — skipping download.\n")
            meta["pdf_file"] = dest.name
            meta["status"]   = "skipped"
            all_meta.append(meta)
            skipped += 1
            continue

        # Find and download PDF
        pdf_url, orig_fname = get_pdf_url(uuid)
        time.sleep(SLEEP)

        if not pdf_url:
            print(f"    ⚠️  No PDF found.\n")
            meta["pdf_file"] = ""
            meta["status"]   = "no_pdf"
            all_meta.append(meta)
            failed += 1
            continue

        print(f"    ⬇️  Downloading {orig_fname}…")
        ok = download_pdf(pdf_url, dest)
        time.sleep(SLEEP)

        if ok:
            size_kb = dest.stat().st_size // 1024
            print(f"    ✅ Saved ({size_kb} KB)\n")
            meta["pdf_file"] = dest.name
            meta["status"]   = "downloaded"
            downloaded += 1
        else:
            meta["pdf_file"] = ""
            meta["status"]   = "failed"
            failed += 1

        all_meta.append(meta)

    # Write CSV
    if all_meta:
        fieldnames = ["title","author","advisor","second_reader","date",
                      "department","degree","keywords","abstract",
                      "pdf_file","status","uuid"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_meta)
        print(f"📋 Metadata saved → {csv_path}")

    print(f"\n{'='*60}")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")
    print(f"{'='*60}")
    print(f"\nNext: python build_vdb.py --reset")


if __name__ == "__main__":
    main()
