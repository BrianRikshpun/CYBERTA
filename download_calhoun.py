"""
CyberTA – Calhoun NPS Thesis Downloader
Downloads IS department theses only, with full metadata.

Usage:
    python download_calhoun.py --n 20
    python download_calhoun.py --n 50 --search "cyber network"
"""

import os, re, sys, csv, time, argparse, requests
from pathlib import Path

BASE     = "https://calhoun.nps.edu"
REST_API = f"{BASE}/server/api"
# Collection UUID for IS theses (from the browse URL scope param)
IS_SCOPE = "8170278c-29e1-4af7-8cd6-c960e36516f9"
SLEEP    = 1.0

HEADERS = {
    "User-Agent": "CyberTA-Research/1.0 (NPS educational use)",
    "Accept"    : "application/json",
}

META_FIELDS = {
    "title"        : ["dc.title"],
    "author"       : ["dc.contributor.author"],
    "advisor"      : ["dc.contributor.advisor","dc.contributor.firstadvisor"],
    "second_reader": ["dc.contributor.secondreader","dc.contributor.reader",
                      "dc.contributor.committeemember"],
    "date"         : ["dc.date.issued"],
    "abstract"     : ["dc.description.abstract"],
    "department"   : ["dc.contributor.department"],
    "degree"       : ["dc.type.degree","thesis.degree.name"],
    "keywords"     : ["dc.subject"],
}


def sanitize(name, max_len=120):
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return re.sub(r'\s+', " ", name).strip()[:max_len]


def extract_metadata(item):
    raw = item.get("metadata", {})
    result = {}
    for field, keys in META_FIELDS.items():
        vals = []
        for k in keys:
            for e in raw.get(k, []):
                v = e.get("value","").strip()
                if v and v not in vals:
                    vals.append(v)
        result[field] = " | ".join(vals)
    return result


def get_full_item(uuid):
    try:
        r = requests.get(f"{REST_API}/core/items/{uuid}",
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ⚠️  metadata fetch failed: {e}")
        return {}


def get_items_from_collection(scope, n, search=""):
    """
    Use the collection endpoint directly — guarantees IS-only results.
    Falls back to search with scope filter if needed.
    """
    items = []
    page  = 0
    size  = min(n, 20)
    print(f"🔍 Fetching IS theses from collection scope {scope}…")

    while len(items) < n:
        if search:
            # Search within the specific collection scope
            url    = f"{REST_API}/discover/search/objects"
            params = {
                "scope"  : scope,
                "query"  : search,
                "dsoType": "item",
                "page"   : page,
                "size"   : size,
            }
        else:
            # Browse the collection directly by handle/items
            url    = f"{REST_API}/core/collections/{scope}/mappedItems"
            params = {"page": page, "size": size}

        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # Fall back to discover search with scope
            print(f"  Collection browse failed ({e}), trying scoped search…")
            url    = f"{REST_API}/discover/search/objects"
            params = {
                "scope"  : scope,
                "query"  : search or "*",
                "dsoType": "item",
                "page"   : page,
                "size"   : size,
            }
            try:
                r    = requests.get(url, headers=HEADERS, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e2:
                print(f"  ⚠️  API error: {e2}")
                break

        # Parse HAL response
        if search or "searchResult" in str(data):
            raw   = (data.get("_embedded",{})
                        .get("searchResult",{})
                        .get("_embedded",{})
                        .get("objects",[]))
            batch = [o.get("_embedded",{}).get("indexableObject",{}) for o in raw]
        else:
            batch = data.get("_embedded",{}).get("mappedItems",
                    data.get("_embedded",{}).get("items",[]))

        if not batch:
            break

        for item in batch:
            if len(items) >= n:
                break
            uuid = item.get("uuid") or item.get("id")
            if uuid:
                items.append({"uuid": uuid, "name": item.get("name","untitled")})

        total_pages = data.get("page",{}).get("totalPages", 1)
        if page >= total_pages - 1:
            break
        page += 1
        time.sleep(SLEEP)

    print(f"  Found {len(items)} items.\n")
    return items


def get_pdf_url(uuid):
    try:
        r       = requests.get(f"{REST_API}/core/items/{uuid}/bundles",
                               headers=HEADERS, timeout=20)
        bundles = r.json().get("_embedded",{}).get("bundles",[])
    except:
        return None, None
    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue
        try:
            r2 = requests.get(f"{REST_API}/core/bundles/{bundle['uuid']}/bitstreams",
                              headers=HEADERS, timeout=20)
            for bs in r2.json().get("_embedded",{}).get("bitstreams",[]):
                fname = bs.get("name","")
                if fname.lower().endswith(".pdf"):
                    link = bs.get("_links",{}).get("content",{}).get("href","")
                    if link:
                        return link, fname
        except:
            continue
    return None, None


def download_pdf(url, dest):
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=180)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    ❌ {e}")
        if dest.exists(): dest.unlink()
        return False


def is_IS_department(meta):
    """Filter: only keep Information Sciences department theses."""
    dept = meta.get("department","").lower()
    return "information science" in dept or dept == "" or "(is)" in dept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=20)
    parser.add_argument("--out",    default="./data")
    parser.add_argument("--search", default="")
    parser.add_argument("--all_depts", action="store_true",
                        help="Download from all departments (no IS filter)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "thesis_metadata.csv"

    print(f"\n{'='*60}")
    print(f"  Calhoun NPS Thesis Downloader")
    print(f"  Target   : {args.n} IS theses")
    print(f"  Output   : {out_dir.resolve()}")
    if args.search: print(f"  Filter   : '{args.search}'")
    print(f"{'='*60}\n")

    # Fetch more than N to account for non-IS filtering
    fetch_n   = args.n * 3 if not args.all_depts else args.n
    stubs     = get_items_from_collection(IS_SCOPE, fetch_n, args.search)
    downloaded = skipped = failed = 0
    all_meta  = []

    for i, stub in enumerate(stubs, 1):
        if downloaded >= args.n:
            break

        uuid = stub["uuid"]
        print(f"[{i}/{len(stubs)}] Fetching metadata…")

        full = get_full_item(uuid)
        time.sleep(SLEEP)

        meta  = extract_metadata(full) if full else {}
        title = meta.get("title") or stub["name"] or "untitled"
        meta["uuid"] = uuid

        # Department filter
        if not args.all_depts and not is_IS_department(meta):
            dept = meta.get("department","unknown")
            print(f"  ⏭️  Skipping — not IS dept ({dept})\n")
            continue

        print(f"  📄 {title[:70]}")
        for lbl, key in [("Author","author"),("Advisor","advisor"),
                         ("Second Reader","second_reader"),("Date","date"),
                         ("Department","department")]:
            v = meta.get(key,"")
            if v: print(f"    {lbl:<14}: {v[:80]}")

        safe = sanitize(title)
        dest = out_dir / f"{safe}.pdf"

        if dest.exists() and dest.stat().st_size > 1000:
            print(f"    ⏭️  Already exists.\n")
            meta["pdf_file"] = dest.name
            meta["status"]   = "skipped"
            all_meta.append(meta)
            skipped += 1
            downloaded += 1
            continue

        pdf_url, orig = get_pdf_url(uuid)
        time.sleep(SLEEP)

        if not pdf_url:
            print(f"    ⚠️  No PDF.\n")
            meta["pdf_file"] = ""
            meta["status"]   = "no_pdf"
            all_meta.append(meta)
            failed += 1
            continue

        print(f"    ⬇️  {orig}…")
        ok = download_pdf(pdf_url, dest)
        time.sleep(SLEEP)

        meta["pdf_file"] = dest.name if ok else ""
        meta["status"]   = "downloaded" if ok else "failed"
        all_meta.append(meta)
        if ok:
            print(f"    ✅ {dest.stat().st_size//1024} KB\n")
            downloaded += 1
        else:
            failed += 1

    # Save CSV
    if all_meta:
        fields = ["title","author","advisor","second_reader","date",
                  "department","degree","keywords","abstract","pdf_file","status","uuid"]
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_meta)
        print(f"📋 Metadata → {csv_path}")

    print(f"\n{'='*60}")
    print(f"  Downloaded: {downloaded}  Skipped: {skipped}  Failed: {failed}")
    print(f"{'='*60}")
    print(f"\nNext: python build_vdb.py --reset")

if __name__ == "__main__":
    main()
