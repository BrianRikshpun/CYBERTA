"""
CyberTA – Calhoun NPS Thesis Downloader
Uses the correct browse/department endpoint to get all 743 IS theses.

Usage:
    python download_calhoun.py --n 200
    python download_calhoun.py --n 743  # all of them
"""

import os, re, sys, csv, time, argparse, requests
from pathlib import Path

BASE     = "https://calhoun.nps.edu"
REST_API = f"{BASE}/server/api"
SLEEP    = 0.8

# Correct endpoint for IS department — 743 theses
IS_BROWSE_URL = (f"{REST_API}/discover/browses/department/items"
                 f"?filterValue=Information%20Sciences%20(IS)")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CyberTA/1.0; educational use)",
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
    raw    = item.get("metadata", {})
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


def get_all_items(n):
    """Paginate through the IS department browse endpoint."""
    items = []
    seen  = set()
    page  = 0
    size  = 20

    print(f"🔍 Fetching IS (Information Sciences) theses — target: {n}…")

    while len(items) < n:
        params = {"size": size, "page": page, "sort": "dc.date.issued,DESC"}
        try:
            r = requests.get(IS_BROWSE_URL, headers=HEADERS,
                             params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ⚠️  API error page {page}: {e}")
            break

        batch = data.get("_embedded",{}).get("items",[])
        if not batch:
            print(f"  No more results at page {page}.")
            break

        added = 0
        for item in batch:
            uuid = item.get("uuid") or item.get("id")
            if uuid and uuid not in seen:
                seen.add(uuid)
                items.append({"uuid": uuid, "name": item.get("name","untitled")})
                added += 1
            if len(items) >= n:
                break

        total = data.get("page",{}).get("totalElements","?")
        pages = data.get("page",{}).get("totalPages","?")
        print(f"  Page {page+1}/{pages} — +{added} items (total: {len(items)}/{total})")

        total_pages = data.get("page",{}).get("totalPages", page+1)
        if page >= total_pages - 1 or len(batch) < size:
            print(f"  Last page reached.")
            break

        page += 1
        time.sleep(SLEEP)

    print(f"  ✅ {len(items)} unique items found.\n")
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
            r2 = requests.get(
                f"{REST_API}/core/bundles/{bundle['uuid']}/bitstreams",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=50,
                        help="Number of theses to download (default: 50, max: 743)")
    parser.add_argument("--out", default="./data",
                        help="Output directory (default: ./data)")
    args = parser.parse_args()

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "thesis_metadata.csv"

    print(f"\n{'='*60}")
    print(f"  Calhoun NPS Thesis Downloader")
    print(f"  Department : Information Sciences (IS) — 743 available")
    print(f"  Target     : {args.n} theses")
    print(f"  Output     : {out_dir.resolve()}")
    print(f"{'='*60}\n")

    stubs      = get_all_items(args.n)
    downloaded = skipped = failed = 0
    all_meta   = []

    for i, stub in enumerate(stubs, 1):
        uuid = stub["uuid"]
        print(f"[{i}/{len(stubs)}] Fetching metadata…")

        full  = get_full_item(uuid)
        time.sleep(SLEEP)

        meta  = extract_metadata(full) if full else {}
        title = meta.get("title") or stub["name"] or "untitled"
        meta["uuid"] = uuid

        print(f"  📄 {title[:70]}")
        for lbl, key in [("Author","author"),("Advisor","advisor"),
                         ("2nd Reader","second_reader"),("Date","date")]:
            v = meta.get(key,"")
            if v: print(f"    {lbl:<12}: {v[:80]}")

        safe = sanitize(title)
        dest = out_dir / f"{safe}.pdf"

        if dest.exists() and dest.stat().st_size > 1000:
            print(f"    ⏭️  Already exists.\n")
            meta.update({"pdf_file": dest.name, "status": "skipped"})
            all_meta.append(meta)
            skipped += 1
            continue

        pdf_url, orig = get_pdf_url(uuid)
        time.sleep(SLEEP)

        if not pdf_url:
            print(f"    ⚠️  No PDF.\n")
            meta.update({"pdf_file": "", "status": "no_pdf"})
            all_meta.append(meta)
            failed += 1
            continue

        print(f"    ⬇️  {orig}…")
        ok = download_pdf(pdf_url, dest)
        time.sleep(SLEEP)

        meta.update({
            "pdf_file": dest.name if ok else "",
            "status"  : "downloaded" if ok else "failed",
        })
        all_meta.append(meta)

        if ok:
            print(f"    ✅ {dest.stat().st_size//1024} KB\n")
            downloaded += 1
        else:
            failed += 1

    # Save CSV
    if all_meta:
        fields = ["title","author","advisor","second_reader","date",
                  "department","degree","keywords","abstract",
                  "pdf_file","status","uuid"]
        write_header = not csv_path.exists()
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(all_meta)
        print(f"📋 Metadata saved → {csv_path}")

    print(f"\n{'='*60}")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")
    print(f"{'='*60}")
    print(f"\nNext: scp data/ to Hamming, then python build_vdb.py --reset")

if __name__ == "__main__":
    main()
