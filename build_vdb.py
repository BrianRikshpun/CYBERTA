"""
CyberTA – VDB Builder
1. Index all docs in ./data/
2. Generate 6 plots: 2D+3D × {source, advisor, second_reader}
3. Save PNGs + HTMLs to ./plots/

Usage:
    python build_vdb.py
    python build_vdb.py --n 20 --reset --viz_method PCA
"""

import os
import sys
import csv
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="CyberTA VDB Builder")
parser.add_argument("--data_dir",   default="./data")
parser.add_argument("--embed",      default="MiniLM-L6-v2 (fast, 384d)")
parser.add_argument("--viz_method", default="PCA", choices=["PCA","UMAP"])
parser.add_argument("--reset",      action="store_true")
args = parser.parse_args()

print("Loading modules…")
from rag_pipeline import index_file, collection_stats, get_all_vectors, reset_collection
from visualize    import build_2d_plot, build_3d_plot, save_plots, save_png
print("OK\n")

SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}


def load_metadata_csv(data_dir: Path) -> dict:
    """
    Load thesis_metadata.csv if it exists.
    Returns {filename -> {advisor, second_reader, ...}}
    """
    csv_path = data_dir / "thesis_metadata.csv"
    if not csv_path.exists():
        return {}
    meta_map = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.get("pdf_file", "").strip()
            if fname:
                meta_map[fname] = {
                    "advisor"      : row.get("advisor",       ""),
                    "second_reader": row.get("second_reader", ""),
                    "author"       : row.get("author",        ""),
                    "date"         : row.get("date",          ""),
                }
    print(f"📋 Loaded metadata for {len(meta_map)} theses from CSV.")
    return meta_map


def main():
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: '{data_dir}' not found. Create it and add PDFs.")
        sys.exit(1)

    files = [f for f in data_dir.rglob("*") if f.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported files in '{data_dir}'.")
        sys.exit(1)

    print(f"Found {len(files)} file(s).")

    if args.reset:
        print("Clearing VDB…")
        reset_collection()

    # ── Index ─────────────────────────────────────────────────────────────────
    for i, fpath in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {fpath.name}")
        def cb(msg): print(f"   {msg}")
        index_file(str(fpath), embed_model_name=args.embed, progress_cb=cb)

    stats = collection_stats()
    print(f"\n{'='*55}")
    print(f"VDB COMPLETE — {stats['total_chunks']} chunks | {len(stats['sources'])} docs")
    print(f"{'='*55}\n")

    # ── Visualise ─────────────────────────────────────────────────────────────
    embeddings, documents, sources = get_all_vectors()
    if len(embeddings) == 0:
        print("No vectors — skipping plots.")
        return

    os.makedirs("./plots", exist_ok=True)
    meta_map = load_metadata_csv(data_dir)
    method   = args.viz_method.lower()

    color_modes = [
        ("source",        " — by Document"),
        ("advisor",       " — by Advisor"),
        ("second_reader", " — by Second Reader"),
    ]

    for color_by, label in color_modes:
        # Skip advisor/second_reader if no metadata
        if color_by != "source" and not meta_map:
            print(f"⏭️  Skipping '{color_by}' plot — no metadata CSV found.")
            continue

        print(f"🎨 Generating plots colored {label}…")
        fig2d = build_2d_plot(embeddings, documents, sources,
                              method=method, color_by=color_by,
                              meta_map=meta_map, title_suffix=label)
        fig3d = build_3d_plot(embeddings, documents, sources,
                              method=method, color_by=color_by,
                              meta_map=meta_map, title_suffix=label)

        suffix   = f"_{color_by}"
        p2d, p3d = save_plots(fig2d, fig3d, suffix=suffix)
        print(f"   HTML: {p2d}")
        print(f"   HTML: {p3d}")

        try:
            pp2d, pp3d = save_png(fig2d, fig3d, suffix=suffix)
            print(f"   PNG : {pp2d}")
            print(f"   PNG : {pp3d}")
        except Exception as e:
            print(f"   PNG skipped (pip install kaleido): {e}")

    print(f"\n✅ All done! Plots saved to ./plots/")
    print(f"   Run the app:  python app.py")


if __name__ == "__main__":
    main()
