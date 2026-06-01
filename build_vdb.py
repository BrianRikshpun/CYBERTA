"""
CyberTA – VDB Builder
Run this ONCE on Hamming to:
  1. Index all documents in ./data/
  2. Save 2D and 3D cluster plots as PNG + HTML
  3. Print a summary

Usage:
    python build_vdb.py
    python build_vdb.py --data_dir ./mydata --embed "BGE-base-en-v1.5 (quality, 768d)"
"""

import os
import sys
import argparse
from pathlib import Path

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CyberTA VDB Builder")
parser.add_argument("--data_dir", default="./data",
                    help="Directory containing PDF/DOCX/PPTX files (default: ./data)")
parser.add_argument("--embed", default="MiniLM-L6-v2 (fast, 384d)",
                    help="Embedding model display name")
parser.add_argument("--viz_method", default="PCA", choices=["PCA", "UMAP"],
                    help="Dimensionality reduction method for plots")
parser.add_argument("--reset", action="store_true",
                    help="Clear existing VDB before indexing")
args = parser.parse_args()

# ── Imports (after argparse so --help works fast) ─────────────────────────────
print("Loading pipeline modules...")
from rag_pipeline import index_file, collection_stats, get_all_vectors, reset_collection
from visualize import build_2d_plot, build_3d_plot, save_plots
import plotly.io as pio
print("OK\n")

SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}

def main():
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory '{data_dir}' not found.")
        print(f"Create it and put your PDFs/PPTXs inside:\n  mkdir -p {data_dir}")
        sys.exit(1)

    # Collect files
    files = [f for f in data_dir.rglob("*") if f.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported files found in '{data_dir}'")
        print(f"Supported: {', '.join(SUPPORTED)}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) in '{data_dir}':")
    for f in files:
        print(f"  - {f.name}")
    print()

    # Optionally reset
    if args.reset:
        print("Clearing existing VDB...")
        reset_collection()
        print("Done.\n")

    # Index each file
    total_chunks = 0
    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {fpath.name}")
        def cb(msg):
            print(f"    {msg}")
        result = index_file(str(fpath), embed_model_name=args.embed, progress_cb=cb)
        total_chunks += result["chunks"]
        print()

    # Summary
    stats = collection_stats()
    print("=" * 50)
    print(f"VDB BUILD COMPLETE")
    print(f"  Total chunks indexed : {stats['total_chunks']}")
    print(f"  Documents            : {len(stats['sources'])}")
    print(f"  Embedding model      : {args.embed}")
    for src in sorted(stats['sources']):
        print(f"    · {src}")
    print("=" * 50)
    print()

    # Build visualizations
    print(f"Generating {args.viz_method} visualizations...")
    embeddings, documents, sources = get_all_vectors()

    if not embeddings:
        print("No vectors found — skipping plots.")
        return

    os.makedirs("./plots", exist_ok=True)

    fig2d = build_2d_plot(embeddings, documents, sources, method=args.viz_method.lower())
    fig3d = build_3d_plot(embeddings, documents, sources, method=args.viz_method.lower())

    # Save HTML (interactive)
    p2d_html, p3d_html = save_plots(fig2d, fig3d)
    print(f"  Saved HTML: {p2d_html}")
    print(f"  Saved HTML: {p3d_html}")

    # Save PNG (static) using kaleido
    try:
        p2d_png = "./plots/clusters_2d.png"
        p3d_png = "./plots/clusters_3d.png"
        fig2d.write_image(p2d_png, width=1200, height=600)
        fig3d.write_image(p3d_png, width=1200, height=700)
        print(f"  Saved PNG : {p2d_png}")
        print(f"  Saved PNG : {p3d_png}")
    except Exception as e:
        print(f"  PNG export skipped (install kaleido for PNG): {e}")

    print()
    print("All done! You can now run the app:")
    print("  python app.py")

if __name__ == "__main__":
    main()
