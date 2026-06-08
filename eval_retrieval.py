"""
CyberTA – Retrieval Evaluation
Metrics 1-4:
  1. Hit Rate @ K  (did correct source appear in top-K?)
  2. Recall @ K    (fraction of relevant chunks retrieved)
  3. MRR           (Mean Reciprocal Rank)
  4. NDCG @ K      (Normalized Discounted Cumulative Gain)
  + Cosine Similarity Distribution (bonus visual)

Strategy: generate test queries by sampling real chunks from the VDB,
masking the source, then checking if retrieval recovers the correct source.
No human labels needed — fully automatic.

Usage:
    python eval_retrieval.py
    python eval_retrieval.py --n_queries 200 --k 10
"""

import os
import argparse
import random
import math
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CyberTA Retrieval Evaluation")
parser.add_argument("--n_queries",  type=int, default=100,
                    help="Number of test queries to generate (default: 100)")
parser.add_argument("--k",          type=int, default=5,
                    help="Top-K for retrieval metrics (default: 5)")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--out_dir",    default="./eval_results",
                    help="Output directory for plots and JSON report")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.out_dir, exist_ok=True)

print("Loading pipeline…")
from rag_pipeline import get_collection, get_embed_model, DEFAULT_EMBED
print("OK\n")

# ── 1. Load all vectors from VDB ───────────────────────────────────────────────

print("Fetching all chunks from ChromaDB…")
col   = get_collection()
total = col.count()
print(f"  Total chunks: {total}")

embeddings_all, docs_all, sources_all, ids_all = [], [], [], []
batch_size = 1000
offset = 0
while offset < total:
    res = col.get(
        include=["embeddings", "documents", "metadatas"],
        limit=batch_size,
        offset=offset,
    )
    embeddings_all.extend(res["embeddings"])
    docs_all.extend(res["documents"])
    sources_all.extend([m["source"] for m in res["metadatas"]])
    ids_all.extend(res["ids"])
    offset += batch_size

embeddings_all = np.array(embeddings_all, dtype=np.float32)
unique_sources  = sorted(set(sources_all))
n_sources       = len(unique_sources)
print(f"  Unique sources: {n_sources}")

# ── 2. Build test queries ──────────────────────────────────────────────────────
# Sample n_queries chunks; each chunk text becomes the query,
# its source is the ground-truth relevant document.

print(f"\nBuilding {args.n_queries} test queries…")

# Sample one chunk per source first, then random fill
per_source = defaultdict(list)
for i, src in enumerate(sources_all):
    per_source[src].append(i)

sampled_indices = []
# Ensure coverage across sources
for src in unique_sources:
    idxs = per_source[src]
    sampled_indices.append(random.choice(idxs))

# Fill up to n_queries with random samples
all_idx = list(range(len(docs_all)))
random.shuffle(all_idx)
for idx in all_idx:
    if len(sampled_indices) >= args.n_queries:
        break
    if idx not in sampled_indices:
        sampled_indices.append(idx)

sampled_indices = sampled_indices[:args.n_queries]
print(f"  Sampled {len(sampled_indices)} queries "
      f"from {len(set(sources_all[i] for i in sampled_indices))} sources")

# ── 3. Run retrieval & compute metrics ────────────────────────────────────────

model = get_embed_model(DEFAULT_EMBED)

print(f"\nRunning retrieval (K={args.k}) for each query…")

K        = args.k
hit_at_k = []
recall_at_k = []
reciprocal_ranks = []
ndcg_at_k = []
all_top1_scores = []   # cosine similarity of top-1 result
all_scores_flat = []   # all retrieved similarity scores

for qi, idx in enumerate(sampled_indices):
    query_text   = docs_all[idx]
    true_source  = sources_all[idx]
    query_id     = ids_all[idx]

    # Embed query
    q_emb = model.encode([query_text], show_progress_bar=False).tolist()

    # Query ChromaDB — fetch K+1 to exclude the exact chunk itself
    res = col.query(
        query_embeddings=q_emb,
        n_results=min(K + 1, total),
        include=["metadatas", "distances", "documents"],
    )

    ret_sources   = [m["source"] for m in res["metadatas"][0]]
    ret_distances = res["distances"][0]   # cosine distance (lower = more similar)
    ret_ids       = res["ids"][0] if "ids" in res else []

    # Convert distance → similarity  (ChromaDB cosine distance = 1 - cosine_sim)
    ret_sims = [1.0 - d for d in ret_distances]

    # Remove exact self-match if present
    filtered = [
        (src, sim) for src, sim in zip(ret_sources, ret_sims)
        if not (src == true_source and sim > 0.999)
    ][:K]

    if not filtered:
        filtered = list(zip(ret_sources[:K], ret_sims[:K]))

    ret_src_k  = [f[0] for f in filtered]
    ret_sim_k  = [f[1] for f in filtered]

    # Relevance: 1 if same source, 0 otherwise
    relevance = [1 if s == true_source else 0 for s in ret_src_k]

    # ── Hit Rate@K ──
    hit_at_k.append(1 if any(r == 1 for r in relevance) else 0)

    # ── Recall@K ──
    # Number of relevant docs in corpus for this source
    n_relevant_total = len(per_source[true_source]) - 1  # minus the query itself
    n_relevant_total = max(n_relevant_total, 1)
    n_retrieved_relevant = sum(relevance)
    recall_at_k.append(min(n_retrieved_relevant / n_relevant_total, 1.0))

    # ── MRR ──
    rr = 0.0
    for rank, r in enumerate(relevance, start=1):
        if r == 1:
            rr = 1.0 / rank
            break
    reciprocal_ranks.append(rr)

    # ── NDCG@K ──
    dcg  = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))
    # Ideal DCG: all relevant at top
    ideal_rels = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    ndcg_at_k.append(dcg / idcg if idcg > 0 else 0.0)

    # ── Similarity scores ──
    if ret_sim_k:
        all_top1_scores.append(ret_sim_k[0])
    all_scores_flat.extend(ret_sim_k)

    if (qi + 1) % 20 == 0:
        print(f"  {qi+1}/{len(sampled_indices)} queries processed…")

# ── 4. Aggregate results ──────────────────────────────────────────────────────

hit_rate = np.mean(hit_at_k)
recall   = np.mean(recall_at_k)
mrr      = np.mean(reciprocal_ranks)
ndcg     = np.mean(ndcg_at_k)
avg_top1_sim = np.mean(all_top1_scores)
avg_sim      = np.mean(all_scores_flat)

print(f"\n{'='*55}")
print(f"  RETRIEVAL EVALUATION RESULTS  (K={K})")
print(f"{'='*55}")
print(f"  Queries evaluated    : {len(sampled_indices)}")
print(f"  Unique sources       : {n_sources}")
print(f"  Hit Rate @ {K}        : {hit_rate:.4f}  ({hit_rate*100:.1f}%)")
print(f"  Recall @ {K}          : {recall:.4f}  ({recall*100:.1f}%)")
print(f"  MRR                  : {mrr:.4f}")
print(f"  NDCG @ {K}            : {ndcg:.4f}")
print(f"  Avg Top-1 Similarity : {avg_top1_sim:.4f}")
print(f"  Avg All Sim (K={K})   : {avg_sim:.4f}")
print(f"{'='*55}")

# Interpretation
def grade(v, thresholds):
    if v >= thresholds[0]: return "Excellent"
    if v >= thresholds[1]: return "Good"
    if v >= thresholds[2]: return "Fair"
    return "Poor"

print(f"\n  Interpretation:")
print(f"    Hit Rate : {grade(hit_rate, [0.9, 0.7, 0.5])}")
print(f"    Recall   : {grade(recall,   [0.8, 0.6, 0.4])}")
print(f"    MRR      : {grade(mrr,      [0.8, 0.6, 0.4])}")
print(f"    NDCG     : {grade(ndcg,     [0.8, 0.6, 0.4])}")

# ── 5. Save JSON report ───────────────────────────────────────────────────────

report = {
    "k": K,
    "n_queries": len(sampled_indices),
    "n_sources": n_sources,
    "hit_rate_at_k":   round(hit_rate, 4),
    "recall_at_k":     round(recall,   4),
    "mrr":             round(mrr,      4),
    "ndcg_at_k":       round(ndcg,     4),
    "avg_top1_sim":    round(avg_top1_sim, 4),
    "avg_sim_at_k":    round(avg_sim,  4),
    "per_query": {
        "hit_at_k":     hit_at_k,
        "recall_at_k":  [round(v,4) for v in recall_at_k],
        "rr":           [round(v,4) for v in reciprocal_ranks],
        "ndcg":         [round(v,4) for v in ndcg_at_k],
        "top1_sim":     [round(v,4) for v in all_top1_scores],
    }
}

report_path = os.path.join(args.out_dir, "retrieval_metrics.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved → {report_path}")

# ── 6. Visualizations ─────────────────────────────────────────────────────────

print("\nGenerating visualizations…")

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK_BG    = "rgba(15,15,25,1)"
GRID_COLOR = "#2a2a3e"
FONT_COLOR = "#e0e0f0"
ACCENT     = ["#4cc9f0","#7b2fbe","#f72585","#4361ee","#06d6a0"]

layout_defaults = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=13),
)

# ── Plot 1: Summary Bar Chart ──────────────────────────────────────────────────
fig1 = go.Figure()
metrics = [f"Hit Rate@{K}", f"Recall@{K}", "MRR", f"NDCG@{K}"]
values  = [hit_rate, recall, mrr, ndcg]
colors  = ACCENT[:4]

fig1.add_trace(go.Bar(
    x=metrics, y=values,
    marker=dict(
        color=colors,
        line=dict(width=0),
        opacity=0.9,
    ),
    text=[f"{v:.3f}" for v in values],
    textposition="outside",
    textfont=dict(size=15, color=FONT_COLOR),
    width=0.5,
))

# Threshold reference lines
for thresh, label, color in [(0.9,"Excellent","#06d6a0"),
                              (0.7,"Good","#f9c74f"),
                              (0.5,"Fair","#f72585")]:
    fig1.add_hline(y=thresh, line_dash="dot", line_color=color,
                   line_width=1.5,
                   annotation_text=label, annotation_position="right",
                   annotation_font=dict(color=color, size=11))

fig1.update_layout(
    **layout_defaults,
    title=dict(text="Retrieval Metrics Summary", font=dict(size=20), x=0.5),
    yaxis=dict(title="Score", range=[0, 1.12], gridcolor=GRID_COLOR,
               tickformat=".2f"),
    xaxis=dict(gridcolor=GRID_COLOR),
    showlegend=False,
    height=480,
)
p1 = os.path.join(args.out_dir, "01_metrics_summary.html")
fig1.write_html(p1, include_plotlyjs="cdn")
print(f"  Saved: {p1}")

# ── Plot 2: Per-Query Distribution (box + violin) ──────────────────────────────
fig2 = make_subplots(rows=2, cols=2,
                     subplot_titles=[f"Hit Rate@{K}", f"Recall@{K}",
                                     "MRR (Reciprocal Rank)", f"NDCG@{K}"],
                     horizontal_spacing=0.12, vertical_spacing=0.18)

data_map = [
    (hit_at_k,           1, 1, ACCENT[0]),
    (recall_at_k,        1, 2, ACCENT[1]),
    (reciprocal_ranks,   2, 1, ACCENT[2]),
    (ndcg_at_k,          2, 2, ACCENT[3]),
]

for data, row, col, color in data_map:
    fig2.add_trace(go.Violin(
        y=data, box_visible=True, meanline_visible=True,
        fillcolor=color, opacity=0.7,
        line_color=FONT_COLOR, line_width=1,
        points="outliers",
        marker=dict(color=color, size=3),
    ), row=row, col=col)

fig2.update_layout(
    **layout_defaults,
    title=dict(text="Per-Query Score Distributions", font=dict(size=20), x=0.5),
    showlegend=False,
    height=600,
)
for i in range(1, 3):
    for j in range(1, 3):
        fig2.update_yaxes(range=[-0.05, 1.1], gridcolor=GRID_COLOR, row=i, col=j)
        fig2.update_xaxes(gridcolor=GRID_COLOR, row=i, col=j)

p2 = os.path.join(args.out_dir, "02_score_distributions.html")
fig2.write_html(p2, include_plotlyjs="cdn")
print(f"  Saved: {p2}")

# ── Plot 3: Cosine Similarity Distribution ─────────────────────────────────────
fig3 = make_subplots(rows=1, cols=2,
                     subplot_titles=["Top-1 Similarity Distribution",
                                     f"All Retrieved Similarities (top {K})"],
                     horizontal_spacing=0.10)

for trace_data, col, color in [(all_top1_scores, 1, ACCENT[0]),
                                (all_scores_flat,  2, ACCENT[4])]:
    arr = np.array(trace_data)
    fig3.add_trace(go.Histogram(
        x=arr, nbinsx=40,
        marker_color=color, opacity=0.8,
        marker_line=dict(width=0.5, color=DARK_BG),
    ), row=1, col=col)
    # Mean line
    fig3.add_vline(x=arr.mean(), line_dash="dash", line_color="#f9c74f",
                   line_width=2,
                   annotation_text=f"mean={arr.mean():.3f}",
                   annotation_font=dict(color="#f9c74f"),
                   row=1, col=col)

fig3.update_layout(
    **layout_defaults,
    title=dict(text="Cosine Similarity Distributions", font=dict(size=20), x=0.5),
    showlegend=False,
    height=420,
)
fig3.update_xaxes(title="Cosine Similarity", gridcolor=GRID_COLOR)
fig3.update_yaxes(title="Count", gridcolor=GRID_COLOR)

p3 = os.path.join(args.out_dir, "03_similarity_distribution.html")
fig3.write_html(p3, include_plotlyjs="cdn")
print(f"  Saved: {p3}")

# ── Plot 4: Hit Rate vs Recall scatter per query ───────────────────────────────
fig4 = go.Figure()
jitter = np.random.uniform(-0.02, 0.02, len(hit_at_k))
fig4.add_trace(go.Scatter(
    x=np.array(recall_at_k) + jitter,
    y=np.array(hit_at_k)    + jitter * 0.5,
    mode="markers",
    marker=dict(
        color=ndcg_at_k, colorscale="Viridis",
        size=7, opacity=0.7,
        colorbar=dict(title="NDCG", thickness=14, len=0.7),
        line=dict(width=0),
    ),
    hovertemplate="Recall: %{x:.3f}<br>Hit: %{y:.3f}<br>NDCG: %{marker.color:.3f}<extra></extra>",
))

fig4.update_layout(
    **layout_defaults,
    title=dict(text=f"Hit Rate vs Recall@{K} (colored by NDCG)", font=dict(size=20), x=0.5),
    xaxis=dict(title=f"Recall@{K}", gridcolor=GRID_COLOR, range=[-0.05, 1.05]),
    yaxis=dict(title=f"Hit Rate@{K}", gridcolor=GRID_COLOR, range=[-0.1, 1.2],
               tickvals=[0, 1], ticktext=["Miss (0)", "Hit (1)"]),
    height=480,
)
p4 = os.path.join(args.out_dir, "04_hit_vs_recall.html")
fig4.write_html(p4, include_plotlyjs="cdn")
print(f"  Saved: {p4}")

# ── Plot 5: MRR breakdown (rank position histogram) ───────────────────────────
ranks_found = []
for idx in sampled_indices:
    true_source = sources_all[idx]
    query_text  = docs_all[idx]
    q_emb = model.encode([query_text], show_progress_bar=False).tolist()
    res = col.query(query_embeddings=q_emb, n_results=min(K+1, total),
                    include=["metadatas","distances"])
    ret_sources = [m["source"] for m in res["metadatas"][0]]
    ret_sims    = [1-d for d in res["distances"][0]]
    filtered = [(s,sim) for s,sim in zip(ret_sources,ret_sims)
                if not (s==true_source and sim>0.999)][:K]
    found_rank = None
    for rank, (s, _) in enumerate(filtered, 1):
        if s == true_source:
            found_rank = rank
            break
    ranks_found.append(found_rank)

rank_counts = defaultdict(int)
not_found   = 0
for r in ranks_found:
    if r is None:
        not_found += 1
    else:
        rank_counts[r] += 1

rank_labels = [f"Rank {r}" for r in range(1, K+1)] + ["Not Found"]
rank_vals   = [rank_counts[r] for r in range(1, K+1)] + [not_found]
rank_colors = [ACCENT[0]]*K + ["#f72585"]

fig5 = go.Figure(go.Bar(
    x=rank_labels, y=rank_vals,
    marker=dict(color=rank_colors, opacity=0.85, line=dict(width=0)),
    text=rank_vals, textposition="outside",
    textfont=dict(size=13, color=FONT_COLOR),
    width=0.6,
))
fig5.update_layout(
    **layout_defaults,
    title=dict(text="First Relevant Result — Rank Position Histogram", font=dict(size=20), x=0.5),
    xaxis=dict(title="Rank of First Correct Retrieval", gridcolor=GRID_COLOR),
    yaxis=dict(title="Number of Queries", gridcolor=GRID_COLOR),
    height=460,
)
p5 = os.path.join(args.out_dir, "05_rank_histogram.html")
fig5.write_html(p5, include_plotlyjs="cdn")
print(f"  Saved: {p5}")

# ── Plot 6: Radar chart summary ────────────────────────────────────────────────
categories = [f"Hit Rate@{K}", f"Recall@{K}", "MRR", f"NDCG@{K}", "Top-1 Sim"]
vals_radar  = [hit_rate, recall, mrr, ndcg, avg_top1_sim]
vals_radar += [vals_radar[0]]  # close polygon
cats_radar  = categories + [categories[0]]

fig6 = go.Figure(go.Scatterpolar(
    r=vals_radar, theta=cats_radar,
    fill="toself",
    fillcolor=f"rgba(76,201,240,0.25)",
    line=dict(color=ACCENT[0], width=2.5),
    marker=dict(size=8, color=ACCENT[0]),
    name="CyberTA VDB",
))
# "Good" reference
good_vals = [0.7]*len(categories) + [0.7]
fig6.add_trace(go.Scatterpolar(
    r=good_vals, theta=cats_radar,
    fill="toself",
    fillcolor="rgba(249,199,79,0.08)",
    line=dict(color="#f9c74f", width=1.5, dash="dot"),
    name="Good threshold",
))

fig6.update_layout(
    **layout_defaults,
    title=dict(text="Retrieval Quality Radar", font=dict(size=20), x=0.5),
    polar=dict(
        bgcolor=DARK_BG,
        radialaxis=dict(range=[0,1], tickformat=".1f",
                        gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        angularaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    ),
    legend=dict(x=0.85, y=1.1),
    height=500,
)
p6 = os.path.join(args.out_dir, "06_radar_summary.html")
fig6.write_html(p6, include_plotlyjs="cdn")
print(f"  Saved: {p6}")

print(f"\n{'='*55}")
print(f"  All 6 plots saved to {args.out_dir}/")
print(f"  Metrics report: {report_path}")
print(f"{'='*55}")
print(f"\nTo download plots to your laptop:")
print(f"  scp -r brian.rikshpun.is@hamming-sub1.uc.nps.edu:~/CYBERTA/eval_results/ ~/Desktop/CyberTA_eval/")
