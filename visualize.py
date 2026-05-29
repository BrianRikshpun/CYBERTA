"""
CyberTA – Vector Visualisation
Reduces high-dim embeddings to 2D & 3D with PCA (fast, no extra deps)
and optionally UMAP (richer topology). Renders interactive Plotly figures
and saves them as HTML + PNG.
"""

from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

SAVE_DIR = "./plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── colour palette (distinct enough for many sources) ────────────────────────
PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Vivid


def _color_map(sources: List[str]) -> Tuple[List[str], dict]:
    unique = sorted(set(sources))
    cmap   = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique)}
    colors = [cmap[s] for s in sources]
    return colors, cmap


def _reduce(embeddings: np.ndarray, n_components: int, method: str = "pca") -> np.ndarray:
    if method == "umap":
        try:
            import umap                          # optional dependency
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(embeddings)
        except ImportError:
            pass   # fall back to PCA silently
    # PCA
    n = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n, random_state=42)
    reduced = pca.fit_transform(embeddings)
    # If we got fewer components than requested (tiny dataset), pad
    if reduced.shape[1] < n_components:
        pad = np.zeros((reduced.shape[0], n_components - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])
    return reduced


def build_2d_plot(
    embeddings: List,
    documents:  List[str],
    sources:    List[str],
    method:     str = "pca",
) -> go.Figure:
    emb = np.array(embeddings, dtype=np.float32)
    xy  = _reduce(emb, 2, method)

    colors, cmap = _color_map(sources)

    # Truncate hover text
    hover = [doc[:120].replace("\n", " ") + "…" for doc in documents]

    traces = []
    for src, color in cmap.items():
        mask = [i for i, s in enumerate(sources) if s == src]
        traces.append(go.Scatter(
            x    = xy[mask, 0],
            y    = xy[mask, 1],
            mode = "markers",
            name = src,
            marker=dict(color=color, size=7, opacity=0.8,
                        line=dict(width=0.5, color="white")),
            text        = [hover[i] for i in mask],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    axis_label = method.upper()
    fig = go.Figure(traces)
    fig.update_layout(
        title      = f"Vector Clusters – 2D ({axis_label})",
        xaxis_title= f"{axis_label} Dim 1",
        yaxis_title= f"{axis_label} Dim 2",
        legend_title="Source",
        template   = "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(20,20,35,1)",
        font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def build_3d_plot(
    embeddings: List,
    documents:  List[str],
    sources:    List[str],
    method:     str = "pca",
) -> go.Figure:
    emb = np.array(embeddings, dtype=np.float32)
    xyz = _reduce(emb, 3, method)

    colors, cmap = _color_map(sources)
    hover = [doc[:120].replace("\n", " ") + "…" for doc in documents]

    traces = []
    for src, color in cmap.items():
        mask = [i for i, s in enumerate(sources) if s == src]
        traces.append(go.Scatter3d(
            x    = xyz[mask, 0],
            y    = xyz[mask, 1],
            z    = xyz[mask, 2],
            mode = "markers",
            name = src,
            marker=dict(color=color, size=4, opacity=0.85,
                        line=dict(width=0)),
            text        = [hover[i] for i in mask],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    axis_label = method.upper()
    scene = dict(
        xaxis=dict(title=f"{axis_label} 1", backgroundcolor="rgba(20,20,35,1)",
                   gridcolor="#333", showbackground=True),
        yaxis=dict(title=f"{axis_label} 2", backgroundcolor="rgba(20,20,35,1)",
                   gridcolor="#333", showbackground=True),
        zaxis=dict(title=f"{axis_label} 3", backgroundcolor="rgba(20,20,35,1)",
                   gridcolor="#333", showbackground=True),
        bgcolor="rgba(20,20,35,1)",
    )

    fig = go.Figure(traces)
    fig.update_layout(
        title   = f"Vector Clusters – 3D ({axis_label})",
        scene   = scene,
        legend_title="Source",
        template= "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def save_plots(fig2d: go.Figure, fig3d: go.Figure) -> Tuple[str, str]:
    """Save both figures as HTML files. Returns (path_2d, path_3d)."""
    p2d = os.path.join(SAVE_DIR, "clusters_2d.html")
    p3d = os.path.join(SAVE_DIR, "clusters_3d.html")
    fig2d.write_html(p2d, include_plotlyjs="cdn")
    fig3d.write_html(p3d, include_plotlyjs="cdn")
    return p2d, p3d
