"""
CyberTA – Vector Visualisation
Reduces high-dim embeddings to 2D & 3D with PCA or UMAP.
Supports coloring by: source file, advisor, or second_reader.
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

SAVE_DIR = "./plots"
os.makedirs(SAVE_DIR, exist_ok=True)

PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Vivid + px.colors.qualitative.Pastel


def _color_map(labels: List[str]):
    unique = sorted(set(labels))
    cmap   = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique)}
    colors = [cmap[s] for s in labels]
    return colors, cmap


def _reduce(embeddings: np.ndarray, n_components: int, method: str = "pca") -> np.ndarray:
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=n_components, random_state=42).fit_transform(embeddings)
        except ImportError:
            pass
    n   = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n, random_state=42)
    reduced = pca.fit_transform(embeddings)
    if reduced.shape[1] < n_components:
        pad     = np.zeros((reduced.shape[0], n_components - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])
    return reduced


def _resolve_labels(
    sources: List[str],
    color_by: str,
    meta_map: Optional[Dict[str, Dict]] = None,
) -> List[str]:
    """
    Return a label list for each chunk based on color_by mode.
    color_by: 'source' | 'advisor' | 'second_reader'
    meta_map: {filename -> {advisor, second_reader, ...}}
    """
    if color_by == "source" or not meta_map:
        return sources

    labels = []
    for src in sources:
        m     = meta_map.get(src, {})
        raw   = m.get(color_by, "").strip()
        # Take first name if multiple (pipe-separated)
        label = raw.split("|")[0].strip() if raw else "Unknown"
        labels.append(label if label else "Unknown")
    return labels


def build_2d_plot(
    embeddings: List,
    documents:  List[str],
    sources:    List[str],
    method:     str = "pca",
    color_by:   str = "source",
    meta_map:   Optional[Dict[str, Dict]] = None,
    title_suffix: str = "",
) -> go.Figure:
    emb    = np.array(embeddings, dtype=np.float32)
    xy     = _reduce(emb, 2, method)
    labels = _resolve_labels(sources, color_by, meta_map)
    _, cmap = _color_map(labels)
    hover  = [doc[:120].replace("\n", " ") + "…" for doc in documents]

    traces = []
    for lbl, color in cmap.items():
        mask = [i for i, l in enumerate(labels) if l == lbl]
        traces.append(go.Scatter(
            x    = xy[mask, 0],
            y    = xy[mask, 1],
            mode = "markers",
            name = lbl,
            marker=dict(color=color, size=7, opacity=0.8,
                        line=dict(width=0.5, color="white")),
            text        = [hover[i] for i in mask],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    axis = method.upper()
    legend_title = {"source": "Document", "advisor": "Advisor",
                    "second_reader": "Second Reader"}.get(color_by, color_by)
    fig = go.Figure(traces)
    fig.update_layout(
        title       = f"Vector Clusters – 2D ({axis}){title_suffix}",
        xaxis_title = f"{axis} Dim 1",
        yaxis_title = f"{axis} Dim 2",
        legend_title= legend_title,
        template    = "plotly_dark",
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
    color_by:   str = "source",
    meta_map:   Optional[Dict[str, Dict]] = None,
    title_suffix: str = "",
) -> go.Figure:
    emb    = np.array(embeddings, dtype=np.float32)
    xyz    = _reduce(emb, 3, method)
    labels = _resolve_labels(sources, color_by, meta_map)
    _, cmap = _color_map(labels)
    hover  = [doc[:120].replace("\n", " ") + "…" for doc in documents]

    traces = []
    for lbl, color in cmap.items():
        mask = [i for i, l in enumerate(labels) if l == lbl]
        traces.append(go.Scatter3d(
            x    = xyz[mask, 0],
            y    = xyz[mask, 1],
            z    = xyz[mask, 2],
            mode = "markers",
            name = lbl,
            marker=dict(color=color, size=4, opacity=0.85),
            text        = [hover[i] for i in mask],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    axis  = method.upper()
    legend_title = {"source": "Document", "advisor": "Advisor",
                    "second_reader": "Second Reader"}.get(color_by, color_by)
    scene = dict(
        xaxis=dict(title=f"{axis} 1", backgroundcolor="rgba(20,20,35,1)",
                   gridcolor="#333", showbackground=True),
        yaxis=dict(title=f"{axis} 2", backgroundcolor="rgba(20,20,35,1)",
                   gridcolor="#333", showbackground=True),
        zaxis=dict(title=f"{axis} 3", backgroundcolor="rgba(20,20,35,1)",
                   gridcolor="#333", showbackground=True),
        bgcolor="rgba(20,20,35,1)",
    )
    fig = go.Figure(traces)
    fig.update_layout(
        title       = f"Vector Clusters – 3D ({axis}){title_suffix}",
        scene       = scene,
        legend_title= legend_title,
        template    = "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def save_plots(fig2d: go.Figure, fig3d: go.Figure,
               suffix: str = "") -> tuple[str, str]:
    p2d = os.path.join(SAVE_DIR, f"clusters_2d{suffix}.html")
    p3d = os.path.join(SAVE_DIR, f"clusters_3d{suffix}.html")
    fig2d.write_html(p2d, include_plotlyjs="cdn")
    fig3d.write_html(p3d, include_plotlyjs="cdn")
    return p2d, p3d


def save_png(fig2d: go.Figure, fig3d: go.Figure,
             suffix: str = "") -> tuple[str, str]:
    p2d = os.path.join(SAVE_DIR, f"clusters_2d{suffix}.png")
    p3d = os.path.join(SAVE_DIR, f"clusters_3d{suffix}.png")
    fig2d.write_image(p2d, width=1400, height=650)
    fig3d.write_image(p3d, width=1400, height=700)
    return p2d, p3d
