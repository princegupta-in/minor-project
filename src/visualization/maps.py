"""Static LULC map visualizations for Deoghar Forest Cover Analysis.

Includes forest-specific colourmap, before-after comparisons,
forest loss/gain hotspot maps, uncertainty overlays,
and NDVI trend plots.

Adapted from ShivekRanjan/bengaluru-lulc-forecast- with a forest-focused
6-class palette and additional forest-specific helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ── Forest-focused 6-class palette ────────────────────────────────────────────
LULC_COLORS = [
    (0.20, 0.41, 0.81),   # 0 water          — deep blue
    (0.03, 0.28, 0.07),   # 1 dense_forest   — dark forest green
    (0.38, 0.70, 0.25),   # 2 open_forest    — light green
    (0.95, 0.85, 0.30),   # 3 agriculture    — golden yellow
    (0.75, 0.14, 0.14),   # 4 built_up       — brick red
    (0.80, 0.70, 0.55),   # 5 barren         — sandy beige
]
LULC_NAMES = ["water", "dense_forest", "open_forest", "agriculture", "built_up", "barren"]
LULC_CMAP  = ListedColormap(LULC_COLORS)

# Forest = classes 1 + 2
FOREST_CLASSES = [1, 2]


def plot_lulc_map(
    label: np.ndarray,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(label, cmap=LULC_CMAP, vmin=0, vmax=len(LULC_COLORS) - 1, interpolation="nearest")
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    if show_legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=n) for c, n in zip(LULC_COLORS, LULC_NAMES)]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    return ax


def plot_before_after(
    before: np.ndarray,
    after: np.ndarray,
    titles: Tuple[str, str] = ("2015 (Historical)", "2035 (Forecast)"),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plot_lulc_map(before, titles[0], ax=axes[0], show_legend=False)
    plot_lulc_map(after,  titles[1], ax=axes[1], show_legend=True)
    fig.suptitle("Deoghar Forest Cover: Before vs. Forecast", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_forest_loss_gain(
    label_t0: np.ndarray,
    label_t1: np.ndarray,
    title: str = "Forest Change Map",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Highlight forest loss (red) and forest gain (green) between two frames."""
    forest_t0 = np.isin(label_t0, FOREST_CLASSES)
    forest_t1 = np.isin(label_t1, FOREST_CLASSES)

    change_map = np.zeros(label_t0.shape, dtype=np.int8)
    change_map[forest_t0 & ~forest_t1] = -1   # loss
    change_map[~forest_t0 & forest_t1] =  1   # gain

    cmap = ListedColormap(["#c0392b", "#ecf0f1", "#27ae60"])  # red, grey, green
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(change_map, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#c0392b", label="Forest loss"),
        Patch(facecolor="#ecf0f1", label="No change"),
        Patch(facecolor="#27ae60", label="Forest gain"),
    ], loc="lower right", fontsize=10)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_forest_area_trend(
    labels: np.ndarray,
    years: Sequence[int],
    out_path: Optional[Path] = None,
    pixel_area_ha: float = 0.01,   # 10m pixel → 0.01 ha
) -> plt.Figure:
    """Line plot of dense + open forest area (ha) over time."""
    dense_ha  = [np.sum(labels[t] == 1) * pixel_area_ha for t in range(len(years))]
    open_ha   = [np.sum(labels[t] == 2) * pixel_area_ha for t in range(len(years))]
    total_ha  = [d + o for d, o in zip(dense_ha, open_ha)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, dense_ha, marker="o", color="#03470c", label="Dense forest")
    ax.plot(years, open_ha,  marker="s", color="#60b33f", label="Open forest")
    ax.plot(years, total_ha, marker="^", color="#1a6e00", linestyle="--", label="Total forest", linewidth=2)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Area (ha)", fontsize=12)
    ax.set_title("Deoghar: Forest Area Trend", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_uncertainty(
    label: np.ndarray,
    entropy: np.ndarray,
    title: str = "Prediction Uncertainty (MC Dropout)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_lulc_map(label, "Predicted class", ax=axes[0], show_legend=True)
    im = axes[1].imshow(entropy, cmap="magma")
    axes[1].set_title("Entropy (higher = less certain)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_transition_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str] = LULC_NAMES,
    title: str = "LULC Transition Probabilities",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("To", fontsize=11)
    ax.set_ylabel("From", fontsize=11)
    ax.set_title(title, fontsize=13)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig
