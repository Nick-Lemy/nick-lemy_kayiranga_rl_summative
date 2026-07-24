"""Shared figure styling.

One place for the palette and the mark specs, so every figure in the report
reads as one system: same colours per algorithm everywhere, thin marks,
recessive grid and axes, direct labels rather than a number on every point.

The four series colours are slots 1, 2, 7 and 3 of the reference categorical
palette. That ordering was picked with the palette validator rather than by eye:
worst adjacent CVD separation 24.7 dE (OKLab x100, target >= 8) and worst
adjacent normal-vision separation 33.6 (floor 15), so the four algorithms stay
distinguishable for colour-blind readers and in greyscale print. Aqua sits below
3:1 contrast on white, so every figure that uses it also carries a legend and
direct labels - identity is never colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "assets" / "figures"

#: Fixed colour per algorithm. Never cycled, never reassigned by rank.
ALGO_COLOR = {
    "DQN": "#2a78d6",        # blue
    "PPO": "#eb6834",        # orange
    "A2C": "#4a3aa7",        # violet
    "REINFORCE": "#1baf7a",  # aqua
}
ALGO_ORDER = ["DQN", "PPO", "A2C", "REINFORCE"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
FAINT = "#c9c8c3"

STATUS = {
    "good": "#1baf7a",
    "warning": "#eda100",
    "critical": "#e34948",
    "neutral": "#8a8985",
}

#: Terminal states, ordered best to worst, with a status-derived colour.
OUTCOME_ORDER = [
    "delivered",
    "missed_zone",
    "timeout",
    "cold_chain_expired",
    "battery_depleted",
    "corridor_breach",
    "crash",
    "loss_of_control",
]
OUTCOME_COLOR = {
    "delivered": "#1baf7a",
    "missed_zone": "#7fcfae",
    "timeout": "#eda100",
    "cold_chain_expired": "#eb6834",
    "battery_depleted": "#d55181",
    "corridor_breach": "#9085e9",
    "crash": "#e34948",
    "loss_of_control": "#8a2b2b",
}


def use_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def tidy(ax) -> None:
    """Strip the chart junk: no top/right spines, horizontal grid only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", alpha=0.9)
    ax.grid(axis="x", visible=False)


def label_line(ax, x, y, text: str, color: str, dx: float = 0.0, **kw) -> None:
    """Direct-label a series at its end, so identity is never colour alone."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6 + dx, 0),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        va="center",
        ha="left",
        **kw,
    )


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def thousands(x, _pos=None) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:g}k"
    return f"{x:g}"
