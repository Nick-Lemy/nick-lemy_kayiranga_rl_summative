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
    "survey_complete",
    "timeout",
    "battery_depleted",
    "lost",
    "collision",
    "capsized",
]
OUTCOME_COLOR = {
    "survey_complete": "#1baf7a",
    "timeout": "#eda100",
    "battery_depleted": "#d55181",
    "lost": "#9085e9",
    "collision": "#e34948",
    "capsized": "#8a2b2b",
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


def label_ends(ax, entries, min_gap_frac: float = 0.055) -> None:
    """Direct-label several series at their end points without overlapping.

    Series that finish at similar values would otherwise print on top of each
    other - two labels colliding into an unreadable smear is worse than no
    direct labels at all. Entries are ``(x, y, text, colour)``; the y positions
    are nudged apart just enough to clear each other, and a short leader keeps
    each label tied to its line.
    """
    if not entries:
        return
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * min_gap_frac

    ordered = sorted(entries, key=lambda e: e[1])
    placed = [list(e) for e in ordered]
    # single upward pass, then clamp back inside the axes
    for i in range(1, len(placed)):
        if placed[i][1] - placed[i - 1][1] < min_gap:
            placed[i][1] = placed[i - 1][1] + min_gap
    overflow = placed[-1][1] - (hi - min_gap * 0.4)
    if overflow > 0:
        for row in placed:
            row[1] -= overflow

    for (x, y_lab, text, colour), (_, y_true, _, _) in zip(placed, ordered):
        if abs(y_lab - y_true) > min_gap * 0.25:
            ax.plot(
                [x, x + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.018],
                [y_true, y_lab],
                color=colour,
                linewidth=0.8,
                alpha=0.55,
                zorder=3,
                clip_on=False,
            )
        ax.annotate(
            text,
            xy=(x, y_lab),
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontsize=8.5,
            fontweight="bold",
            va="center",
            ha="left",
            clip_on=False,
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
