"""Every figure used in the report, regenerated from the training logs.

    uv run main.py plots

Each function reads only from ``logs/`` and writes a PNG into
``assets/figures/``. Figures whose inputs are missing are skipped with a note
rather than crashing, so this is safe to run part-way through a sweep.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from analysis.style import (
    ALGO_COLOR,
    ALGO_ORDER,
    FAINT,
    FIG_DIR,
    GRID,
    INK_2,
    MUTED,
    OUTCOME_COLOR,
    OUTCOME_ORDER,
    ROOT,
    STATUS,
    label_ends,
    label_line,
    save,
    thousands,
    tidy,
    use_report_style,
)

LOG_DIR = ROOT / "logs"
RESULTS_DIR = LOG_DIR / "results"

#: Reference scores from ``tests/scripted_pilot.py`` and the trivial policies,
#: measured over 40 held-out episodes. Quoted in the comparison figure so the
#: learned curves have something to be better (or worse) than.
BASELINES = {"random policy": -35.4, "do nothing": -214.7, "hand-written pilot": 142.5}


# ------------------------------------------------------------------ log loading


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str, default=np.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def load_evals(algo: str, run_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(timesteps, mean_return, success_rate) for one run."""
    rows = _read_csv(LOG_DIR / algo.lower() / run_id / "evals.csv")
    if not rows:
        return np.array([]), np.array([]), np.array([])
    return (
        np.array([_f(r, "timesteps") for r in rows]),
        np.array([_f(r, "mean_return") for r in rows]),
        np.array([_f(r, "success_rate") for r in rows]),
    )


def load_progress(algo: str, run_id: str) -> list[dict[str, str]]:
    return _read_csv(LOG_DIR / algo.lower() / run_id / "progress.csv")


def sweep_rows(algo: str) -> list[dict[str, str]]:
    rows = _read_csv(RESULTS_DIR / f"{algo.lower()}_sweep.csv")
    return [r for r in rows if r.get("run_id") != "final"]


def best_run(algo: str) -> str | None:
    rows = sweep_rows(algo)
    if not rows:
        return None
    return max(rows, key=lambda r: _f(r, "mean_return", -1e9))["run_id"]


def run_ids(algo: str) -> list[str]:
    return sorted(r["run_id"] for r in sweep_rows(algo))


def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving average with edge padding.

    ``np.convolve(..., mode="same")`` divides the first and last few points by
    the full window even though only part of it overlaps the data, which drags
    both ends of every curve towards zero and invents a spike that is not in the
    training run. Reflecting the ends instead keeps the endpoints honest.
    """
    if len(y) < 3 or window < 2:
        return y
    window = min(window, len(y) if len(y) % 2 else len(y) - 1)
    if window < 2:
        return y
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


# ----------------------------------------------------------------- the figures


def fig_learning_curves() -> None:
    """Cumulative reward per algorithm, every configuration overlaid."""
    use_report_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True)

    drew = False
    for ax, algo in zip(axes.ravel(), ALGO_ORDER):
        colour = ALGO_COLOR[algo]
        best = best_run(algo)
        ids = run_ids(algo)
        for rid in ids:
            steps, ret, _ = load_evals(algo, rid)
            if len(steps) == 0:
                continue
            drew = True
            if rid == best:
                continue
            ax.plot(steps, ret, color=FAINT, linewidth=1.0, alpha=0.85, zorder=2)
        if best:
            steps, ret, _ = load_evals(algo, best)
            if len(steps):
                ax.plot(steps, ret, color=colour, linewidth=2.2, zorder=4)
                label_line(ax, steps[-1], ret[-1], best, colour)
        ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)
        ax.set_title(f"{algo}   ({len(ids)} configurations)", loc="left")
        tidy(ax)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))

    if not drew:
        plt.close(fig)
        print("  [skip] learning curves - no eval logs yet")
        return

    for ax in axes[1]:
        ax.set_xlabel("environment steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean return (held-out seeds)")

    fig.suptitle(
        "Learning curves by algorithm - grey lines are the other configurations,"
        " coloured line is the best",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig01_learning_curves.png")


def fig_algorithm_comparison() -> None:
    """Best configuration of each algorithm on one axis, against the baselines."""
    use_report_style()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    drew = False
    left_labels, right_labels = [], []
    for algo in ALGO_ORDER:
        best = best_run(algo)
        if not best:
            continue
        steps, ret, succ = load_evals(algo, best)
        if len(steps) == 0:
            continue
        drew = True
        colour = ALGO_COLOR[algo]
        ax.plot(steps, _smooth(ret), color=colour, linewidth=2.0, zorder=4)
        left_labels.append((steps[-1], _smooth(ret)[-1], algo, colour))
        ax2.plot(steps, 100 * _smooth(succ), color=colour, linewidth=2.0, zorder=4)
        right_labels.append((steps[-1], 100 * _smooth(succ)[-1], algo, colour))

    if not drew:
        plt.close(fig)
        print("  [skip] algorithm comparison - no eval logs yet")
        return

    for name, value in BASELINES.items():
        ax.axhline(value, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            name,
            xy=(0, value),
            xytext=(2, 3),
            textcoords="offset points",
            color=MUTED,
            fontsize=7.5,
        )

    label_ends(ax, left_labels)
    label_ends(ax2, right_labels)

    ax.set_title("Mean episode return", loc="left")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("return on held-out seeds")
    ax2.set_title("Delivery rate", loc="left")
    ax2.set_xlabel("environment steps")
    ax2.set_ylabel("% of episodes delivered in the zone")
    for a in (ax, ax2):
        tidy(a)
        a.xaxis.set_major_formatter(FuncFormatter(thousands))
        a.margins(x=0.12)

    fig.suptitle(
        "Best configuration per algorithm",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig02_algorithm_comparison.png")


def fig_dqn_objective() -> None:
    """DQN's own objective: TD loss, the value estimate, and exploration."""
    use_report_style()
    algo = "DQN"
    ids = run_ids(algo)
    if not ids:
        print("  [skip] DQN objective - no DQN runs yet")
        return
    best = best_run(algo) or ids[0]
    colour = ALGO_COLOR[algo]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))
    ax_loss, ax_q, ax_eps = axes

    # --- TD loss, every configuration ------------------------------------
    for rid in ids:
        rows = load_progress(algo, rid)
        if not rows:
            continue
        steps = [_f(r, "time/total_timesteps") for r in rows]
        loss = [_f(r, "train/loss") for r in rows]
        pairs = [(s, v) for s, v in zip(steps, loss) if np.isfinite(s) and np.isfinite(v)]
        if not pairs:
            continue
        s, v = zip(*pairs)
        is_best = rid == best
        ax_loss.plot(
            s,
            v,
            color=colour if is_best else FAINT,
            linewidth=2.0 if is_best else 0.9,
            zorder=4 if is_best else 2,
        )
    ax_loss.set_yscale("log")
    ax_loss.set_title("Temporal-difference loss", loc="left")
    ax_loss.set_ylabel("train/loss (log scale)")

    # --- value estimate on a fixed reference batch -------------------------
    for rid in ids:
        rows = _read_csv(LOG_DIR / algo.lower() / rid / "evals.csv")
        pairs = [
            (_f(r, "timesteps"), _f(r, "mean_q"))
            for r in rows
            if np.isfinite(_f(r, "mean_q", np.nan))
        ]
        if not pairs:
            continue
        s, v = zip(*pairs)
        is_best = rid == best
        ax_q.plot(
            s,
            v,
            color=colour if is_best else FAINT,
            linewidth=2.0 if is_best else 0.9,
            zorder=4 if is_best else 2,
        )
    ax_q.set_title("Mean greedy Q on fixed states", loc="left")
    ax_q.set_ylabel("max_a Q(s,a)")

    # --- exploration schedule ----------------------------------------------
    for rid in ids:
        rows = load_progress(algo, rid)
        pairs = [
            (_f(r, "time/total_timesteps"), _f(r, "rollout/exploration_rate"))
            for r in rows
            if np.isfinite(_f(r, "rollout/exploration_rate", np.nan))
        ]
        if not pairs:
            continue
        s, v = zip(*pairs)
        is_best = rid == best
        ax_eps.plot(
            s,
            v,
            color=colour if is_best else FAINT,
            linewidth=2.0 if is_best else 0.9,
            zorder=4 if is_best else 2,
        )
    ax_eps.set_title("Exploration rate (epsilon)", loc="left")
    ax_eps.set_ylabel("epsilon")

    for ax in axes:
        tidy(ax)
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))

    fig.suptitle(
        f"DQN objective curves - coloured line is the best configuration ({best}),"
        " grey are the other nine",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig03_dqn_objective.png")


def _entropy_series(algo: str, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Policy entropy in nats, however the algorithm happens to record it."""
    if algo == "REINFORCE":
        path = LOG_DIR / algo.lower() / run_id / "train_log.json"
        if not path.exists():
            return np.array([]), np.array([])
        blob = json.loads(path.read_text())
        return (
            np.array([r["timesteps"] for r in blob]),
            np.array([r["entropy"] for r in blob]),
        )
    rows = load_progress(algo, run_id)
    # SB3 logs entropy_loss = -mean(entropy), so flip the sign back
    pairs = [
        (_f(r, "time/total_timesteps"), -_f(r, "train/entropy_loss"))
        for r in rows
        if np.isfinite(_f(r, "train/entropy_loss", np.nan))
    ]
    if not pairs:
        return np.array([]), np.array([])
    s, v = zip(*pairs)
    return np.array(s), np.array(v)


def fig_pg_entropy() -> None:
    """Policy entropy over training - the exploration/exploitation trace."""
    use_report_style()
    pg = ["PPO", "A2C", "REINFORCE"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), sharey=True)

    drew = False
    max_entropy = np.log(10)  # uniform over the ten actions
    for ax, algo in zip(axes, pg):
        colour = ALGO_COLOR[algo]
        best = best_run(algo)
        for rid in run_ids(algo):
            steps, ent = _entropy_series(algo, rid)
            if len(steps) == 0:
                continue
            drew = True
            is_best = rid == best
            ax.plot(
                steps,
                _smooth(ent, 5),
                color=colour if is_best else FAINT,
                linewidth=2.0 if is_best else 0.9,
                zorder=4 if is_best else 2,
            )
        ax.axhline(max_entropy, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            "uniform policy (ln 10)",
            xy=(0, max_entropy),
            xytext=(2, 3),
            textcoords="offset points",
            color=MUTED,
            fontsize=7.5,
        )
        ax.set_title(algo, loc="left", color=colour)
        ax.set_xlabel("environment steps")
        tidy(ax)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))

    if not drew:
        plt.close(fig)
        print("  [skip] entropy curves - no policy-gradient logs yet")
        return

    axes[0].set_ylabel("policy entropy (nats)")
    fig.suptitle(
        "Policy-gradient entropy - how fast each method stops exploring",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig04_pg_entropy.png")


def fig_convergence() -> None:
    """How quickly each configuration got most of the way to its own best."""
    use_report_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    labels, values, colours, finals = [], [], [], []
    for algo in ALGO_ORDER:
        for row in sorted(sweep_rows(algo), key=lambda r: r["run_id"]):
            conv = _f(row, "convergence_step", np.nan)
            if not np.isfinite(conv):
                continue
            labels.append(row["run_id"])
            values.append(conv)
            colours.append(ALGO_COLOR[algo])
            finals.append(_f(row, "mean_return"))

    if not labels:
        plt.close(fig)
        print("  [skip] convergence - no sweep results yet")
        return

    y = np.arange(len(labels))
    ax.barh(y, values, color=colours, height=0.68)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("environment steps to reach 90% of the run's own best")
    ax.set_title("Convergence speed", loc="left")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    tidy(ax)
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)

    # convergence speed against what it actually converged to
    for algo in ALGO_ORDER:
        rows = [r for r in sweep_rows(algo) if np.isfinite(_f(r, "convergence_step", np.nan))]
        if not rows:
            continue
        xs = [_f(r, "convergence_step") for r in rows]
        ys = [_f(r, "mean_return") for r in rows]
        ax2.scatter(
            xs,
            ys,
            s=46,
            color=ALGO_COLOR[algo],
            edgecolor="white",
            linewidth=1.4,
            zorder=4,
            label=algo,
        )
    ax2.set_xlabel("steps to 90% of best")
    ax2.set_ylabel("final mean return")
    ax2.set_title("Fast is not the same as good", loc="left")
    ax2.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax2.legend(loc="lower right", ncol=2)
    tidy(ax2)

    fig.suptitle(
        "Convergence across all configurations",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig05_convergence.png")


def fig_generalization() -> None:
    """Held-out seeds versus deliberately harder conditions."""
    use_report_style()
    path = RESULTS_DIR / "generalization.csv"
    rows = _read_csv(path)
    if not rows:
        print("  [skip] generalisation - run `uv run main.py evaluate` first")
        return

    conditions = []
    for r in rows:
        if r["condition"] not in conditions:
            conditions.append(r["condition"])
    algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in rows)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    width = 0.8 / max(len(algos), 1)
    x = np.arange(len(conditions))

    for i, algo in enumerate(algos):
        by_cond = {r["condition"]: r for r in rows if r["algo"] == algo}
        rets = [_f(by_cond.get(c, {}), "mean_return") for c in conditions]
        succ = [100 * _f(by_cond.get(c, {}), "success_rate") for c in conditions]
        offset = (i - (len(algos) - 1) / 2) * width
        ax.bar(x + offset, rets, width * 0.88, color=ALGO_COLOR[algo], label=algo)
        ax2.bar(x + offset, succ, width * 0.88, color=ALGO_COLOR[algo], label=algo)

    for a, title, ylab in (
        (ax, "Mean return", "return"),
        (ax2, "Delivery rate", "% delivered"),
    ):
        a.set_xticks(x, [c.replace("_", "\n") for c in conditions], fontsize=7.5)
        a.set_title(title, loc="left")
        a.set_ylabel(ylab)
        a.axhline(0, color=GRID, linewidth=1.0)
        tidy(a)
    ax.legend(loc="upper right", ncol=2)

    fig.suptitle(
        "Generalisation: unseen seeds, unseen terrain, and harder weather",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig06_generalization.png")


def fig_outcomes() -> None:
    """What actually happened at the end of each episode."""
    use_report_style()
    path = RESULTS_DIR / "outcomes.csv"
    rows = _read_csv(path)
    if not rows:
        print("  [skip] outcome breakdown - run `uv run main.py evaluate` first")
        return

    agents = [r["agent"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.4, 0.62 * len(agents) + 1.9))
    y = np.arange(len(agents))

    left = np.zeros(len(agents))
    for outcome in OUTCOME_ORDER:
        vals = np.array([100 * _f(r, outcome, 0.0) for r in rows])
        if not vals.any():
            continue
        ax.barh(
            y,
            vals,
            left=left,
            height=0.66,
            color=OUTCOME_COLOR[outcome],
            label=outcome.replace("_", " "),
            edgecolor="#fcfcfb",
            linewidth=1.6,  # 2px surface gap between stacked segments
        )
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 9:  # only label segments wide enough to read
                ax.text(
                    l + v / 2,
                    yi,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                )
        left += vals

    ax.set_yticks(y, agents, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of held-out episodes")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    tidy(ax)
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)

    fig.suptitle(
        "How episodes ended - a delivery is the only full success",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig07_outcomes.png")


def fig_hyperparameter_effects() -> None:
    """The two hyperparameters shared by every algorithm, against final score."""
    use_report_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    drew = False
    for algo in ALGO_ORDER:
        rows = sweep_rows(algo)
        if not rows:
            continue
        lrs = [_f(r, "learning_rate") for r in rows]
        gammas = [_f(r, "gamma") for r in rows]
        rets = [_f(r, "mean_return") for r in rows]
        colour = ALGO_COLOR[algo]
        pairs = [(a, b) for a, b in zip(lrs, rets) if np.isfinite(a) and np.isfinite(b)]
        if pairs:
            drew = True
            ax.scatter(
                *zip(*pairs),
                s=46,
                color=colour,
                edgecolor="white",
                linewidth=1.4,
                zorder=4,
                label=algo,
            )
        pairs = [(a, b) for a, b in zip(gammas, rets) if np.isfinite(a) and np.isfinite(b)]
        if pairs:
            ax2.scatter(
                *zip(*pairs),
                s=46,
                color=colour,
                edgecolor="white",
                linewidth=1.4,
                zorder=4,
                label=algo,
            )

    if not drew:
        plt.close(fig)
        print("  [skip] hyperparameter effects - no sweep results yet")
        return

    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("final mean return")
    ax.set_title("Learning rate", loc="left")
    ax.legend(loc="lower center", ncol=2)
    ax2.set_xlabel("discount factor")
    ax2.set_ylabel("final mean return")
    ax2.set_title("Discount factor", loc="left")
    for a in (ax, ax2):
        a.axhline(0, color=GRID, linewidth=1.0)
        tidy(a)

    fig.suptitle(
        "Hyperparameter sensitivity across all 40 configurations",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig08_hyperparameter_effects.png")


ALL_FIGURES = [
    fig_learning_curves,
    fig_algorithm_comparison,
    fig_dqn_objective,
    fig_pg_entropy,
    fig_convergence,
    fig_generalization,
    fig_outcomes,
    fig_hyperparameter_effects,
]


def main(args=None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nRegenerating figures into {FIG_DIR.relative_to(ROOT)}/")
    for fn in ALL_FIGURES:
        try:
            fn()
        except Exception as exc:  # keep going; a missing log should not kill the batch
            print(f"  [fail] {fn.__name__}: {exc}")
    print()


if __name__ == "__main__":
    main()
