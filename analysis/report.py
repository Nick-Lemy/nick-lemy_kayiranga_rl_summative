"""Build the report as a self-contained, print-ready HTML document.

    uv run python -m analysis.report      # writes report/report.html

Open it in a browser and print to PDF. Every table and every number is read
back out of ``logs/`` at build time, so the report cannot drift away from the
runs that produced it; figures are inlined as base64 so the file travels as one
document.

The prose that interprets the results lives in ``analysis/report_text.py``, next
to the numbers it describes.
"""

from __future__ import annotations

import base64
import csv
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "assets" / "figures"
RESULTS = ROOT / "logs" / "results"
OUT_DIR = ROOT / "report"

#: Columns to show per algorithm, in order: the hyperparameters that were varied
#: plus the outcome columns. Anything held constant across all ten runs is left
#: out of the table and stated in the caption instead.
TABLE_COLUMNS = {
    "DQN": [
        ("run_id", "Run"),
        ("learning_rate", "LR"),
        ("gamma", "γ"),
        ("buffer_size", "Buffer"),
        ("batch_size", "Batch"),
        ("exploration_fraction", "ε-frac"),
        ("exploration_final_eps", "ε-final"),
        ("target_update_interval", "Target"),
        ("tau", "τ"),
        ("net_arch", "Net"),
    ],
    "PPO": [
        ("run_id", "Run"),
        ("learning_rate", "LR"),
        ("gamma", "γ"),
        ("n_steps", "n_steps"),
        ("batch_size", "Batch"),
        ("n_epochs", "Epochs"),
        ("clip_range", "Clip"),
        ("gae_lambda", "λ"),
        ("ent_coef", "Entropy"),
        ("net_arch", "Net"),
    ],
    "A2C": [
        ("run_id", "Run"),
        ("learning_rate", "LR"),
        ("gamma", "γ"),
        ("n_steps", "n_steps"),
        ("gae_lambda", "λ"),
        ("ent_coef", "Entropy"),
        ("vf_coef", "VF"),
        ("use_rms_prop", "RMSProp"),
        ("normalize_advantage", "Norm adv"),
        ("net_arch", "Net"),
    ],
    "REINFORCE": [
        ("run_id", "Run"),
        ("learning_rate", "LR"),
        ("gamma", "γ"),
        ("episodes_per_update", "Eps/update"),
        ("use_baseline", "Baseline"),
        ("normalize_returns", "Norm ret"),
        ("ent_coef", "Entropy"),
        ("max_grad_norm", "Grad clip"),
        ("net_arch", "Net"),
    ],
}

RESULT_COLUMNS = [
    ("mean_return", "Return"),
    ("std_return", "± SD"),
    ("success_rate", "Delivered"),
    ("mean_miss", "Miss (m)"),
    ("convergence_step", "Steps to 90%"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def fmt(key: str, value: str) -> str:
    """Render one cell: compact numbers, human-readable flags."""
    if value in ("", None):
        return "—"
    if key == "success_rate":
        try:
            return f"{100 * float(value):.0f}%"
        except ValueError:
            return value
    if key == "convergence_step":
        try:
            v = float(value)
            return "—" if not np.isfinite(v) else f"{v / 1000:.0f}k"
        except ValueError:
            return value
    if key in ("use_baseline", "normalize_returns", "use_rms_prop", "normalize_advantage"):
        return "yes" if str(value).lower() in ("true", "1", "yes") else "no"
    try:
        v = float(value)
    except ValueError:
        return str(value)
    if v.is_integer() and abs(v) >= 1000:
        return f"{v / 1000:.0f}k" if abs(v) >= 10_000 else f"{int(v):,}"
    if v.is_integer():
        return str(int(v))
    if abs(v) < 0.001:
        return f"{v:.0e}"
    if abs(v) < 1:
        return f"{v:g}"
    return f"{v:.2f}"


def sweep_table(algo: str) -> str:
    rows = [r for r in read_csv(RESULTS / f"{algo.lower()}_sweep.csv") if r.get("run_id") != "final"]
    if not rows:
        return f'<p class="missing">No {algo} sweep results found — run the sweep first.</p>'
    rows.sort(key=lambda r: r["run_id"])

    best = max(rows, key=lambda r: float(r.get("mean_return", "-inf")))
    cols = TABLE_COLUMNS[algo]

    head = "".join(f"<th>{label}</th>" for _, label in cols)
    head += "".join(f'<th class="res">{label}</th>' for _, label in RESULT_COLUMNS)

    body = []
    for r in rows:
        cls = ' class="best"' if r["run_id"] == best["run_id"] else ""
        cells = "".join(f"<td>{fmt(k, r.get(k, ''))}</td>" for k, _ in cols)
        cells += "".join(
            f'<td class="res">{fmt(k, r.get(k, ""))}</td>' for k, _ in RESULT_COLUMNS
        )
        body.append(f"<tr{cls}>{cells}</tr>")

    return (
        f'<table class="sweep"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def figure(name: str, caption: str, cls: str = "") -> str:
    path = FIG_DIR / name
    if not path.exists():
        return f'<p class="missing">Figure {name} not generated yet.</p>'
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<figure class="{cls}"><img src="data:image/png;base64,{b64}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def generalization_table() -> str:
    rows = read_csv(RESULTS / "generalization.csv")
    if not rows:
        return '<p class="missing">Run <code>uv run main.py evaluate</code> first.</p>'
    conditions, agents = [], []
    for r in rows:
        if r["condition"] not in conditions:
            conditions.append(r["condition"])
        if r["agent"] not in agents:
            agents.append(r["agent"])

    head = "<th>Agent</th>" + "".join(
        f'<th>{c.replace("_", " ")}</th>' for c in conditions
    )
    body = []
    for agent in agents:
        by = {r["condition"]: r for r in rows if r["agent"] == agent}
        cells = "".join(
            f'<td>{fmt("mean_return", by.get(c, {}).get("mean_return", ""))}'
            f' <span class="sub">({fmt("success_rate", by.get(c, {}).get("success_rate", ""))})</span></td>'
            for c in conditions
        )
        body.append(f"<tr><td class='agent'>{agent}</td>{cells}</tr>")
    return (
        f'<table class="sweep gen"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def summary_table() -> str:
    """Best configuration per algorithm, side by side."""
    rows = []
    for algo in ("DQN", "PPO", "A2C", "REINFORCE"):
        sweep = [
            r for r in read_csv(RESULTS / f"{algo.lower()}_sweep.csv") if r.get("run_id") != "final"
        ]
        final = [
            r for r in read_csv(RESULTS / f"{algo.lower()}_sweep.csv") if r.get("run_id") == "final"
        ]
        if not sweep:
            continue
        best = max(sweep, key=lambda r: float(r.get("mean_return", "-inf")))
        source = final[0] if final else best
        rows.append(
            f"<tr><td class='agent'>{algo}</td>"
            f"<td>{best['run_id']}</td>"
            f"<td>{fmt('mean_return', source.get('mean_return', ''))}</td>"
            f"<td>{fmt('success_rate', source.get('success_rate', ''))}</td>"
            f"<td>{fmt('mean_miss', source.get('mean_miss', ''))}</td>"
            f"<td>{fmt('convergence_step', best.get('convergence_step', ''))}</td>"
            f"<td>{fmt('wall_time_s', best.get('wall_time_s', ''))} s</td></tr>"
        )
    if not rows:
        return '<p class="missing">No results yet.</p>'
    return (
        '<table class="sweep"><thead><tr><th>Algorithm</th><th>Best config</th>'
        "<th>Return</th><th>Delivered</th><th>Miss (m)</th>"
        "<th>Steps to 90%</th><th>Train time</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


CSS = """
@page { size: A4; margin: 15mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 9.6pt; line-height: 1.46; color: #16181d; margin: 0;
  max-width: 190mm; margin-inline: auto; padding: 10mm 6mm;
}
h1 { font-size: 19pt; margin: 0 0 2mm; line-height: 1.15; letter-spacing: -.2pt; }
h2 {
  font-size: 12.5pt; margin: 7mm 0 2.5mm; padding-bottom: 1.2mm;
  border-bottom: 1.6px solid #16181d; letter-spacing: -.1pt;
}
h3 { font-size: 10.4pt; margin: 4.5mm 0 1.5mm; }
p { margin: 0 0 2.4mm; text-align: justify; hyphens: auto; }
.byline { color: #55595f; font-size: 9pt; margin-bottom: 5mm; }
.lead { font-size: 10.2pt; }

table.sweep {
  width: 100%; border-collapse: collapse; font-size: 7.3pt;
  margin: 2.5mm 0 2mm; font-family: "SF Mono", "Menlo", monospace;
  font-variant-numeric: tabular-nums;
}
table.sweep th {
  text-align: right; padding: 1.3mm 1.1mm; border-bottom: 1.2px solid #16181d;
  font-weight: 600; white-space: nowrap;
}
table.sweep td { text-align: right; padding: 1.15mm 1.1mm; border-bottom: .4px solid #dcdde0; }
table.sweep th:first-child, table.sweep td:first-child { text-align: left; }
table.sweep th.res, table.sweep td.res { background: #f4f5f7; }
table.sweep tr.best td { font-weight: 700; background: #eaf3ff; }
table.sweep tr.best td.res { background: #dcebff; }
td.agent { font-weight: 600; }
.sub { color: #6a6e75; font-size: 6.6pt; }

figure { margin: 3mm 0 4mm; page-break-inside: avoid; }
figure img { width: 100%; display: block; border: .5px solid #e2e3e6; }
figcaption { font-size: 7.7pt; color: #4a4e55; margin-top: 1.3mm; line-height: 1.4; }
figcaption b { color: #16181d; }

.caption { font-size: 7.7pt; color: #4a4e55; margin: -1mm 0 3mm; }
.missing { color: #b4442f; font-style: italic; font-size: 8.4pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt; background: #f2f3f5; padding: .3mm 1mm; }

.two { display: grid; grid-template-columns: 1fr 1fr; gap: 0 6mm; }
.three { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0 5mm; }
.box {
  border-left: 2.4px solid #16181d; padding: 1.5mm 0 1.5mm 3mm;
  margin: 2.5mm 0; font-size: 9pt; background: #fafafb;
}
ul { margin: 0 0 2.4mm; padding-left: 4.5mm; }
li { margin-bottom: .9mm; }
.pagebreak { page-break-before: always; }
.kv { font-size: 8.6pt; }
.kv td { padding: .7mm 2mm .7mm 0; vertical-align: top; }
.kv td:first-child { font-weight: 600; white-space: nowrap; }
"""


def build() -> Path:
    from analysis import report_text as T

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Mission-Based RL — Blood-Delivery UAV</title>
<style>{CSS}</style></head><body>

<h1>Optimising a Blood-Delivery UAV with Reinforcement Learning</h1>
<div class="byline">
  Nick Lemy Kayiranga &nbsp;·&nbsp; n.kayiranga@alustudent.com &nbsp;·&nbsp;
  Mission-Based Reinforcement Learning — Summative &nbsp;·&nbsp; {date.today():%d %B %Y}<br>
  Repository: <code>nick-lemy_kayiranga_rl_summative</code>
</div>

<p class="lead">{T.ABSTRACT}</p>

<h2>1. Environment</h2>
{T.ENV_INTRO}
{figure("env_overview.png", T.CAP_ENV)}

<h3>1.1 Agent, action space and observation space</h3>
{T.ENV_SPACES}

<h3>1.2 Reward structure</h3>
{T.ENV_REWARD}

<h3>1.3 Start state, termination, and stochasticity</h3>
{T.ENV_TERMINAL}

<div class="box">{T.BASELINE_BOX}</div>

<h2 class="pagebreak">2. Implementation</h2>
{T.IMPLEMENTATION}

<h2>3. Hyperparameter experiments</h2>
{T.HP_INTRO}

<h3>3.1 DQN (value-based)</h3>
{sweep_table("DQN")}
<p class="caption">{T.CAP_DQN}</p>
{T.DQN_ANALYSIS}

<h3 class="pagebreak">3.2 PPO</h3>
{sweep_table("PPO")}
<p class="caption">{T.CAP_PPO}</p>
{T.PPO_ANALYSIS}

<h3>3.3 A2C</h3>
{sweep_table("A2C")}
<p class="caption">{T.CAP_A2C}</p>
{T.A2C_ANALYSIS}

<h3>3.4 REINFORCE</h3>
{sweep_table("REINFORCE")}
<p class="caption">{T.CAP_REINFORCE}</p>
{T.REINFORCE_ANALYSIS}

<h2 class="pagebreak">4. Results and discussion</h2>
<h3>4.1 Cumulative reward</h3>
{figure("fig01_learning_curves.png", T.CAP_FIG1)}
{figure("fig02_algorithm_comparison.png", T.CAP_FIG2)}
{T.DISCUSSION_REWARD}

<h3 class="pagebreak">4.2 Training objectives and exploration</h3>
{figure("fig03_dqn_objective.png", T.CAP_FIG3)}
{figure("fig04_pg_entropy.png", T.CAP_FIG4)}
{T.DISCUSSION_OBJECTIVE}

<h3>4.3 Convergence</h3>
{figure("fig05_convergence.png", T.CAP_FIG5)}
{T.DISCUSSION_CONVERGENCE}

<h3 class="pagebreak">4.4 Generalisation</h3>
{generalization_table()}
<p class="caption">{T.CAP_GEN_TABLE}</p>
{figure("fig06_generalization.png", T.CAP_FIG6)}
{T.DISCUSSION_GENERALIZATION}

<h3>4.5 What the agents actually do</h3>
{figure("fig07_outcomes.png", T.CAP_FIG7)}
{T.DISCUSSION_BEHAVIOUR}

<h2>5. Summary and conclusion</h2>
{summary_table()}
<p class="caption">{T.CAP_SUMMARY}</p>
{T.CONCLUSION}

</body></html>
"""
    out = OUT_DIR / "report.html"
    out.write_text(html)
    print(f"  wrote {out.relative_to(ROOT)}  ({len(html) / 1024:.0f} KB)")
    print("  open it in a browser and print to PDF (A4, default margins)")
    return out


def main(args=None) -> None:
    build()


if __name__ == "__main__":
    main()
