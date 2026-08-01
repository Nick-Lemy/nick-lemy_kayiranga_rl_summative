"""Build the report as a print-ready HTML document that matches the ALU template.

    uv run python -m analysis.report      # writes report/report.html

Open it in a browser and print to PDF (A4). The layout copies the assignment
report template: the grey ALU header block on every page, Times New Roman body
text, and plain bordered tables. Every number is read back out of ``logs/`` at
build time, and the figures are inlined as base64, so the file is one document.

The words that interpret the results live in ``analysis/report_text.py``.
"""

from __future__ import annotations

import base64
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "assets" / "figures"
RESULTS = ROOT / "logs" / "results"
OUT_DIR = ROOT / "report"


# --------------------------------------------------------------- cell helpers


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _f(value: str, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def num(value: str) -> str:
    """Compact number for a table cell."""
    v = _f(value)
    if not np.isfinite(v):
        return str(value) if value not in ("", None) else ""
    if v.is_integer() and abs(v) >= 1000:
        return f"{v / 1000:.0f}k"
    if v.is_integer():
        return str(int(v))
    if abs(v) < 0.001:
        return f"{v:.0e}".replace("e-0", "e-")
    return f"{v:g}"


def lr(value: str) -> str:
    v = _f(value)
    if not np.isfinite(v):
        return str(value)
    return f"{v:.0e}".replace("e-0", "e-")


def yesno(value: str) -> str:
    return "yes" if str(value).lower() in ("true", "1", "yes") else "no"


def reward(value: str) -> str:
    v = _f(value)
    return "" if not np.isfinite(v) else f"{v:.0f}"


def survey(value: str) -> str:
    v = _f(value)
    return "" if not np.isfinite(v) else f"{100 * v:.0f}%"


def exploration(r: dict) -> str:
    final = num(r.get("exploration_final_eps", ""))
    frac = _f(r.get("exploration_fraction", ""))
    frac_s = f"{100 * frac:.0f}%" if np.isfinite(frac) else "?"
    return f"1.0 to {final}, over {frac_s}"


#: Per-algorithm table layout: (column label, cell getter). A getter is either a
#: CSV key with an optional formatter, or a function of the whole row. The last
#: two columns are the measured outcome. Columns follow the template's named
#: fields for DQN and add the varied parameters for the others.
TABLES = {
    "DQN": [
        ("Learning Rate", ("learning_rate", lr)),
        ("Gamma", ("gamma", num)),
        ("Replay Buffer Size", ("buffer_size", num)),
        ("Batch Size", ("batch_size", num)),
        ("Exploration Strategy", exploration),
        ("Mean Reward", ("mean_return", reward)),
        ("Survey %", ("success_rate", survey)),
    ],
    "REINFORCE": [
        ("Learning Rate", ("learning_rate", lr)),
        ("Gamma", ("gamma", num)),
        ("Episodes / Update", ("episodes_per_update", num)),
        ("Baseline", ("use_baseline", yesno)),
        ("Normalize Returns", ("normalize_returns", yesno)),
        ("Entropy Coef", ("ent_coef", num)),
        ("Mean Reward", ("mean_return", reward)),
        ("Survey %", ("success_rate", survey)),
    ],
    "PPO": [
        ("Learning Rate", ("learning_rate", lr)),
        ("Gamma", ("gamma", num)),
        ("Rollout (n_steps)", ("n_steps", num)),
        ("Batch Size", ("batch_size", num)),
        ("Clip Range", ("clip_range", num)),
        ("Entropy Coef", ("ent_coef", num)),
        ("Mean Reward", ("mean_return", reward)),
        ("Survey %", ("success_rate", survey)),
    ],
    "A2C": [
        ("Learning Rate", ("learning_rate", lr)),
        ("Gamma", ("gamma", num)),
        ("Rollout (n_steps)", ("n_steps", num)),
        ("GAE Lambda", ("gae_lambda", num)),
        ("Entropy Coef", ("ent_coef", num)),
        ("Normalize Adv.", ("normalize_advantage", yesno)),
        ("Mean Reward", ("mean_return", reward)),
        ("Survey %", ("success_rate", survey)),
    ],
}


def _cell(getter, row: dict) -> str:
    if callable(getter):
        return getter(row)
    key, fmt = getter
    return fmt(row.get(key, ""))


def sweep_table(algo: str) -> str:
    rows = [r for r in read_csv(RESULTS / f"{algo.lower()}_sweep.csv") if r.get("run_id") != "final"]
    if not rows:
        return f'<p class="missing">No {algo} results yet. Run the sweep first.</p>'
    rows.sort(key=lambda r: r["run_id"])
    best = max(rows, key=lambda r: _f(r.get("mean_return", "-inf")))
    cols = TABLES[algo]

    head = "".join(f"<th>{label}</th>" for label, _ in cols)
    body = []
    for r in rows:
        cls = ' class="best"' if r["run_id"] == best["run_id"] else ""
        cells = "".join(f"<td>{_cell(getter, r)}</td>" for _, getter in cols)
        body.append(f"<tr{cls}>{cells}</tr>")
    return (
        f'<table class="grid"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def obs_table() -> str:
    from analysis.report_text import OBS_TABLE

    head = (
        "<tr><th>Observation</th><th>Description</th>"
        "<th>Source (Sensor / Camera / API / Dataset)</th>"
        "<th>Encoding / Data Type</th><th>Range</th></tr>"
    )
    body = "".join(
        f"<tr><td>{o}</td><td>{d}</td><td>{s}</td><td>{e}</td><td>{rng}</td></tr>"
        for o, d, s, e, rng in OBS_TABLE
    )
    return f'<table class="grid obs"><thead>{head}</thead><tbody>{body}</tbody></table>'


def summary_table() -> str:
    rows = []
    for algo in ("DQN", "PPO", "A2C", "REINFORCE"):
        data = read_csv(RESULTS / f"{algo.lower()}_sweep.csv")
        sweep = [r for r in data if r.get("run_id") != "final"]
        final = [r for r in data if r.get("run_id") == "final"]
        if not sweep:
            continue
        best = max(sweep, key=lambda r: _f(r.get("mean_return", "-inf")))
        src = final[0] if final else best
        rows.append(
            f"<tr><td>{algo}</td><td>{best['run_id']}</td>"
            f"<td>{reward(src.get('mean_return', ''))}</td>"
            f"<td>{survey(src.get('success_rate', ''))}</td>"
            f"<td>{num(src.get('mean_miss', ''))}</td></tr>"
        )
    if not rows:
        return '<p class="missing">No results yet.</p>'
    return (
        '<table class="grid"><thead><tr><th>Algorithm</th><th>Best Setting</th>'
        "<th>Mean Reward</th><th>Survey %</th><th>Scan Offset (m)</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def generalization_table() -> str:
    rows = read_csv(RESULTS / "generalization.csv")
    if not rows:
        return '<p class="missing">Run the evaluation step first.</p>'
    conditions, agents = [], []
    for r in rows:
        if r["condition"] not in conditions:
            conditions.append(r["condition"])
        if r["agent"] not in agents:
            agents.append(r["agent"])
    head = "<th>Agent</th>" + "".join(f'<th>{c.replace("_", " ")}</th>' for c in conditions)
    body = []
    for agent in agents:
        by = {r["condition"]: r for r in rows if r["agent"] == agent}
        cells = "".join(
            f"<td>{reward(by.get(c, {}).get('mean_return', ''))} "
            f"({survey(by.get(c, {}).get('success_rate', ''))})</td>"
            for c in conditions
        )
        body.append(f"<tr><td>{agent}</td>{cells}</tr>")
    return (
        f'<table class="grid gen"><thead><tr>{head}</tr></thead>'
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


# ------------------------------------------------------------------------ CSS
# The look copies the template: grey Arial ALU header repeated on every page,
# Times New Roman body, black text, plain black-bordered tables.

CSS = """
@page { size: A4; margin: 12mm 18mm 14mm 18mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 11pt; line-height: 1.35; color: #000; margin: 0;
}

/* running header via the table-header-group trick: a thead repeats at the top
   of every printed page in Chromium, which is how the ALU block appears on each
   page just like the template. */
table.layout { width: 100%; border-collapse: collapse; }
table.layout > thead { display: table-header-group; }
table.layout > thead > tr > td { border: none; padding: 0 0 5mm 0; }
table.layout > tbody > tr > td { border: none; padding: 0; }
.pagehead { font-family: Arial, Helvetica, sans-serif; }
.pagehead .u   { font-size: 8.5pt; font-weight: bold; color: #595959; letter-spacing: .3pt; }
.pagehead .bse { font-size: 19pt; font-weight: bold; color: #808080; margin: .3mm 0; }
.pagehead .s   { font-size: 10.5pt; color: #a6a6a6; line-height: 1.25; }

h1 { font-size: 12pt; font-weight: bold; margin: 0 0 3mm; }
h2 { font-size: 12pt; font-weight: bold; margin: 5mm 0 2mm; }
h3 { font-size: 11pt; font-weight: bold; margin: 3.5mm 0 1.5mm; padding-left: 6mm; }
p  { margin: 0 0 2.4mm; text-align: justify; }
ol, ul { margin: 0 0 2.4mm; padding-left: 8mm; }
li { margin-bottom: .8mm; }
sup { font-size: 7.5pt; }

.meta { margin: 0 0 4mm; }
.meta div { margin-bottom: 1mm; }
.meta b { font-weight: bold; }

table.grid {
  width: 100%; border-collapse: collapse; margin: 2mm 0 2.5mm;
  font-size: 9.5pt; font-family: "Times New Roman", Times, serif;
}
table.grid th, table.grid td {
  border: 0.7px solid #000; padding: 1.5mm 1.8mm; text-align: left; vertical-align: top;
}
table.grid th { font-weight: bold; }
table.grid tr.best td { font-weight: bold; background: #e9e9e9; }
table.grid.obs th { text-align: center; font-style: italic; }
table.grid.gen { font-size: 9pt; }

figure { margin: 2mm 0 2.5mm; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; max-height: 68mm; object-fit: contain; display: block; }
figure.env img { max-height: 78mm; }
figcaption { font-size: 9pt; color: #000; margin-top: 1mm; text-align: left; line-height: 1.28; }

.caption { font-size: 9pt; margin: -1mm 0 2.5mm; }
.missing { color: #b00; font-style: italic; }
code { font-family: "Courier New", monospace; font-size: 9.5pt; }
.note { border: 0.7px solid #000; padding: 2mm 2.5mm; margin: 2.5mm 0; font-size: 10pt; }
.pagebreak { page-break-before: always; }
"""


def build() -> Path:
    from analysis import report_text as T

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        '<div class="pagehead"><div class="u">AFRICAN LEADERSHIP UNIVERSITY</div>'
        '<div class="bse">[BSE]</div><div class="s">[ML TECHNIQUES II]</div>'
        '<div class="s">[SUMMATIVE ASSIGNMENT]</div></div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Reinforcement Learning Summative Assignment Report</title>
<style>{CSS}</style></head><body>
<table class="layout"><thead><tr><td>{header}</td></tr></thead>
<tbody><tr><td>

<h1>Reinforcement Learning Summative Assignment Report</h1>
<div class="meta">
  <div><b>Student Name:</b> Nick Lemy Kayiranga</div>
  <div><b>Video Recording:</b> [add your video link here, 3 minutes max, camera on, share the entire screen]</div>
  <div><b>GitHub Repository:</b> https://github.com/&lt;your-username&gt;/nick-lemy_kayiranga_rl_summative</div>
</div>

<h2>1. Project Overview</h2>
<p>{T.OVERVIEW.strip()}</p>

<h2>2. Environment Description</h2>
<h3>a. Agent(s)</h3>
{T.AGENTS}
{figure("env_overview.png", T.CAP_ENV, cls="env")}

<h3>b. Action Space</h3>
{T.ACTION_SPACE}

<h3>c. Observation Space</h3>
{T.OBS_INTRO}
{obs_table()}

<h3>d. Reward Structure</h3>
{T.REWARD}
<div class="note">{T.BASELINE_BOX}</div>

<h2>3. System Analysis And Design</h2>
<h3>a. Deep Q-Network (DQN)</h3>
{T.SYS_DQN}
<h3>b. Policy Gradient Methods (REINFORCE, PPO, A2C)</h3>
{T.SYS_PG}

<h2>4. Implementation</h2>
{T.IMPL_INTRO}

<h3>a. DQN</h3>
{sweep_table("DQN")}
<p class="caption"><b>Table 1. DQN.</b> Ten settings, 200,000 steps each. Held fixed: 4 parallel
environments, target update every 2,000 steps. The best row is shaded.</p>
{T.DQN_ANALYSIS}

<h3>b. REINFORCE</h3>
{sweep_table("REINFORCE")}
<p class="caption"><b>Table 2. REINFORCE.</b> Ten settings, 200,000 steps each. Updates use complete
episodes only, so "Episodes / Update" is the real batch size.</p>
{T.REINFORCE_ANALYSIS}

<h3>c. PPO</h3>
{sweep_table("PPO")}
<p class="caption"><b>Table 3. PPO.</b> Ten settings, 200,000 steps each, 8 parallel environments.
Held fixed: value coefficient 0.5, gradient clip 0.5.</p>
{T.PPO_ANALYSIS}

<h3>d. A2C</h3>
{sweep_table("A2C")}
<p class="caption"><b>Table 4. A2C.</b> Ten settings, 200,000 steps each, 8 parallel environments.
"Rollout" is per environment, so the real batch is eight times larger.</p>
{T.A2C_ANALYSIS}

<h2>5. Results Discussion</h2>
<h3>a. Cumulative Rewards</h3>
{figure("fig01_learning_curves.png", T.CAP_FIG1)}
{figure("fig02_algorithm_comparison.png", T.CAP_FIG2)}
{T.DISCUSSION_CUMULATIVE}
{figure("fig03_dqn_objective.png", T.CAP_FIG3)}
{figure("fig04_pg_entropy.png", T.CAP_FIG4)}
{T.DISCUSSION_OBJECTIVE}

<h3>b. Episodes To Converge</h3>
{figure("fig05_convergence.png", T.CAP_FIG5)}
{T.DISCUSSION_CONVERGE}
{summary_table()}
<p class="caption"><b>Table 5. Longer final runs.</b> The best setting of each algorithm, trained
again for 1,500,000 steps and scored on the held-out seeds.</p>

<h3>c. Generalization</h3>
{generalization_table()}
<p class="caption">{T.CAP_GEN_TABLE}</p>
{figure("fig06_generalization.png", T.CAP_FIG6)}
{T.DISCUSSION_GENERALIZATION}

<h2>6. Conclusion and Discussion</h2>
{T.CONCLUSION}

</td></tr></tbody></table>
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
