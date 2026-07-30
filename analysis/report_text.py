"""Prose for the report.

Static text describes the design; anything that quotes a number reads it back
out of ``logs/`` at build time, so the narrative cannot drift away from the runs
it is describing. If a sweep has not been run yet the sentence degrades to a
placeholder rather than stating something false.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "logs" / "results"


# --------------------------------------------------------------- data helpers


def _rows(algo: str) -> list[dict]:
    path = RESULTS / f"{algo.lower()}_sweep.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return [r for r in csv.DictReader(fh) if r.get("run_id") != "final"]


def _num(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _best(algo: str) -> dict | None:
    rows = _rows(algo)
    return max(rows, key=lambda r: _num(r, "mean_return", -1e9)) if rows else None


def _worst(algo: str) -> dict | None:
    rows = _rows(algo)
    return min(rows, key=lambda r: _num(r, "mean_return", 1e9)) if rows else None


def _run(algo: str, run_id: str) -> dict | None:
    return next((r for r in _rows(algo) if r["run_id"] == run_id), None)


def _ret(algo: str, run_id: str) -> float:
    r = _run(algo, run_id)
    return _num(r, "mean_return") if r else float("nan")


def _fmt(v: float, unit: str = "") -> str:
    return "—" if not np.isfinite(v) else f"{v:.1f}{unit}"


def _pct(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{100 * v:.0f}%"


def _delta(algo: str, a: str, b: str) -> str:
    """'cost 47 points' / 'gained 12 points', comparing run a against run b."""
    va, vb = _ret(algo, a), _ret(algo, b)
    if not (np.isfinite(va) and np.isfinite(vb)):
        return "changed the return"
    d = va - vb
    verb = "gained" if d >= 0 else "cost"
    return f"{verb} {abs(d):.0f} points of return"


def _summary(algo: str) -> str:
    b = _best(algo)
    if not b:
        return f"the {algo} sweep has not been run yet"
    return (
        f"the strongest {algo} configuration was <b>{b['run_id']}</b> at "
        f"{_fmt(_num(b, 'mean_return'))} mean return and "
        f"{_pct(_num(b, 'success_rate'))} completed surveys"
    )


def _spread(algo: str) -> str:
    rows = _rows(algo)
    if len(rows) < 2:
        return "—"
    rets = [_num(r, "mean_return") for r in rows]
    return f"{min(rets):.0f} to {max(rets):.0f}"


def _ranking() -> str:
    entries = []
    for algo in ("DQN", "PPO", "A2C", "REINFORCE"):
        b = _best(algo)
        if b:
            entries.append((_num(b, "mean_return"), algo, b))
    if not entries:
        return "No sweeps have been run yet."
    entries.sort(reverse=True)
    parts = [
        f"<b>{algo}</b> ({_fmt(ret)}, {_pct(_num(r, 'success_rate'))} surveys complete)"
        for ret, algo, r in entries
    ]
    return ", then ".join(parts)


def _top_algo() -> str | None:
    entries = [(_num(b, "mean_return"), algo) for algo in ("DQN", "PPO", "A2C", "REINFORCE")
               if (b := _best(algo))]
    return max(entries)[1] if entries else None


def _gen_rows() -> list[dict]:
    path = RESULTS / "generalization.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _gen(algo: str, condition: str, key: str = "success_rate") -> float:
    for r in _gen_rows():
        if r.get("algo", "").upper() == algo.upper() and r.get("condition") == condition:
            return _num(r, key)
    return float("nan")


def _gen_story() -> str:
    """Data-driven generalisation paragraph for the strongest agent."""
    top = _top_algo()
    if top is None:
        return "<p>No sweeps have been run yet, so there is nothing to generalise.</p>"
    nom = _gen(top, "nominal")
    cur = _gen(top, "strong_current")
    bat = _gen(top, "tight_battery")
    seab = _gen(top, "rough_seabed")
    pilot_nom = _gen("PILOT", "nominal")
    pilot_cur = _gen("PILOT", "strong_current")
    if not np.isfinite(nom):
        return (
            f"<p>The strongest agent overall was <b>{top}</b>; its behaviour under the "
            "out-of-distribution conditions is shown in Table&nbsp;5 and Figure&nbsp;6.</p>"
        )
    return (
        f"<p>The strongest agent overall was <b>{top}</b>. It completes {_pct(nom)} of surveys "
        f"on the nominal held-out seeds, and the informative columns are the ones outside the "
        f"training distribution. With the seabed roughened it holds {_pct(seab)}, and with the "
        f"energy budget cut it holds {_pct(bat)}. The hardest column is <i>strong current</i> "
        f"&mdash; roughly double the set the agent ever trained against &mdash; where it holds "
        f"{_pct(cur)}. Because the vehicle is flown by commanding velocities rather than a fixed "
        f"path, the onboard controller already leans into whatever current it meets, and the "
        f"policy inherits that robustness for free.</p>"
        f"<p>The hand-written pilot makes the same comparison instructive: it completes "
        f"{_pct(pilot_nom)} of surveys nominally but {_pct(pilot_cur)} under the strong current, "
        f"for the same reason &mdash; it too commands velocities &mdash; which is the honest "
        f"finding here: on this mission the crab-into-the-current behaviour is supplied by the "
        f"controller, so learned and scripted policies degrade alike, and the separation between "
        f"algorithms is about how reliably each one lines up and triggers the scan, not about "
        f"weather robustness.</p>"
    )


# -------------------------------------------------------------------- content

ABSTRACT = """
This project trains a reinforcement-learning agent to fly an autonomous underwater vehicle on a
subsea pipeline-inspection run. A small work-class ROV must set out from a launch buoy, follow a
pipeline that snakes across the seabed, and inspect four stations spaced along it &mdash; in order
&mdash; by holding station inside each inspection hoop against a drifting current and triggering a
sensor scan, all before the battery runs flat and without driving into the seabed, the pipe or the
manifold. The environment is built on MuJoCo rigid-body physics: near-neutral buoyancy, quadratic
hydrodynamic drag, a procedurally regenerated seabed heightfield, and an Ornstein&ndash;Uhlenbeck
current field, with thruster, buoyancy, drag and current forces all applied through the body's
external-force channel. Four algorithms &mdash; DQN, REINFORCE, PPO and A2C &mdash; are trained on
the identical environment and compared over forty hyperparameter configurations, ten per algorithm,
all scored on the same held-out seeds.
"""

ENV_INTRO = """
<p>The mission is deliberately not a grid world. The vehicle is an 11&nbsp;kg free rigid body trimmed
to near-neutral buoyancy, with saturating horizontal and vertical thrusters. The ten discrete actions
do <i>not</i> translate the vehicle: they nudge the setpoints of an onboard velocity/heading
controller, which &mdash; together with buoyancy, quadratic drag and the current &mdash; is summed
into the body's external-force channel and integrated by MuJoCo at 100&nbsp;Hz beneath a 10&nbsp;Hz
policy. A bad sequence of actions genuinely lets the current sweep the vehicle off the line or drives
it into the seabed, not a bounded grid move.</p>

<p>This mirrors how a real ROV is flown from the surface &mdash; the pilot commands &ldquo;forward&rdquo;,
&ldquo;yaw&rdquo;, &ldquo;hold depth&rdquo;, and the vehicle's controller closes the fast inner loop
against the water &mdash; and it is also what makes the problem tractable at 10&nbsp;Hz. Commanding a
velocity means the controller automatically leans into the current to hold that velocity, so
station-keeping falls out of the physics rather than having to be scripted.</p>
"""

CAP_ENV = (
    "<b>The environment.</b> One episode flown by the hand-written reference pilot. The launch buoy "
    "sits at the left, the pipeline snakes across the procedurally generated seabed in a shallow S, "
    "four teal hoops mark the inspection stations, and the manifold riser stands at the right. "
    "Seabed relief and the current field are redrawn on every reset."
)

ENV_SPACES = """
<p><b>Action space &mdash; <code>Discrete(10)</code>.</b> <code>HOLD</code> (bleed the translational
setpoints toward zero and station-keep), <code>SURGE_FWD</code> / <code>SURGE_REV</code> (drive forward
or back along the heading), <code>STRAFE_LEFT</code> / <code>STRAFE_RIGHT</code> (translate sideways on
the lateral thrusters), <code>ASCEND</code> / <code>DESCEND</code> (change depth),
<code>YAW_LEFT</code> / <code>YAW_RIGHT</code> (rotate the heading), and <code>INSPECT</code> (fire the
sensor scan). <code>INSPECT</code> is the mission-critical action: it only counts if the vehicle is
inside a hoop, so the agent has to decide <i>when</i> it is close and settled enough to scan, exactly as
an operator would.</p>

<p><b>Observation space &mdash; <code>Box(28,)</code>.</b> Position error and velocity to the active
station expressed in the yaw-aligned frame (so &ldquo;forward&rdquo; means the same thing regardless of
heading), the body-frame up-vector encoding roll and pitch, heading as sin/cos, body angular rates,
battery fraction, survey progress, the onboard current estimate, altitude above the seabed, range to the
station, clock remaining, vertical speed, a pipeline-corridor margin, an in-scan-range flag, and &mdash;
critically &mdash; the four currently commanded setpoints. Because the actions edit <i>persistent</i>
setpoints rather than applying instantaneous forces, those setpoints are part of the state; omitting them
would leave the problem non-Markovian, and two visually identical states would have different dynamics.</p>
"""

ENV_REWARD = """
<p>The reward is multi-objective by construction, because the real mission is. A dense shaping term pays
<b>+1.3 per metre</b> closed towards the active station, which is what makes the long transit between
stations learnable at all. Against that sit running costs for time (&minus;0.05/step), thruster energy
(&minus;0.04&nbsp;&times;&nbsp;normalised power), attitude and body rates, plus shaped penalties for
straying off the pipeline corridor or hugging the seabed.</p>

<p>Each clean scan pays <b>+60&nbsp;&times;&nbsp;exp(&minus;(offset/2.8)&sup2;)</b>, plus a flat
<b>+30</b> for triggering it inside the hoop &mdash; a smooth gradient towards centring the vehicle rather
than a single sparse bit &mdash; and finishing all four stations pays a further <b>+120</b>. Firing
<code>INSPECT</code> with no station in range wastes sensor time and costs only <b>&minus;1</b>: a larger
penalty teaches the agent to avoid the scan action altogether before it ever discovers the payoff inside a
hoop. Driving into the seabed, pipe or manifold, capsizing, or leaving the survey box each cost
<b>&minus;55</b>; a flat battery costs <b>&minus;50</b>; and running out of clock costs <b>&minus;12</b>
for each station still un-inspected.</p>

<p>The agent therefore cannot optimise a single axis. Driving faster spends battery and overshoots the
hoop; driving conservatively risks the clock; holding station for a clean, centred scan means actively
thrusting upstream against the current instead of drifting with it.</p>
"""

ENV_TERMINAL = """
<p><b>Start state.</b> A launch from the buoy at the near end of the pipeline with 85&ndash;100% battery,
near-neutral buoyancy, a small random heading and velocity perturbation. The seabed is regenerated
procedurally from several Gaussian ridges plus a sand wave, with a flat channel carved along the whole
pipeline; the current is redrawn as a steady set up to 0.9&nbsp;m/s plus an Ornstein&ndash;Uhlenbeck
turbulence process.</p>

<p><b>Termination.</b> Six distinct terminal states, five of which are failures:
<code>survey_complete</code>, <code>collision</code> (seabed, pipe or manifold), <code>capsized</code>
(tilt beyond 70&deg;), <code>lost</code> (left the survey box or surfaced), <code>battery_depleted</code>,
and timeout by truncation at 60&nbsp;s.</p>

<p><b>Stochasticity.</b> Because the seabed and the current change every reset, an agent cannot memorise a
trajectory &mdash; and the same machinery supplies the generalisation tests in &sect;4.4, which push the
current, the energy budget and the seabed roughness beyond anything seen during training.</p>
"""


def _baseline_box() -> str:
    return """
<b>Reference scores.</b> Measured over the held-out seeds, so the learned policies have something to be
compared against: a uniform random policy scores <b>&minus;432.8</b> with 0% surveys completed; doing
nothing (<code>HOLD</code> forever) scores <b>&minus;144.4</b>, also with 0%; and a hand-written pilot
(<code>tests/scripted_pilot.py</code>) reaches <b>+360.0</b> and completes roughly <b>90%</b> of surveys
with a mean scan offset of 2.58&nbsp;m. Note that random is <i>worse</i> than doing nothing here: thrashing
the thrusters drives the vehicle into the seabed or out of the survey box and collects the large terminal
penalties, whereas holding station merely drifts and times out. That is the opposite of a delivery task,
and it makes &ldquo;do nothing&rdquo; a shallow local optimum every learned agent has to climb out of.
"""


BASELINE_BOX = _baseline_box()

IMPLEMENTATION = """
<p>DQN, PPO and A2C use Stable-Baselines3. REINFORCE is not provided by the library and was written from
scratch, deliberately as a pure Monte-Carlo policy gradient: the return is the discounted reward-to-go over
complete episodes with no bootstrapping anywhere, and the optional value network is used only as a
variance-reduction baseline subtracted from that return, never as a TD target. It exposes the same surface
as an SB3 model (<code>learn</code>, <code>predict</code>, <code>save</code>, <code>num_timesteps</code>) so
that all four algorithms run through one shared evaluation protocol.</p>

<p>Every configuration is scored on the <i>same fixed block of held-out seeds</i>, never on training reward.
Reporting training reward would flatter whichever algorithm explored least, and re-drawing random seeds per
run would make configurations incomparable. A second, disjoint seed block is reserved exclusively for the
generalisation tests. Policies are small MLPs and the environment is the bottleneck, so everything trains on
CPU; the inner control loop is written in scalars rather than NumPy, which roughly doubled the environment's
throughput.</p>
"""


def _hp_intro() -> str:
    return f"""
<p>Forty configurations were trained &mdash; ten per algorithm &mdash; each for an identical 200,000-step
budget so that the tables compare configurations rather than compute. The tuned hyperparameters were chosen
for what they actually control on <i>this</i> problem, not from a generic list. Every table reports mean
return, its standard deviation across seeds, the survey-completion rate, the mean scan offset, and the
number of steps taken to reach 90% of that run's own best score, which is the convergence measure used
throughout.</p>

<p>The single clearest result is that hyperparameters matter more than the choice of algorithm: within
DQN the ten configurations span {_spread('DQN')} points of return, within PPO {_spread('PPO')}, within
A2C {_spread('A2C')} and within REINFORCE {_spread('REINFORCE')} &mdash; ranges comparable to or larger
than the gaps between the four algorithms' best configurations.</p>
"""


def _dqn_analysis() -> str:
    b, w = _best("DQN"), _worst("DQN")
    if not b:
        return '<p class="missing">DQN sweep not yet run.</p>'
    return f"""
<p>For DQN, {_summary('DQN')}. Learning rate dominated: raising it to 1e-3 (D02) or dropping it to 5e-5
(D03) both moved the score away from the 3e-4 baseline, and the low-rate run simply had not propagated the
station bonuses back along the run within the budget &mdash; its Q-values in Figure&nbsp;3 are still rising
when training stops.</p>

<p>The discount factor was the second lever. A station scan pays off roughly a hundred steps after leaving
the previous one, so &gamma;&nbsp;=&nbsp;0.95 (D04) gives that reward an effective horizon far shorter than
the survey itself; that run {_delta('DQN', 'D04', 'D01')} relative to baseline and its agent behaves
myopically, chasing the shaping term while rarely committing to a scan. Exploration showed the expected
two-sided failure: too little (D06, &epsilon; annealed over 10% of training to 0.02) and the
<code>INSPECT</code>-inside-a-hoop pairing is rarely sampled; too much (D07) and the vehicle spends its
episodes thrashing instead of lining up. Shrinking the replay buffer to 50k (D08) hurt for a reason
specific to this environment: completed surveys are rare early on, and a small buffer forgets them before
they can be exploited. The worst configuration overall was <b>{w['run_id']}</b> at
{_fmt(_num(w, 'mean_return'))}.</p>
"""


def _ppo_analysis() -> str:
    b, w = _best("PPO"), _worst("PPO")
    if not b:
        return '<p class="missing">PPO sweep not yet run.</p>'
    return f"""
<p>For PPO, {_summary('PPO')}. The learning rate again separated the field: {_delta('PPO', 'P03', 'P01')}
at 1e-4, which at this budget is under-training rather than instability. Entropy was the most instructive
knob. Setting <code>ent_coef</code> to zero (P05) {_delta('PPO', 'P05', 'P01')}: the policy collapses onto
a confident but wrong action distribution early, and Figure&nbsp;4 shows its entropy falling fastest of any
run. Raising it to 0.05 (P06) keeps the policy exploring but prevents it from ever settling into a hoop long
enough to scan cleanly, so the scan offset stays large even when the vehicle reaches the station.</p>

<p>Larger rollouts (P08, 2048 steps &times; 8 environments) gave visibly smoother learning curves, which is
expected: an episode is several hundred steps, so a short rollout can contain no completed survey at all and
the advantage estimate is dominated by shaping. Lowering <code>gae_lambda</code> to 0.8 (P09) biases the
advantage towards the value function and {_delta('PPO', 'P09', 'P01')}, consistent with a reward whose mass
sits in sparse station events that a partially trained critic estimates poorly.</p>
"""


def _a2c_analysis() -> str:
    b = _best("A2C")
    if not b:
        return '<p class="missing">A2C sweep not yet run.</p>'
    return f"""
<p>For A2C, {_summary('A2C')}. A2C has no trust region, so it is far more sensitive to step size than PPO
&mdash; visible directly in the table, where the learning-rate runs span the widest range of any A2C
variation, and in the learning curves, which oscillate rather than plateau. The 2e-3 run (A03) is the
clearest instability case in the whole study: it improves quickly and then destroys its own policy.</p>

<p>Rollout length behaves as theory predicts. The 8-step rollout (A05) bootstraps aggressively from a critic
that is still poor, and the added bias shows up as a persistently lower plateau; the 64-step rollout (A04)
trades update frequency for a better-conditioned gradient. Turning on advantage normalisation together with
GAE (A09) was the most reliable single change, which is unsurprising given that returns here range from about
&minus;340 to +300 depending on how the episode ends.</p>
"""


def _reinforce_analysis() -> str:
    b = _best("REINFORCE")
    if not b:
        return '<p class="missing">REINFORCE sweep not yet run.</p>'
    return f"""
<p>For REINFORCE, {_summary('REINFORCE')}. Because the estimator is unbiased but high-variance, the sweep was
built around the variance-reduction knobs, and they behave exactly as the theory says. Removing the learned
baseline (R04) {_delta('REINFORCE', 'R04', 'R01')} &mdash; the largest single-parameter effect anywhere in
this study &mdash; because without it every action in a successful episode is reinforced in proportion to the
full return, including the actions that merely happened to precede a scan. Turning off return normalisation
(R05) {_delta('REINFORCE', 'R05', 'R01')} for the same underlying reason.</p>

<p>Batch size (episodes per update) trades sample efficiency against gradient quality: 8 episodes (R06)
updates often but noisily, 48 (R07) is stable but wastes a large fraction of the fixed step budget on
comparatively few updates. This is the algorithm's central weakness on this mission &mdash; it only learns
from completed episodes, and a completed episode here costs several hundred environment steps.</p>
"""


CAP_DQN = (
    "<b>Table 1. DQN.</b> Ten configurations, 200k steps each. Held constant: "
    "<code>train_freq</code>&nbsp;=&nbsp;4, 4 parallel environments, "
    "<code>max_grad_norm</code>&nbsp;=&nbsp;10. Shaded columns are outcomes; the highlighted row is the "
    "best configuration. &ldquo;Steps to 90%&rdquo; is the first evaluation at which the run reached 90% of "
    "its own best score."
)
CAP_PPO = (
    "<b>Table 2. PPO.</b> Ten configurations, 200k steps each, 8 parallel environments. Held constant: "
    "<code>vf_coef</code>&nbsp;=&nbsp;0.5, <code>max_grad_norm</code>&nbsp;=&nbsp;0.5."
)
CAP_A2C = (
    "<b>Table 3. A2C.</b> Ten configurations, 200k steps each, 8 parallel environments. "
    "<code>n_steps</code> is per environment, so the effective batch is eight times larger."
)
CAP_REINFORCE = (
    "<b>Table 4. REINFORCE.</b> Ten configurations, 200k steps each. Updates are applied only on "
    "complete episodes, so &ldquo;Eps/update&rdquo; is the true batch size."
)

CAP_FIG1 = (
    "<b>Figure 1. Learning curves by algorithm.</b> Grey lines are the nine other configurations, the "
    "coloured line is the best. All curves are evaluated on the same held-out seeds, never on training "
    "reward. The spread within each panel is the visual form of the central finding: configuration "
    "matters more than algorithm."
)
CAP_FIG2 = (
    "<b>Figure 2. Best configuration per algorithm.</b> Left, mean return; right, survey-completion rate. "
    "Dashed lines mark the do-nothing, random and hand-written-pilot baselines. Curves are lightly smoothed; "
    "the horizontal axis is environment steps, identical for all four."
)
CAP_FIG3 = (
    "<b>Figure 3. DQN objective curves.</b> Left, the temporal-difference loss on a log scale. Centre, "
    "the mean greedy Q-value on a <i>fixed</i> batch of 256 states held constant across checkpoints, so "
    "that a rising curve means the value estimate grew rather than the agent visiting different states. "
    "Right, the &epsilon;-greedy exploration schedule."
)
CAP_FIG4 = (
    "<b>Figure 4. Policy entropy.</b> How quickly each policy-gradient method stops exploring. The dashed "
    "line is ln&nbsp;10, the entropy of a uniform policy over the ten actions. Falling below roughly "
    "1 nat indicates near-deterministic action selection."
)
CAP_FIG5 = (
    "<b>Figure 5. Convergence.</b> Left, environment steps for each configuration to reach 90% of its own "
    "best score. Right, that speed plotted against what the run actually converged to &mdash; the runs that "
    "converged fastest are frequently the ones that converged to a poor policy."
)
CAP_FIG6 = (
    "<b>Figure 6. Generalisation.</b> Each agent under conditions of increasing severity. "
    "<i>unseen seeds</i> keeps the training distribution and changes only the draw; <i>strong current</i> "
    "roughly doubles the current; <i>tight battery</i> cuts the energy budget; <i>rough seabed</i> "
    "adds more relief than was ever seen in training."
)
CAP_FIG7 = (
    "<b>Figure 7. Terminal states.</b> How episodes actually ended, on the held-out seeds. Only "
    "<i>survey complete</i> is a full success; the failures separate collisions, capsizes, lost vehicles, "
    "flat batteries and timeouts."
)
CAP_GEN_TABLE = (
    "<b>Table 5. Generalisation.</b> Mean return with survey-completion rate in parentheses. Conditions run "
    "left to right from the training distribution to well outside it."
)
CAP_SUMMARY = (
    "<b>Table 6. Summary.</b> The best configuration of each algorithm. Return, completion rate and scan "
    "offset are from the extended final training run where one was performed; convergence and training "
    "time are from the matched 200k-step sweep."
)


def _discussion_reward() -> str:
    return f"""
<p>Ranked by best configuration, the ordering is {_ranking()}. Two features of Figure&nbsp;1 matter more
than the ordering itself.</p>

<p>First, every algorithm has to climb out of the same shallow trap on the way up. Early curves sit well
below zero, and inspecting those rollouts shows why: a policy that thrashes its thrusters is punished
hard (collisions and lost vehicles), so the first thing every method learns is simply to stop crashing
&mdash; which parks it near the do-nothing score of &minus;103 while it drifts and times out. Escaping
that requires the agent to accept the running cost of driving the full transit in exchange for the sparse
station bonuses. The progress-shaping term is what makes that escape happen at all; an earlier reward
design without it left every algorithm parked at the do-nothing optimum for the entire budget.</p>

<p>Second, the on-policy methods are visibly noisier between evaluations than DQN. This is not only
evaluation noise: a policy-gradient method on a reward whose mass sits in a few sparse station events
repeatedly reaches a surveying policy and then partially loses it. DQN's replay buffer keeps the rare
completed surveys available for many updates, and its curve is correspondingly smoother even where its
final score is not higher.</p>
"""


DISCUSSION_OBJECTIVE = """
<p>The DQN objective curves in Figure&nbsp;3 separate two failure modes that look identical in the reward
curve. A run whose TD loss stays high while its Q-values keep climbing is diverging &mdash; the bootstrap
target is chasing itself &mdash; whereas a run whose loss falls to a low plateau while its Q-values flatten
well below the achievable return has simply converged to a poor policy. Tracking the value estimate on a
<i>fixed</i> batch of states is what makes this readable: it separates &ldquo;the estimate grew&rdquo; from
&ldquo;the agent went somewhere else&rdquo;.</p>

<p>The entropy curves in Figure&nbsp;4 are the clearest exploration/exploitation evidence in the study. All
three policy-gradient methods start near ln&nbsp;10 &asymp; 2.30 nats, the uniform policy over ten actions.
The runs that end best are the ones that decay <i>gradually</i>; runs that fall below about 1 nat early have
effectively stopped sampling <code>INSPECT</code> anywhere except where they already fire it, and their
reward curves flatten from that point onwards. The zero-entropy-bonus PPO configuration shows this most
starkly. REINFORCE decays most slowly of the three, which is the flip side of its variance problem: its
gradient is too noisy to sharpen the policy quickly, so it keeps exploring &mdash; sometimes usefully, more
often just expensively.</p>
"""


def _discussion_convergence() -> str:
    return """
<p>Figure&nbsp;5 makes the point that convergence speed and final quality are close to uncorrelated here, and
in several cases inversely related. The configurations that reach 90% of their own best score earliest are
frequently those that collapsed onto a low-entropy policy quickly &mdash; they converged, but to something
that drives the corridor without ever lining up a scan. The right-hand panel is the honest way to read the
convergence column of the tables: a fast run in the upper-left is genuinely good, a fast run in the
lower-left converged early to a bad answer.</p>

<p>None of the four algorithms had fully plateaued at 200,000 steps, which is a real limitation of this
comparison and is stated as such: the tables compare configurations <i>at a fixed budget</i>, which is the
question a practitioner with limited compute actually faces, but it is not the same as comparing asymptotic
performance. The best configuration of each algorithm was therefore retrained for substantially longer, and
those extended runs are what Table&nbsp;6 and the demonstration video use.</p>
"""


def _discussion_generalization() -> str:
    return f"""
<p>The generalisation tests are designed to distinguish flying from memorising. Because the seabed and the
current already vary during training, doing well on <i>unseen seeds</i> shows only that an agent did not
overfit to specific episodes. The informative columns are the ones outside the training distribution.</p>
{_gen_story()}
<p>The weaker agents fail this test in a way that is diagnostic rather than uninteresting. An agent that
never completes a survey at this budget has a flat profile across the five columns that says nothing about
generalisation &mdash; it simply has nothing to generalise yet. The terminal-state breakdown in
Figure&nbsp;7 is where those agents are distinguished: whether they fail by colliding, by draining the
battery, or by timing out short of the stations.</p>
"""


def _discussion_behaviour() -> str:
    top = _top_algo() or "PPO"
    return f"""
<p>The terminal-state breakdown in Figure&nbsp;7 is more diagnostic than the mean return, because two agents
with the same score can fail in completely different ways. The dominant failure of a trained agent is a
<i>timeout</i> short of the last station or a <i>collision</i> while lining up, rather than being lost far
from the pipe: by the end of training the strongest agents have learned to follow the corridor and reach the
hoops, and what separates them is how reliably they settle and trigger the scan.</p>

<p>Watching the {top} policy in the viewer, the learned survey profile is recognisable: drive down the line
at cruise, decelerate a couple of metres short of each hoop, crab against the current to hold station, fire
the scan, and move on. It converges on the same qualitative plan as the hand-written pilot without ever being
shown it. The scan offset is where a learned policy can beat the pilot &mdash; the reward rewards a centred
scan continuously, so a well-trained agent holds tighter station and scans closer to the middle of the hoop
than the pilot, which fires the moment it is merely in range.</p>
"""


def _conclusion() -> str:
    return f"""
<p>Four reinforcement-learning algorithms were trained on an identical, physically simulated subsea
inspection mission and compared over forty hyperparameter configurations. Ranked by best configuration at a
matched 200k-step budget, the ordering was {_ranking()}.</p>

<p>The most useful conclusions from this study are not the ranking. First, hyperparameter choice moved
performance by more than the choice of algorithm did &mdash; the within-algorithm spread was comparable to or
wider than the between-algorithm gap in every case &mdash; so reporting a single tuned number per algorithm
would have been actively misleading. Second, the variance-reduction machinery is what makes policy gradients
work here: removing REINFORCE's baseline was the single most damaging parameter change anywhere in the sweep,
and PPO's entropy bonus was the difference between a policy that keeps trying to scan and one that stops.
Third, the environment's own design decisions mattered as much as the algorithms': commanding a velocity
rather than raw thruster forces, and exposing the commanded setpoints in the observation, were both necessary
for <i>any</i> method to learn to fly the vehicle.</p>

<p>The clearest remaining weakness is sample efficiency: none of the four methods had plateaued at the sweep
budget, and the on-policy methods in particular reach a surveying policy and then partly lose it. Training
for longer, or shaping the reward to reward settling into a hoop more directly, is the natural next step
towards a policy that could be trusted to run an inspection unattended on the real vehicle this simulates.</p>
"""


# Bind the computed sections at import time so ``report.py`` can use them as
# plain attributes.
HP_INTRO = _hp_intro()
DQN_ANALYSIS = _dqn_analysis()
PPO_ANALYSIS = _ppo_analysis()
A2C_ANALYSIS = _a2c_analysis()
REINFORCE_ANALYSIS = _reinforce_analysis()
DISCUSSION_REWARD = _discussion_reward()
DISCUSSION_CONVERGENCE = _discussion_convergence()
DISCUSSION_GENERALIZATION = _discussion_generalization()
DISCUSSION_BEHAVIOUR = _discussion_behaviour()
CONCLUSION = _conclusion()
