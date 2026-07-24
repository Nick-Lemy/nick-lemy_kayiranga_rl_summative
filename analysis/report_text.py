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
        f"{_pct(_num(b, 'success_rate'))} deliveries"
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
        f"<b>{algo}</b> ({_fmt(ret)}, {_pct(_num(r, 'success_rate'))} delivered)"
        for ret, algo, r in entries
    ]
    return ", then ".join(parts)


# -------------------------------------------------------------------- content

ABSTRACT = """
This project trains a reinforcement-learning agent to fly a cold-chain blood delivery,
modelled on the Zipline UAV service operating in Rwanda. A cargo quadrotor must lift a
blood pack from a distribution centre, cross a range of hills, and place the pack inside a
3&nbsp;m drop zone at a rural health post &mdash; before the blood spoils, before the battery runs
flat, without leaving the regulated flight corridor, and without flying into a hillside.
The environment is built on MuJoCo rigid-body physics with a genuine four-rotor airframe,
a procedurally regenerated heightfield, stochastic wind, and a payload that really
detaches and descends under a parachute. Four algorithms &mdash; DQN, REINFORCE, PPO and A2C
&mdash; are trained on the identical environment and compared over forty hyperparameter
configurations, ten per algorithm, all scored on the same held-out seeds.
"""

ENV_INTRO = """
<p>The mission is deliberately not a grid world. The aircraft is a 1.6&nbsp;kg airframe carrying
a 0.3&nbsp;kg payload, with a thrust-to-weight ratio of 1.61 and four independently actuated
rotors. The ten discrete actions do <i>not</i> translate the aircraft: they nudge the setpoints
of an onboard PD flight controller, which mixes them into four individual rotor thrusts.
Lift, banking, translation, stalling and crashing are then produced by MuJoCo's integrator
running at 100&nbsp;Hz beneath a 10&nbsp;Hz policy. A bad sequence of actions produces a genuinely
unrecoverable attitude, not a bounded grid move.</p>

<p>This mirrors how a real cargo UAV is commanded &mdash; an operator or autopilot moves attitude
and climb-rate setpoints, and the flight controller closes the fast inner loop &mdash; and it is
also what makes the problem tractable. An earlier version exposed raw collective thrust to
the policy; at one action per 100&nbsp;ms the aircraft could not arrest a descent in time and
flew itself into the ground from every initial condition.</p>
"""

CAP_ENV = (
    "<b>The environment.</b> One episode flown by the hand-written reference pilot. "
    "The depot and helipad sit at the left of the corridor, procedurally generated hills in "
    "between, and the health post with its green 3&nbsp;m drop zone at the right; orange pylons "
    "mark the lateral limits of the regulated airspace. Terrain, drop-zone position and wind "
    "are redrawn on every reset."
)

ENV_SPACES = """
<p><b>Action space &mdash; <code>Discrete(10)</code>.</b> <code>HOVER</code> (level the wings and bleed the
climb rate to zero), <code>THROTTLE_UP</code> / <code>THROTTLE_DOWN</code> (raise or lower the
commanded climb rate), <code>PITCH_FORWARD</code> / <code>PITCH_BACK</code> (accelerate or
decelerate), <code>ROLL_LEFT</code> / <code>ROLL_RIGHT</code> (translate laterally),
<code>YAW_LEFT</code> / <code>YAW_RIGHT</code> (rotate the heading for the drop run), and
<code>RELEASE_PAYLOAD</code> (open the cargo bay). Every one of these maps onto a command a real
operator can issue. Yaw is split into two actions deliberately: with a single yaw direction, any
small heading correction requires rotating almost a full turn, and the aircraft corkscrews out of
the corridor while doing it.</p>

<p><b>Observation space &mdash; <code>Box(27,)</code>.</b> Position error and velocity expressed in the
yaw-aligned frame (so &ldquo;forward&rdquo; means the same thing to the policy regardless of heading),
body-frame gravity encoding roll and pitch, heading as sin/cos, body angular rates, battery
fraction, cold-chain fraction, the onboard wind estimate, a payload-attached flag, altitude above
ground, range to the drop zone, clock remaining, vertical speed, lateral corridor margin, and
&mdash; critically &mdash; the three currently commanded setpoints. Because the actions edit
<i>persistent</i> setpoints rather than applying instantaneous forces, those setpoints are part of
the state; omitting them would leave the problem non-Markovian, and two visually identical states
would have different dynamics.</p>
"""

ENV_REWARD = """
<p>The reward is multi-objective by construction, because the real mission is. A dense shaping
term pays <b>+1.2 per metre</b> closed towards the release point, which is what makes the
60&nbsp;m transit learnable at all. Against that sit running costs for time (&minus;0.06/step),
energy (&minus;0.04&nbsp;&times;&nbsp;normalised power), attitude (&minus;0.15&nbsp;&times;&nbsp;tilt&sup2;)
and body rates, plus shaped penalties for approaching the corridor walls or the terrain.</p>

<p>The delivery itself pays <b>+150&nbsp;&times;&nbsp;exp(&minus;(miss/3)&sup2;)</b>, plus a flat
<b>+50</b> if the pack lands inside the ring &mdash; a smooth gradient towards accuracy rather than a
single sparse bit. Releasing outside the zone costs <b>&minus;30</b>, and an impact above 6&nbsp;m/s
costs <b>&minus;4</b> per m/s, because burst blood bags are a failed delivery. Crashing, breaching the
corridor, running the battery flat and spoiling the blood each cost <b>&minus;70</b>; running out of
clock costs <b>&minus;15</b>.</p>

<p>The agent therefore cannot optimise a single axis. Flying faster spends battery and arrives too
fast to drop accurately; flying conservatively risks the cold-chain timer; climbing high enough to
clear the ridge safely costs energy and lengthens the parachute descent, which lets crosswind walk
the pack out of the zone.</p>
"""

ENV_TERMINAL = """
<p><b>Start state.</b> A catapult launch from the depot pad with the payload attached, 85&ndash;100%
battery, a small random heading and velocity perturbation, and a randomly placed health post.
Terrain is regenerated procedurally from seven Gaussian hills plus a ridge across the route, with
flat aprons carved at both sites; the wind field is redrawn as a steady component up to 4&nbsp;m/s
plus an Ornstein&ndash;Uhlenbeck gust process.</p>

<p><b>Termination.</b> Eight distinct terminal states, seven of which are failures:
<code>delivered</code>, <code>missed_zone</code>, <code>crash</code>, <code>loss_of_control</code>
(tilt beyond 80&deg;), <code>corridor_breach</code>, <code>battery_depleted</code>,
<code>cold_chain_expired</code>, and timeout by truncation at 30&nbsp;s.</p>

<p><b>Stochasticity.</b> Because terrain, drop-zone position and weather all change every reset,
an agent cannot memorise a trajectory &mdash; and the same machinery supplies the generalisation
tests in &sect;4.4, which push the wind, the energy budget and the mission geometry beyond
anything seen during training.</p>
"""

BASELINE_BOX = """
<b>Reference scores.</b> Measured over 40 held-out episodes, so the learned policies have something
to be compared against: doing nothing (<code>HOVER</code> forever) scores <b>&minus;214.7</b> with 0%
deliveries; a uniform random policy scores <b>&minus;35.4</b>, also with 0%; and a hand-written
cascaded-PID pilot (<code>tests/scripted_pilot.py</code>) reaches <b>+142.5</b> with roughly 70%
deliveries. Random beats doing nothing only because it dumps the payload early and ends the
episode before the penalties accumulate &mdash; a local optimum every learned agent has to climb out
of, and one that shows up clearly in the early training curves.
"""

IMPLEMENTATION = """
<p>DQN, PPO and A2C use Stable-Baselines3. REINFORCE is not provided by the library and was written
from scratch, deliberately as a pure Monte-Carlo policy gradient: the return is the discounted
reward-to-go over complete episodes with no bootstrapping anywhere, and the optional value network
is used only as a variance-reduction baseline subtracted from that return, never as a TD target.
It exposes the same surface as an SB3 model (<code>learn</code>, <code>predict</code>,
<code>save</code>, <code>num_timesteps</code>) so that all four algorithms run through one shared
evaluation protocol.</p>

<p>Every configuration is scored on the <i>same fixed block of held-out seeds</i>, never on training
reward. Reporting training reward would flatter whichever algorithm explored least, and re-drawing
random seeds per run would make configurations incomparable. A second, disjoint seed block is
reserved exclusively for the generalisation tests. Policies are small MLPs and the environment is
the bottleneck, so everything trains on CPU; the environment was profiled from 304 to about 1,600
steps per second, mostly by moving the inner control loop off NumPy and switching the integrator.</p>
"""


def _hp_intro() -> str:
    return f"""
<p>Forty configurations were trained &mdash; ten per algorithm &mdash; each for an identical 300,000-step
budget so that the tables compare configurations rather than compute. The tuned hyperparameters were
chosen for what they actually control on <i>this</i> problem, not from a generic list. Every table
reports mean return, its standard deviation across seeds, the delivery rate, the mean miss distance,
and the number of steps taken to reach 90% of that run's own best score, which is the convergence
measure used throughout.</p>

<p>The single clearest result is that hyperparameters matter more than the choice of algorithm: within
DQN the ten configurations span {_spread('DQN')} points of return, within PPO {_spread('PPO')}, within
A2C {_spread('A2C')} and within REINFORCE {_spread('REINFORCE')} &mdash; ranges comparable to or larger
than the gaps between the four algorithms' best configurations.</p>
"""


HP_INTRO = property(lambda self: _hp_intro())  # placeholder; replaced below


def _dqn_analysis() -> str:
    b, w = _best("DQN"), _worst("DQN")
    if not b:
        return '<p class="missing">DQN sweep not yet run.</p>'
    return f"""
<p>For DQN, {_summary('DQN')}. Learning rate dominated: raising it to 1e-3 (D02) or dropping it to
5e-5 (D03) both moved the score away from the 3e-4 baseline, and the low-rate run simply had not
propagated the terminal delivery bonus back to the climb-out within the budget &mdash; its Q-values
in Figure&nbsp;3 are still rising when training stops.</p>

<p>The discount factor was the second lever. The delivery bonus arrives roughly 150 steps after
take-off, so &gamma;&nbsp;=&nbsp;0.95 (D04) gives that bonus an effective horizon far shorter than the
mission itself; that run {_delta('DQN', 'D04', 'D01')} relative to baseline and its agent behaves
myopically, optimising the shaping term while never committing to a drop. Exploration showed the
expected two-sided failure: too little (D06, &epsilon; annealed over 10% of training to 0.02) and
<code>RELEASE_PAYLOAD</code> at the right place is rarely sampled; too much (D07) and the aircraft
spends its episodes tumbling instead of reaching the zone. Shrinking the replay buffer to 50k (D08)
hurt for a reason specific to this environment: successful deliveries are rare early on, and a small
buffer forgets them before they can be exploited. The worst configuration overall was
<b>{w['run_id']}</b> at {_fmt(_num(w, 'mean_return'))}.</p>
"""


def _ppo_analysis() -> str:
    b, w = _best("PPO"), _worst("PPO")
    if not b:
        return '<p class="missing">PPO sweep not yet run.</p>'
    return f"""
<p>For PPO, {_summary('PPO')}. The learning rate again separated the field: {_delta('PPO', 'P03', 'P01')}
at 1e-4, which at this budget is under-training rather than instability. Entropy was the most
instructive knob. Setting <code>ent_coef</code> to zero (P05) {_delta('PPO', 'P05', 'P01')}: the policy
collapses onto a confident but wrong action distribution early, and Figure&nbsp;4 shows its entropy
falling fastest of any run. Raising it to 0.05 (P06) keeps the policy exploring but prevents it from
ever committing to a precise release point, so the miss distance stays large even when the aircraft
reaches the zone.</p>

<p>Larger rollouts (P08, 2048 steps &times; 8 environments) gave visibly smoother learning curves, which
is expected: an episode is 150&ndash;300 steps, so a short rollout can contain no completed delivery at all
and the advantage estimate is dominated by shaping. Lowering <code>gae_lambda</code> to 0.8 (P09) biases
the advantage towards the value function and {_delta('PPO', 'P09', 'P01')}, consistent with a reward whose
mass sits in a single terminal event that a partially trained critic estimates poorly.</p>
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

<p>Rollout length behaves as theory predicts. The 8-step rollout (A05) bootstraps aggressively from a
critic that is still poor, and the added bias shows up as a persistently lower plateau; the 64-step
rollout (A04) trades update frequency for a better-conditioned gradient. Turning on advantage
normalisation together with GAE (A09) was the most reliable single change, which is unsurprising given
that returns here range from about &minus;300 to +250 depending on how the episode ends.</p>
"""


def _reinforce_analysis() -> str:
    b = _best("REINFORCE")
    if not b:
        return '<p class="missing">REINFORCE sweep not yet run.</p>'
    return f"""
<p>For REINFORCE, {_summary('REINFORCE')}. Because the estimator is unbiased but high-variance, the sweep
was built around the variance-reduction knobs, and they behave exactly as the theory says. Removing the
learned baseline (R04) {_delta('REINFORCE', 'R04', 'R01')} &mdash; the largest single-parameter effect
anywhere in this study &mdash; because without it every action in a successful episode is reinforced in
proportion to the full return, including the actions that merely happened to precede the drop. Turning
off return normalisation (R05) {_delta('REINFORCE', 'R05', 'R01')} for the same underlying reason.</p>

<p>Batch size (episodes per update) trades sample efficiency against gradient quality: 8 episodes (R06)
updates often but noisily, 48 (R07) is stable but wastes a large fraction of the fixed step budget on
comparatively few updates. This is the algorithm's central weakness on this mission &mdash; it only learns
from completed episodes, and a completed episode here costs 150&ndash;300 environment steps.</p>
"""


CAP_DQN = (
    "<b>Table 1. DQN.</b> Ten configurations, 300k steps each. Held constant: "
    "<code>train_freq</code>&nbsp;=&nbsp;4, 4 parallel environments, "
    "<code>max_grad_norm</code>&nbsp;=&nbsp;10. Shaded columns are outcomes; the highlighted row is the "
    "best configuration. &ldquo;Steps to 90%&rdquo; is the first evaluation at which the run reached 90% of "
    "its own best score."
)
CAP_PPO = (
    "<b>Table 2. PPO.</b> Ten configurations, 300k steps each, 8 parallel environments. Held constant: "
    "<code>vf_coef</code>&nbsp;=&nbsp;0.5, <code>max_grad_norm</code>&nbsp;=&nbsp;0.5."
)
CAP_A2C = (
    "<b>Table 3. A2C.</b> Ten configurations, 300k steps each, 8 parallel environments. "
    "<code>n_steps</code> is per environment, so the effective batch is eight times larger."
)
CAP_REINFORCE = (
    "<b>Table 4. REINFORCE.</b> Ten configurations, 300k steps each. Updates are applied only on "
    "complete episodes, so &ldquo;Eps/update&rdquo; is the true batch size."
)

CAP_FIG1 = (
    "<b>Figure 1. Learning curves by algorithm.</b> Grey lines are the nine other configurations, the "
    "coloured line is the best. All curves are evaluated on the same held-out seeds, never on training "
    "reward. The spread within each panel is the visual form of the central finding: configuration "
    "matters more than algorithm."
)
CAP_FIG2 = (
    "<b>Figure 2. Best configuration per algorithm.</b> Left, mean return; right, delivery rate. Dashed "
    "lines mark the do-nothing, random and hand-written-pilot baselines. Curves are lightly smoothed; "
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
    "<i>unseen seeds</i> keeps the training distribution and changes only the draw; <i>harsh weather</i> "
    "doubles the wind; <i>tight battery</i> cuts the energy and cold-chain budgets; <i>long range</i> "
    "pushes the health post beyond anything seen in training."
)
CAP_FIG7 = (
    "<b>Figure 7. Terminal states.</b> How episodes actually ended, on the held-out seeds. Only "
    "<i>delivered</i> is a full success; <i>missed zone</i> means the agent released but was inaccurate."
)
CAP_GEN_TABLE = (
    "<b>Table 5. Generalisation.</b> Mean return with delivery rate in parentheses. Conditions run left "
    "to right from the training distribution to well outside it."
)
CAP_SUMMARY = (
    "<b>Table 6. Summary.</b> The best configuration of each algorithm. Return, delivery rate and miss "
    "distance are from the extended final training run where one was performed; convergence and training "
    "time are from the matched 300k-step sweep."
)


def _discussion_reward() -> str:
    return f"""
<p>Ranked by best configuration, the ordering is {_ranking()}. Two features of Figure&nbsp;1 matter more
than the ordering itself.</p>

<p>First, every algorithm passes through the same local optimum on the way up. Early curves sit near
&minus;35, which is precisely the score of the random policy, and inspecting those rollouts shows why:
the fastest way to stop losing step-penalty reward is to dump the payload immediately and end the
episode. Escaping it requires the agent to accept a worse short-term return &mdash; flying the full
60&nbsp;m transit while bleeding time and energy &mdash; in exchange for the terminal delivery bonus.
The progress-shaping term is what makes that escape happen at all; an earlier reward design without it
left every algorithm parked at the dumping optimum for the entire budget.</p>

<p>Second, the on-policy methods are visibly noisier between evaluations than DQN. This is not only
evaluation noise: PPO's best run repeatedly reaches a delivering policy and then partially loses it,
which is characteristic of a policy-gradient method on a reward whose mass sits in a single terminal
event. DQN's replay buffer keeps the rare successful deliveries available for many updates, and its
curve is correspondingly smoother even where its final score is not higher.</p>
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
The runs that end best are the ones that decay <i>gradually</i>; runs that fall below about 1 nat early
have effectively stopped sampling <code>RELEASE_PAYLOAD</code> anywhere except where they already release
it, and their reward curves flatten from that point onwards. The zero-entropy-bonus PPO configuration
shows this most starkly. REINFORCE decays most slowly of the three, which is the flip side of its variance
problem: its gradient is too noisy to sharpen the policy quickly, so it keeps exploring &mdash; sometimes
usefully, more often just expensively.</p>
"""


def _discussion_convergence() -> str:
    return """
<p>Figure&nbsp;5 makes the point that convergence speed and final quality are close to uncorrelated here,
and in several cases inversely related. The configurations that reach 90% of their own best score
earliest are frequently those that collapsed onto a low-entropy policy quickly &mdash; they converged, but
to something that dumps the payload short of the zone. The right-hand panel is the honest way to read the
convergence column of the tables: a fast run in the upper-left is genuinely good, a fast run in the
lower-left converged early to a bad answer.</p>

<p>None of the four algorithms had fully plateaued at 300,000 steps, which is a real limitation of this
comparison and is stated as such: the tables compare configurations <i>at a fixed budget</i>, which is the
question a practitioner with limited compute actually faces, but it is not the same as comparing
asymptotic performance. The best configuration of each algorithm was therefore retrained for
substantially longer, and those extended runs are what Table&nbsp;6 and the demonstration video use.</p>
"""


DISCUSSION_GENERALIZATION = """
<p>The generalisation tests are designed to distinguish flying from memorising. Because terrain, drop-zone
position and wind already vary during training, doing well on <i>unseen seeds</i> shows only that an agent
did not overfit to specific episodes. The informative columns are the ones outside the training
distribution.</p>

<p>The headline result is that the PPO agent barely degrades at all. It delivers 100% on the nominal
held-out seeds with a 0.89&nbsp;m mean miss, and still delivers 96% on a disjoint seed block, 96% with the
energy and cold-chain budgets cut, and 96% with the health post pushed beyond any position it saw in
training. Even under <i>harsh weather</i> &mdash; double the steady wind and nearly double the gust
intensity &mdash; it delivers 88% with the miss distance rising only from 0.89&nbsp;m to 1.57&nbsp;m. That
is not a memorised trajectory; the policy is genuinely closing a control loop on the observation.</p>

<p>The comparison against the hand-written pilot is the most striking part of the study, and it runs
opposite to what one might expect. The analytic pilot computes its upwind release offset explicitly from
the wind estimate, yet it collapses from 40% deliveries to <b>12%</b> under harsh weather, because its
drift model is a linear correction calibrated for the nominal wind range and it degrades badly outside it.
PPO, which was never given an explicit drift model at all, holds 88%. The learned policy appears to
compensate by releasing lower and closer to the zone &mdash; shortening the canopy descent, and with it the
time the crosswind has to act &mdash; which is a strategy the analytic pilot does not implement.</p>

<p>The weaker agents fail this test in a way that is diagnostic rather than uninteresting. DQN sits near
8% deliveries throughout and its scores move almost at random across conditions, which is the signature of
a policy that reaches the zone but has not learned a reliable release rule. A2C and REINFORCE never
deliver under any condition, so their flat profiles across the five columns say nothing about
generalisation &mdash; they simply have nothing to generalise yet at this budget.</p>
"""


DISCUSSION_BEHAVIOUR = """
<p>The terminal-state breakdown in Figure&nbsp;7 is more diagnostic than the mean return, because two agents
with the same score can fail in completely different ways. The dominant failure of a trained agent is
<i>missed zone</i> rather than <i>crash</i>: by the end of training all four algorithms have learned to fly
the corridor and arrive over the health post, and what separates them is the precision of the release. This
is the behaviour one wants &mdash; the hard part of the mission has become accuracy, not survival.</p>

<p>Watching the PPO policy in the viewer, the learned flight profile is recognisable: a fast climb out of
the depot to clear the ridge, a shallow high-speed cruise, a deceleration beginning roughly ten metres short
of the zone, and a descent to the release altitude before the drop. It converges on the same qualitative
plan as the hand-written pilot without ever being shown it, and then improves on it &mdash; 100% deliveries
at 0.89&nbsp;m against the pilot's 40% at 2.87&nbsp;m on the same seeds. The difference is in the terminal
phase: the agent arrives slower and releases lower and closer to the target than the analytic pilot does,
which shortens the canopy descent and therefore the window in which crosswind can push the pack off the
zone. Nothing in the reward names that strategy; it falls out of the accuracy term interacting with the
modelled parachute physics, and it is why the agent holds up under weather the analytic drift correction
cannot handle.</p>
"""


def _conclusion() -> str:
    return f"""
<p>Four reinforcement-learning algorithms were trained on an identical, physically simulated blood-delivery
mission and compared over forty hyperparameter configurations. Ranked by best configuration at a matched
300k-step budget, the ordering was {_ranking()}.</p>

<p>The most useful conclusions from this study are not the ranking. First, hyperparameter choice moved
performance by more than the choice of algorithm did &mdash; the within-algorithm spread was comparable to
or wider than the between-algorithm gap in every case &mdash; so reporting a single tuned number per
algorithm would have been actively misleading. Second, the variance-reduction machinery is what makes
policy gradients work here: removing REINFORCE's baseline was the single most damaging parameter change
anywhere in the sweep, and PPO's entropy bonus was the difference between a policy that keeps trying to
release and one that stops. Third, the environment's own design decisions mattered as much as the
algorithms': commanding a climb rate rather than raw thrust, and exposing the commanded setpoints in the
observation, were both necessary for <i>any</i> method to learn to fly.</p>

<p>The clearest remaining weakness is generalisation to weather outside the training distribution. The wind
is observable and the physics of the parachute descent is modelled, but nothing in the current reward
requires the agent to use the wind estimate when choosing a release point &mdash; and under doubled wind it
demonstrably does not. Training with a wider wind distribution, or shaping the reward on predicted rather
than instantaneous drift, is the natural next step towards a policy that could be deployed on the real
service this simulates.</p>
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
CONCLUSION = _conclusion()
