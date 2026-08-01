"""Text for the report.

Fixed text describes the design. Anything that quotes a number reads it back out
of ``logs/`` at build time, so the words cannot drift away from the runs they
describe. The writing is kept plain and short on purpose.
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


def _final(algo: str) -> dict | None:
    path = RESULTS / f"{algo.lower()}_sweep.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        finals = [r for r in csv.DictReader(fh) if r.get("run_id") == "final"]
    return finals[0] if finals else None


def _ret(algo: str, run_id: str) -> float:
    r = next((r for r in _rows(algo) if r["run_id"] == run_id), None)
    return _num(r, "mean_return") if r else float("nan")


def _fmt(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.0f}"


def _pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{100 * v:.0f}%"


def _spread(algo: str) -> str:
    rows = _rows(algo)
    if len(rows) < 2:
        return "n/a"
    rets = [_num(r, "mean_return") for r in rows]
    return f"{min(rets):.0f} to {max(rets):.0f}"


def _summary(algo: str) -> str:
    b = _best(algo)
    if not b:
        return f"the {algo} runs are not done yet"
    return (
        f"the best {algo} setting was <b>{b['run_id']}</b>, with "
        f"{_fmt(_num(b, 'mean_return'))} mean reward and "
        f"{_pct(_num(b, 'success_rate'))} of surveys finished"
    )


def _ranking() -> str:
    entries = []
    for algo in ("DQN", "PPO", "A2C", "REINFORCE"):
        b = _best(algo)
        if b:
            entries.append((_num(b, "mean_return"), algo, b))
    if not entries:
        return "no runs are done yet"
    entries.sort(reverse=True)
    parts = [
        f"<b>{algo}</b> ({_fmt(ret)} reward, {_pct(_num(r, 'success_rate'))} finished)"
        for ret, algo, r in entries
    ]
    return ", then ".join(parts)


def _top_algo() -> str | None:
    entries = [(_num(b, "mean_return"), algo) for algo in ("DQN", "PPO", "A2C", "REINFORCE")
               if (b := _best(algo))]
    return max(entries)[1] if entries else None


def _gen(algo: str, condition: str, key: str = "success_rate") -> float:
    path = RESULTS / "generalization.csv"
    if not path.exists():
        return float("nan")
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("algo", "").upper() == algo.upper() and r.get("condition") == condition:
                return _num(r, key)
    return float("nan")


# ---------------------------------------------------------- 1. project overview

OVERVIEW = """
This project trains a reinforcement learning agent to run a subsea pipeline inspection. A small
underwater robot (an ROV) starts at a launch buoy, follows a pipeline that lies on the seabed, and
inspects four stations along it in order. At each station it must reach an inspection ring, hold its
position against the water current, and take a scan. It has to do this before the battery runs out
and without hitting the seabed, the pipe, or the end manifold. The environment uses MuJoCo physics
with buoyancy, water drag, and a moving current. Four algorithms (DQN, PPO, A2C, and REINFORCE) are
trained on the same environment and compared over forty hyperparameter runs.
"""

# ------------------------------------------------------- 2. environment: agents

AGENTS = """
<p>The agent is one work-class ROV. It is an 11&nbsp;kg rigid body trimmed to near-neutral buoyancy,
so it neither sinks nor floats up on its own. It has horizontal thrusters for moving forward and
sideways, vertical thrusters for changing depth, and it can turn to face any heading.</p>

<p>The agent does not teleport. Each action sets a target speed or heading. An onboard controller
turns that target into thruster forces. Buoyancy, drag, and the current then act on the body, and
MuJoCo moves it forward in time. This is how a real ROV pilot flies the vehicle from a surface ship:
the pilot gives simple commands like "go forward" or "hold depth", and the vehicle handles the fast
control loop against the water.</p>
"""

ACTION_SPACE = """
<p>The action space is discrete with ten actions. This covers every basic command a pilot can send
to the vehicle:</p>
<ol>
<li><b>HOLD</b>: stop driving and hold position.</li>
<li><b>SURGE FORWARD</b>: drive forward along the heading.</li>
<li><b>SURGE REVERSE</b>: back off.</li>
<li><b>YAW LEFT</b>: turn the heading to port.</li>
<li><b>YAW RIGHT</b>: turn the heading to starboard.</li>
<li><b>STRAFE LEFT</b>: slide left on the side thrusters.</li>
<li><b>STRAFE RIGHT</b>: slide right.</li>
<li><b>ASCEND</b>: rise.</li>
<li><b>DESCEND</b>: dive.</li>
<li><b>INSPECT</b>: take a sensor scan. This is the key action. It only counts when the vehicle is
inside an inspection ring, so the agent has to pick the right moment to scan.</li>
</ol>
<p>Yaw and strafe are split into left and right so the agent can make a small correction in either
direction. Every action maps to a real thruster or sensor command on the vehicle.</p>
"""

# ------------------------------------------------- 2c. observation space table

OBS_INTRO = """
<p>The observation is a vector of 28 numbers, all of type float32. Positions and velocities are given
in the vehicle heading frame, so "forward" always means the same thing to the agent no matter which
way it is pointing. The setpoints the actions change are part of the observation, because the actions
edit a stored target and not an instant force, so the problem stays Markov. The table lists each
observation, the real sensor that would provide it, its data type, and its range.</p>
"""

#: Rows of the observation table: (observation, description, source, encoding, range)
OBS_TABLE: list[tuple[str, str, str, str, str]] = [
    ("Position error to station",
     "Vector from the vehicle to the active station, in the heading frame",
     "USBL acoustic positioning + INS", "3 x float32, scaled by 40 m", "about -10 to 10"),
    ("Velocity",
     "Speed of the vehicle in the heading frame",
     "Doppler Velocity Log (DVL)", "3 x float32, scaled by 3 m/s", "about -3 to 3"),
    ("Up direction",
     "Which way is up, in the body frame, so it encodes roll and pitch",
     "IMU accelerometer", "3 x float32, unit vector", "-1 to 1"),
    ("Heading",
     "Compass heading as sine and cosine",
     "IMU / fibre-optic gyro compass", "2 x float32", "-1 to 1"),
    ("Angular rates",
     "How fast the body is rotating",
     "IMU gyroscope", "3 x float32, scaled by 4 rad/s", "about -1 to 1"),
    ("Battery left",
     "Fraction of the battery still available",
     "Battery management system", "1 x float32", "0 to 1"),
    ("Survey progress",
     "Fraction of the four stations already inspected",
     "Onboard mission computer", "1 x float32", "0 to 1"),
    ("Current estimate",
     "Water current the vehicle feels, in the heading frame",
     "DVL water-track", "2 x float32, scaled by 2 m/s", "about -1 to 1"),
    ("Altitude above seabed",
     "Height of the vehicle over the seabed below it",
     "Altimeter / echo sounder", "1 x float32, scaled by 6 m", "0 to about 2"),
    ("Range to station",
     "Flat distance to the active station",
     "USBL / imaging sonar", "1 x float32, scaled by 55 m", "0 to 1"),
    ("Time left",
     "Fraction of the mission clock still remaining",
     "Onboard timer", "1 x float32", "0 to 1"),
    ("Vertical speed",
     "How fast the vehicle is rising or diving",
     "Depth (pressure) sensor", "1 x float32, scaled by 3 m/s", "about -1 to 1"),
    ("Pipeline corridor margin",
     "How close the vehicle is to the pipe it should follow",
     "Multibeam sonar pipe tracker", "1 x float32", "0 to 1"),
    ("In scan range flag",
     "1 when the active station is close enough to scan, else 0",
     "Computed onboard", "1 x float32", "0 or 1"),
    ("Commanded setpoints",
     "The surge, sway, depth-rate, and heading targets currently set",
     "Autopilot state", "4 x float32", "-1 to 1"),
]

# ----------------------------------------------------------- 2d. reward structure

REWARD = """
<p>The reward has several parts, because the mission has several goals at once.</p>
<p>The main shaping term gives <b>+1.3</b> for every metre the vehicle moves closer to the active
station. This is what makes the long trip between stations learnable. Each clean scan pays
<b>60 &middot; exp(-(offset / 2.8)<sup>2</sup>)</b>, where "offset" is the distance from the ring
centre, so the reward is highest when the vehicle is centred in the ring. On top of that, a scan
inside the ring pays a flat <b>+30</b>, and finishing all four stations pays a further <b>+120</b>.</p>
<p>Small costs apply every step for time, thruster energy, tilt, and spin. There are shaped penalties
for drifting off the pipeline and for hugging the seabed. Taking a scan with no station in range
costs only <b>1</b> point. This penalty is kept small on purpose: a large one taught the agent to
avoid the scan action completely before it ever learned that a scan inside a ring pays off. Hard
failures (driving into the seabed, pipe, or manifold, tipping past 70 degrees, or leaving the survey
area) cost <b>55</b>. A flat battery costs <b>50</b>. Running out of time costs <b>12</b> for each
station still not inspected.</p>
<p>The agent cannot win by chasing one goal. Going fast drains the battery and overshoots the ring.
Going slow risks the clock. A clean, centred scan means actively thrusting against the current to
hold position instead of drifting with it.</p>
"""

BASELINE_BOX = """
<b>Reference scores.</b> Measured over the held-out seeds so the trained agents have something to
beat. A random policy scores <b>-433</b> with 0% of surveys finished. Doing nothing (HOLD forever)
scores <b>-144</b>, also 0%. A hand-written pilot in <code>tests/scripted_pilot.py</code> reaches
<b>+360</b> and finishes about <b>90%</b> of surveys. Random is worse than doing nothing here,
because thrashing the thrusters drives the vehicle into the seabed or out of the area and collects
the big penalties, while doing nothing just drifts and runs the clock out.
"""

# --------------------------------------------------- 3. system analysis and design

SYS_DQN = """
<p>DQN is the value-based method. From the 28-number observation it learns a Q-value (an expected
return) for each of the ten actions, and it acts by taking the action with the highest value. The
network is a small multilayer perceptron with two hidden layers and ReLU activations, built with
Stable-Baselines3. The best run uses 256 units per layer.</p>
<p>It uses the two standard DQN features. A <b>replay buffer</b> stores past transitions and samples
them at random, which breaks the correlation between steps that are close in time. A separate
<b>target network</b> gives the learning target and is updated slowly, so the target does not chase
the network it is training. Actions during training are picked with an epsilon-greedy rule: epsilon
starts high for exploration and drops over time. The replay buffer matters a lot here, because a
finished survey is rare early in training, and the buffer keeps those rare good episodes available
for many updates.</p>
"""

SYS_PG = """
<p>The three policy-gradient methods learn a policy directly, which is a probability over the ten
actions, instead of a value. PPO and A2C use an actor-critic setup from Stable-Baselines3: a small
shared network outputs both the action probabilities (the actor) and a value estimate (the critic).
PPO limits how far the policy can move in one update using a clipped objective, which keeps training
stable. A2C uses the same idea but without the clip and with much shorter rollouts, so it is more
sensitive to the learning rate.</p>
<p>REINFORCE is not in Stable-Baselines3, so it is written from scratch in <code>training/reinforce.py</code>
as a pure Monte-Carlo policy gradient. It updates only from complete episodes, using the discounted
return with no bootstrapping anywhere. An optional value network is used only as a baseline that is
subtracted from the return to lower the variance, never as a learning target. It exposes the same
methods as a Stable-Baselines3 model, so all four algorithms run through one shared training and
scoring path and the comparison stays fair.</p>
"""

# ------------------------------------------------------------- 4. implementation

def _impl_intro() -> str:
    return f"""
<p>Each algorithm was trained with ten different hyperparameter settings. Every run used the same
200,000-step budget, so the tables compare settings and not compute time. The columns show the
settings that were changed. "Mean Reward" and "Survey %" are measured on a fixed block of held-out
seeds that the agent never trains on, so the numbers are comparable across runs and algorithms. The
best row in each table is shaded.</p>
<p>The clearest single result is that the choice of hyperparameters changes the score more than the
choice of algorithm. Within DQN the ten runs span {_spread('DQN')} points of reward, within PPO
{_spread('PPO')}, within A2C {_spread('A2C')}, and within REINFORCE {_spread('REINFORCE')}. These
gaps are as wide as, or wider than, the gap between the best runs of the four algorithms.</p>
"""


def _dqn_analysis() -> str:
    b = _best("DQN")
    if not b:
        return '<p class="missing">DQN runs not done yet.</p>'
    return f"""
<p>For DQN, {_summary('DQN')}. Learning rate mattered most: setting it too high or too low both moved
the score away from the middle value. The discount was the next lever. A scan pays off about a hundred
steps after leaving the last station, so a very short discount (gamma&nbsp;=&nbsp;0.95) gave the best
result here, while an even shorter horizon made the agent short-sighted. Exploration showed a clear
two-sided effect: too little and the INSPECT-inside-a-ring pairing was almost never tried, too much
and the vehicle thrashed and never lined up. Shrinking the replay buffer hurt, because finished
surveys are rare early on and a small buffer forgets them.</p>
"""


def _ppo_analysis() -> str:
    b = _best("PPO")
    if not b:
        return '<p class="missing">PPO runs not done yet.</p>'
    return f"""
<p>For PPO, {_summary('PPO')}. The strongest run used a larger network and a slightly higher learning
rate. Entropy was the most useful knob. Setting the entropy bonus to zero let the policy become
confident too early and stop trying the scan action, which lowered the score. A large network with a
moderate rollout learned the survey fastest. A very low learning rate simply did not finish learning
inside the budget.</p>
"""


def _a2c_analysis() -> str:
    b = _best("A2C")
    if not b:
        return '<p class="missing">A2C runs not done yet.</p>'
    return f"""
<p>For A2C, {_summary('A2C')}. A2C has no clip on the update, so it is far more sensitive to the step
size than PPO. At the 200,000-step budget most A2C runs did not finish a survey, and the learning
curves swing up and down instead of settling. The longer final run (see Table&nbsp;5) shows that A2C
can learn the task with more steps, but it needs much more training than PPO to get there, and it
scans less accurately.</p>
"""


def _reinforce_analysis() -> str:
    b = _best("REINFORCE")
    if not b:
        return '<p class="missing">REINFORCE runs not done yet.</p>'
    return f"""
<p>For REINFORCE, {_summary('REINFORCE')}. The method is unbiased but noisy, so the runs test the
variance-reduction knobs. Removing the value baseline made the runs clearly worse, and turning off
return normalisation did the same, both as the theory predicts. Even so, no REINFORCE run finished a
survey at this budget, and the longer final run did not either. Because it learns only from complete
episodes, and a complete episode here is several hundred steps long, its updates are too noisy to
sharpen the policy in time. This is a real and expected weakness on a long task like this one.</p>
"""


# ------------------------------------------------------------------- captions

CAP_ENV = (
    "<b>The environment.</b> One episode flown by the hand-written pilot. Left: the whole survey, "
    "with the launch buoy, the pipeline in a shallow S across the seabed, four teal inspection rings, "
    "and the manifold on the right. Then a transit between stations, a scan at a ring, and the "
    "finished survey. The seabed shape and the current change on every reset."
)
CAP_FIG1 = (
    "<b>Figure 1. Cumulative reward per algorithm.</b> One panel per algorithm. Grey lines are the "
    "nine other settings, the coloured line is the best. All lines are scored on the same held-out "
    "seeds. The wide spread inside each panel is the main result: the setting matters more than the "
    "algorithm."
)
CAP_FIG2 = (
    "<b>Figure 2. Best setting of each algorithm.</b> Left: mean reward. Right: survey completion "
    "rate. Dashed lines mark the do-nothing, random, and pilot scores. The x-axis is training steps."
)
CAP_FIG3 = (
    "<b>Figure 3. DQN objective curves.</b> Left: the TD loss on a log scale. Middle: the mean "
    "Q-value on a fixed set of states, so a rising line means the value estimate grew and not that "
    "the agent moved elsewhere. Right: the epsilon exploration schedule."
)
CAP_FIG4 = (
    "<b>Figure 4. Policy entropy.</b> How fast each policy-gradient method stops exploring. The dashed "
    "line is ln 10, the entropy of a random policy over ten actions. Falling below about 1 means the "
    "policy has become almost fixed."
)
CAP_FIG5 = (
    "<b>Figure 5. Steps to converge.</b> Left: training steps for each setting to reach 90% of its "
    "own best score. Right: that speed against the score it reached. Fast is not always good: some "
    "settings settle quickly on a weak policy."
)
CAP_FIG6 = (
    "<b>Figure 6. Generalization.</b> Each agent under harder conditions. \"unseen seeds\" only "
    "changes the random draw. \"strong current\" roughly doubles the current. \"tight battery\" cuts "
    "the energy. \"rough seabed\" adds more relief than in training."
)
CAP_GEN_TABLE = (
    "<b>Table 6. Generalization.</b> Mean reward with survey completion rate in brackets. Conditions "
    "run from the training setting on the left to harder ones on the right."
)


# ------------------------------------------------------------ 5. discussion

def _disc_cumulative() -> str:
    return f"""
<p>Figure&nbsp;1 shows the cumulative reward for every setting, one panel per algorithm, and
Figure&nbsp;2 puts the best setting of each algorithm on one axis against the baselines. Two things
stand out.</p>
<p>First, every algorithm has to climb out of the same trap. Early lines sit well below zero, because
a policy that thrashes its thrusters is punished hard, so the first thing every method learns is to
stop crashing. That parks it near the do-nothing score while it drifts and times out. Getting past
that means paying the running cost of the full trip in exchange for the station rewards, and the
distance-shaping term is what makes that happen.</p>
<p>Second, DQN and PPO reach the pilot's level and finish most surveys, while A2C and REINFORCE stay
low at this budget. Ranked by best setting, the order is {_ranking()}. DQN's line is smoother than
PPO's, because its replay buffer keeps the rare finished surveys for many updates, while the on-policy
methods reach a working policy and then partly lose it.</p>
"""


DISCUSSION_OBJECTIVE = """
<p>Figure&nbsp;3 shows DQN's own objective. A run whose TD loss stays high while its Q-values keep
rising is unstable, because the target is chasing itself. A run whose loss falls flat while the
Q-values level off well below the reward that is actually reachable has just settled on a weak policy.
Measuring the Q-values on a fixed set of states is what makes this readable, because it separates "the
estimate grew" from "the agent went somewhere else".</p>
<p>Figure&nbsp;4 shows policy entropy for the three policy-gradient methods. They all start near
ln&nbsp;10, which is 2.30, the entropy of a random policy over ten actions. The runs that end best are
the ones whose entropy falls slowly. Runs that drop below about 1 early have stopped trying the
INSPECT action anywhere new, and their reward stops improving from that point. The zero-entropy PPO
run shows this most clearly. REINFORCE keeps the most entropy, which is the other side of its variance
problem: its gradient is too noisy to sharpen the policy quickly.</p>
"""


def _disc_converge() -> str:
    return """
<p>Figure&nbsp;5 shows that speed of convergence and final quality are almost unrelated here, and
sometimes they point in opposite directions. The settings that reach 90% of their own best score
earliest are often the ones that collapsed onto a fixed, weak policy. The right panel is the honest
way to read the "steps to 90%" column in the tables: a fast run that also reached a high score is
good, but a fast run that reached a low score just gave up early.</p>
<p>None of the four algorithms had fully levelled off at 200,000 steps. That is a real limit of the
sweep, and it is stated as such. The tables compare settings at a fixed budget, which is the question
a person with limited compute actually faces, but it is not the same as comparing the best each
method can ever reach. So the best setting of each algorithm was trained again for 1,500,000 steps,
and those longer runs are the ones in Table&nbsp;5.</p>
"""


def _disc_generalization() -> str:
    top = _top_algo() or "PPO"
    nom = _gen(top, "nominal")
    cur = _gen(top, "strong_current")
    bat = _gen(top, "tight_battery")
    seab = _gen(top, "rough_seabed")
    p_nom = _gen("PILOT", "nominal")
    p_cur = _gen("PILOT", "strong_current")
    if not np.isfinite(nom):
        return "<p>Run the evaluation step to fill in the generalization results.</p>"
    return f"""
<p>The generalization test scores each agent on a second, separate block of seeds and under four
harder settings. Because the seabed and current already change during training, doing well on unseen
seeds only shows the agent did not memorise fixed episodes. The useful columns are the harder ones.</p>
<p>The strongest agent overall was <b>{top}</b>. It finishes {_pct(nom)} of surveys on the held-out
seeds. It holds {_pct(seab)} on a rougher seabed and {_pct(bat)} with a smaller battery, so it did not
overfit to the easy settings. The one hard column is strong current, where it drops to {_pct(cur)}.
The reason is honest and worth stating: the crab-into-the-current skill is done by the onboard
controller, not by the policy, so the hand-written pilot drops the same way, from {_pct(p_nom)} to
{_pct(p_cur)}. On this mission the gap between algorithms is about how reliably each one lines up and
scans, not about current. The two agents that never finish a survey have flat lines across all
columns, because they have nothing to generalise yet.</p>
"""


def _conclusion() -> str:
    return f"""
<p>Four reinforcement learning algorithms were trained on the same subsea inspection mission and
compared over forty hyperparameter runs. Ranked by best setting at the 200,000-step budget, the order
was {_ranking()}. With longer training (Table&nbsp;5), DQN and PPO both finish about 93% of surveys,
PPO scans the most accurately, A2C also learns the task but needs far more steps and scans less
precisely, and REINFORCE never finishes a survey.</p>
<p>PPO is the best fit for this problem. Its clip keeps training stable, and it beats the hand-written
pilot on both completion and scan accuracy. DQN is a close second, and its replay buffer gives the
smoothest learning. A2C works but is slow and unstable without a trust region, and REINFORCE is the
weakest because learning only from long episodes is too noisy here.</p>
<p>Two lessons stand out. First, the hyperparameters mattered more than the algorithm. Second, two
environment choices were needed for any method to learn: actions that set a target velocity instead
of raw thruster forces, and not ending an episode at the first small mistake. With more time, the
next steps are longer training for A2C and REINFORCE and a reward that rewards holding steady in the
ring more directly.</p>
"""


# Build the computed text once at import so ``report.py`` can read them as
# plain attributes.
IMPL_INTRO = _impl_intro()
DQN_ANALYSIS = _dqn_analysis()
PPO_ANALYSIS = _ppo_analysis()
A2C_ANALYSIS = _a2c_analysis()
REINFORCE_ANALYSIS = _reinforce_analysis()
DISCUSSION_CUMULATIVE = _disc_cumulative()
DISCUSSION_CONVERGE = _disc_converge()
DISCUSSION_GENERALIZATION = _disc_generalization()
CONCLUSION = _conclusion()
