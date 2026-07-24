# Blood-Delivery UAV — Mission-Based Reinforcement Learning

A cargo quadrotor has to carry a cold-chain blood pack from a distribution centre,
across a range of Rwandan hills, and put it inside a 3 m drop zone at a rural health
post — before the blood spoils, before the battery runs flat, without leaving the
regulated flight corridor, and without flying into a hillside.

Four RL algorithms are trained on that mission and compared: **DQN** (value-based),
and **REINFORCE**, **PPO** and **A2C** (policy-gradient).

Simulated in **MuJoCo** with real rigid-body physics, a four-rotor airframe, a
procedurally generated heightfield, stochastic wind, and a payload that is genuinely
released and genuinely falls under a parachute.

![The mission](assets/figures/env_overview.png)

---

## Quickstart

```bash
git clone <repo-url> && cd nick-lemy_kayiranga_rl_summative
uv sync
uv run main.py
```

`uv sync` creates the environment and installs everything (Python is pinned to 3.11
and torch to the CPU wheels, so this stays a few hundred MB rather than pulling the
whole CUDA stack). `uv run main.py` opens the interactive 3D simulation flying the
best trained agent.

No other setup is needed — no manual venv, no `pip install`.

### Everything else

| Command | What it does |
|---|---|
| `uv run main.py` | Live 3D simulation with the best trained agent, in real time |
| `uv run main.py env-info` | Print the full action / observation / reward specification |
| `uv run main.py play --algo ppo --render` | Verbose rollout: 3D viewer **and** step-by-step terminal telemetry |
| `uv run main.py evaluate` | Score every agent, including the generalisation tests |
| `uv run main.py plots` | Regenerate every figure in the report |
| `uv run main.py video --algo ppo` | Record an MP4 with a telemetry HUD |
| `uv run main.py export-trace` | Write a JSON episode for the browser viewer |
| `uv run main.py train --algo ppo` | Run the 10-configuration PPO sweep |
| `uv run main.py train --algo ppo --final` | Retrain the sweep winner for longer |
| `uv run pytest -q` | Environment contract tests |

`uv run python play.py --algo dqn --render` works directly too.

The viewer plays back at **real time** by default: one second of simulated flight
takes one second of wall clock, so a delivery runs about 15-20 s. `--speed 0.5`
gives slow motion and `--speed 3` skips ahead, on both `demo` and `play`.

---

## The environment

### Why it is not a grid world

The nine directional actions do **not** move the aircraft. They nudge the setpoints of
an onboard PD flight controller, which mixes them into four individual rotor thrusts;
lift, banking, translation, stalling and crashing are then produced by MuJoCo's
integrator. A bad sequence of actions produces a genuinely unrecoverable attitude.

Two design decisions came out of actually flying it (see `tests/scripted_pilot.py`):

- **Collective commands a climb rate, not raw thrust.** Raw thrust needs faster
  corrections than one action per 100 ms can supply, and the aircraft simply falls out
  of the sky. Commanding vertical speed and letting an inner loop find the thrust is
  the standard altitude-hold mode of a real cargo UAV.
- **The commanded setpoints are part of the observation.** Because the actions edit
  persistent state, omitting them would make the problem non-Markovian.

### Action space — `Discrete(10)`

| # | Action | Real-world meaning |
|---|---|---|
| 0 | `HOVER` | level the wings, bleed the climb rate to zero |
| 1 | `THROTTLE_UP` | increase commanded climb rate |
| 2 | `THROTTLE_DOWN` | decrease commanded climb rate |
| 3 | `PITCH_FORWARD` | nose down → accelerate forward |
| 4 | `PITCH_BACK` | nose up → decelerate / back up |
| 5 | `ROLL_LEFT` | bank left → translate left |
| 6 | `ROLL_RIGHT` | bank right → translate right |
| 7 | `YAW_LEFT` | rotate heading to port |
| 8 | `YAW_RIGHT` | rotate heading to starboard |
| 9 | `RELEASE_PAYLOAD` | open the cargo bay — the mission-critical act |

### Observation space — `Box(27,)`

Position error and velocity in the yaw-aligned frame, body-frame gravity (attitude),
heading, body rates, battery, cold-chain time, wind estimate, payload-attached flag,
altitude above ground, range to the zone, clock remaining, vertical speed, corridor
margin, and the three commanded setpoints. Run `uv run main.py env-info` for the exact
index map.

### Reward

| Term | Value |
|---|---|
| Progress towards the release point | **+1.2** per metre closed |
| Accurate delivery | **+150 · exp(−(miss/3)²)** |
| Payload inside the 3 m zone | **+50** |
| Released outside the zone | **−30** |
| Impact above 6 m/s | **−4** per m/s |
| Time / energy / tilt / spin | −0.06 / −0.04 / −0.15 / −0.02 per step |
| Corridor and terrain proximity | shaped penalties |
| Crash, corridor breach, flat battery, spoiled blood | **−70** each |
| Ran out of time | **−15** |

### Start state and termination

Catapult launch from the depot pad with the payload attached and 85–100% battery.
Every reset draws a **new health-post position, new procedural terrain and a new wind
field**, which is what the generalisation test exploits.

Episodes end on `delivered`, `missed_zone`, `crash`, `loss_of_control`,
`corridor_breach`, `battery_depleted`, `cold_chain_expired`, or timeout.

### Reference scores

Measured over 40 held-out episodes, so the learned policies have something to be
compared against:

| Policy | Mean return | Delivered |
|---|---|---|
| Do nothing (`HOVER` forever) | −214.7 | 0% |
| Uniform random | −35.4 | 0% |
| Hand-written pilot (`tests/scripted_pilot.py`) | +142.5 | 70% |

Random beats doing nothing because it dumps the payload early and ends the episode —
a local optimum the learned agents have to climb out of.

---

## Browser replay viewer

Episodes serialise to a self-contained JSON document and replay in the browser, which
is what an operations dashboard would do against a `GET /episodes/:id` endpoint. The
page rebuilds the terrain, drop zone, corridor and both bodies from the JSON alone —
it knows nothing about MuJoCo.

```bash
uv run main.py export-trace          # writes viewer/episode.json
python3 -m http.server 8000 -d viewer
# open http://localhost:8000
```

Scrub, change playback speed, switch between chase / orbit / mission cameras, and
watch the reward, battery and cold-chain traces update per step. three.js is vendored
locally, so it works offline.

The schema is `zipline-rl-trace/1`: a header (terrain heightfield, drop target, zone
radius, corridor limits, action names) plus one record per control step with both body
poses, the action index, the step reward, the consumables and the wind vector.

---

## Repository layout

```
├── pyproject.toml            # uv project definition (CPU torch pinned)
├── uv.lock                   # locked dependency graph
├── main.py                   # entry point / CLI
├── play.py                   # run a trained agent with verbose telemetry
│
├── environment/
│   ├── custom_env.py         # the Gymnasium environment
│   └── rendering.py          # MuJoCo viewer, HUD renderer, MP4 recording
│
├── training/
│   ├── common.py             # shared env factory, evaluation protocol, sweep runner
│   ├── reinforce.py          # REINFORCE, written from scratch (SB3 has none)
│   ├── dqn_training.py       # 10-configuration DQN sweep
│   └── pg_training.py        # 10-configuration PPO / A2C / REINFORCE sweeps
│
├── analysis/
│   ├── evaluate.py           # scoring and the generalisation tests
│   ├── plots.py              # every figure in the report
│   └── style.py              # shared palette and figure styling
│
├── models/{dqn,pg}/          # trained policies
├── logs/                     # eval histories, SB3 scalars, results tables
├── assets/                   # MuJoCo scene, figures, recorded video
├── viewer/                   # three.js replay viewer
└── tests/                    # contract tests + the hand-written reference pilot
```

---

## Reproducing the results

```bash
uv run main.py train --algo ppo          # 10 configurations x 300k steps
uv run main.py train --algo a2c
uv run main.py train --algo reinforce
uv run main.py train --algo dqn

uv run main.py train --algo ppo --final  # retrain the winner for 1.5M steps
uv run main.py evaluate                  # scores + generalisation
uv run main.py plots                     # figures
```

Every configuration is scored on the **same held-out block of seeds**, never on
training reward, so the four algorithms stay comparable. A second, disjoint seed block
is reserved for the generalisation test, which also perturbs the wind, the energy
budget and the mission geometry beyond anything seen in training.

Results land in `logs/results/*_sweep.csv` (the hyperparameter tables),
`logs/results/generalization.csv`, and `assets/figures/`.

---

## Notes

- Training runs on CPU by design — the policies are small MLPs and the environment is
  the bottleneck. `torch.set_num_threads(1)` measured ~1.9× faster end-to-end than the
  default, because torch's thread pool otherwise fights the environment workers.
- The MuJoCo scene regenerates its heightfield on every reset; both renderers re-upload
  it to the GPU, or the render would show hills the physics is no longer using.

**Nick Lemy Kayiranga** — n.kayiranga@alustudent.com
