"""DQN sweep for the blood-delivery mission.

Ten configurations, each varying hyperparameters that matter for a *value-based*
method on this problem specifically:

``learning_rate``
    The Q-target moves as the policy improves, so too large a step makes the
    bootstrapped target chase itself.
``buffer_size`` / ``learning_starts``
    Episodes here end in one of seven very different terminal states. A small
    buffer forgets the rare successful deliveries and the agent regresses to
    dumping the payload early.
``exploration_fraction`` / ``exploration_final_eps``
    Delivery needs a specific action (RELEASE_PAYLOAD) at a specific place. Too
    little exploration and it is never discovered; too much and the aircraft
    never survives long enough to reach the zone.
``gamma``
    The delivery bonus arrives ~150 steps after take-off, so the discount has to
    be long enough to propagate it back to the climb-out.
``target_update_interval`` / ``tau``
    Directly controls how stale the bootstrap target is, i.e. stability.
``train_freq`` / ``gradient_steps`` / ``batch_size``
    Replay reuse versus wall-clock.

Run ``uv run python -m training.dqn_training`` for the whole sweep, or
``--runs D01 D04`` for a subset.
"""

from __future__ import annotations

import argparse

from stable_baselines3 import DQN

from training.common import RunSpec, print_header, run_sb3

#: Timesteps per sweep run. Long enough to separate the configurations, short
#: enough that forty runs fit in a sitting; the best configuration is then
#: retrained for longer in ``final``.
SWEEP_STEPS = 300_000
FINAL_STEPS = 1_500_000
N_ENVS = 4


def sweep_specs(total_timesteps: int = SWEEP_STEPS) -> list[RunSpec]:
    """The ten DQN configurations reported in the hyperparameter table."""
    base = dict(
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=128,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        tau=1.0,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
    )

    def spec(run_id: str, net_arch=(256, 256), **overrides) -> RunSpec:
        return RunSpec(
            run_id=run_id,
            algo="DQN",
            params={**base, **overrides},
            net_arch=net_arch,
            total_timesteps=total_timesteps,
            n_envs=N_ENVS,
        )

    return [
        # --- baseline -------------------------------------------------------
        spec("D01"),
        # --- learning rate ---------------------------------------------------
        spec("D02", learning_rate=1e-3),
        spec("D03", learning_rate=5e-5),
        # --- discount --------------------------------------------------------
        spec("D04", gamma=0.95),
        spec("D05", gamma=0.999),
        # --- exploration schedule --------------------------------------------
        spec("D06", exploration_fraction=0.1, exploration_final_eps=0.02),
        spec("D07", exploration_fraction=0.7, exploration_final_eps=0.15),
        # --- replay capacity and target staleness -----------------------------
        spec("D08", buffer_size=50_000, learning_starts=5_000),
        spec("D09", target_update_interval=500, tau=0.01),
        # --- capacity and replay reuse ----------------------------------------
        spec(
            "D10",
            net_arch=(400, 300),
            batch_size=256,
            gradient_steps=2,
            learning_rate=5e-4,
        ),
    ]


def final_spec() -> RunSpec:
    """Longer run used for the demo policy; filled in from the sweep winner."""
    from training.common import read_results

    rows = read_results("DQN")
    best_id = "D01"
    if rows:
        best = max(rows, key=lambda r: float(r.get("mean_return", "-inf")))
        best_id = best["run_id"]
    spec = next(s for s in sweep_specs(FINAL_STEPS) if s.run_id == best_id)
    spec.run_id = "final"
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN hyperparameter sweep")
    parser.add_argument("--runs", nargs="*", help="subset of run ids, e.g. D01 D04")
    parser.add_argument("--timesteps", type=int, default=SWEEP_STEPS)
    parser.add_argument("--final", action="store_true", help="retrain the sweep winner")
    args = parser.parse_args()

    if args.final:
        spec = final_spec()
        print_header(f"DQN final run ({spec.total_timesteps:,} steps)")
        run_sb3(spec, DQN)
        return

    specs = sweep_specs(args.timesteps)
    if args.runs:
        wanted = {r.upper() for r in args.runs}
        specs = [s for s in specs if s.run_id.upper() in wanted]

    print_header(f"DQN sweep: {len(specs)} runs x {args.timesteps:,} steps")
    for i, spec in enumerate(specs, 1):
        changed = {
            k: v for k, v in spec.params.items() if v != sweep_specs()[0].params.get(k)
        }
        print(f"\n[{i}/{len(specs)}] {spec.run_id}  {changed or 'baseline'}  net={spec.net_arch}")
        run_sb3(spec, DQN)


if __name__ == "__main__":
    main()
