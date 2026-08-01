"""Policy-gradient sweeps: PPO, A2C and REINFORCE.

Ten configurations each. The hyperparameters chosen for each algorithm are the
ones that actually bite on this mission:

PPO
    ``n_steps``/``batch_size`` set how much on-policy data backs each update, and
    a completed station only shows up in the data every ~100+ steps. ``clip_range``
    and ``n_epochs`` control how far the policy may move on that data before it
    goes stale. ``gae_lambda`` trades bias against variance on a reward that mixes
    dense progress shaping with sparse station bonuses. ``ent_coef`` decides
    whether the INSPECT action keeps getting tried at the hoops.
A2C
    No clipping and a much shorter rollout, so it is far more sensitive to
    ``learning_rate`` and to ``normalize_advantage``; ``use_rms_prop`` and
    ``vf_coef`` are the other two levers that visibly change stability here.
REINFORCE
    Unbiased but high-variance. ``use_baseline``, ``episodes_per_update`` and
    ``normalize_returns`` are precisely the variance-reduction knobs, so the
    sweep is built to expose how much each one is worth.

Run ``uv run python -m training.pg_training`` for everything, or
``--algo ppo`` / ``--runs P01 A03`` for a subset.
"""

from __future__ import annotations

import argparse

from stable_baselines3 import A2C, PPO

from training.common import RunSpec, print_header, run_reinforce, run_sb3

SWEEP_STEPS = 200_000
FINAL_STEPS = 800_000
N_ENVS = 8


# ------------------------------------------------------------------------- PPO


def ppo_specs(total_timesteps: int = SWEEP_STEPS) -> list[RunSpec]:
    base = dict(
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    def spec(run_id: str, net_arch=(128, 128), **overrides) -> RunSpec:
        return RunSpec(
            run_id=run_id,
            algo="PPO",
            params={**base, **overrides},
            net_arch=net_arch,
            total_timesteps=total_timesteps,
            n_envs=N_ENVS,
        )

    return [
        spec("P01"),
        spec("P02", learning_rate=1e-3),
        spec("P03", learning_rate=1e-4),
        spec("P04", gamma=0.95),
        spec("P05", ent_coef=0.0),
        spec("P06", ent_coef=0.05),
        spec("P07", clip_range=0.1, n_epochs=20),
        spec("P08", n_steps=2048, batch_size=512),
        spec("P09", gae_lambda=0.8),
        spec("P10", net_arch=(256, 256), learning_rate=5e-4, n_steps=1024, batch_size=256),
    ]


# ------------------------------------------------------------------------- A2C


def a2c_specs(total_timesteps: int = SWEEP_STEPS) -> list[RunSpec]:
    base = dict(
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.995,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        normalize_advantage=False,
    )

    def spec(run_id: str, net_arch=(128, 128), **overrides) -> RunSpec:
        return RunSpec(
            run_id=run_id,
            algo="A2C",
            params={**base, **overrides},
            net_arch=net_arch,
            total_timesteps=total_timesteps,
            n_envs=N_ENVS,
        )

    return [
        spec("A01"),
        spec("A02", learning_rate=3e-4),
        spec("A03", learning_rate=2e-3),
        spec("A04", n_steps=64),
        spec("A05", n_steps=8),
        spec("A06", gamma=0.95),
        spec("A07", ent_coef=0.0),
        spec("A08", ent_coef=0.05),
        spec("A09", normalize_advantage=True, gae_lambda=0.95),
        spec("A10", net_arch=(256, 256), use_rms_prop=False, learning_rate=3e-4, n_steps=32),
    ]


# ------------------------------------------------------------------- REINFORCE


def reinforce_specs(total_timesteps: int = SWEEP_STEPS) -> list[RunSpec]:
    base = dict(
        learning_rate=3e-4,
        gamma=0.995,
        episodes_per_update=16,
        use_baseline=True,
        normalize_returns=True,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    def spec(run_id: str, net_arch=(128, 128), **overrides) -> RunSpec:
        return RunSpec(
            run_id=run_id,
            algo="REINFORCE",
            params={**base, **overrides},
            net_arch=net_arch,
            total_timesteps=total_timesteps,
            n_envs=N_ENVS,
        )

    return [
        spec("R01"),
        spec("R02", learning_rate=1e-3),
        spec("R03", learning_rate=1e-4),
        spec("R04", use_baseline=False),
        spec("R05", normalize_returns=False),
        spec("R06", episodes_per_update=8),
        spec("R07", episodes_per_update=48),
        spec("R08", gamma=0.95),
        spec("R09", ent_coef=0.05),
        spec("R10", net_arch=(256, 256), learning_rate=5e-4, episodes_per_update=32),
    ]


ALGOS = {
    "ppo": (ppo_specs, lambda s: run_sb3(s, PPO)),
    "a2c": (a2c_specs, lambda s: run_sb3(s, A2C)),
    "reinforce": (reinforce_specs, run_reinforce),
}


def final_spec(algo: str) -> RunSpec:
    from training.common import read_results

    specs_fn, _ = ALGOS[algo]
    rows = read_results(algo)
    specs = specs_fn(FINAL_STEPS)
    best_id = specs[0].run_id
    if rows:
        best_id = max(rows, key=lambda r: float(r.get("mean_return", "-inf")))["run_id"]
    spec = next(s for s in specs if s.run_id == best_id)
    spec.run_id = "final"
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-gradient hyperparameter sweeps")
    parser.add_argument("--algo", nargs="*", choices=sorted(ALGOS), default=sorted(ALGOS))
    parser.add_argument("--runs", nargs="*", help="subset of run ids, e.g. P01 A03")
    parser.add_argument("--timesteps", type=int, default=SWEEP_STEPS)
    parser.add_argument("--final", action="store_true", help="retrain each sweep winner")
    args = parser.parse_args()

    for algo in args.algo:
        specs_fn, runner = ALGOS[algo]
        if args.final:
            spec = final_spec(algo)
            print_header(f"{algo.upper()} final run ({spec.total_timesteps:,} steps)")
            runner(spec)
            continue

        specs = specs_fn(args.timesteps)
        if args.runs:
            wanted = {r.upper() for r in args.runs}
            specs = [s for s in specs if s.run_id.upper() in wanted]
        if not specs:
            continue

        baseline = specs_fn()[0].params
        print_header(f"{algo.upper()} sweep: {len(specs)} runs x {args.timesteps:,} steps")
        for i, spec in enumerate(specs, 1):
            changed = {k: v for k, v in spec.params.items() if v != baseline.get(k)}
            print(
                f"\n[{i}/{len(specs)}] {spec.run_id}  {changed or 'baseline'}"
                f"  net={spec.net_arch}"
            )
            runner(spec)


if __name__ == "__main__":
    main()
