"""Score every trained agent, and test whether any of it generalises.

    uv run main.py evaluate

Three things happen here:

1. every trained policy is scored on the held-out evaluation seeds,
2. the same policies are re-scored under conditions they were never trained on,
3. the terminal-state breakdown is written out for the outcome figure.

The generalisation conditions are deliberately different in kind, not just in
seed. ``unseen_seeds`` keeps the training distribution and only changes the
random draw. ``strong_current`` roughly doubles the current the agent ever saw,
which tests whether it learned to hold station or merely memorised a trajectory.
``tight_battery`` and ``rough_seabed`` squeeze the energy budget and roughen the
terrain. An agent that only holds up on the first condition has overfitted to
the training distribution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from environment.custom_env import EnvConfig
from training.common import (
    EVAL_SEEDS,
    GENERALIZATION_SEEDS,
    RESULTS_DIR,
    available_agents,
    evaluate_policy,
    load_agent,
    write_csv,
)

ROOT = Path(__file__).resolve().parent.parent


def conditions() -> dict[str, tuple[EnvConfig, tuple[int, ...]]]:
    """The evaluation conditions, from nominal to deliberately hostile.

    The conditions are different in kind, not just in seed. ``unseen_seeds``
    keeps the training distribution and only changes the random draw.
    ``strong_current`` roughly doubles the water current the agent ever trained
    against, which tests whether it learned to hold station or merely memorised a
    trajectory. ``tight_battery`` and ``rough_seabed`` squeeze the energy budget
    and roughen the terrain the vehicle has to stay clear of.
    """
    nominal = EnvConfig()
    return {
        # the distribution the agent was trained on, on its held-out seeds
        "nominal": (nominal, EVAL_SEEDS),
        # same distribution, a completely disjoint block of seeds
        "unseen_seeds": (nominal, GENERALIZATION_SEEDS),
        # current well outside anything seen in training
        "strong_current": (
            replace(nominal, current_mean_max=1.8, current_sigma=0.8),
            GENERALIZATION_SEEDS,
        ),
        # less energy than the mission was tuned for
        "tight_battery": (
            replace(nominal, battery_endurance_s=55.0),
            GENERALIZATION_SEEDS,
        ),
        # a rougher, more cluttered seabed than in training
        "rough_seabed": (
            replace(nominal, n_ridges=12),
            GENERALIZATION_SEEDS,
        ),
    }


def _pilot_predict():
    """The hand-written pilot, wrapped to look like a policy over observations."""
    from environment.custom_env import SubseaInspectionEnv
    from tests.scripted_pilot import scripted_policy

    holder: dict[str, SubseaInspectionEnv] = {}

    def bind(env: SubseaInspectionEnv):
        holder["env"] = env
        return lambda _obs: scripted_policy(holder["env"])

    return bind


def main(args=None) -> None:
    episodes = getattr(args, "episodes", 30)
    agents = available_agents()

    print(f"\n{'=' * 74}\n EVALUATION\n{'=' * 74}")
    if not agents:
        print("  No trained models found in models/. Train one first:")
        print("      uv run main.py train --algo ppo --final")
        print("  Scoring the reference pilot only.\n")

    conds = conditions()
    gen_rows: list[dict] = []
    outcome_rows: list[dict] = []

    # ---------------------------------------------------------------- agents
    for algo in sorted(agents):
        try:
            predict, label = load_agent(algo)
        except Exception as exc:
            print(f"  [skip] {algo}: {exc}")
            continue

        print(f"\n  {label}")
        for cond_name, (cfg, seeds) in conds.items():
            res = evaluate_policy(predict, seeds=seeds[:episodes], config=cfg)
            gen_rows.append(
                {
                    "agent": label,
                    "algo": algo.upper(),
                    "condition": cond_name,
                    **res.as_row(),
                }
            )
            print(
                f"    {cond_name:<16} return {res.mean_return:8.2f}"
                f"   survey {100 * res.success_rate:5.1f}%"
                f"   scan-off {res.mean_miss:5.2f} m"
            )
            if cond_name == "nominal":
                total = max(1, sum(res.outcomes.values()))
                outcome_rows.append(
                    {
                        "agent": algo.upper(),
                        **{k: round(v / total, 4) for k, v in res.outcomes.items()},
                    }
                )

    # ------------------------------------------------------- reference pilot
    from environment.custom_env import SubseaInspectionEnv
    from tests.scripted_pilot import scripted_policy

    print("\n  hand-written pilot (reference)")
    for cond_name, (cfg, seeds) in conds.items():
        env = SubseaInspectionEnv(config=cfg)
        res = evaluate_policy(lambda _o: scripted_policy(env), seeds=seeds[:episodes], env=env)
        env.close()
        gen_rows.append(
            {
                "agent": "hand-written pilot",
                "algo": "PILOT",
                "condition": cond_name,
                **res.as_row(),
            }
        )
        print(
            f"    {cond_name:<16} return {res.mean_return:8.2f}"
            f"   survey {100 * res.success_rate:5.1f}%"
            f"   scan-off {res.mean_miss:5.2f} m"
        )
        if cond_name == "nominal":
            total = max(1, sum(res.outcomes.values()))
            outcome_rows.append(
                {
                    "agent": "PILOT",
                    **{k: round(v / total, 4) for k, v in res.outcomes.items()},
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / "generalization.csv", gen_rows)
    write_csv(RESULTS_DIR / "outcomes.csv", outcome_rows)
    print(f"\n  wrote {RESULTS_DIR / 'generalization.csv'}")
    print(f"  wrote {RESULTS_DIR / 'outcomes.csv'}\n")


if __name__ == "__main__":
    main()
