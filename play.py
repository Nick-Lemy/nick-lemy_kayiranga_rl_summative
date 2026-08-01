"""Run a trained agent and narrate what it is doing.

    uv run python play.py --algo ppo --render

Prints a step-by-step telemetry trace plus a per-episode summary, so the
terminal shows *why* the agent scored what it scored - which action it chose,
how much battery it had left, how many stations it had inspected, and how the
episode ended. ``--render`` opens the interactive 3D viewer alongside the trace,
which is the combination used in the demonstration video.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from environment.custom_env import SubseaInspectionEnv  # noqa: E402

# terminal colours, so successes and failures are distinguishable on video
_GREEN, _RED, _YELLOW, _CYAN, _DIM, _RESET = (
    "\033[92m",
    "\033[91m",
    "\033[93m",
    "\033[96m",
    "\033[2m",
    "\033[0m",
)

_GOOD_OUTCOMES = {"survey_complete"}


def _resolve_agent(algo: str | None):
    from training.common import best_available_agent, load_agent

    if algo:
        predict, label = load_agent(algo)
        return predict, label, None
    return best_available_agent()


def run(args) -> None:
    predict, label, fallback = _resolve_agent(getattr(args, "algo", None))

    print(f"\n{_CYAN}{'=' * 74}")
    print(f" AGENT: {label}")
    print(f"{'=' * 74}{_RESET}")
    if fallback is not None:
        print(f"{_YELLOW}  No trained model on disk - flying the reference pilot.{_RESET}")
        print("  Train one with:  uv run main.py train --algo ppo --final\n")

    render = getattr(args, "render", False)
    env = SubseaInspectionEnv(
        render_mode="human" if render else None,
        playback_speed=getattr(args, "speed", 1.0),
    )
    returns, outcomes, misses = [], [], []

    try:
        for episode in range(args.episodes):
            seed = (args.seed + episode) if args.seed is not None else None
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            action_counts: Counter[str] = Counter()
            info: dict = {}

            print(f"{_DIM}  --- episode {episode + 1}/{args.episodes}"
                  f"  (seed {seed if seed is not None else 'random'})"
                  f"  {env._info()['waypoints_total']} stations to inspect"
                  f"  current {np.linalg.norm(env.current_mean):.2f} m/s ---{_RESET}")

            prev_done = 0
            while not done:
                action = fallback(env) if fallback is not None else predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
                step += 1
                action_counts[info["action"]] += 1

                # call out each inspection so the terminal shows the scans
                if info["waypoints_done"] > prev_done:
                    prev_done = info["waypoints_done"]
                    print(f"    {_GREEN}*** SCAN CAPTURED at station {prev_done}"
                          f"/{info['waypoints_total']} ***{_RESET}")

                if step % 5 == 0:
                    scan = "SCANNING" if info["scanning"] else ("READY" if info["scan_ready"] else "")
                    print(
                        f"    t={info['mission_time']:5.1f}s"
                        f"  {info['action']:<12}"
                        f"  range {info['range_to_wp']:6.2f} m"
                        f"  bat {info['battery']:.2f}"
                        f"  stn {info['waypoints_done']}/{info['waypoints_total']}"
                        f"  {scan:<8}"
                        f"  R {total:8.2f}"
                    )
                if env._viewer is not None and not env._viewer.running:
                    done = True

            good = info["outcome"] in _GOOD_OUTCOMES
            colour = _GREEN if good else _RED
            miss = info["inspect_error"]
            print(
                f"  {colour}=> {info['outcome'].upper()}{_RESET}"
                + (f"   scan offset {miss:.2f} m" if not math.isnan(miss) else "")
                + f"   return {total:8.2f}"
                f"   time {info['mission_time']:.1f} s"
                f"   battery left {100 * info['battery']:.0f}%"
            )
            top = ", ".join(f"{a} x{n}" for a, n in action_counts.most_common(4))
            print(f"  {_DIM}most used actions: {top}{_RESET}\n")

            returns.append(total)
            outcomes.append(info["outcome"])
            if not math.isnan(miss):
                misses.append(miss)

            if env._viewer is not None and not env._viewer.running:
                break
    finally:
        env.close()

    if not returns:
        return
    completed = sum(o in _GOOD_OUTCOMES for o in outcomes)
    print(f"{_CYAN}{'=' * 74}")
    print(f" SUMMARY over {len(returns)} episodes - {label}")
    print(f"{'=' * 74}{_RESET}")
    print(f"  mean return       {np.mean(returns):8.2f}  (std {np.std(returns):.2f})")
    print(f"  survey complete   {100 * completed / len(returns):7.1f} %")
    if misses:
        print(f"  mean scan offset  {np.mean(misses):8.2f} m")
    print(f"  outcomes          {dict(Counter(outcomes))}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a trained subsea-inspection ROV agent.")
    p.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default=None,
                   help="which trained policy to fly (default: best available)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--render", action="store_true", help="open the interactive 3D viewer")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback rate when rendering, 1.0 = real time")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
