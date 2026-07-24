"""Entry point for the blood-delivery RL project.

    uv sync && uv run main.py

opens the interactive 3D simulation with the best trained agent available. Every
other task is a subcommand; run ``uv run main.py --help`` to list them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner(text: str) -> None:
    print(f"\n\033[1;36m{'=' * 74}\n {text}\n{'=' * 74}\033[0m")


# --------------------------------------------------------------------- commands


def cmd_env_info(args) -> None:
    """Print the environment's contract: actions, observation layout, rewards."""
    import mujoco

    from environment.custom_env import ACTION_MEANING, OBS_LAYOUT, ZiplineDeliveryEnv

    env = ZiplineDeliveryEnv()
    cfg = env.cfg
    integrator = mujoco.mjtIntegrator(env.model.opt.integrator).name.removeprefix("mjINT_")
    _banner("ENVIRONMENT: Zipline-style blood-delivery quadrotor")
    print(f"  observation space : {env.observation_space}")
    print(f"  action space      : {env.action_space}")
    print(f"  control rate      : {1 / env.dt:.0f} Hz   ({env.max_steps} steps max)")
    print(
        f"  physics timestep  : {env.model.opt.timestep * 1000:.0f} ms"
        f"  ({integrator.lower()} integrator)"
    )

    print("\n  ACTIONS")
    for idx, name in sorted(ACTION_MEANING.items()):
        print(f"    {idx:>2}  {name}")

    print("\n  OBSERVATION (27 values)")
    for idx, desc in OBS_LAYOUT:
        print(f"    {idx:>6}  {desc}")

    print("\n  REWARD")
    print(f"    progress towards the aim point       +{cfg.w_progress} per metre closed")
    print(
        f"    accurate delivery                    +{cfg.r_delivery}"
        f" * exp(-(miss/{cfg.zone_radius:.0f})^2)"
    )
    print(f"    payload inside the {cfg.zone_radius:.0f} m zone           +{cfg.r_in_zone}")
    print(
        f"    time / energy / tilt / spin          -{cfg.w_step} / -{cfg.w_energy}"
        f" / -{cfg.w_tilt} / -{cfg.w_spin} per step"
    )
    print(f"    corridor and terrain proximity       shaped, up to -{cfg.w_corridor} / -{cfg.w_terrain}")
    print(f"    released outside the zone            -{cfg.p_failed_drop}")
    print(f"    impact above {cfg.safe_impact_v:.0f} m/s                -{cfg.p_impact} per m/s")
    print(f"    crash / corridor breach              -{cfg.p_crash} / -{cfg.p_corridor_breach}")
    print(f"    flat battery / spoiled blood         -{cfg.p_battery} / -{cfg.p_cold_chain}")
    print(f"    ran out of time                      -{cfg.p_timeout}")

    print("\n  TERMINAL STATES")
    for name in (
        "delivered            payload landed inside the drop zone",
        "missed_zone          payload landed outside it",
        "crash                flew into terrain or scenery",
        "loss_of_control      tilted past 80 degrees",
        "corridor_breach      left the regulated airspace",
        "battery_depleted     flat battery before delivery",
        "cold_chain_expired   blood spoiled before delivery",
        "timeout              ran out of clock (truncation)",
    ):
        print(f"    {name}")

    print("\n  START STATE")
    print("    Catapult launch from the depot pad at x=-30 m with the payload attached and")
    print("    85-100% battery. Every reset draws a new health-post position")
    print(f"    (x in {cfg.post_x_range}, y in {cfg.post_y_range}), fresh procedural")
    print(f"    terrain, and a fresh wind field (up to {cfg.wind_mean_max:.0f} m/s steady + gusts).")
    env.close()


def cmd_demo(args) -> None:
    """Fly the best available agent in the interactive 3D viewer."""
    from environment.custom_env import ZiplineDeliveryEnv
    from training.common import best_available_agent

    predict, label, fallback = best_available_agent()
    _banner(f"LIVE SIMULATION - {label}")
    if fallback is not None:
        print("  No trained model found. Train one with:")
        print("      uv run main.py train --algo ppo --final")
        print("  Flying the hand-written reference pilot instead.\n")

    env = ZiplineDeliveryEnv(
        render_mode="human",
        record_trace=args.trace,
        playback_speed=args.speed,
    )
    print(f"  playing back at {args.speed}x real time"
          f"{'' if args.speed == 1.0 else '  (use --speed 1 for real time)'}\n")
    info: dict = {}
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode if args.seed is not None else None
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                action = fallback(env) if fallback is not None else predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
                step += 1
                if args.verbose and step % 5 == 0:
                    print(
                        f"  t={info['flight_time']:5.1f}s  {info['action']:<15}"
                        f"  range {info['range_to_zone']:6.2f} m"
                        f"  alt {info['altitude_agl']:5.1f} m"
                        f"  bat {info['battery']:.2f}"
                        f"  cold {info['cold_chain']:.2f}"
                        f"  R {total:8.2f}"
                    )
                if env._viewer is not None and not env._viewer.running:
                    done = True
            miss = info["miss_distance"]
            print(
                f"\n  episode {episode + 1}: {info['outcome'].upper()}"
                + (f"  miss {miss:.2f} m" if miss == miss else "")
                + f"  return {total:.2f}  flight {info['flight_time']:.1f} s\n"
            )
            if env._viewer is not None and not env._viewer.running:
                break
    finally:
        if args.trace:
            out = env.export_trace(ROOT / "logs" / "traces" / "demo.json")
            print(f"  trace written to {out}")
        env.close()


def cmd_play(args) -> None:
    """Non-interactive rollout of a chosen agent, printing verbose telemetry."""
    import play

    play.run(args)


def cmd_train(args) -> None:
    from training import dqn_training, pg_training

    if args.algo == "dqn":
        sys.argv = ["dqn_training"] + (["--final"] if args.final else [])
        dqn_training.main()
    else:
        sys.argv = ["pg_training", "--algo", args.algo] + (["--final"] if args.final else [])
        pg_training.main()


def cmd_evaluate(args) -> None:
    from analysis.evaluate import main as evaluate_main

    evaluate_main(args)


def cmd_plots(args) -> None:
    from analysis.plots import main as plots_main

    plots_main(args)


def cmd_video(args) -> None:
    from environment.custom_env import ZiplineDeliveryEnv
    from environment.rendering import record_video
    from training.common import best_available_agent, load_agent

    if args.algo:
        predict, label = load_agent(args.algo)
        fallback = None
    else:
        predict, label, fallback = best_available_agent()
    _banner(f"RECORDING - {label}")

    env = ZiplineDeliveryEnv()
    policy = (lambda obs: fallback(env)) if fallback is not None else predict
    out = record_video(env, policy, args.out, seed=args.seed, camera=args.camera)
    print(f"  wrote {out}")
    env.close()


def cmd_export(args) -> None:
    """Write a JSON episode trace for the browser replay viewer."""
    from environment.custom_env import ZiplineDeliveryEnv
    from training.common import best_available_agent, load_agent

    if args.algo:
        predict, label = load_agent(args.algo)
        fallback = None
    else:
        predict, label, fallback = best_available_agent()

    env = ZiplineDeliveryEnv(record_trace=True)
    obs, _ = env.reset(seed=args.seed)
    done = False
    info: dict = {}
    while not done:
        action = fallback(env) if fallback is not None else predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    out = env.export_trace(args.out, extra={"agent": label, "seed": args.seed})
    env.close()
    _banner("JSON EXPORT")
    print(f"  agent    : {label}")
    print(f"  outcome  : {info['outcome']}")
    print(f"  written  : {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("\n  Open viewer/index.html in a browser to replay it in 3D.")


# ------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Mission-based RL: a blood-delivery quadrotor over Rwandan terrain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run main.py                      live 3D demo with the best agent\n"
            "  uv run main.py env-info             print the action/observation/reward spec\n"
            "  uv run main.py train --algo ppo     run the PPO hyperparameter sweep\n"
            "  uv run main.py play --algo ppo      verbose rollout of the PPO agent\n"
            "  uv run main.py evaluate             score every trained agent\n"
            "  uv run main.py plots                regenerate every figure in the report\n"
            "  uv run main.py video --algo ppo     record an MP4\n"
            "  uv run main.py export-trace         write JSON for the web viewer\n"
        ),
    )
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("demo", help="interactive 3D simulation (default)")
    d.add_argument("--episodes", type=int, default=3)
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--verbose", action="store_true", default=True)
    d.add_argument("--quiet", dest="verbose", action="store_false")
    d.add_argument("--trace", action="store_true", help="also write a JSON trace")
    d.add_argument("--speed", type=float, default=1.0,
                   help="playback rate, 1.0 = real time (try 0.5 for slow motion)")
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("env-info", help="print the environment specification")
    i.set_defaults(func=cmd_env_info)

    pl = sub.add_parser("play", help="verbose rollout of a trained agent")
    pl.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default=None)
    pl.add_argument("--episodes", type=int, default=5)
    pl.add_argument("--seed", type=int, default=None)
    pl.add_argument("--render", action="store_true", help="open the 3D viewer as well")
    pl.add_argument("--speed", type=float, default=1.0,
                   help="playback rate when rendering, 1.0 = real time")
    pl.set_defaults(func=cmd_play)

    t = sub.add_parser("train", help="run a hyperparameter sweep")
    t.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], required=True)
    t.add_argument("--final", action="store_true", help="retrain the sweep winner for longer")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="score every trained agent, incl. generalisation")
    e.add_argument("--episodes", type=int, default=30)
    e.set_defaults(func=cmd_evaluate)

    g = sub.add_parser("plots", help="regenerate every figure used in the report")
    g.set_defaults(func=cmd_plots)

    v = sub.add_parser("video", help="record an MP4 of an agent flying")
    v.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default=None)
    v.add_argument("--out", default="assets/agent_demo.mp4")
    v.add_argument("--seed", type=int, default=9001)
    v.add_argument("--camera", choices=["chase", "mission"], default="chase")
    v.set_defaults(func=cmd_video)

    x = sub.add_parser("export-trace", help="write a JSON episode for the web viewer")
    x.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default=None)
    x.add_argument("--out", default="viewer/episode.json")
    x.add_argument("--seed", type=int, default=9001)
    x.set_defaults(func=cmd_export)

    return p


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:] or ["demo"]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
