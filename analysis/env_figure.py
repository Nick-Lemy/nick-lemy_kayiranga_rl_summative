"""Render the environment overview figure used in the report.

    uv run python -m analysis.env_figure

Flies one successful survey with the reference pilot and grabs four moments from
it - the whole pipeline, a transit between stations, a scan at a hoop, and the
finished survey - so the reader can see what the agent is actually controlling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.style import INK, INK_2, MUTED, save, use_report_style
from environment.custom_env import SubseaInspectionEnv
from environment.rendering import OffscreenRenderer
from tests.scripted_pilot import scripted_policy

W, H = 900, 560


def _find_survey_seed(env: SubseaInspectionEnv, seeds=range(120, 200)) -> int | None:
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(scripted_policy(env))
            done = term or trunc
        if info["outcome"] == "survey_complete":
            return seed
    return None


def capture(seed: int) -> tuple[list[np.ndarray], list[str]]:
    """Four frames from one completed survey, with captions."""
    env = SubseaInspectionEnv()
    mission = OffscreenRenderer(env, width=W, height=H, camera="mission", hud=False)
    chase = OffscreenRenderer(env, width=W, height=H, camera="chase", hud=False)
    station = OffscreenRenderer(env, width=W, height=H, camera="station", hud=False)

    env.reset(seed=seed)
    frames: dict[str, np.ndarray] = {}
    prev_wp = 0
    step = 0
    done = False
    while not done:
        _, _, term, trunc, info = env.step(scripted_policy(env))
        done = term or trunc
        step += 1
        if step == 6:
            frames["overview"] = mission.frame()
        if step == 40:
            frames["transit"] = chase.frame()
        # grab a scan the moment a station is logged (active index advances)
        if info["waypoints_done"] > prev_wp and info["waypoints_done"] == 2:
            frames["scan"] = station.frame()
        prev_wp = info["waypoints_done"]
        if done:
            frames["final"] = mission.frame()

    env.close()
    order = ["overview", "transit", "scan", "final"]
    captions = [
        "1. The survey. Launch buoy on the left, the pipeline snaking across the seabed\n"
        "in a shallow S, four teal inspection hoops, and the manifold on the right.",
        "2. Transit. The policy commands surge / sway / yaw / depth setpoints; an onboard\n"
        "controller turns them into thruster forces and leans into the current.",
        "3. Inspection. Holding station inside a hoop against the drifting current, the\n"
        "vehicle fires the INSPECT action to log a scan and unlock the next station.",
        f"4. Survey complete. Mean scan offset {info['inspect_error']:.2f} m, mission time "
        f"{info['mission_time']:.1f} s,\nbattery remaining {100 * info['battery']:.0f}%.",
    ]
    return [frames[k] for k in order if k in frames], [
        c for k, c in zip(order, captions) if k in frames
    ]


def main() -> None:
    use_report_style()
    env = SubseaInspectionEnv()
    seed = _find_survey_seed(env)
    env.close()
    if seed is None:
        print("  [fail] the reference pilot did not complete a survey on any probed seed")
        return
    print(f"  filming survey on seed {seed}")

    images, captions = capture(seed)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    for ax, img, cap in zip(axes.ravel(), images, captions):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d9d8d3")
        ax.text(
            0.0,
            -0.035,
            cap,
            transform=ax.transAxes,
            fontsize=8.2,
            color=INK_2,
            va="top",
            ha="left",
            linespacing=1.6,
        )
    for ax in axes.ravel()[len(images) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Subsea inspection ROV: one episode, flown by the reference pilot",
        x=0.012,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.012,
        0.945,
        "MuJoCo rigid-body physics - buoyancy, quadratic drag, a procedurally "
        "regenerated seabed and a drifting current field",
        ha="left",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=4.2)
    save(fig, "env_overview.png")


if __name__ == "__main__":
    main()
