"""Render the environment overview figure used in the report.

    uv run python -m analysis.env_figure

Flies one successful delivery with the reference pilot and grabs four moments
from it - the whole corridor, the climb-out, the approach, and the payload under
canopy - so the reader can see what the agent is actually controlling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.style import INK, INK_2, MUTED, save, use_report_style
from environment.custom_env import ZiplineDeliveryEnv
from environment.rendering import OffscreenRenderer
from tests.scripted_pilot import scripted_policy

W, H = 900, 560


def _find_delivering_seed(env: ZiplineDeliveryEnv, seeds=range(120, 200)) -> int | None:
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(scripted_policy(env))
            done = term or trunc
        if info["outcome"] == "delivered":
            return seed
    return None


def capture(seed: int) -> tuple[list[np.ndarray], list[str]]:
    """Four frames from one delivery, with captions."""
    env = ZiplineDeliveryEnv()
    mission = OffscreenRenderer(env, width=W, height=H, camera="mission", hud=False)
    chase = OffscreenRenderer(env, width=W, height=H, camera="chase", hud=False)
    drop = OffscreenRenderer(env, width=W, height=H, camera="payload", hud=False)

    env.reset(seed=seed)
    frames: dict[str, np.ndarray] = {}
    release_step = None
    step = 0
    done = False
    while not done:
        _, _, term, trunc, info = env.step(scripted_policy(env))
        done = term or trunc
        step += 1
        if step == 6:
            frames["overview"] = mission.frame()
        if step == 26:
            frames["climb"] = chase.frame()
        if not env.attached and release_step is None:
            release_step = step
        if release_step is not None and step == release_step + 3:
            frames["drop"] = drop.frame()
        if done:
            frames["final"] = mission.frame()

    env.close()
    order = ["overview", "climb", "drop", "final"]
    captions = [
        "1. The mission. Depot and helipad on the left, hills in between, health post\n"
        "and the green 3 m drop zone on the right; orange pylons mark the corridor.",
        "2. Climb-out. The policy commands attitude and climb-rate setpoints, and an\n"
        "onboard PD loop turns them into four individual rotor thrusts.",
        "3. Release. The pack detaches with the aircraft's momentum and descends under\n"
        "a canopy, so any crosswind walks it downwind for the whole fall.",
        f"4. Delivered. Miss distance {info['miss_distance']:.2f} m, flight time "
        f"{info['flight_time']:.1f} s, battery\nremaining {100 * info['battery']:.0f}%, "
        f"cold chain {100 * info['cold_chain']:.0f}% unused.",
    ]
    return [frames[k] for k in order if k in frames], captions[: len(frames)]


def main() -> None:
    use_report_style()
    env = ZiplineDeliveryEnv()
    seed = _find_delivering_seed(env)
    env.close()
    if seed is None:
        print("  [fail] the reference pilot did not deliver on any probed seed")
        return
    print(f"  filming delivery on seed {seed}")

    images, captions = capture(seed)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    for ax, img, cap in zip(axes.ravel(), images, captions):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d9d8d3")
        # captions go under the axes as free text: set_xlabel gets clipped by
        # tight_layout once the string wraps to a second line
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
        "Blood-delivery quadrotor: one episode, flown by the reference pilot",
        x=0.012,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.012,
        0.945,
        "MuJoCo rigid-body physics - 4 rotors, procedurally regenerated terrain, "
        "stochastic wind, a payload that really detaches",
        ha="left",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=4.2)
    save(fig, "env_overview.png")


if __name__ == "__main__":
    main()
