"""A hand-written pilot used to prove the mission is flyable.

This is not part of the agent. It exists so that we can answer "is the reward
reachable at all, and how much of it?" without waiting on a training run, and so
that regressions in the physics or the reward show up immediately. The RL
results in the report are only meaningful if a competent controller can score
well here, and if doing nothing scores badly.

The pilot flies the survey the obvious way: point at the active station, drive
the heading-frame velocity setpoints to close the range, ease off and hold
station as it arrives, and trigger the scan once it is inside the hoop and slow
enough. Because it commands a velocity, the onboard controller automatically
leans into the current to hold that velocity, so no explicit crab term is
needed - the current shows up as a steady thrust bias, exactly as it does for a
learned policy.
"""

from __future__ import annotations

import math

import numpy as np

from environment.custom_env import Action, SubseaInspectionEnv


def scripted_policy(env: SubseaInspectionEnv) -> int:
    """Station-to-station pilot: transit, slow to a stop in the ring, hold, scan.

    The scan is not instant. The vehicle has to hold station inside the ring for
    a short dwell before the scan can be captured, so the pilot drives to each
    ring, brakes to a near stop, nulls its velocity so the controller holds it
    against the current, waits for the scan to be ready, and only then fires
    INSPECT.
    """
    cfg = env.cfg
    pos = env._rov_pos()
    vel = env._rov_vel()
    mat = env.data.xmat[env._bid_rov].reshape(3, 3)
    yaw = math.atan2(mat[1, 0], mat[0, 0])

    target = env._active_target()
    to_target = target - pos
    horiz = float(np.linalg.norm(to_target[:2]))

    # 1. capture the scan the moment the hold is complete.
    if env._scan_ready:
        return int(Action.INSPECT)

    # 2. hold the working depth of the active station.
    if to_target[2] > 0.35 and env.vz_cmd < cfg.max_vspeed - 0.01:
        return int(Action.ASCEND)
    if to_target[2] < -0.35 and env.vz_cmd > -cfg.max_vspeed + 0.01:
        return int(Action.DESCEND)

    # near the ring centre: null the velocity setpoints so the onboard
    # controller holds the vehicle on the spot against the current while the scan
    # builds. Holding near the centre (not the edge) keeps it in range through
    # the whole dwell even as the current nudges it.
    if horiz <= 1.0:
        if env.surge_cmd > 0.05:
            return int(Action.SURGE_REV)
        if env.surge_cmd < -0.05:
            return int(Action.SURGE_FWD)
        if env.sway_cmd > 0.05:
            return int(Action.STRAFE_RIGHT)
        if env.sway_cmd < -0.05:
            return int(Action.STRAFE_LEFT)
        return int(Action.HOLD)

    # transit: point at the station, then drive a range-scheduled speed that
    # decays to almost zero at the centre so the vehicle arrives slow.
    bearing = math.atan2(to_target[1], to_target[0])
    heading_err = _wrap(bearing - yaw)
    if abs(heading_err) > 0.2:
        return int(Action.YAW_LEFT if heading_err > 0 else Action.YAW_RIGHT)

    surge_target = float(np.clip(0.6 * horiz - 0.3, 0.0, cfg.max_surge))
    if env.surge_cmd < surge_target - 0.2:
        return int(Action.SURGE_FWD)
    if env.surge_cmd > surge_target + 0.2:
        return int(Action.SURGE_REV)

    lat_err = float(-to_target[0] * math.sin(yaw) + to_target[1] * math.cos(yaw))
    if lat_err > 0.8 and env.sway_cmd < cfg.max_sway - 0.01:
        return int(Action.STRAFE_LEFT)
    if lat_err < -0.8 and env.sway_cmd > -cfg.max_sway + 0.01:
        return int(Action.STRAFE_RIGHT)

    return int(Action.HOLD)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(episodes: int = 20, seed: int = 0, verbose: bool = True) -> dict:
    env = SubseaInspectionEnv()
    returns, misses, outcomes, times, batteries = [], [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(scripted_policy(env))
            total += r
            done = term or trunc
        returns.append(total)
        outcomes.append(info["outcome"])
        times.append(info["mission_time"])
        batteries.append(info["battery"])
        if not math.isnan(info["inspect_error"]):
            misses.append(info["inspect_error"])
        if verbose:
            print(
                f"ep {ep:3d}  return {total:8.2f}  {info['outcome']:<18}"
                f"  stations {info['waypoints_done']}/{info['waypoints_total']}"
                f"  scan-off {info['inspect_error']:5.2f} m  t {info['mission_time']:5.1f} s"
                f"  bat {info['battery']:.2f}"
            )
    env.close()

    complete = sum(o == "survey_complete" for o in outcomes)
    summary = {
        "mean_return": float(np.mean(returns)),
        "success_rate": complete / episodes,
        "mean_miss": float(np.mean(misses)) if misses else float("nan"),
        "mean_time": float(np.mean(times)),
        "mean_battery_left": float(np.mean(batteries)),
        "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
    }
    if verbose:
        print("\n--- scripted pilot summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    run()
