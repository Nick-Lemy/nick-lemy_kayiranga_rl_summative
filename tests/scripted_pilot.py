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
    """Greedy station-to-station pilot: face it, close in, hold, scan."""
    cfg = env.cfg
    pos = env._rov_pos()
    vel = env._rov_vel()
    mat = env.data.xmat[env._bid_rov].reshape(3, 3)
    yaw = math.atan2(mat[1, 0], mat[0, 0])

    target = env._active_target()
    to_target = target - pos
    horiz = float(np.linalg.norm(to_target[:2]))
    speed = float(np.linalg.norm(vel))

    # 1. scan the moment we are inside the hoop and settled enough for a clean
    #    reading; holding station against the current is what makes this
    #    non-trivial, so we wait until the drift is bled off.
    if horiz <= cfg.inspect_radius and abs(to_target[2]) < 0.8 and speed < 1.1:
        return int(Action.INSPECT)

    # 2. hold the working depth of the active station.
    if to_target[2] > 0.35 and env.vz_cmd < cfg.max_vspeed - 0.01:
        return int(Action.ASCEND)
    if to_target[2] < -0.35 and env.vz_cmd > -cfg.max_vspeed + 0.01:
        return int(Action.DESCEND)

    # 3. point the nose at the station during the transit; close in, the bearing
    #    swings fast and chasing it just wastes actions, and the vehicle can
    #    strafe sideways without turning anyway.
    bearing = math.atan2(to_target[1], to_target[0])
    heading_err = _wrap(bearing - yaw)
    if horiz > cfg.inspect_radius and abs(heading_err) > 0.2:
        return int(Action.YAW_LEFT if heading_err > 0 else Action.YAW_RIGHT)

    # 4. drive a range-scheduled forward speed so the approach decelerates and
    #    arrives slow enough to hold station and scan.
    surge_target = float(np.clip(0.5 * horiz, 0.0, cfg.max_surge))
    fwd_speed = float(vel[0] * math.cos(yaw) + vel[1] * math.sin(yaw))
    if env.surge_cmd < surge_target - 0.25 and fwd_speed < surge_target:
        return int(Action.SURGE_FWD)
    if env.surge_cmd > surge_target + 0.25:
        return int(Action.SURGE_REV)

    # 5. trim lateral offset to the station line with the strafe thrusters.
    lat_err = float(-to_target[0] * math.sin(yaw) + to_target[1] * math.cos(yaw))
    if lat_err > 1.0 and env.sway_cmd < cfg.max_sway - 0.01:
        return int(Action.STRAFE_LEFT)
    if lat_err < -1.0 and env.sway_cmd > -cfg.max_sway + 0.01:
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
