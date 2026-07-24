"""A hand-written pilot used to prove the mission is flyable.

This is not part of the agent. It exists so that we can answer "is the reward
reachable at all, and how much of it?" without waiting on a training run, and so
that regressions in the physics or the reward show up immediately. The RL
results in the report are only meaningful if a competent controller can score
well here, and if doing nothing scores badly.
"""

from __future__ import annotations

import math

import numpy as np

from environment.custom_env import Action, ZiplineDeliveryEnv


#: Upwind aim offset per second of canopy descent per m/s of steady wind.
#: Swept over 0.4-1.3 against 24 fixed seeds; delivery rate peaks at ~1.0
#: (83%) and falls off on either side, so the drop really is drift-limited.
DRIFT_K = 1.0


def scripted_policy(env: ZiplineDeliveryEnv) -> int:
    """Greedy waypoint pilot: climb, run the corridor, line up, drop."""
    pos = env._drone_pos()
    vel = env._drone_vel()
    mat = env.data.xmat[env._bid_drone].reshape(3, 3)
    yaw = math.atan2(mat[1, 0], mat[0, 0])

    if not env.attached:
        return int(Action.HOVER)

    to_target = env.aim_point - pos
    horiz = float(np.linalg.norm(to_target[:2]))

    # Aim upwind: under canopy the box drifts downwind for the whole descent,
    # so the release point is offset against the wind by roughly the drift it
    # will accumulate on the way down. The offset is computed from the *steady*
    # wind and a fixed release altitude, not from the instantaneous gust and the
    # current altitude - an operator knows the prevailing wind, not each gust,
    # and feeding the raw OU signal back in swings the aim point by ten metres a
    # second and sends the aircraft chasing its own tail.
    fall_time = env.cfg.release_altitude / 5.0
    drift = env.wind_mean * fall_time * DRIFT_K
    aim_xy = env.aim_point[:2] - drift
    to_aim = np.array([aim_xy[0] - pos[0], aim_xy[1] - pos[1]])
    horiz_aim = float(np.linalg.norm(to_aim))

    # keep a safe clearance over the ridge until we are close to the zone
    ground = env.terrain_height(pos[0], pos[1])
    lookahead = max(
        env.terrain_height(pos[0] + d * math.cos(yaw), pos[1] + d * math.sin(yaw))
        for d in (4.0, 8.0, 12.0)
    )
    desired_z = max(ground, lookahead) + (8.0 if horiz > 14.0 else 6.0)
    if horiz < 6.0:
        desired_z = env.aim_point[2]

    # 1. release when parked over the (wind-corrected) release point and settled
    if (
        horiz_aim < 1.5
        and abs(pos[2] - env.aim_point[2]) < 2.0
        and float(np.linalg.norm(vel[:2])) < 2.0
    ):
        return int(Action.RELEASE_PAYLOAD)

    # 2. Cascaded altitude hold: an outer loop turns altitude error into a
    #    climb-rate target, the inner one trims collective towards it. Driving
    #    throttle straight off altitude error makes this airframe pilot-induce
    #    oscillations and fly itself into the ground.
    vz_des = float(np.clip(0.9 * (desired_z - pos[2]), -3.0, 4.0))
    if env.vz_cmd < vz_des - 0.5:
        return int(Action.THROTTLE_UP)
    if env.vz_cmd > vz_des + 0.5:
        return int(Action.THROTTLE_DOWN)

    # 3. Point the nose at the release point, but only during the cruise. Close
    #    in, the bearing swings quickly and chasing it starves the braking logic
    #    of actions, which is how this pilot used to sail straight over the zone
    #    at cruise speed. A quadrotor can translate without yawing anyway.
    bearing = math.atan2(to_aim[1], to_aim[0])
    heading_err = _wrap(bearing - yaw)
    if horiz_aim > 12.0 and abs(heading_err) > 0.25:
        return int(Action.YAW_LEFT if heading_err > 0 else Action.YAW_RIGHT)

    # 4. Track a velocity *vector*: aim for a speed proportional to the range,
    #    so the approach decelerates on a schedule and arrives slow enough to
    #    drop.
    speed_cap = float(np.clip(0.6 * horiz_aim, 0.5, 9.0))
    v_des = to_aim / max(horiz_aim, 1e-6) * speed_cap
    err = v_des - vel[:2]
    e_fwd = float(err[0] * math.cos(yaw) + err[1] * math.sin(yaw))
    e_lat = float(-err[0] * math.sin(yaw) + err[1] * math.cos(yaw))

    # Close the loop on the *setpoint*, not on the raw error. The tilt commands
    # are integrators - they persist until something walks them back - so
    # bang-banging them straight off the velocity error builds up a bank the
    # pilot then has to unwind, and the aircraft ends up orbiting the zone at
    # cruise speed instead of arriving at it. Converting the velocity error into
    # a target bank angle and trimming towards that is what makes the approach
    # settle, exactly as the climb-rate loop above already does for altitude.
    max_tilt = env.cfg.max_tilt
    pitch_target = float(np.clip(0.12 * e_fwd, -max_tilt, max_tilt))
    roll_target = float(np.clip(-0.12 * e_lat, -max_tilt, max_tilt))
    d_pitch = pitch_target - env.pitch_cmd
    d_roll = roll_target - env.roll_cmd
    tol = env.cfg.tilt_delta * 0.75

    if abs(d_pitch) >= abs(d_roll):
        if d_pitch > tol:
            return int(Action.PITCH_FORWARD)
        if d_pitch < -tol:
            return int(Action.PITCH_BACK)
    else:
        if d_roll < -tol:
            return int(Action.ROLL_LEFT)
        if d_roll > tol:
            return int(Action.ROLL_RIGHT)

    return int(Action.HOVER)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(episodes: int = 20, seed: int = 0, verbose: bool = True) -> dict:
    env = ZiplineDeliveryEnv()
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
        times.append(info["flight_time"])
        batteries.append(info["battery"])
        if not math.isnan(info["miss_distance"]):
            misses.append(info["miss_distance"])
        if verbose:
            print(
                f"ep {ep:3d}  return {total:8.2f}  {info['outcome']:<18}"
                f"  miss {info['miss_distance']:6.2f} m  t {info['flight_time']:5.1f} s"
                f"  bat {info['battery']:.2f}"
            )
    env.close()

    delivered = sum(o == "delivered" for o in outcomes)
    summary = {
        "mean_return": float(np.mean(returns)),
        "success_rate": delivered / episodes,
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
