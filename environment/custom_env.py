"""Zipline-style blood-delivery quadrotor environment.

Mission
-------
A cargo UAV lifts a cold-chain blood pack from a distribution centre and has to
put it inside a 3 m drop zone at a rural health post on the far side of a range
of Rwandan hills. It has to do that

  * before the blood spoils      (cold-chain timer),
  * before the battery runs flat (energy budget that scales with how hard it flies),
  * without leaving the regulated flight corridor,
  * without flying into a hillside, and
  * without slamming the payload into the ground hard enough to burst the bags.

Everything above is a real constraint on the real service this is modelled on,
and each one shows up in the reward as a separate term, so the agent has to
trade speed against energy against accuracy rather than optimise a single axis.

Why the dynamics are not scripted
---------------------------------
The nine discrete actions do **not** teleport the aircraft. They nudge the
setpoints of an onboard attitude controller (exactly the "angle mode" of a real
flight controller), which mixes them into four individual rotor thrusts. Lift,
banking, translation, stalling and crashing are then produced by MuJoCo's
rigid-body integrator. A bad sequence of actions produces a genuinely
unrecoverable attitude, not a bounded grid move.

Coordinate conventions
----------------------
World is z-up. Body x is forward, y is left, z is up. With that convention a
*positive* pitch angle tilts the nose down and therefore accelerates the
aircraft forwards, and a *positive* roll angle banks it to the right. The
observation is expressed in the yaw-aligned frame, so "forward" always means
the same thing to the policy regardless of which way the aircraft is pointing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_PATH = Path(__file__).resolve().parent.parent / "assets" / "zipline_scene.xml"


class Action(IntEnum):
    """The nine commands an operator (or the policy) can issue to the aircraft."""

    HOVER = 0            # level the wings, hold current collective
    THROTTLE_UP = 1      # more collective  -> climb
    THROTTLE_DOWN = 2    # less collective  -> descend
    PITCH_FORWARD = 3    # nose down        -> accelerate along body +x
    PITCH_BACK = 4       # nose up          -> decelerate / back up
    ROLL_LEFT = 5        # bank left        -> translate along body +y
    ROLL_RIGHT = 6       # bank right       -> translate along body -y
    YAW_LEFT = 7         # rotate heading to port, to line up the drop run
    YAW_RIGHT = 8        # rotate heading to starboard
    RELEASE_PAYLOAD = 9  # open the cargo bay - the mission-critical act


ACTION_MEANING: dict[int, str] = {
    Action.HOVER: "HOVER",
    Action.THROTTLE_UP: "THROTTLE_UP",
    Action.THROTTLE_DOWN: "THROTTLE_DOWN",
    Action.PITCH_FORWARD: "PITCH_FORWARD",
    Action.PITCH_BACK: "PITCH_BACK",
    Action.ROLL_LEFT: "ROLL_LEFT",
    Action.ROLL_RIGHT: "ROLL_RIGHT",
    Action.YAW_LEFT: "YAW_LEFT",
    Action.YAW_RIGHT: "YAW_RIGHT",
    Action.RELEASE_PAYLOAD: "RELEASE_PAYLOAD",
}


@dataclass
class EnvConfig:
    """Every tunable in one place so experiments can perturb the mission, not the code."""

    # --- timing -------------------------------------------------------------
    frame_skip: int = 10              # 10 x 10 ms  ->  a 10 Hz guidance loop
    max_episode_seconds: float = 30.0

    # --- flight envelope ----------------------------------------------------
    max_tilt: float = 0.45            # rad, largest commandable bank/pitch (~26 deg)
    tilt_delta: float = 0.08          # rad added per attitude action
    yaw_delta: float = 0.12           # rad per YAW action
    rotor_max: float = 7.5            # N per rotor (matches the XML ctrlrange)
    max_climb_rate: float = 4.0       # m/s, largest commandable vertical speed
    vz_delta: float = 1.0             # m/s added per throttle action

    # --- onboard attitude / altitude-hold loop ------------------------------
    kp_att: float = 12.0
    kd_att: float = 2.2
    kp_yaw: float = 4.0
    kd_yaw: float = 1.0
    kp_vz: float = 1.2                # N per rotor per (m/s) of climb-rate error
    mass_ff: float = 1.9              # kg assumed by the collective feed-forward
    mix_roll: float = 1.6             # N of differential thrust at full command
    mix_pitch: float = 1.6
    mix_yaw: float = 0.8

    # --- consumables --------------------------------------------------------
    battery_endurance_s: float = 50.0  # seconds of hover on a full pack
    cold_chain_s: float = 26.0         # seconds before the blood is unusable

    # --- weather ------------------------------------------------------------
    wind_mean_max: float = 4.0        # m/s, steady component drawn per episode
    wind_sigma: float = 2.0           # m/s, gust intensity
    wind_theta: float = 0.6           # OU mean reversion
    wind_drag_k: float = 0.06         # N per (m/s)^2 of relative airspeed
    # sized so the 0.3 kg pack settles at ~5 m/s under canopy
    chute_drag_k: float = 0.1176      # N per (m/s)^2 on the released payload

    # --- geometry -----------------------------------------------------------
    zone_radius: float = 3.0
    release_altitude: float = 6.0     # m AGL the aim point sits at
    corridor_half_width: float = 20.0
    corridor_x_min: float = -38.0
    corridor_x_max: float = 44.0
    ceiling: float = 28.0
    min_agl: float = 2.0              # below this the terrain-proximity penalty bites

    # --- domain randomisation ----------------------------------------------
    post_x_range: tuple[float, float] = (26.0, 34.0)
    post_y_range: tuple[float, float] = (-9.0, 9.0)
    randomize_terrain: bool = True
    n_hills: int = 7

    # --- reward weights -----------------------------------------------------
    w_progress: float = 1.2
    w_step: float = 0.06
    w_energy: float = 0.04
    w_tilt: float = 0.15
    w_spin: float = 0.02
    w_corridor: float = 0.5
    w_terrain: float = 0.6
    r_delivery: float = 150.0         # peak accuracy reward, decays with miss distance
    r_in_zone: float = 50.0           # flat bonus for landing inside the ring
    p_failed_drop: float = 30.0       # releasing outside the zone
    p_impact: float = 4.0             # per m/s of impact above the safe threshold
    safe_impact_v: float = 6.0        # a canopy descent lands at ~5 m/s
    p_crash: float = 70.0
    p_corridor_breach: float = 70.0
    p_battery: float = 70.0
    p_cold_chain: float = 70.0
    p_timeout: float = 15.0


# observation normalisation constants
POS_SCALE = 35.0
VEL_SCALE = 14.0
ANGVEL_SCALE = 8.0
WIND_SCALE = 10.0
ALT_SCALE = 25.0
RANGE_SCALE = 70.0

OBS_DIM = 27

#: Human-readable index map for the observation vector. Kept next to the builder
#: so the report and the code can never drift apart.
OBS_LAYOUT: list[tuple[str, str]] = [
    ("0-2", "position error to the aim point, yaw-frame, / 35 m"),
    ("3-5", "velocity, yaw-frame, / 14 m/s"),
    ("6-8", "gravity direction in the body frame (encodes roll and pitch)"),
    ("9-10", "sin/cos of heading"),
    ("11-13", "body angular rates / 8 rad/s"),
    ("14", "battery remaining, 1 = full"),
    ("15", "cold-chain time remaining, 1 = fresh"),
    ("16-17", "estimated wind, yaw-frame, / 10 m/s"),
    ("18", "payload still attached (1/0)"),
    ("19", "altitude above ground level / 25 m"),
    ("20", "horizontal range to the drop zone / 70 m"),
    ("21", "fraction of the episode clock left"),
    ("22", "vertical speed / 8 m/s"),
    ("23", "lateral corridor margin, 1 = centred, 0 = at the wall"),
    ("24-26", "the roll / pitch / climb-rate setpoints currently commanded"),
]


def _euler_from_mat(mat: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (roll, pitch, yaw) from a 3x3 rotation matrix."""
    pitch = math.asin(max(-1.0, min(1.0, -mat[2, 0])))
    roll = math.atan2(mat[2, 1], mat[2, 2])
    yaw = math.atan2(mat[1, 0], mat[0, 0])
    return roll, pitch, yaw


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _to_yaw_frame(vec: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world vector into the yaw-aligned (heading) frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1], vec[2]])


class ZiplineDeliveryEnv(gym.Env):
    """Gymnasium environment for the blood-delivery mission.

    Parameters
    ----------
    render_mode:
        ``"human"`` opens the interactive MuJoCo viewer, ``"rgb_array"`` returns
        annotated frames for video capture, ``None`` runs headless for training.
    config:
        An :class:`EnvConfig`; the defaults describe the nominal mission.
    record_trace:
        When true every transition is appended to an in-memory trace that
        :meth:`export_trace` writes out as JSON for the browser replay viewer.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        config: EnvConfig | None = None,
        record_trace: bool = False,
        realtime: bool = True,
        playback_speed: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.record_trace = record_trace
        # Only affects render_mode="human": the viewer sleeps so simulated time
        # tracks wall-clock time. Without it the flight replays at roughly a
        # hundred times real speed. Training never renders, so this costs
        # nothing there.
        self.realtime = realtime
        self.playback_speed = playback_speed

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep * self.cfg.frame_skip
        self.max_steps = int(round(self.cfg.max_episode_seconds / self.dt))

        # --- cache ids so the hot loop never does string lookups -------------
        mj = mujoco
        self._bid_drone = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "drone")
        self._bid_payload = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "payload")
        self._bid_post = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "health_post")
        self._mocap_post = int(self.model.body_mocapid[self._bid_post])
        self._eq_weld = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_EQUALITY, "payload_weld")
        self._gid_payload = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "payload_box")
        self._qadr_drone = self.model.jnt_qposadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "drone_free")
        ]
        self._qadr_payload = self.model.jnt_qposadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "payload_free")
        ]
        self._dadr_drone = self.model.jnt_dofadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "drone_free")
        ]
        self._dadr_payload = self.model.jnt_dofadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "payload_free")
        ]
        #: every geom that belongs to the airframe, for crash detection
        self._drone_geoms = {
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self._bid_drone
        }

        # --- heightfield bookkeeping ----------------------------------------
        self._hf_nrow = int(self.model.hfield_nrow[0])
        self._hf_ncol = int(self.model.hfield_ncol[0])
        self._hf_adr = int(self.model.hfield_adr[0])
        self._hf_size = np.array(self.model.hfield_size[0])  # rx, ry, elev, base
        self._hfield = np.zeros((self._hf_nrow, self._hf_ncol), dtype=np.float64)
        self.terrain_dirty = True

        self.launch_xy = np.array(
            self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "launch_site")][:2]
        )

        # --- spaces ----------------------------------------------------------
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # --- episode state ---------------------------------------------------
        self._renderer = None
        self._viewer = None
        self.trace: list[dict[str, Any]] = []
        self.trace_header: dict[str, Any] = {}
        self._reset_episode_state()

    # ------------------------------------------------------------------ setup

    def _reset_episode_state(self) -> None:
        self.step_count = 0
        self.attached = True
        self.battery = 1.0
        self.cold_chain = 1.0
        self.vz_cmd = 0.0
        self.pitch_cmd = 0.0
        self.roll_cmd = 0.0
        self.yaw_cmd = 0.0
        self.wind = np.zeros(2)
        self.wind_mean = np.zeros(2)
        self.release_step = -1
        self.outcome = "flying"
        self._prev_dist = 0.0
        self._payload_landed = False
        self._last_action = int(Action.HOVER)
        self._last_reward = 0.0
        self._episode_return = 0.0
        self._last_thrusts = np.zeros(4)
        self._miss_distance = float("nan")
        self._impact_speed = float("nan")

    def _generate_terrain(self, post_xy: np.ndarray) -> None:
        """Build a fresh hill field and carve flat aprons at both sites."""
        rx, ry, elev = self._hf_size[0], self._hf_size[1], self._hf_size[2]
        xs = np.linspace(-rx, rx, self._hf_ncol)
        ys = np.linspace(-ry, ry, self._hf_nrow)
        gx, gy = np.meshgrid(xs, ys)

        h = np.zeros_like(gx)
        if self.cfg.randomize_terrain:
            for _ in range(self.cfg.n_hills):
                cx = self.np_random.uniform(-rx * 0.85, rx * 0.85)
                cy = self.np_random.uniform(-ry * 0.85, ry * 0.85)
                amp = self.np_random.uniform(0.25, 0.75)
                sx = self.np_random.uniform(6.0, 15.0)
                sy = self.np_random.uniform(6.0, 15.0)
                h += amp * np.exp(-(((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2))
            # a ridge across the middle of the route, so there is always
            # something between the depot and the clinic
            ridge_x = self.np_random.uniform(-6.0, 6.0)
            h += self.np_random.uniform(0.35, 0.6) * np.exp(-(((gx - ridge_x) / 7.0) ** 2))
        else:
            h += 0.45 * np.exp(-((gx / 7.0) ** 2))

        h = np.clip(h, 0.0, 1.0)

        # flatten a landing apron at the depot and at the health post
        for centre, radius in ((self.launch_xy, 6.5), (post_xy, 8.0)):
            d = np.sqrt((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2)
            blend = np.clip((d - radius) / (radius * 0.9), 0.0, 1.0)
            h = h * blend  # ground level (0) inside the apron, terrain outside

        self._hfield = h
        self.model.hfield_data[self._hf_adr : self._hf_adr + h.size] = h.ravel()
        self.terrain_dirty = True
        self._terrain_elev = elev

    def terrain_height(self, x: float, y: float) -> float:
        """Bilinearly interpolated ground height under a world point."""
        rx, ry, elev = self._hf_size[0], self._hf_size[1], self._hf_size[2]
        fc = (x + rx) / (2.0 * rx) * (self._hf_ncol - 1)
        fr = (y + ry) / (2.0 * ry) * (self._hf_nrow - 1)
        if not (0.0 <= fc <= self._hf_ncol - 1 and 0.0 <= fr <= self._hf_nrow - 1):
            return 0.0
        c0, r0 = int(fc), int(fr)
        c1 = min(c0 + 1, self._hf_ncol - 1)
        r1 = min(r0 + 1, self._hf_nrow - 1)
        tc, tr = fc - c0, fr - r0
        top = self._hfield[r0, c0] * (1 - tc) + self._hfield[r0, c1] * tc
        bot = self._hfield[r1, c0] * (1 - tc) + self._hfield[r1, c1] * tc
        return float((top * (1 - tr) + bot * tr) * elev)

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_episode_state()

        mujoco.mj_resetData(self.model, self.data)

        # --- place the health post, then shape terrain around it -------------
        post_x = self.np_random.uniform(*self.cfg.post_x_range)
        post_y = self.np_random.uniform(*self.cfg.post_y_range)
        self.post_xy = np.array([post_x, post_y])
        self.data.mocap_pos[self._mocap_post] = np.array([post_x, post_y, 0.0])
        self._generate_terrain(self.post_xy)

        zone_z = self.terrain_height(post_x, post_y)
        self.drop_target = np.array([post_x, post_y, zone_z + 0.12])
        self.aim_point = np.array([post_x, post_y, zone_z + self.cfg.release_altitude])

        # --- catapult launch: airborne, wings level, a little forward speed ---
        start = np.array(
            [
                self.launch_xy[0] + self.np_random.uniform(-0.4, 0.4),
                self.launch_xy[1] + self.np_random.uniform(-0.4, 0.4),
                1.35 + self.np_random.uniform(-0.1, 0.2),
            ]
        )
        self.data.qpos[self._qadr_drone : self._qadr_drone + 3] = start
        yaw0 = self.np_random.uniform(-0.25, 0.25)
        self.data.qpos[self._qadr_drone + 3 : self._qadr_drone + 7] = np.array(
            [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
        )
        self.data.qvel[self._dadr_drone : self._dadr_drone + 3] = np.array(
            [self.np_random.uniform(1.0, 2.5), 0.0, 0.0]
        )
        self.data.qpos[self._qadr_payload : self._qadr_payload + 3] = start + np.array(
            [0.0, 0.0, -0.155]
        )
        self.data.qpos[self._qadr_payload + 3 : self._qadr_payload + 7] = np.array([1.0, 0, 0, 0])
        self.yaw_cmd = yaw0

        # --- weather ---------------------------------------------------------
        wdir = self.np_random.uniform(-math.pi, math.pi)
        wmag = self.np_random.uniform(0.0, self.cfg.wind_mean_max)
        self.wind_mean = np.array([wmag * math.cos(wdir), wmag * math.sin(wdir)])
        self.wind = self.wind_mean.copy()

        # --- consumables start partly used, so the agent cannot assume a full tank
        self.battery = float(self.np_random.uniform(0.85, 1.0))

        self.data.eq_active[self._eq_weld] = 1
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = float(np.linalg.norm(self._drone_pos() - self.aim_point))

        if self.record_trace:
            self._begin_trace()

        obs = self._observe()
        return obs, self._info()

    # ------------------------------------------------------------------- step

    def step(self, action: int):
        action = int(action)
        cfg = self.cfg
        self._last_action = action

        # --- 1. the action edits the setpoints the flight controller tracks ---
        if action == Action.THROTTLE_UP:
            self.vz_cmd = min(cfg.max_climb_rate, self.vz_cmd + cfg.vz_delta)
        elif action == Action.THROTTLE_DOWN:
            self.vz_cmd = max(-cfg.max_climb_rate, self.vz_cmd - cfg.vz_delta)
        elif action == Action.PITCH_FORWARD:
            self.pitch_cmd = min(cfg.max_tilt, self.pitch_cmd + cfg.tilt_delta)
        elif action == Action.PITCH_BACK:
            self.pitch_cmd = max(-cfg.max_tilt, self.pitch_cmd - cfg.tilt_delta)
        elif action == Action.ROLL_LEFT:
            self.roll_cmd = max(-cfg.max_tilt, self.roll_cmd - cfg.tilt_delta)
        elif action == Action.ROLL_RIGHT:
            self.roll_cmd = min(cfg.max_tilt, self.roll_cmd + cfg.tilt_delta)
        elif action == Action.YAW_LEFT:
            self.yaw_cmd = _wrap_pi(self.yaw_cmd + cfg.yaw_delta)
        elif action == Action.YAW_RIGHT:
            self.yaw_cmd = _wrap_pi(self.yaw_cmd - cfg.yaw_delta)
        elif action == Action.HOVER:
            # bleed the wings back towards level and the climb rate towards
            # zero, rather than snapping to them
            self.pitch_cmd *= 0.6
            self.roll_cmd *= 0.6
            self.vz_cmd *= 0.4

        released_now = False
        if action == Action.RELEASE_PAYLOAD and self.attached:
            self.data.eq_active[self._eq_weld] = 0
            self.attached = False
            self.release_step = self.step_count
            released_now = True

        # --- 2. roll the physics forward with the inner loop closed ----------
        energy_acc = 0.0
        # The gust field is resampled once per control step and the resulting
        # force left standing on the body: xfrc_applied persists across
        # mj_step, and re-deriving it every 10 ms bought nothing but Python
        # overhead in the innermost loop of the whole project.
        self._apply_wind()
        ctrl = self.data.ctrl
        for _ in range(cfg.frame_skip):
            total = self._attitude_loop(ctrl)
            mujoco.mj_step(self.model, self.data)
            energy_acc += total**1.5
        self._last_thrusts = np.array(ctrl)

        # --- 3. consumables --------------------------------------------------
        hover_power = (1.9 * 9.81) ** 1.5
        power_norm = energy_acc / (cfg.frame_skip * hover_power)
        self.battery -= self.dt * power_norm / cfg.battery_endurance_s
        self.battery = max(0.0, self.battery)
        if self.attached:
            self.cold_chain = max(0.0, self.cold_chain - self.dt / cfg.cold_chain_s)

        self.step_count += 1

        # --- 4. reward and termination ---------------------------------------
        reward, terminated, truncated = self._reward_and_done(released_now, power_norm)
        self._last_reward = reward
        self._episode_return += reward

        obs = self._observe()
        info = self._info()

        if self.record_trace:
            self._append_trace()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------- flight control + physics

    def _attitude_loop(self, ctrl) -> float:
        """PD attitude hold, mixed into four rotor thrusts.

        This is the aircraft's own autopilot, not part of the policy: the policy
        only moves the setpoints. Running it inside the frame-skip loop at 100 Hz
        is what makes a 10 Hz discrete policy able to fly at all, and mirrors how
        a real cargo UAV is actually commanded.

        Writes the four thrusts straight into ``ctrl`` and returns their sum.
        This runs ten times per environment step and tens of millions of times
        per training sweep, so it is written in scalars: the numpy version of the
        same arithmetic spent more time allocating four-element arrays than
        MuJoCo spent integrating the physics.
        """
        cfg = self.cfg
        mat = self.data.xmat[self._bid_drone]
        # xmat is row-major 3x3: indices 0,1,2 / 3,4,5 / 6,7,8
        pitch = math.asin(max(-1.0, min(1.0, -mat[6])))
        roll = math.atan2(mat[7], mat[8])
        yaw = math.atan2(mat[3], mat[0])

        qvel = self.data.qvel
        i = self._dadr_drone
        wx, wy, wz = qvel[i + 3], qvel[i + 4], qvel[i + 5]

        u_roll = cfg.kp_att * (self.roll_cmd - roll) - cfg.kd_att * wx
        u_pitch = cfg.kp_att * (self.pitch_cmd - pitch) - cfg.kd_att * wy
        u_yaw = cfg.kp_yaw * _wrap_pi(self.yaw_cmd - yaw) - cfg.kd_yaw * wz
        u_roll = -1.0 if u_roll < -1.0 else (1.0 if u_roll > 1.0 else u_roll)
        u_pitch = -1.0 if u_pitch < -1.0 else (1.0 if u_pitch > 1.0 else u_pitch)
        u_yaw = -1.0 if u_yaw < -1.0 else (1.0 if u_yaw > 1.0 else u_yaw)

        # Collective runs in altitude-hold: the throttle actions command a climb
        # *rate*, and this loop finds the thrust that holds it. A gravity
        # feed-forward divided by the tilt cosine keeps vertical lift constant as
        # the aircraft banks. This is the standard flight mode of a cargo UAV,
        # and it is what makes a 10 Hz discrete policy able to fly the aircraft
        # at all: commanding raw thrust needs faster corrections than one action
        # per 100 ms can supply, so the airframe simply falls out of the sky.
        tilt_cos = mat[8] if mat[8] > 0.4 else 0.4
        base = (cfg.mass_ff * 9.81 / 4.0) / tilt_cos + cfg.kp_vz * (self.vz_cmd - qvel[i + 2])

        r = cfg.mix_roll * u_roll
        p = cfg.mix_pitch * u_pitch
        y = cfg.mix_yaw * u_yaw
        hi = cfg.rotor_max
        total = 0.0
        for k, v in enumerate(
            (base + r - p - y, base + r + p + y, base - r + p - y, base - r - p + y)
        ):
            v = 0.0 if v < 0.0 else (hi if v > hi else v)
            ctrl[k] = v
            total += v
        return total

    def _apply_wind(self) -> None:
        """Ornstein-Uhlenbeck gusts pushing on the airframe."""
        cfg = self.cfg
        h = self.dt
        self.wind += cfg.wind_theta * (self.wind_mean - self.wind) * h + cfg.wind_sigma * math.sqrt(
            h
        ) * self.np_random.normal(size=2)

        vel = self._drone_vel()
        rel = np.array([self.wind[0] - vel[0], self.wind[1] - vel[1], 0.0])
        force = cfg.wind_drag_k * np.linalg.norm(rel) * rel
        self.data.xfrc_applied[self._bid_drone, :3] = force

        if not self.attached and not self._payload_landed:
            # Once released the box descends under a small paper parachute, the
            # way the real service delivers. That caps the impact speed, but it
            # also means the canopy is pushed downwind for the whole descent, so
            # releasing high or in a crosswind walks the box out of the zone.
            # Wind therefore has to be flown *into*, not merely survived.
            pv = self.data.qvel[self._dadr_payload : self._dadr_payload + 3]
            air = np.array([pv[0] - self.wind[0], pv[1] - self.wind[1], pv[2]])
            self.data.xfrc_applied[self._bid_payload, :3] = (
                -cfg.chute_drag_k * np.linalg.norm(air) * air
            )

    # ------------------------------------------------------------ observations

    def _drone_pos(self) -> np.ndarray:
        return self.data.xpos[self._bid_drone]

    def _drone_vel(self) -> np.ndarray:
        return self.data.qvel[self._dadr_drone : self._dadr_drone + 3]

    def _payload_pos(self) -> np.ndarray:
        return self.data.xpos[self._bid_payload]

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        pos = self._drone_pos()
        vel = self._drone_vel()
        mat = self.data.xmat[self._bid_drone].reshape(3, 3)
        _, _, yaw = _euler_from_mat(mat)
        omega = self.data.qvel[self._dadr_drone + 3 : self._dadr_drone + 6]

        target = self.aim_point if self.attached else self.drop_target
        err_yaw = _to_yaw_frame(target - pos, yaw)
        vel_yaw = _to_yaw_frame(vel, yaw)
        wind_yaw = _to_yaw_frame(np.array([self.wind[0], self.wind[1], 0.0]), yaw)

        gravity_body = mat.T @ np.array([0.0, 0.0, -1.0])
        agl = pos[2] - self.terrain_height(pos[0], pos[1])
        ground_range = float(np.linalg.norm((self.drop_target - pos)[:2]))
        corridor_margin = 1.0 - min(1.0, abs(pos[1]) / cfg.corridor_half_width)
        time_left = 1.0 - self.step_count / self.max_steps

        obs = np.concatenate(
            [
                err_yaw / POS_SCALE,
                vel_yaw / VEL_SCALE,
                gravity_body,
                [math.sin(yaw), math.cos(yaw)],
                omega / ANGVEL_SCALE,
                [self.battery, self.cold_chain],
                wind_yaw[:2] / WIND_SCALE,
                [1.0 if self.attached else 0.0],
                [agl / ALT_SCALE],
                [ground_range / RANGE_SCALE],
                [time_left],
                [vel[2] / 8.0],
                [corridor_margin],
                # the actions edit persistent setpoints, so those setpoints are
                # part of the state and have to be observable for the problem to
                # stay Markovian
                [
                    self.roll_cmd / cfg.max_tilt,
                    self.pitch_cmd / cfg.max_tilt,
                    self.vz_cmd / cfg.max_climb_rate,
                ],
            ]
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # ------------------------------------------------------------------ reward

    def _reward_and_done(self, released_now: bool, power_norm: float):
        cfg = self.cfg
        pos = self._drone_pos()
        vel = self._drone_vel()
        mat = self.data.xmat[self._bid_drone].reshape(3, 3)
        omega = self.data.qvel[self._dadr_drone + 3 : self._dadr_drone + 6]

        reward = 0.0
        terminated = False
        truncated = False

        # --- dense guidance: close the range to the aim point ----------------
        target = self.aim_point if self.attached else self.drop_target
        dist = float(np.linalg.norm(target - pos))
        if self.attached:
            reward += cfg.w_progress * (self._prev_dist - dist)
        self._prev_dist = dist

        # --- running costs ----------------------------------------------------
        reward -= cfg.w_step
        reward -= cfg.w_energy * power_norm
        tilt = math.acos(max(-1.0, min(1.0, mat[2, 2])))
        reward -= cfg.w_tilt * tilt * tilt
        reward -= cfg.w_spin * min(9.0, float(np.dot(omega, omega)))

        # --- staying inside the regulated corridor ---------------------------
        lateral = abs(pos[1]) / cfg.corridor_half_width
        if lateral > 0.75:
            reward -= cfg.w_corridor * (lateral - 0.75) ** 2 * 16.0

        # --- terrain proximity -------------------------------------------------
        agl = pos[2] - self.terrain_height(pos[0], pos[1])
        if agl < cfg.min_agl:
            reward -= cfg.w_terrain * (cfg.min_agl - agl)

        # --- hard failures ----------------------------------------------------
        speed = float(np.linalg.norm(vel))
        if self.attached:
            if self._airframe_contact() and speed > 1.5:
                self.outcome = "crash"
                return reward - cfg.p_crash, True, False
            if tilt > 1.4:
                self.outcome = "loss_of_control"
                return reward - cfg.p_crash, True, False
            if agl < 0.05:
                self.outcome = "crash"
                return reward - cfg.p_crash, True, False

        out_of_corridor = (
            abs(pos[1]) > cfg.corridor_half_width
            or pos[0] < cfg.corridor_x_min
            or pos[0] > cfg.corridor_x_max
            or pos[2] > cfg.ceiling
        )
        if out_of_corridor and self.attached:
            self.outcome = "corridor_breach"
            return reward - cfg.p_corridor_breach, True, False

        if self.battery <= 0.0 and self.attached:
            self.outcome = "battery_depleted"
            return reward - cfg.p_battery, True, False

        if self.cold_chain <= 0.0 and self.attached:
            self.outcome = "cold_chain_expired"
            return reward - cfg.p_cold_chain, True, False

        # --- the delivery itself ----------------------------------------------
        if not self.attached and not self._payload_landed:
            ppos = self._payload_pos()
            pvel = self.data.qvel[self._dadr_payload : self._dadr_payload + 3]
            ground = self.terrain_height(ppos[0], ppos[1])
            if ppos[2] <= ground + 0.12 or self._payload_contact():
                self._payload_landed = True
                miss = float(np.linalg.norm(ppos[:2] - self.drop_target[:2]))
                impact = float(np.linalg.norm(pvel))
                self._miss_distance = miss
                self._impact_speed = impact

                acc = cfg.r_delivery * math.exp(-((miss / cfg.zone_radius) ** 2))
                reward += acc
                if miss <= cfg.zone_radius:
                    reward += cfg.r_in_zone
                    self.outcome = "delivered"
                else:
                    reward -= cfg.p_failed_drop
                    self.outcome = "missed_zone"
                reward -= cfg.p_impact * max(0.0, impact - cfg.safe_impact_v)
                return reward, True, False

        # --- clock ------------------------------------------------------------
        if self.step_count >= self.max_steps:
            self.outcome = "timeout"
            reward -= cfg.p_timeout
            truncated = True

        return reward, terminated, truncated

    def _airframe_contact(self) -> bool:
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 in self._drone_geoms or con.geom2 in self._drone_geoms:
                return True
        return False

    def _payload_contact(self) -> bool:
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == self._gid_payload or con.geom2 == self._gid_payload:
                return True
        return False

    # -------------------------------------------------------------------- info

    def _info(self) -> dict[str, Any]:
        pos = self._drone_pos()
        return {
            "outcome": self.outcome,
            "delivered": self.outcome == "delivered",
            "miss_distance": self._miss_distance,
            "impact_speed": self._impact_speed,
            "battery": float(self.battery),
            "cold_chain": float(self.cold_chain),
            "attached": bool(self.attached),
            "altitude_agl": float(pos[2] - self.terrain_height(pos[0], pos[1])),
            "range_to_zone": float(np.linalg.norm((self.drop_target - pos)[:2])),
            "flight_time": float(self.step_count * self.dt),
            "action": ACTION_MEANING[self._last_action],
            "episode_return": float(self._episode_return),
        }

    # ------------------------------------------------------------ JSON tracing

    def _begin_trace(self) -> None:
        """Capture everything a browser needs to rebuild the scene from scratch."""
        step = max(1, self._hf_nrow // 48)
        coarse = self._hfield[::step, ::step]
        self.trace_header = {
            "schema": "zipline-rl-trace/1",
            "dt": self.dt,
            "terrain": {
                "rows": int(coarse.shape[0]),
                "cols": int(coarse.shape[1]),
                "size_x": float(self._hf_size[0]),
                "size_y": float(self._hf_size[1]),
                "elevation": float(self._hf_size[2]),
                "heights": [round(float(v), 4) for v in coarse.ravel()],
            },
            "launch": [float(self.launch_xy[0]), float(self.launch_xy[1])],
            "drop_target": [round(float(v), 3) for v in self.drop_target],
            "zone_radius": self.cfg.zone_radius,
            "corridor": {
                "half_width": self.cfg.corridor_half_width,
                "x_min": self.cfg.corridor_x_min,
                "x_max": self.cfg.corridor_x_max,
                "ceiling": self.cfg.ceiling,
            },
            "actions": [ACTION_MEANING[a] for a in sorted(ACTION_MEANING)],
        }
        self.trace = []

    def _append_trace(self) -> None:
        q = self.data.xquat
        self.trace.append(
            {
                "t": round(self.step_count * self.dt, 3),
                "drone": {
                    "p": [round(float(v), 3) for v in self._drone_pos()],
                    "q": [round(float(v), 4) for v in q[self._bid_drone]],
                },
                "payload": {
                    "p": [round(float(v), 3) for v in self._payload_pos()],
                    "q": [round(float(v), 4) for v in q[self._bid_payload]],
                },
                "a": int(self._last_action),
                "r": round(float(self._last_reward), 3),
                "bat": round(float(self.battery), 4),
                "cold": round(float(self.cold_chain), 4),
                "att": bool(self.attached),
                "wind": [round(float(v), 3) for v in self.wind],
            }
        )

    def export_trace(self, path: str | Path, extra: dict | None = None) -> Path:
        """Write the recorded episode to JSON for the Three.js replay viewer."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.trace_header)
        payload["summary"] = {
            **{k: v for k, v in self._info().items() if not isinstance(v, np.ndarray)},
            "steps": len(self.trace),
        }
        if extra:
            payload["summary"].update(extra)
        payload["frames"] = self.trace
        path.write_text(json.dumps(payload, separators=(",", ":")))
        return path

    # ----------------------------------------------------------------- render

    def render(self):
        from environment import rendering

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = rendering.PassiveViewer(
                    self, realtime=self.realtime, speed=self.playback_speed
                )
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = rendering.OffscreenRenderer(self)
            return self._renderer.frame()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(**kwargs) -> ZiplineDeliveryEnv:
    """Factory used by the training scripts and by :mod:`main`."""
    return ZiplineDeliveryEnv(**kwargs)


__all__ = [
    "ZiplineDeliveryEnv",
    "EnvConfig",
    "Action",
    "ACTION_MEANING",
    "OBS_LAYOUT",
    "OBS_DIM",
    "make_env",
]
