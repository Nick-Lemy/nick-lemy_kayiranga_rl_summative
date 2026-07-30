"""Autonomous underwater vehicle (ROV) subsea pipeline-inspection environment.

Mission
-------
A small work-class ROV starts at a launch buoy at one end of a submerged
pipeline and has to inspect four stations spaced along it, *in order*, finishing
at the manifold at the far end. At each station it must fly up to the inspection
hoop and trigger a sensor scan. It has to do that

  * before the battery runs flat (energy budget that scales with how hard it
    drives its thrusters),
  * without straying far from the pipeline it is meant to be surveying,
  * without driving into the seabed, the pipe or the manifold, and
  * against a drifting water current that constantly pushes it off the line.

Each of those is a separate term in the reward, so the agent has to trade survey
speed against energy against staying on the pipe rather than optimising a single
axis.

Why the dynamics are not scripted
---------------------------------
The ten discrete actions do **not** teleport the vehicle. They nudge the
setpoints of an onboard velocity/heading controller (exactly how a real ROV is
flown from the surface: "go forward", "yaw right", "hold depth"). That
controller, buoyancy, quadratic hydrodynamic drag and the current field are all
summed into the body's external-force slot, and MuJoCo's rigid-body integrator
produces the motion. A bad sequence of actions genuinely drifts the vehicle into
the seabed or lets the current sweep it off the pipe; nothing is on rails.

There are no MuJoCo actuators: underwater a thruster is simply a force, so every
force is written straight into ``xfrc_applied`` from here, which keeps the
physics identical to what the reward and the observation are computed from.

Coordinate conventions
----------------------
World is z-up. Body x is forward, y is port (left), z is up. The observation is
expressed in the yaw-aligned (heading) frame, so "forward" always means the same
thing to the policy regardless of which way the vehicle is pointing.
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

SCENE_PATH = Path(__file__).resolve().parent.parent / "assets" / "subsea_scene.xml"

#: Pipeline centreline nodes (x, y). Mirrors the capsule <geom>s in the scene
#: XML exactly, so the corridor reward and the seabed channel are computed from
#: the same geometry the vehicle can actually collide with.
PIPE_NODES: np.ndarray = np.array(
    [
        [-28.0, 0.0],
        [-20.0, 4.0],
        [-12.0, 5.0],
        [-4.0, 2.0],
        [4.0, -2.0],
        [12.0, -5.0],
        [20.0, -4.0],
        [28.0, 0.0],
    ]
)

#: The four inspection stations, in the order they must be visited (x, y, z).
#: The first three are hoops on the pipe; the last is the manifold riser.
WAYPOINTS: np.ndarray = np.array(
    [
        [-20.0, 4.0, 1.1],
        [-4.0, 2.0, 1.1],
        [12.0, -5.0, 1.1],
        [28.0, 0.0, 1.5],
    ]
)
N_WAYPOINTS = len(WAYPOINTS)


class Action(IntEnum):
    """The ten commands the pilot (or the policy) can issue to the ROV."""

    HOLD = 0            # thrusters to idle - the vehicle coasts and drifts
    SURGE_FWD = 1       # drive forward along the heading
    SURGE_REV = 2       # back off
    YAW_LEFT = 3        # rotate heading to port
    YAW_RIGHT = 4       # rotate heading to starboard
    STRAFE_LEFT = 5     # translate to port without turning (lateral thrusters)
    STRAFE_RIGHT = 6    # translate to starboard
    ASCEND = 7          # rise
    DESCEND = 8         # dive
    INSPECT = 9         # trigger a scan of the active station - mission-critical


ACTION_MEANING: dict[int, str] = {
    Action.HOLD: "HOLD",
    Action.SURGE_FWD: "SURGE_FWD",
    Action.SURGE_REV: "SURGE_REV",
    Action.YAW_LEFT: "YAW_LEFT",
    Action.YAW_RIGHT: "YAW_RIGHT",
    Action.STRAFE_LEFT: "STRAFE_LEFT",
    Action.STRAFE_RIGHT: "STRAFE_RIGHT",
    Action.ASCEND: "ASCEND",
    Action.DESCEND: "DESCEND",
    Action.INSPECT: "INSPECT",
}


@dataclass
class EnvConfig:
    """Every tunable in one place so experiments can perturb the mission, not the code."""

    # --- timing -------------------------------------------------------------
    frame_skip: int = 10               # 10 x 10 ms  ->  a 10 Hz guidance loop
    max_episode_seconds: float = 55.0

    # --- manoeuvring envelope ----------------------------------------------
    max_surge: float = 2.4             # m/s, largest commandable forward speed
    max_sway: float = 1.4              # m/s, largest lateral speed
    max_vspeed: float = 1.2            # m/s, largest vertical speed
    surge_delta: float = 0.6           # m/s added per SURGE action
    sway_delta: float = 0.5            # m/s added per STRAFE action
    vspeed_delta: float = 0.4          # m/s added per ASCEND/DESCEND action
    yaw_delta: float = 0.16            # rad per YAW action

    # --- onboard controller / hydrodynamics --------------------------------
    mass: float = 11.0                 # kg (matches the XML inertial)
    kp_vel: float = 6.0                # thrust per (m/s) of velocity error, per kg
    thr_horiz_max: float = 45.0        # N, saturation of the horizontal thrusters
    thr_vert_max: float = 34.0         # N, saturation of the vertical thrusters
    kp_yaw: float = 6.0
    kd_yaw: float = 2.2
    k_right: float = 22.0              # passive metacentric righting stiffness
    k_angdrag: float = 6.0             # angular drag (keeps the hull settled)
    drag_lin: float = 6.0              # N per (m/s), low-speed damping
    drag_quad: float = 7.0             # N per (m/s)^2, dominant at cruise

    # --- consumables --------------------------------------------------------
    battery_endurance_s: float = 95.0  # seconds of full-thrust driving on a pack

    # --- current ------------------------------------------------------------
    current_mean_max: float = 0.9      # m/s, steady set drawn per episode
    current_sigma: float = 0.45        # m/s, gust/turbulence intensity
    current_theta: float = 0.5         # OU mean reversion

    # --- geometry -----------------------------------------------------------
    inspect_radius: float = 2.8        # m, must be this close to log a scan
    scan_speed_max: float = 1.0        # m/s, must be slower than this to auto-log
    scan_dwell_steps: int = 4          # control steps of station-keeping to auto-log
    corridor_half_width: float = 9.0   # m off the pipe before the survey degrades
    survey_x_min: float = -32.0
    survey_x_max: float = 32.0
    survey_y_abs: float = 16.0
    depth_ceiling: float = 10.0        # z above which the vehicle has surfaced
    depth_floor: float = 0.3           # z below which it is on the seabed
    min_agl: float = 0.6               # below this the seabed-proximity penalty bites

    # --- domain randomisation ----------------------------------------------
    randomize_terrain: bool = True
    n_ridges: int = 7

    # --- reward weights -----------------------------------------------------
    w_progress: float = 1.3            # per metre closed toward the active station
    w_step: float = 0.05
    w_energy: float = 0.04
    w_tilt: float = 0.2
    w_spin: float = 0.02
    w_corridor: float = 0.4            # for straying off the pipeline
    w_seabed: float = 0.6              # for hugging the seabed
    w_bounds: float = 0.7              # per metre outside the (soft) survey box
    r_inspect: float = 60.0            # peak per-station scan reward (decays with offset)
    r_station_bonus: float = 30.0      # flat bonus for a clean scan inside the hoop
    r_complete: float = 120.0          # finishing the whole survey
    p_bad_scan: float = 1.0            # triggering a scan with no station in range
    #: kept deliberately small: a large bad-scan penalty teaches the agent to
    #: avoid INSPECT altogether before it ever discovers that a scan inside a
    #: hoop pays off, which stalls learning at zero completed surveys.
    p_collision: float = 55.0          # driving into the seabed, pipe or manifold
    p_lost: float = 55.0               # leaving the survey box / surfacing
    p_battery: float = 50.0            # flat battery mid-survey
    p_capsize: float = 55.0            # tumbled past 70 degrees
    p_timeout: float = 12.0            # per un-inspected station left at the buzzer


# observation normalisation constants
POS_SCALE = 40.0
VEL_SCALE = 3.0
ANGVEL_SCALE = 4.0
CURRENT_SCALE = 2.0
ALT_SCALE = 6.0
RANGE_SCALE = 55.0

OBS_DIM = 28

#: Human-readable index map for the observation vector. Kept next to the builder
#: so the report and the code can never drift apart.
OBS_LAYOUT: list[tuple[str, str]] = [
    ("0-2", "position error to the active station, yaw-frame, / 40 m"),
    ("3-5", "velocity, yaw-frame, / 3 m/s"),
    ("6-8", "up direction in the body frame (encodes roll and pitch)"),
    ("9-10", "sin/cos of heading"),
    ("11-13", "body angular rates / 4 rad/s"),
    ("14", "battery remaining, 1 = full"),
    ("15", "survey progress, fraction of stations inspected"),
    ("16-17", "estimated current, yaw-frame, / 2 m/s"),
    ("18", "altitude above the seabed / 6 m"),
    ("19", "horizontal range to the active station / 55 m"),
    ("20", "fraction of the mission clock left"),
    ("21", "vertical speed / 3 m/s"),
    ("22", "pipeline corridor margin, 1 = on the pipe, 0 = at the edge"),
    ("23", "1 when the active station is within scan range, else 0"),
    ("24-27", "commanded surge / sway / vertical-speed / relative-heading setpoints"),
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


def _dist_to_polyline(x: float, y: float, nodes: np.ndarray) -> float:
    """Shortest distance in the xy-plane from a point to a polyline."""
    p = np.array([x, y])
    best = float("inf")
    for a, b in zip(nodes[:-1], nodes[1:]):
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom == 0.0 else max(0.0, min(1.0, float((p - a) @ ab) / denom))
        d = float(np.linalg.norm(p - (a + t * ab)))
        if d < best:
            best = d
    return best


def _grid_dist_to_polyline(gx: np.ndarray, gy: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Vectorised point-to-polyline distance over a whole meshgrid.

    Called once per reset over the 96x96 seabed grid, so the per-segment maths is
    done in numpy rather than looping in python.
    """
    best = np.full(gx.shape, np.inf)
    for a, b in zip(nodes[:-1], nodes[1:]):
        ax, ay = a
        abx, aby = b - a
        denom = abx * abx + aby * aby
        if denom == 0.0:
            d = np.hypot(gx - ax, gy - ay)
        else:
            t = np.clip(((gx - ax) * abx + (gy - ay) * aby) / denom, 0.0, 1.0)
            d = np.hypot(gx - (ax + t * abx), gy - (ay + t * aby))
        best = np.minimum(best, d)
    return best


class SubseaInspectionEnv(gym.Env):
    """Gymnasium environment for the ROV pipeline-inspection mission.

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
        # tracks wall-clock time. Training never renders, so this costs nothing
        # there.
        self.realtime = realtime
        self.playback_speed = playback_speed

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep * self.cfg.frame_skip
        self.max_steps = int(round(self.cfg.max_episode_seconds / self.dt))

        # --- cache ids so the hot loop never does string lookups -------------
        mj = mujoco
        self._bid_rov = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "rov")
        self._qadr_rov = self.model.jnt_qposadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "rov_free")
        ]
        self._dadr_rov = self.model.jnt_dofadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "rov_free")
        ]
        #: every geom that belongs to the vehicle, for collision detection
        self._rov_geoms = {
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self._bid_rov
        }

        # --- heightfield bookkeeping ----------------------------------------
        self._hf_nrow = int(self.model.hfield_nrow[0])
        self._hf_ncol = int(self.model.hfield_ncol[0])
        self._hf_adr = int(self.model.hfield_adr[0])
        self._hf_size = np.array(self.model.hfield_size[0])  # rx, ry, elev, base
        self._hfield = np.zeros((self._hf_nrow, self._hf_ncol), dtype=np.float64)
        self.terrain_dirty = True

        self.launch_xy = PIPE_NODES[0].copy()

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
        self.battery = 1.0
        self.surge_cmd = 0.0
        self.sway_cmd = 0.0
        self.vz_cmd = 0.0
        self.yaw_cmd = 0.0
        self.current = np.zeros(2)
        self.current_mean = np.zeros(2)
        self.active_wp = 0
        self.outcome = "surveying"
        self._prev_dist = 0.0
        self._last_action = int(Action.HOLD)
        self._last_reward = 0.0
        self._episode_return = 0.0
        self._last_thrust = 0.0
        self._dwell = 0
        self._scan_offsets: list[float] = []
        self._scanned_at: dict[int, float] = {}

    def _generate_terrain(self) -> None:
        """Build a fresh ridged seabed and carve a flat channel under the pipe."""
        rx, ry, elev = self._hf_size[0], self._hf_size[1], self._hf_size[2]
        xs = np.linspace(-rx, rx, self._hf_ncol)
        ys = np.linspace(-ry, ry, self._hf_nrow)
        gx, gy = np.meshgrid(xs, ys)

        h = np.zeros_like(gx)
        if self.cfg.randomize_terrain:
            for _ in range(self.cfg.n_ridges):
                cx = self.np_random.uniform(-rx * 0.85, rx * 0.85)
                cy = self.np_random.uniform(-ry * 0.85, ry * 0.85)
                amp = self.np_random.uniform(0.3, 0.85)
                sx = self.np_random.uniform(5.0, 13.0)
                sy = self.np_random.uniform(5.0, 13.0)
                h += amp * np.exp(-(((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2))
            # a low sand wave running across the survey box, so there is always
            # relief near the line the vehicle has to hold
            wave = self.np_random.uniform(-4.0, 4.0)
            h += self.np_random.uniform(0.25, 0.5) * np.exp(-(((gy - wave) / 6.0) ** 2))
        else:
            h += 0.4 * np.exp(-((gy / 6.0) ** 2))

        h = np.clip(h, 0.0, 1.0)

        # flatten a channel along the whole pipeline so the pipe sits on the
        # seabed at a known level and the vehicle has a clear survey corridor
        dist = _grid_dist_to_polyline(gx, gy, PIPE_NODES)
        channel_r = 5.5
        blend = np.clip((dist - channel_r) / (channel_r * 0.9), 0.0, 1.0)
        h = h * blend  # seabed level (0) inside the channel, ridges outside

        self._hfield = h
        self.model.hfield_data[self._hf_adr : self._hf_adr + h.size] = h.ravel()
        self.terrain_dirty = True
        self._terrain_elev = elev

    def terrain_height(self, x: float, y: float) -> float:
        """Bilinearly interpolated seabed height under a world point."""
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
        self._generate_terrain()

        # --- launch: hovering just off the buoy, pointing down the line -------
        start = np.array(
            [
                self.launch_xy[0] + self.np_random.uniform(-0.4, 0.4),
                self.launch_xy[1] + self.np_random.uniform(-0.4, 0.4),
                1.2 + self.np_random.uniform(-0.15, 0.25),
            ]
        )
        self.data.qpos[self._qadr_rov : self._qadr_rov + 3] = start
        yaw0 = self.np_random.uniform(-0.2, 0.2)
        self.data.qpos[self._qadr_rov + 3 : self._qadr_rov + 7] = np.array(
            [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
        )
        self.data.qvel[self._dadr_rov : self._dadr_rov + 3] = np.array(
            [self.np_random.uniform(0.0, 0.4), 0.0, 0.0]
        )
        self.yaw_cmd = yaw0

        # --- current ---------------------------------------------------------
        cdir = self.np_random.uniform(-math.pi, math.pi)
        cmag = self.np_random.uniform(0.0, self.cfg.current_mean_max)
        self.current_mean = np.array([cmag * math.cos(cdir), cmag * math.sin(cdir)])
        self.current = self.current_mean.copy()

        # --- consumables start partly used, so the agent cannot assume a full pack
        self.battery = float(self.np_random.uniform(0.85, 1.0))

        mujoco.mj_forward(self.model, self.data)
        self._prev_dist = float(np.linalg.norm(self._rov_pos() - self._active_target()))

        if self.record_trace:
            self._begin_trace()

        return self._observe(), self._info()

    # ------------------------------------------------------------------- step

    def step(self, action: int):
        action = int(action)
        cfg = self.cfg
        self._last_action = action

        # --- 1. the action edits the setpoints the onboard controller tracks --
        if action == Action.SURGE_FWD:
            self.surge_cmd = min(cfg.max_surge, self.surge_cmd + cfg.surge_delta)
        elif action == Action.SURGE_REV:
            self.surge_cmd = max(-cfg.max_surge, self.surge_cmd - cfg.surge_delta)
        elif action == Action.STRAFE_LEFT:
            self.sway_cmd = min(cfg.max_sway, self.sway_cmd + cfg.sway_delta)
        elif action == Action.STRAFE_RIGHT:
            self.sway_cmd = max(-cfg.max_sway, self.sway_cmd - cfg.sway_delta)
        elif action == Action.ASCEND:
            self.vz_cmd = min(cfg.max_vspeed, self.vz_cmd + cfg.vspeed_delta)
        elif action == Action.DESCEND:
            self.vz_cmd = max(-cfg.max_vspeed, self.vz_cmd - cfg.vspeed_delta)
        elif action == Action.YAW_LEFT:
            self.yaw_cmd = _wrap_pi(self.yaw_cmd + cfg.yaw_delta)
        elif action == Action.YAW_RIGHT:
            self.yaw_cmd = _wrap_pi(self.yaw_cmd - cfg.yaw_delta)
        elif action == Action.HOLD:
            # bleed the translational setpoints back toward zero (station-keep)
            self.surge_cmd *= 0.55
            self.sway_cmd *= 0.55
            self.vz_cmd *= 0.55

        scanned_now = action == Action.INSPECT

        # --- 2. roll the physics forward with the inner control loop closed ---
        self._resample_current()
        energy_acc = 0.0
        for _ in range(cfg.frame_skip):
            thr = self._hydro_loop()
            mujoco.mj_step(self.model, self.data)
            energy_acc += thr
        self._last_thrust = energy_acc / cfg.frame_skip

        # --- 3. consumables --------------------------------------------------
        power_norm = self._last_thrust / (cfg.thr_horiz_max * 2.0)
        self.battery -= self.dt * power_norm / cfg.battery_endurance_s
        self.battery = max(0.0, self.battery)

        self.step_count += 1

        # --- 4. reward and termination ---------------------------------------
        reward, terminated, truncated = self._reward_and_done(scanned_now, power_norm)
        self._last_reward = reward
        self._episode_return += reward

        obs = self._observe()
        info = self._info()

        if self.record_trace:
            self._append_trace()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -------------------------------------------- onboard control + hydro physics

    def _resample_current(self) -> None:
        """Ornstein-Uhlenbeck turbulence around the episode's steady current."""
        cfg = self.cfg
        h = self.dt
        self.current += cfg.current_theta * (
            self.current_mean - self.current
        ) * h + cfg.current_sigma * math.sqrt(h) * self.np_random.normal(size=2)

    def _hydro_loop(self) -> float:
        """Sum buoyancy, drag, current and thruster forces onto the body.

        This is the vehicle's own controller plus the water it moves through, not
        part of the policy: the policy only moves the setpoints. It runs inside
        the frame-skip loop at 100 Hz, which is what lets a 10 Hz discrete policy
        fly the vehicle smoothly. Written in scalars/small arrays because it runs
        ten times per environment step and tens of millions of times per sweep.

        Returns the horizontal thrust magnitude, used for the energy budget.

        Written in scalars rather than small numpy arrays: it runs ten times per
        environment step and tens of millions of times per sweep, and the numpy
        version spent more time allocating three-element vectors than MuJoCo
        spent integrating the physics.
        """
        cfg = self.cfg
        mat = self.data.xmat[self._bid_rov]  # row-major 3x3: 0,1,2 / 3,4,5 / 6,7,8
        yaw = math.atan2(mat[3], mat[0])
        qvel = self.data.qvel
        i = self._dadr_rov
        vx, vy, vz = qvel[i], qvel[i + 1], qvel[i + 2]
        wx, wy, wz = qvel[i + 3], qvel[i + 4], qvel[i + 5]

        # velocity relative to the moving water (current is horizontal)
        rx, ry, rz = vx - self.current[0], vy - self.current[1], vz
        speed_rel = math.sqrt(rx * rx + ry * ry + rz * rz)
        drag = cfg.drag_lin + cfg.drag_quad * speed_rel

        # buoyancy exactly cancels weight (near-neutral trim); drag opposes v_rel
        fx = -drag * rx
        fy = -drag * ry
        fz = cfg.mass * 9.81 - drag * rz

        # thruster forces from the velocity setpoints, expressed in world axes
        c, s = math.cos(yaw), math.sin(yaw)
        vdx = c * self.surge_cmd - s * self.sway_cmd
        vdy = s * self.surge_cmd + c * self.sway_cmd
        k = cfg.mass * cfg.kp_vel
        tx = k * (vdx - vx)
        ty = k * (vdy - vy)
        tz = k * (self.vz_cmd - vz)
        th = math.sqrt(tx * tx + ty * ty)
        if th > cfg.thr_horiz_max:
            scale = cfg.thr_horiz_max / th
            tx *= scale
            ty *= scale
            th = cfg.thr_horiz_max
        tz = -cfg.thr_vert_max if tz < -cfg.thr_vert_max else (
            cfg.thr_vert_max if tz > cfg.thr_vert_max else tz
        )
        fx += tx
        fy += ty
        fz += tz

        # yaw tracking + passive righting so the hull stays level and settled.
        # up = body z-axis in world = (mat[2], mat[5], mat[8]); the righting
        # torque is k_right * (up x world_up), which is (up_y, -up_x, 0).
        xf = self.data.xfrc_applied
        b = self._bid_rov
        xf[b, 0] = fx
        xf[b, 1] = fy
        xf[b, 2] = fz
        xf[b, 3] = cfg.k_right * mat[5] - cfg.k_angdrag * wx
        xf[b, 4] = -cfg.k_right * mat[2] - cfg.k_angdrag * wy
        xf[b, 5] = cfg.kp_yaw * _wrap_pi(self.yaw_cmd - yaw) - cfg.kd_yaw * wz
        return th + abs(tz)

    # ------------------------------------------------------------ observations

    def _rov_pos(self) -> np.ndarray:
        return self.data.xpos[self._bid_rov]

    def _rov_vel(self) -> np.ndarray:
        return self.data.qvel[self._dadr_rov : self._dadr_rov + 3]

    def _active_target(self) -> np.ndarray:
        return WAYPOINTS[min(self.active_wp, N_WAYPOINTS - 1)]

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        pos = self._rov_pos()
        vel = self._rov_vel()
        mat = self.data.xmat[self._bid_rov].reshape(3, 3)
        _, _, yaw = _euler_from_mat(mat)
        omega = self.data.qvel[self._dadr_rov + 3 : self._dadr_rov + 6]

        target = self._active_target()
        err_yaw = _to_yaw_frame(target - pos, yaw)
        vel_yaw = _to_yaw_frame(vel, yaw)
        current_yaw = _to_yaw_frame(np.array([self.current[0], self.current[1], 0.0]), yaw)

        up_body = mat.T @ np.array([0.0, 0.0, 1.0])
        agl = pos[2] - self.terrain_height(pos[0], pos[1])
        ground_range = float(np.linalg.norm((target - pos)[:2]))
        pipe_dist = _dist_to_polyline(pos[0], pos[1], PIPE_NODES)
        corridor_margin = 1.0 - min(1.0, pipe_dist / cfg.corridor_half_width)
        time_left = 1.0 - self.step_count / self.max_steps
        in_range = 1.0 if ground_range <= cfg.inspect_radius else 0.0

        obs = np.concatenate(
            [
                err_yaw / POS_SCALE,
                vel_yaw / VEL_SCALE,
                up_body,
                [math.sin(yaw), math.cos(yaw)],
                omega / ANGVEL_SCALE,
                [self.battery, self.active_wp / N_WAYPOINTS],
                current_yaw[:2] / CURRENT_SCALE,
                [agl / ALT_SCALE],
                [ground_range / RANGE_SCALE],
                [time_left],
                [vel[2] / VEL_SCALE],
                [corridor_margin],
                [in_range],
                # the actions edit persistent setpoints, so those setpoints are
                # part of the state and must be observable for the problem to
                # stay Markovian
                [
                    self.surge_cmd / cfg.max_surge,
                    self.sway_cmd / cfg.max_sway,
                    self.vz_cmd / cfg.max_vspeed,
                    _wrap_pi(self.yaw_cmd - yaw) / math.pi,
                ],
            ]
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # ------------------------------------------------------------------ reward

    def _reward_and_done(self, scanned_now: bool, power_norm: float):
        cfg = self.cfg
        pos = self._rov_pos()
        vel = self._rov_vel()
        mat = self.data.xmat[self._bid_rov].reshape(3, 3)
        omega = self.data.qvel[self._dadr_rov + 3 : self._dadr_rov + 6]

        reward = 0.0
        terminated = False
        truncated = False

        # --- dense guidance: close the range to the active station -----------
        target = self._active_target()
        dist = float(np.linalg.norm(target - pos))
        reward += cfg.w_progress * (self._prev_dist - dist)
        self._prev_dist = dist

        # --- running costs ----------------------------------------------------
        reward -= cfg.w_step
        reward -= cfg.w_energy * power_norm
        tilt = math.acos(max(-1.0, min(1.0, mat[2, 2])))
        reward -= cfg.w_tilt * tilt * tilt
        reward -= cfg.w_spin * min(9.0, float(np.dot(omega, omega)))

        # --- staying on the pipeline -----------------------------------------
        pipe_dist = _dist_to_polyline(pos[0], pos[1], PIPE_NODES)
        if pipe_dist > cfg.corridor_half_width * 0.55:
            reward -= cfg.w_corridor * (pipe_dist - cfg.corridor_half_width * 0.55)

        # --- seabed proximity -------------------------------------------------
        agl = pos[2] - self.terrain_height(pos[0], pos[1])
        if agl < cfg.min_agl:
            reward -= cfg.w_seabed * (cfg.min_agl - agl)

        speed = float(np.linalg.norm(vel))

        # --- hard failures ----------------------------------------------------
        # Only a genuine hard impact or a full tumble ends the episode. Merely
        # brushing the seabed or drifting outside the box is turned into a shaped
        # penalty below, not a termination: ending episodes at the first mistake
        # starves the agent of the full-horizon experience the progress shaping
        # needs to teach the transit, and learning stalls at zero completions.
        if self._rov_contact() and speed > 1.6:
            self.outcome = "collision"
            return reward - cfg.p_collision, True, False
        if tilt > 1.22:  # ~70 degrees
            self.outcome = "capsized"
            return reward - cfg.p_capsize, True, False

        # --- soft survey box: penalise leaving, keep the episode alive --------
        over = (
            max(0.0, pos[0] - cfg.survey_x_max)
            + max(0.0, cfg.survey_x_min - pos[0])
            + max(0.0, abs(pos[1]) - cfg.survey_y_abs)
            + max(0.0, pos[2] - cfg.depth_ceiling)
            + max(0.0, cfg.depth_floor - pos[2])
        )
        if over > 0.0:
            reward -= cfg.w_bounds * over
        if over > 5.0:  # only give up once it has escaped the survey area entirely
            self.outcome = "lost"
            return reward - cfg.p_lost, True, False

        if self.battery <= 0.0:
            self.outcome = "battery_depleted"
            return reward - cfg.p_battery, True, False

        # --- the inspection itself -------------------------------------------
        # A scan is logged either by firing INSPECT inside the hoop, or by
        # holding station inside it for a few control steps (the vehicle's
        # auto-log-on-station-keeping behaviour). INSPECT is the faster, manual
        # route a trained agent uses; the dwell fallback keeps the survey
        # completable by navigation alone.
        ground_range = float(np.linalg.norm((target - pos)[:2]))
        in_range = ground_range <= cfg.inspect_radius
        if in_range and speed < cfg.scan_speed_max:
            self._dwell += 1
        else:
            self._dwell = 0

        do_scan = in_range and (scanned_now or self._dwell >= cfg.scan_dwell_steps)
        if scanned_now and not in_range:
            # a manual scan with nothing in range wastes energy and sensor time
            reward -= cfg.p_bad_scan

        if do_scan:
            offset = float(np.linalg.norm(target - pos))
            self._scan_offsets.append(offset)
            self._scanned_at[self.active_wp] = float(self.step_count * self.dt)
            reward += cfg.r_inspect * math.exp(-((offset / cfg.inspect_radius) ** 2))
            reward += cfg.r_station_bonus
            self.active_wp += 1
            self._dwell = 0
            if self.active_wp >= N_WAYPOINTS:
                self.outcome = "survey_complete"
                return reward + cfg.r_complete, True, False
            # retarget: recompute the progress baseline to the next station
            self._prev_dist = float(np.linalg.norm(self._active_target() - pos))

        # --- clock ------------------------------------------------------------
        if self.step_count >= self.max_steps:
            self.outcome = "timeout"
            reward -= cfg.p_timeout * (N_WAYPOINTS - self.active_wp)
            truncated = True

        return reward, terminated, truncated

    def _rov_contact(self) -> bool:
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 in self._rov_geoms or con.geom2 in self._rov_geoms:
                return True
        return False

    # -------------------------------------------------------------------- info

    def _info(self) -> dict[str, Any]:
        pos = self._rov_pos()
        target = self._active_target()
        offsets = self._scan_offsets
        return {
            "outcome": self.outcome,
            "success": self.outcome == "survey_complete",
            "inspect_error": float(np.mean(offsets)) if offsets else float("nan"),
            "waypoints_done": int(self.active_wp),
            "waypoints_total": int(N_WAYPOINTS),
            "survey_progress": self.active_wp / N_WAYPOINTS,
            "battery": float(self.battery),
            "altitude_agl": float(pos[2] - self.terrain_height(pos[0], pos[1])),
            "range_to_wp": float(np.linalg.norm((target - pos)[:2])),
            "current_speed": float(np.linalg.norm(self.current)),
            "mission_time": float(self.step_count * self.dt),
            "speed": float(np.linalg.norm(self._rov_vel())),
            "action": ACTION_MEANING[self._last_action],
            "episode_return": float(self._episode_return),
        }

    # ------------------------------------------------------------ JSON tracing

    def _begin_trace(self) -> None:
        """Capture everything a browser needs to rebuild the scene from scratch."""
        step = max(1, self._hf_nrow // 48)
        coarse = self._hfield[::step, ::step]
        self.trace_header = {
            "schema": "subsea-rl-trace/1",
            "dt": self.dt,
            "seabed": {
                "rows": int(coarse.shape[0]),
                "cols": int(coarse.shape[1]),
                "size_x": float(self._hf_size[0]),
                "size_y": float(self._hf_size[1]),
                "elevation": float(self._hf_size[2]),
                "heights": [round(float(v), 4) for v in coarse.ravel()],
            },
            "launch": [float(self.launch_xy[0]), float(self.launch_xy[1])],
            "pipeline": [[round(float(x), 3), round(float(y), 3)] for x, y in PIPE_NODES],
            "waypoints": [[round(float(v), 3) for v in wp] for wp in WAYPOINTS],
            "inspect_radius": self.cfg.inspect_radius,
            "survey_box": {
                "x_min": self.cfg.survey_x_min,
                "x_max": self.cfg.survey_x_max,
                "y_abs": self.cfg.survey_y_abs,
                "ceiling": self.cfg.depth_ceiling,
            },
            "actions": [ACTION_MEANING[a] for a in sorted(ACTION_MEANING)],
        }
        self.trace = []

    def _append_trace(self) -> None:
        q = self.data.xquat
        self.trace.append(
            {
                "t": round(self.step_count * self.dt, 3),
                "rov": {
                    "p": [round(float(v), 3) for v in self._rov_pos()],
                    "q": [round(float(v), 4) for v in q[self._bid_rov]],
                },
                "a": int(self._last_action),
                "r": round(float(self._last_reward), 3),
                "bat": round(float(self.battery), 4),
                "wp": int(self.active_wp),
                "cur": [round(float(v), 3) for v in self.current],
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


def make_env(**kwargs) -> SubseaInspectionEnv:
    """Factory used by the training scripts and by :mod:`main`."""
    return SubseaInspectionEnv(**kwargs)


__all__ = [
    "SubseaInspectionEnv",
    "EnvConfig",
    "Action",
    "ACTION_MEANING",
    "OBS_LAYOUT",
    "OBS_DIM",
    "PIPE_NODES",
    "WAYPOINTS",
    "N_WAYPOINTS",
    "make_env",
]
