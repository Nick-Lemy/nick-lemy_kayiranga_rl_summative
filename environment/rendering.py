"""Visualisation for the subsea inspection environment.

Three ways to look at the same simulation:

``PassiveViewer``
    MuJoCo's native interactive 3D window. Used for the live demo - you can
    orbit, zoom and pause while the policy flies the vehicle.
``OffscreenRenderer``
    Headless RGB frames with a telemetry HUD burned in, for recording MP4s and
    for the still that goes in the report.
``record_video``
    Convenience wrapper that rolls out one episode and writes an MP4.

Both renderers have to re-upload the heightfield to the GPU whenever the
environment regenerates the seabed, otherwise the vehicle moves over relief
that is no longer the one the physics is using.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from environment.custom_env import SubseaInspectionEnv

# HUD colours (R, G, B)
_WHITE = (245, 247, 250)
_DIM = (170, 178, 190)
_GOOD = (90, 220, 130)
_WARN = (255, 190, 70)
_BAD = (255, 95, 95)
_ACCENT = (120, 190, 255)


def _chase_camera(env: "SubseaInspectionEnv", cam: mujoco.MjvCamera, mode: str = "chase") -> None:
    """Point a free camera at the vehicle.

    ``chase`` trails the ROV along its heading, which is what reads best on
    video; ``mission`` pulls back to frame the whole pipeline so the launch buoy,
    the survey line and the manifold are visible at once; ``station`` frames the
    active inspection hoop the vehicle is closing on.
    """
    pos = env._rov_pos()
    if mode == "mission":
        cam.lookat[:] = np.array([0.0, 0.0, 2.0])
        cam.distance = 72.0
        cam.azimuth = 74.0
        cam.elevation = -26.0
        return

    if mode == "station":
        # Frames the vehicle with the station it is inspecting, from slightly
        # above so the seabed and the hoop both stay in shot.
        cam.lookat[:] = 0.5 * (pos + env._active_target())
        cam.distance = 12.0
        cam.azimuth = 58.0
        cam.elevation = -18.0
        return

    # MuJoCo's azimuth is the direction the camera *looks along*, so trailing the
    # vehicle means matching its yaw, not opposing it.
    mat = env.data.xmat[env._bid_rov].reshape(3, 3)
    yaw = math.atan2(mat[1, 0], mat[0, 0])
    cam.lookat[:] = pos
    cam.distance = 6.5
    cam.azimuth = math.degrees(yaw)
    cam.elevation = -12.0


class PassiveViewer:
    """Thin wrapper over ``mujoco.viewer.launch_passive``.

    ``launch_passive`` does not pace anything - it just draws whatever is in
    ``MjData`` when you call ``sync``. Driving the loop from python therefore
    replays the flight as fast as the CPU can integrate it, which here is on the
    order of a hundred times real time and looks like the vehicle teleporting.
    This wrapper sleeps the difference so one second of simulated flight takes
    one second of wall clock, at ``speed`` times normal.
    """

    def __init__(
        self,
        env: "SubseaInspectionEnv",
        camera: str = "chase",
        realtime: bool = True,
        speed: float = 1.0,
    ) -> None:
        import mujoco.viewer as mjv

        self.env = env
        self.camera = camera
        self.realtime = realtime
        self.speed = max(0.05, float(speed))
        self._handle = mjv.launch_passive(
            env.model, env.data, show_left_ui=False, show_right_ui=False
        )
        self._handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._upload_terrain()
        self._wall_ref = time.perf_counter()
        self._sim_ref = float(env.data.time)

    def _upload_terrain(self) -> None:
        try:
            self._handle.update_hfield(0)
        except Exception:  # pragma: no cover - viewer already gone
            pass
        self.env.terrain_dirty = False

    def _pace(self) -> None:
        """Hold the frame until wall clock catches up with simulated time."""
        sim_t = float(self.env.data.time)
        # mj_resetData rewinds the clock, so re-baseline on every episode
        if sim_t < self._sim_ref:
            self._wall_ref = time.perf_counter()
            self._sim_ref = sim_t
            return
        target = self._wall_ref + (sim_t - self._sim_ref) / self.speed
        lag = target - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        elif lag < -0.5:
            # we fell badly behind (a slow frame, or the window was dragged);
            # resync rather than sprinting to catch up
            self._wall_ref = time.perf_counter()
            self._sim_ref = sim_t

    def sync(self) -> None:
        if not self._handle.is_running():
            return
        if self.env.terrain_dirty:
            self._upload_terrain()
        if self.camera != "free":
            _chase_camera(self.env, self._handle.cam, self.camera)
        self._handle.sync()
        if self.realtime:
            self._pace()

    @property
    def running(self) -> bool:
        return self._handle.is_running()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # pragma: no cover
            pass


class OffscreenRenderer:
    """Renders annotated RGB frames without needing a window."""

    def __init__(
        self,
        env: "SubseaInspectionEnv",
        width: int = 1280,
        height: int = 720,
        camera: str = "chase",
        hud: bool = True,
    ) -> None:
        self.env = env
        self.camera = camera
        self.hud = hud
        self._renderer = mujoco.Renderer(env.model, height=height, width=width)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._font = _load_font(int(height * 0.026))
        self._font_small = _load_font(int(height * 0.020))
        self._upload_terrain()

    def _upload_terrain(self) -> None:
        # The Renderer keeps its MjrContext private, but the terrain buffer has
        # to be pushed to the GPU after every reset or the render shows stale
        # hills. Falling back silently is fine: the frame is still correct
        # everywhere except the terrain mesh.
        try:
            mujoco.mjr_uploadHField(self.env.model, self._renderer._mjr_context, 0)
        except Exception:  # pragma: no cover
            pass
        self.env.terrain_dirty = False

    def frame(self) -> np.ndarray:
        if self.env.terrain_dirty:
            self._upload_terrain()
        _chase_camera(self.env, self._cam, self.camera)
        self._renderer.update_scene(self.env.data, camera=self._cam)
        img = self._renderer.render()
        if self.hud:
            img = self._draw_hud(img)
        return img

    def _draw_hud(self, img: np.ndarray) -> np.ndarray:
        """Burn telemetry onto the frame so the video shows the agent's state."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover
            return img

        env = self.env
        info = env._info()
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil, "RGBA")
        w, h = pil.size
        pad = int(w * 0.016)
        line = int(h * 0.038)

        # ---- left panel: mission telemetry --------------------------------
        panel_w, panel_h = int(w * 0.30), int(line * 6.6)
        draw.rounded_rectangle(
            [pad, pad, pad + panel_w, pad + panel_h], radius=10, fill=(12, 16, 24, 190)
        )
        x, y = pad + int(w * 0.014), pad + int(line * 0.35)

        draw.text((x, y), "SUBSEA INSPECTION ROV", font=self._font, fill=_ACCENT)
        y += line
        draw.text(
            (x, y),
            f"t {info['mission_time']:5.1f} s     range {info['range_to_wp']:5.1f} m",
            font=self._font_small,
            fill=_WHITE,
        )
        y += int(line * 0.85)
        draw.text(
            (x, y),
            f"depth {info['altitude_agl']:4.1f} m AB   "
            f"station {info['waypoints_done']}/{info['waypoints_total']}"
            f"   set {info['current_speed']:.2f} m/s",
            font=self._font_small,
            fill=_WHITE,
        )
        y += int(line * 1.0)

        bar_w = panel_w - int(w * 0.028) - int(w * 0.075)
        self._bar(draw, x, y, bar_w, int(line * 0.42), info["battery"], "BATTERY")
        y += int(line * 1.05)
        self._bar(draw, x, y, bar_w, int(line * 0.42), info["survey_progress"], "SURVEY")
        y += int(line * 1.15)

        draw.text(
            (x, y),
            f"return {info['episode_return']:8.1f}",
            font=self._font_small,
            fill=_WHITE,
        )

        # ---- right panel: the action being taken --------------------------
        act_txt = info["action"]
        tw = draw.textlength(act_txt, font=self._font)
        bw = int(tw + w * 0.045)
        draw.rounded_rectangle(
            [w - pad - bw, pad, w - pad, pad + int(line * 1.5)],
            radius=10,
            fill=(12, 16, 24, 190),
        )
        draw.text(
            (w - pad - bw + int(w * 0.022), pad + int(line * 0.36)),
            act_txt,
            font=self._font,
            fill=_WARN,
        )

        # ---- outcome banner -------------------------------------------------
        if info["outcome"] not in ("surveying",):
            good = info["outcome"] == "survey_complete"
            txt = info["outcome"].replace("_", " ").upper()
            if good and not math.isnan(info["inspect_error"]):
                txt += f"   scan offset {info['inspect_error']:.2f} m"
            tw = draw.textlength(txt, font=self._font)
            bx = (w - tw) / 2
            draw.rounded_rectangle(
                [bx - pad, h - pad - int(line * 1.6), bx + tw + pad, h - pad],
                radius=10,
                fill=(12, 40, 20, 210) if good else (48, 12, 12, 210),
            )
            draw.text(
                (bx, h - pad - int(line * 1.22)),
                txt,
                font=self._font,
                fill=_GOOD if good else _BAD,
            )

        return np.asarray(pil)

    def _bar(self, draw, x: int, y: int, w: int, h: int, frac: float, label: str) -> None:
        frac = float(np.clip(frac, 0.0, 1.0))
        colour = _GOOD if frac > 0.5 else (_WARN if frac > 0.2 else _BAD)
        draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=(60, 66, 78, 255))
        if frac > 0.01:
            draw.rounded_rectangle(
                [x, y, x + int(w * frac), y + h], radius=h // 2, fill=colour + (255,)
            )
        draw.text((x + w + int(h * 0.6), y - h * 0.35), label, font=self._font_small, fill=_DIM)

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:  # pragma: no cover
            pass


def _load_font(size: int):
    """Best-effort truetype lookup; PIL's bitmap default ignores size."""
    try:
        from PIL import ImageFont
    except ImportError:  # pragma: no cover
        return None
    candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # pragma: no cover
                continue
    try:
        from PIL import ImageFont

        return ImageFont.load_default(size)
    except Exception:  # pragma: no cover
        return None


def record_video(
    env: "SubseaInspectionEnv",
    policy,
    path: str | Path,
    max_steps: int | None = None,
    fps: int = 20,
    seed: int | None = None,
    camera: str = "chase",
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Roll one episode out to an MP4.

    ``policy`` takes an observation and returns an action, so an SB3 model's
    ``predict`` can be passed through a one-line lambda.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    renderer = OffscreenRenderer(env, width=width, height=height, camera=camera)
    obs, _ = env.reset(seed=seed)
    frames = [renderer.frame()]
    steps = max_steps or env.max_steps
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        frames.append(renderer.frame())
        if terminated or truncated:
            break
    # hold the final frame so the outcome banner is readable
    frames.extend([frames[-1]] * fps)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)
    renderer.close()
    return path
