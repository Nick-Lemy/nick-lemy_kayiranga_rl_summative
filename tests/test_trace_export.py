"""The JSON episode trace is a published contract.

The browser viewer in ``viewer/`` rebuilds the entire scene from this document
and knows nothing about MuJoCo, so the schema is the interface between the two
halves of the project. These tests pin it down: if a field is renamed or a shape
changes, the viewer breaks silently, and this is what catches it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from environment.custom_env import Action, SubseaInspectionEnv


@pytest.fixture(scope="module")
def trace(tmp_path_factory) -> dict:
    env = SubseaInspectionEnv(record_trace=True)
    env.reset(seed=120)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        done = term or trunc
    path = env.export_trace(tmp_path_factory.mktemp("trace") / "ep.json", extra={"agent": "test"})
    env.close()
    return json.loads(path.read_text())


def test_header_describes_the_whole_scene(trace):
    assert trace["schema"] == "subsea-rl-trace/1"
    assert trace["dt"] == pytest.approx(0.1)
    for key in ("seabed", "launch", "pipeline", "waypoints", "inspect_radius", "survey_box", "actions"):
        assert key in trace, f"missing header field {key}"

    t = trace["seabed"]
    assert len(t["heights"]) == t["rows"] * t["cols"], "heightfield is not rectangular"
    assert all(0.0 <= h <= 1.0 for h in t["heights"]), "heights are not normalised to 0..1"
    assert t["size_x"] > 0 and t["elevation"] > 0

    assert len(trace["waypoints"]) >= 2
    assert all(len(wp) == 3 for wp in trace["waypoints"]), "waypoints must be xyz"
    assert all(len(node) == 2 for node in trace["pipeline"]), "pipeline nodes must be xy"
    assert trace["actions"] == [a.name for a in sorted(Action)]


def test_every_frame_is_renderable(trace):
    frames = trace["frames"]
    assert len(frames) > 10
    n_wp = len(trace["waypoints"])
    for f in frames:
        assert len(f["rov"]["p"]) == 3, "position must be xyz"
        assert len(f["rov"]["q"]) == 4, "orientation must be a quaternion"
        assert abs(np.linalg.norm(f["rov"]["q"]) - 1.0) < 1e-2, "quaternion not unit"
        assert 0 <= f["a"] < len(Action)
        assert 0.0 <= f["bat"] <= 1.0
        assert 0 <= f["wp"] <= n_wp
        assert len(f["cur"]) == 2


def test_time_advances_monotonically(trace):
    times = [f["t"] for f in trace["frames"]]
    assert times == sorted(times)
    assert times[0] == pytest.approx(trace["dt"])


def test_survey_progress_only_advances(trace):
    """`wp` is the active-station index; it must be monotonic non-decreasing."""
    wps = [f["wp"] for f in trace["frames"]]
    assert all(b >= a for a, b in zip(wps, wps[1:])), "survey progress went backwards"


def test_summary_reports_the_outcome(trace):
    s = trace["summary"]
    assert s["agent"] == "test"
    assert s["steps"] == len(trace["frames"])
    assert s["outcome"] in {
        "survey_complete",
        "collision",
        "capsized",
        "lost",
        "battery_depleted",
        "timeout",
        "surveying",
    }
