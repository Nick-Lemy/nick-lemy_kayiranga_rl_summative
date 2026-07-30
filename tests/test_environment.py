"""Contract tests for the subsea inspection environment.

These are the properties the RL results depend on. If any of them break, the
numbers in the report stop meaning what they say, so they are asserted rather
than assumed: the spaces match what the algorithms were configured against, the
episode really is Markovian in the observation, the physics is deterministic
given a seed, every terminal condition is reachable, and the reward actually
prefers a clean inspection to a wasted one.

    uv run pytest -q
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environment.custom_env import (
    ACTION_MEANING,
    OBS_DIM,
    WAYPOINTS,
    Action,
    EnvConfig,
    SubseaInspectionEnv,
)


@pytest.fixture(scope="module")
def env():
    e = SubseaInspectionEnv()
    yield e
    e.close()


# ------------------------------------------------------------------- contract


def test_spaces_match_the_trained_configuration(env):
    assert env.action_space.n == len(Action) == 10
    assert env.observation_space.shape == (OBS_DIM,) == (28,)
    assert set(ACTION_MEANING) == {int(a) for a in Action}


def test_observation_is_finite_and_bounded(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(400):
        obs, reward, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        assert np.all(np.isfinite(obs)), "non-finite observation"
        assert np.all(np.abs(obs) <= 10.0), "observation outside the clipped range"
        assert np.isfinite(reward)
        if term or trunc:
            obs, _ = env.reset()


def test_setpoints_are_observable():
    """The actions edit persistent setpoints, so they have to be in the state.

    Without this the problem is not Markovian: two identical-looking states can
    have completely different commanded velocities and therefore different
    dynamics.
    """
    env = SubseaInspectionEnv()
    env.reset(seed=3)
    for _ in range(4):
        env.step(int(Action.SURGE_FWD))
    obs_fwd, *_ = env.step(int(Action.YAW_LEFT))

    env.reset(seed=3)
    for _ in range(4):
        env.step(int(Action.SURGE_REV))
    obs_rev, *_ = env.step(int(Action.YAW_LEFT))

    # index 24 is the surge setpoint
    assert obs_fwd[24] > 0 > obs_rev[24]
    env.close()


# ---------------------------------------------------------------- determinism


def test_same_seed_gives_the_same_episode():
    a, b = SubseaInspectionEnv(), SubseaInspectionEnv()
    obs_a, _ = a.reset(seed=1234)
    obs_b, _ = b.reset(seed=1234)
    np.testing.assert_allclose(obs_a, obs_b)
    for action in [1, 3, 3, 0, 6, 2, 7, 9, 0, 5]:
        oa, ra, ta, ua, _ = a.step(action)
        ob, rb, tb, ub, _ = b.step(action)
        np.testing.assert_allclose(oa, ob, atol=1e-6)
        assert ra == pytest.approx(rb) and ta == tb and ua == ub
    a.close()
    b.close()


def test_reset_randomises_the_mission():
    """Seabed and current must actually vary, or there is nothing to generalise to."""
    env = SubseaInspectionEnv()
    terrains, currents = [], []
    for seed in range(6):
        env.reset(seed=seed)
        terrains.append(env._hfield.copy())
        currents.append(tuple(env.current_mean))
    assert len({c for c in currents}) == len(currents), "current did not change"
    assert not np.allclose(terrains[0], terrains[1]), "seabed did not change"
    env.close()


# ------------------------------------------------------------------ mechanics


def _teleport(env, xyz):
    """Place the vehicle at a world point, at rest, and settle the physics."""
    env.data.qpos[env._qadr_rov : env._qadr_rov + 3] = xyz
    env.data.qpos[env._qadr_rov + 3 : env._qadr_rov + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_inspect_in_range_advances_the_survey():
    env = SubseaInspectionEnv()
    env.reset(seed=7)
    assert env.active_wp == 0
    _teleport(env, WAYPOINTS[0])
    _, reward, term, trunc, info = env.step(int(Action.INSPECT))
    assert env.active_wp == 1, "a scan inside the hoop did not advance the survey"
    assert reward > 20.0, f"a clean scan should pay well, got {reward:.1f}"
    assert not (term or trunc)
    env.close()


def test_inspect_out_of_range_is_penalised_and_changes_nothing():
    env = SubseaInspectionEnv()
    env.reset(seed=7)
    # far from station 0, well outside the hoop
    _teleport(env, [WAYPOINTS[0][0], WAYPOINTS[0][1] + 12.0, 1.2])
    before = env.active_wp
    _, reward, *_ = env.step(int(Action.INSPECT))
    assert env.active_wp == before, "a scan with nothing in range advanced the survey"
    assert reward < 0, "a wasted scan should cost something"
    env.close()


def test_full_survey_via_teleport_scores_high():
    """Inspecting all four stations in order must terminate as survey_complete."""
    env = SubseaInspectionEnv()
    env.reset(seed=9)
    total, outcome = 0.0, None
    for wp in WAYPOINTS:
        # sit just off the hoop centre so we are still inside scan range but
        # clear of the pipe and the manifold riser that occupy the exact points
        _teleport(env, [wp[0], wp[1] + 1.5, wp[2]])
        _, r, term, trunc, info = env.step(int(Action.INSPECT))
        total += r
        outcome = info["outcome"]
        if term or trunc:
            break
    assert outcome == "survey_complete", outcome
    assert info["success"] is True
    assert total > 200.0, f"a perfect survey should score well, got {total:.1f}"
    env.close()


def test_battery_only_runs_down():
    env = SubseaInspectionEnv()
    env.reset(seed=5)
    prev = env.battery
    for _ in range(60):
        env.step(int(Action.SURGE_FWD))
        assert env.battery <= prev + 1e-9
        prev = env.battery
    assert env.battery < 1.0
    env.close()


# -------------------------------------------------------------------- rewards


def test_idling_runs_the_clock_out_and_is_clearly_bad():
    """Doing nothing must be clearly bad, or the agent can farm shaping reward."""
    env = SubseaInspectionEnv()
    env.reset(seed=2)
    total, done, info = 0.0, False, {}
    while not done:
        _, r, term, trunc, info = env.step(int(Action.HOLD))
        total += r
        done = term or trunc
    assert total < -20, f"idling scored {total:.1f}, which is not punishing enough"
    assert info["outcome"] != "survey_complete"
    env.close()


def test_leaving_the_survey_box_terminates():
    env = SubseaInspectionEnv()
    env.reset(seed=13)
    outcome = None
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(int(Action.ASCEND))  # climb until surfaced
        if term or trunc:
            outcome = info["outcome"]
            break
    assert outcome == "lost", f"surfacing should end the episode as 'lost', got {outcome}"
    env.close()


def test_env_config_is_respected():
    tight = EnvConfig(battery_endurance_s=6.0)
    env = SubseaInspectionEnv(config=tight)
    env.reset(seed=1)
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(int(Action.SURGE_FWD))
        done = term or trunc
    assert info["outcome"] == "battery_depleted", info["outcome"]
    env.close()
