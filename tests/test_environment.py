"""Contract tests for the delivery environment.

These are the properties the RL results depend on. If any of them break, the
numbers in the report stop meaning what they say, so they are asserted rather
than assumed: the spaces match what the algorithms were configured against, the
episode really is Markovian in the observation, the physics is deterministic
given a seed, every terminal condition is reachable, and the reward actually
prefers a good delivery to a bad one.

    uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.custom_env import (
    ACTION_MEANING,
    OBS_DIM,
    Action,
    EnvConfig,
    ZiplineDeliveryEnv,
)


@pytest.fixture(scope="module")
def env():
    e = ZiplineDeliveryEnv()
    yield e
    e.close()


# ------------------------------------------------------------------- contract


def test_spaces_match_the_trained_configuration(env):
    assert env.action_space.n == len(Action) == 10
    assert env.observation_space.shape == (OBS_DIM,) == (27,)
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
    have completely different commanded attitudes and therefore different
    dynamics.
    """
    env = ZiplineDeliveryEnv()
    obs, _ = env.reset(seed=3)
    for _ in range(4):
        env.step(int(Action.ROLL_RIGHT))
    obs_right, *_ = env.step(int(Action.HOVER))

    env.reset(seed=3)
    for _ in range(4):
        env.step(int(Action.ROLL_LEFT))
    obs_left, *_ = env.step(int(Action.HOVER))

    # index 24 is the roll setpoint
    assert obs_right[24] > 0 > obs_left[24]
    env.close()


# ---------------------------------------------------------------- determinism


def test_same_seed_gives_the_same_episode():
    a, b = ZiplineDeliveryEnv(), ZiplineDeliveryEnv()
    obs_a, _ = a.reset(seed=1234)
    obs_b, _ = b.reset(seed=1234)
    np.testing.assert_allclose(obs_a, obs_b)
    for action in [1, 3, 3, 0, 6, 2, 7, 9, 0, 0]:
        oa, ra, ta, ua, _ = a.step(action)
        ob, rb, tb, ub, _ = b.step(action)
        np.testing.assert_allclose(oa, ob, atol=1e-6)
        assert ra == pytest.approx(rb) and ta == tb and ua == ub
    a.close()
    b.close()


def test_reset_randomises_the_mission():
    """Terrain, drop zone and weather must actually vary, or there is nothing
    to generalise to."""
    env = ZiplineDeliveryEnv()
    posts, terrains, winds = [], [], []
    for seed in range(6):
        env.reset(seed=seed)
        posts.append(tuple(env.post_xy))
        terrains.append(env._hfield.copy())
        winds.append(tuple(env.wind_mean))
    assert len({p for p in posts}) == len(posts), "drop zone did not move"
    assert len({w for w in winds}) == len(winds), "wind did not change"
    assert not np.allclose(terrains[0], terrains[1]), "terrain did not change"
    env.close()


# ------------------------------------------------------------------ mechanics


def test_release_detaches_the_payload():
    env = ZiplineDeliveryEnv()
    env.reset(seed=7)
    assert env.attached and env.data.eq_active[env._eq_weld] == 1

    gap_before = np.linalg.norm(env._drone_pos() - env._payload_pos())
    env.step(int(Action.RELEASE_PAYLOAD))
    assert not env.attached and env.data.eq_active[env._eq_weld] == 0

    for _ in range(12):
        env.step(int(Action.HOVER))
    gap_after = np.linalg.norm(env._drone_pos() - env._payload_pos())
    assert gap_after > gap_before + 0.5, "payload did not fall away from the aircraft"
    env.close()


def test_payload_descends_under_canopy_not_free_fall():
    """The chute is what makes the wind matter at release; check it is doing work."""
    env = ZiplineDeliveryEnv()
    env.reset(seed=11)
    for _ in range(30):  # climb first so there is room to fall
        env.step(int(Action.THROTTLE_UP))
    env.step(int(Action.RELEASE_PAYLOAD))
    speeds = []
    for _ in range(40):
        env.step(int(Action.HOVER))
        speeds.append(abs(env.data.qvel[env._dadr_payload + 2]))
    # free fall over 4 s would pass 30 m/s; the canopy should cap it near 5
    assert max(speeds) < 12.0, f"payload fell too fast: {max(speeds):.1f} m/s"
    env.close()


def test_cold_chain_and_battery_only_run_down():
    env = ZiplineDeliveryEnv()
    env.reset(seed=5)
    prev_bat, prev_cold = env.battery, env.cold_chain
    for _ in range(60):
        env.step(int(Action.HOVER))
        assert env.battery <= prev_bat + 1e-9
        assert env.cold_chain <= prev_cold + 1e-9
        prev_bat, prev_cold = env.battery, env.cold_chain
    assert env.battery < 1.0 and env.cold_chain < 1.0
    env.close()


# -------------------------------------------------------------------- rewards


def test_hovering_at_the_pad_runs_the_clock_out():
    """Doing nothing must be clearly bad, or the agent can farm shaping reward."""
    env = ZiplineDeliveryEnv()
    env.reset(seed=2)
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, info = env.step(int(Action.HOVER))
        total += r
        done = term or trunc
    assert total < -50, f"idling scored {total:.1f}, which is not punishing enough"
    assert info["outcome"] != "delivered"
    env.close()


def test_accurate_delivery_beats_dumping_the_payload():
    """The reward has to rank a good drop above a bad one by a wide margin."""
    cfg = EnvConfig()
    env = ZiplineDeliveryEnv(config=cfg)

    # dump it immediately, at the launch pad, ~60 m from the zone
    env.reset(seed=21)
    dumped, done = 0.0, False
    _, r, term, trunc, _ = env.step(int(Action.RELEASE_PAYLOAD))
    dumped += r
    done = term or trunc
    while not done:
        _, r, term, trunc, dump_info = env.step(int(Action.HOVER))
        dumped += r
        done = term or trunc

    # teleport the aircraft over the zone, then release
    env.reset(seed=21)
    env.data.qpos[env._qadr_drone : env._qadr_drone + 3] = env.aim_point
    env.data.qpos[env._qadr_payload : env._qadr_payload + 3] = env.aim_point - [0, 0, 0.155]
    env.data.qvel[:] = 0
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    good, done = 0.0, False
    _, r, term, trunc, _ = env.step(int(Action.RELEASE_PAYLOAD))
    good += r
    done = term or trunc
    while not done:
        _, r, term, trunc, good_info = env.step(int(Action.HOVER))
        good += r
        done = term or trunc

    assert good_info["outcome"] == "delivered", good_info["outcome"]
    assert dump_info["outcome"] == "missed_zone", dump_info["outcome"]
    assert good > dumped + 100, f"good drop {good:.1f} vs dump {dumped:.1f}"
    env.close()


def test_corridor_breach_terminates():
    env = ZiplineDeliveryEnv()
    env.reset(seed=13)
    # fly sideways until the regulated airspace runs out
    outcome = None
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(int(Action.ROLL_LEFT))
        if term or trunc:
            outcome = info["outcome"]
            break
    assert outcome is not None, "flying sideways for a whole episode never terminated"
    env.close()


def test_env_config_is_respected():
    tight = EnvConfig(cold_chain_s=4.0)
    env = ZiplineDeliveryEnv(config=tight)
    env.reset(seed=1)
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(int(Action.HOVER))
        done = term or trunc
    assert info["outcome"] == "cold_chain_expired"
    assert info["flight_time"] <= 5.0
    env.close()
