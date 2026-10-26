"""Tests for the SimToolReal-style baseline env (strl_env.py). Run: pytest -q tests/test_strl_env.py"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mathutils as mu  # noqa: E402
from strl_config import StrlCfg  # noqa: E402
from strl_env import StrlEnv  # noqa: E402


@pytest.fixture
def env():
    e = StrlEnv()
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture
def env_nodr():
    s = StrlCfg()
    s.dr = False
    e = StrlEnv(scfg=s)
    e.reset(seed=1)
    yield e
    e.close()


def test_spaces_and_finite(env):
    obs, info = env.reset(seed=3)
    assert env.action_space.shape == (6,)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert info["priv"].shape == (env.priv_dim,)
    assert np.all(np.isfinite(info["priv"]))


def test_random_rollout_finite(env):
    rng = np.random.default_rng(0)
    ends = 0
    for _ in range(800):
        obs, rew, term, trunc, info = env.step(rng.uniform(-1, 1, 6))
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        if term or trunc:
            assert "episode" in info
            ends += 1
            obs, info = env.reset()
            assert np.all(np.isfinite(obs))
    assert env.abs_goal_fallbacks == 0


def test_goal_at_current_pose_is_reached(env_nodr):
    env = env_nodr
    pos, _ = env.tool_pose()
    env.set_goal_pose(pos, env.data.xquat[env.tool_id], env.data.qpos[env.arm_qadr])
    assert env.keypoint_distance() < 1e-6
    old_goal = env.goal.p.copy()
    zero = np.zeros(6)
    rewards = []
    for _ in range(env.scfg.success_steps + 3):
        _, rew, term, trunc, info = env.step(zero)
        rewards.append(rew)
        if env.successes:
            break
    assert env.successes == 1
    assert not np.allclose(env.goal.p, old_goal)  # the next goal was sampled
    assert max(rewards) > 50.0  # near-goal bonus (1000 / 10 per step)


def test_delta_goals_reachable(env_nodr):
    env = env_nodr
    for _ in range(10):
        g = env._sample_delta_goal()
        env._scratch_pose(g.q_ref)
        pos = env._scratch.xpos[env.tool_id]
        quat = env._scratch.xquat[env.tool_id]
        assert np.linalg.norm(pos - g.p) < 0.015
        assert np.degrees(np.linalg.norm(mu.quat_error(g.quat, quat))) < 3.0
        assert env.goal_sampler.in_region(g.p)
        env.goal = g


def test_tolerance_changes_threshold(env):
    env.set_tolerance(0.02)
    assert abs(env.near_threshold() - 0.02 * env.scfg.keypoint_scale) < 1e-12


def test_tool_reuse_between_episodes(env_nodr):
    env = env_nodr
    every = env.scfg.new_tool_every
    model = env.model
    for k in range(1, every):
        obs, info = env.reset()
        assert info["reset_kind"] == "rehome"
        assert env.model is model  # no recompile
        assert np.all(np.isfinite(obs))
        assert env.successes == 0 and env.t == 0
        for _ in range(5):
            env.step(np.zeros(6))
    obs, info = env.reset()
    assert info["reset_kind"] == "full"
    assert env.model is not model
