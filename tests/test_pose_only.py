"""Pose-only baseline (EnvCfg.goal_velocity False): observation, success rules, bonus and the PoseOracle."""
import dataclasses

import numpy as np
import pytest


def _pose_env(cfg, rule: str, seed: int = 0):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(dataclasses.replace(cfg, goal_velocity=False, pose_success=rule))
    e.reset(seed=seed)
    return e


def _goal_here(env, d=(0.0, 0.0, -1.0), offset=0.0):
    """A goal at the current tool pose (shifted by offset along d), with a goal velocity the head doesn't have."""
    from goals import Goal

    d = np.asarray(d, dtype=np.float64)
    head = env.data.site_xpos[env.head_site].copy()
    env.goal = Goal(p=head + offset * d, v=0.4 * d, quat=env._tool_quat(), axis=d)


def test_pose_only_drops_the_goal_velocity_from_the_observation(env, cfg):
    for rule, extra_priv in (("hold", 1), ("touch", 0)):
        e = _pose_env(cfg, rule)
        assert e.observation_space.shape[0] == env.observation_space.shape[0] - 10
        assert e.priv_dim == env.priv_dim + extra_priv
        # The observation doesn't change with the goal velocity.
        o1 = e._get_obs()
        e.goal.v = -3.0 * e.goal.v
        assert np.array_equal(o1, e._get_obs())
        e.close()


def test_bad_pose_success_rule_is_rejected(cfg):
    from env import PoseVelocityEnv

    with pytest.raises(ValueError):
        PoseVelocityEnv(dataclasses.replace(cfg, goal_velocity=False, pose_success="nope"))


def test_touch_succeeds_on_the_first_near_step_at_any_velocity(cfg):
    e = _pose_env(cfg, "touch")
    _goal_here(e)
    _, _, term, _, info = e.step(np.zeros(e.action_space.shape))
    assert term and info["passed"] and info["success"]
    assert np.isclose(info["episode"]["terms"]["bonus"], e.cfg.reward.reach_goal_bonus)
    e.close()


def test_hold_pays_per_near_step_and_succeeds_after_success_steps(cfg):
    e = _pose_env(cfg, "hold")
    n, bonus = e.cfg.success_steps, e.cfg.reward.reach_goal_bonus
    zero = np.zeros(e.action_space.shape)
    _goal_here(e)
    goal = e.goal
    _, _, term, _, info = e.step(zero)
    assert not term and not info["success"] and e.near_steps == 1
    assert np.isclose(e.ep_terms["bonus"], bonus / n)
    # A step away from the goal pays no bonus and keeps the count (near steps need not be consecutive).
    _goal_here(e, offset=0.3)
    _, _, term, _, _ = e.step(zero)
    assert not term and e.near_steps == 1 and np.isclose(e.ep_terms["bonus"], bonus / n)
    e.goal = goal
    for k in range(2, n + 1):
        _, _, term, _, info = e.step(zero)
        assert e.near_steps == k
        assert term == (k == n) and info["success"] == (k == n)
    assert np.isclose(info["episode"]["terms"]["bonus"], bonus)
    assert info["episode"]["near_steps"] == n
    e.close()


@pytest.mark.parametrize("rule", ["hold", "touch"])
def test_pose_oracle_succeeds(cfg, rule):
    """The scripted reach (oracle.PoseOracle) succeeds on most weld goals under both success rules."""
    e = _pose_env(cfg, rule, seed=1)
    successes = 0
    for _ in range(12):
        while True:
            _, _, term, trunc, info = e.step(e.oracle_action())
            if term or trunc:
                break
        successes += info["episode"]["success"]
        e.reset()
    assert successes >= 10, successes
    e.close()


def _run_task(cfg, task: str, rule: str, episodes: int):
    from config import TaskCfg
    from oracle import PoseOracle
    from tasks import TaskEnv

    env = TaskEnv(task, dataclasses.replace(cfg, goal_velocity=False, pose_success=rule), TaskCfg())
    env.set_curriculum(1.0)
    pol, out = PoseOracle(), []
    env.reset(seed=0)
    for _ in range(episodes):
        while True:
            _, _, term, trunc, info = env.step(pol(env))
            if term or trunc:
                break
        out.append(info["task"])
        env.reset()
    env.close()
    return out


@pytest.mark.parametrize("rule", ["hold", "touch"])
def test_pose_only_task_keypoints_advance_by_the_pose_rule(cfg, rule):
    """A pose-only policy stops at each keypoint instead of crossing its plane; the task env still advances."""
    for m in _run_task(cfg, "sweep", rule, 2):
        assert m["keypoints_passed"] == m["keypoints"] and not m["timeout"], m


def test_pose_only_hammer_ends_on_the_contact_not_near_the_hit_pose(cfg):
    """The hit keypoint isn't passed by getting near it: the episode ends when the hammer touches the target."""
    for m in _run_task(cfg, "nail", "touch", 2):
        assert m["touched"] and m["keypoints_passed"] == m["keypoints"] - 1, m
