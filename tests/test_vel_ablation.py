"""Goal-velocity ablations: velocity_obs, velocity_reward and pose_chain (timed pose sub-goals)."""
import dataclasses

import numpy as np
import pytest


def _env(cfg, seed: int = 0, **kw):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(dataclasses.replace(cfg, **kw))
    e.reset(seed=seed)
    return e


def _goal_here(env, v=(0.0, 0.0, -0.4)):
    """A goal at the current tool pose with a goal velocity the (still) head doesn't have."""
    from goals import Goal

    v = np.asarray(v, dtype=np.float64)
    head = env.data.site_xpos[env.head_site].copy()
    env.goal = Goal(p=head, v=v, quat=env._tool_quat(), axis=v / np.linalg.norm(v))


def test_no_vel_obs_gives_the_pose_only_observation_and_vg_to_the_critic(env, cfg):
    e = _env(cfg, velocity_obs=False)
    pose = _env(cfg, goal_velocity=False, pose_success="touch")
    assert e.observation_space.shape == pose.observation_space.shape
    assert e.priv_dim == env.priv_dim + 3
    o1, p1 = e._get_obs(), e._get_priv()
    e.goal.v = -3.0 * e.goal.v
    assert np.array_equal(o1, e._get_obs())
    assert not np.array_equal(p1, e._get_priv())
    e.close()
    pose.close()


def test_no_vel_reward_pays_a_flat_bonus_but_success_still_needs_the_velocity(cfg):
    zero = None
    for vel_reward in (True, False):
        e = _env(cfg, velocity_reward=vel_reward)
        zero = np.zeros(e.action_space.shape)
        _goal_here(e)
        _, _, term, _, info = e.step(zero)
        assert term and info["passed"] and not info["success"]  # attempt made, at the wrong velocity
        bonus = info["episode"]["terms"]["bonus"]
        if vel_reward:
            assert bonus < 0.1 * e.cfg.reward.reach_goal_bonus
        else:
            assert np.isclose(bonus, e.cfg.reward.reach_goal_bonus)
        e.close()


def test_ablations_of_the_pose_only_goal_are_rejected(cfg):
    from env import PoseVelocityEnv

    with pytest.raises(ValueError):
        PoseVelocityEnv(dataclasses.replace(cfg, pose_chain=2))  # needs pose-only
    with pytest.raises(ValueError):
        PoseVelocityEnv(dataclasses.replace(cfg, goal_velocity=False, pose_success="hold", pose_chain=2))


def test_pose_chain_builds_timed_sub_goals_on_the_strike_line(cfg):
    n = 4
    e = _env(cfg, goal_velocity=False, pose_success="touch", pose_chain=n)
    g = e.goal_full
    assert len(e.sub_goals) == n + 1 and e.goal is e.sub_goals[0]
    assert np.allclose(e.sub_goals[0].p, g.p - e.cfg.goal.lead_in * g.direction)
    assert np.allclose(e.sub_goals[-1].p, g.p)
    dt = e.model.opt.timestep * e.cfg.frame_skip
    assert np.isclose(e.sub_budget[1] * dt, e.cfg.goal.lead_in / n / g.speed) or e.sub_budget[1] == 1.0
    # Same observation layout as plain pose-only touch.
    plain = _env(cfg, goal_velocity=False, pose_success="touch")
    assert e.observation_space.shape == plain.observation_space.shape
    e.close()
    plain.close()


def test_pose_chain_advances_without_bonus_and_ends_late_sub_goals(cfg):
    e = _env(cfg, goal_velocity=False, pose_success="touch", pose_chain=2)
    zero = np.zeros(e.action_space.shape)
    head = e.data.site_xpos[e.head_site].copy()
    e.goal.p = head.copy()  # sub-goal 0 is reached on the next step
    e.goal.quat = e._tool_quat()
    _, _, term, trunc, info = e.step(zero)
    assert not (term or trunc) and e.sub_idx == 1 and e.t_sub == 0
    assert e.ep_terms["bonus"] == 0.0
    # Sitting still, sub-goal 1 is missed: the episode is cut once it is pose_chain_slack x late.
    steps = 0
    while not (term or trunc):
        _, _, term, trunc, info = e.step(zero)
        steps += 1
    assert trunc and not info["episode"]["success"]
    assert steps == int(np.ceil(e.cfg.pose_chain_slack * e.sub_budget[1]))
    e.close()


def test_chain_pose_oracle_reaches_the_goal_through_its_sub_goals(cfg):
    e = _env(cfg, seed=1, goal_velocity=False, pose_success="touch", pose_chain=4)
    e.set_curriculum(1.0)
    e.reset()
    successes, dv = 0, []
    for _ in range(12):
        while True:
            _, _, term, trunc, info = e.step(e.oracle_action())
            if term or trunc:
                break
        ep = info["episode"]
        successes += ep["success"]
        if ep["pass"]:
            dv.append(ep["pass"]["dv_frac"])
        e.reset()
    assert successes >= 8, successes
    # The final sub-goal's diagnostics are against the sampled goal velocity, which the timing conveys roughly.
    assert np.median(dv) < 1.0, dv
    e.close()
