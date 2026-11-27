"""goal_mode "trajectory": a smooth random joint-space reference into each strike goal (PoseVelocityEnv._build_reference).
The goal and its success rule are those of "points"; the reference adds observations and the tracking reward."""
import dataclasses

import numpy as np
import pytest

import mathutils as mu
from rewards import keypoint_distance, keypoint_offsets


def _env(cfg, level=0.0, action_mode="joint_vel", seed=0):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(dataclasses.replace(cfg, action_mode=action_mode, goal_mode="trajectory"))
    e.set_curriculum(level)
    e.reset(seed=seed)
    return e


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_reference_shape(cfg, level):
    """Starts at the arm's current state, stays within the joint limits and speeds, and runs through the goal along
    the lead-in at v_g in the goal orientation."""
    e = _env(cfg, level)
    dt = e.model.opt.timestep * e.cfg.frame_skip
    for _ in range(15):
        ref, g = e.ref, e.goal
        assert not e.ref_fallback
        assert np.all(np.isfinite(ref.p)) and np.all(np.isfinite(ref.v)) and np.all(np.isfinite(e.ref_q))
        assert len(ref) <= e.cfg.max_steps
        np.testing.assert_allclose(ref.p[0], e.data.site_xpos[e.head_site], atol=2e-3)
        np.testing.assert_allclose(e.ref_q[0], e.data.qpos[e.arm_qadr], atol=1e-9)
        lo, hi = e.arm_range[:, 0], e.arm_range[:, 1]
        assert np.all(e.ref_q >= lo - 1e-6) and np.all(e.ref_q <= hi + 1e-6)
        assert np.all(np.abs(e.ref_qd[1:]) <= e._joint_v_max() + 1e-9)
        k = e.ref_goal_idx
        np.testing.assert_allclose(ref.p[k], g.p, atol=1e-9)
        n_lead = int(round(e.cfg.goal.lead_in / (g.speed * dt)))
        for j in range(k - n_lead, k + 1):  # the lead-in: on the line at v_g, goal orientation; FK agrees
            np.testing.assert_allclose(ref.v[j], g.v, atol=1e-9)
            head, _, quat, _ = e._scratch_pose(e.ref_q[j])
            assert np.linalg.norm(head - ref.p[j]) < 1e-3
            assert np.linalg.norm(mu.quat_error(g.quat, quat)) < 5e-3
        # The approach stays out of the attempt region, so the attempt is made on the lead-in.
        off = keypoint_offsets(e.cfg.reward)
        d = [keypoint_distance(ref.p[j], ref.quat[j], g.p, g.quat, off) for j in range(k - n_lead)]
        assert min(d) > e.eps_kp()
        # Smooth: consecutive approach samples move about as fast as the reference velocity says.
        steps = np.linalg.norm(np.diff(ref.p[: k - n_lead], axis=0), axis=1)
        assert steps.max() < 1.5 * max(np.linalg.norm(ref.v, axis=1).max(), 1e-3) * dt + 1e-4
        e.reset()
    e.close()


def test_obs_layout_and_check_env(cfg):
    from gymnasium.utils.env_checker import check_env

    from env import PoseVelocityEnv

    pts = PoseVelocityEnv(dataclasses.replace(cfg, action_mode="joint_vel"))
    e = _env(cfg)
    # Reference now (4 x 3), each preview offset (2 x 3), joint reference (2 x 6); the kin features lose one column.
    extra = 12 + 6 * len(e.tcfg.preview) + 12 - 6
    assert e.observation_space.shape[0] == pts.observation_space.shape[0] + extra
    assert e.priv_dim == pts.priv_dim + 1
    shape = e.observation_space.shape
    for _ in range(20):
        obs, _, term, trunc, _ = e.step(e.action_space.sample())
        assert obs.shape == shape and np.all(np.isfinite(obs))
        if term or trunc:
            obs, _ = e.reset()
            assert obs.shape == shape
    check_env(e.unwrapped, skip_render_check=True)
    pts.close()
    e.close()


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_follower_succeeds(cfg, level):
    """The scripted reference follower (env.oracle_action in trajectory mode) makes the strike: the references are
    feasible and the success rule is reachable by tracking them."""
    e = _env(cfg, level, seed=1)
    wins, n = 0, 20
    for _ in range(n):
        done = False
        while not done:
            _, r, term, trunc, info = e.step(e.oracle_action())
            done = term or trunc
        ep = info["episode"]
        wins += ep["success"]
        assert "track" in ep["terms"] and "keypoint" not in ep["terms"]
        e.reset()
    assert wins / n >= 0.8
    e.close()


def test_lost_terminates(cfg):
    """Doing nothing while the reference moves away ends the episode as a failure ("lost")."""
    e = _env(cfg)
    for _ in range(e.cfg.max_steps):
        _, _, term, trunc, info = e.step(np.zeros(e.n_act))
        if term or trunc:
            break
    assert term and info["lost"] and info["episode"]["lost"] == 1.0 and not info["episode"]["success"]
    e.close()


def test_checks(cfg):
    from env import PoseVelocityEnv

    for bad in (dict(grip_mode="physical"), dict(goal_velocity=False), dict(action_mode="joint")):
        with pytest.raises(ValueError):
            PoseVelocityEnv(dataclasses.replace(cfg, goal_mode="trajectory", **bad))
    with pytest.raises(ValueError):
        PoseVelocityEnv(dataclasses.replace(cfg, goal_mode="spline"))
