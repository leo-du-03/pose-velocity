import numpy as np
import pytest

import mathutils as mu
from config import TaskCfg


def _run(env, policy=None):
    while True:
        _, _, term, trunc, info = env.step((policy or env.trace_action)())
        if term or trunc:
            return info["task"]


def test_trace_velocities_match_positions():
    """Analytic velocities agree with the sampled poses: (p[k+1] - p[k]) / dt is the mean of v[k] and v[k+1], and
    likewise for the orientation."""
    from traces import TraceBuilder

    b = TraceBuilder(0.02, [0.3, 0.1, 0.4], mu.yaw_quat(0.3), [1.0, 0, 0])
    q1 = mu.quat_mul(mu.axis_angle_quat([0, 1, 0], 1.0), mu.yaw_quat(-0.5))
    b.move_to([0.5, -0.1, 0.3], q1, 0.8, "approach")
    b.hold(0.1)
    b.ramp([0, 1, 0], 0.0, 0.5, 0.3)
    b.const([0, 1, 0], 0.5, 6, "strike")
    b.ramp([0, 1, 0], 0.5, 0.0, 0.2, "contact")
    tr = b.build()
    fd = np.diff(tr.p, axis=0) / tr.dt
    np.testing.assert_allclose(fd, 0.5 * (tr.v[1:] + tr.v[:-1]), atol=0.02)
    for k in range(len(tr) - 1):
        rv = mu.quat_error(tr.quat[k + 1], tr.quat[k]) / tr.dt
        np.testing.assert_allclose(rv, 0.5 * (tr.w[k] + tr.w[k + 1]), atol=0.05)
    assert np.all(np.abs(np.sum(tr.quat[1:] * tr.quat[:-1], axis=1)) > 0.99)  # no jumps


def test_task_obs_matches_training_env(env, cfg):
    """Checkpoints trained on PoseVelocityEnv must fit the task env's observation."""
    from tasks import TaskEnv

    t = TaskEnv("plate", cfg)
    assert t.observation_space.shape == env.observation_space.shape
    assert t.action_space.shape == env.action_space.shape
    t.close()


@pytest.mark.parametrize("task", ["plate", "nail"])
def test_hammer_trace_hits_target_force(cfg, task):
    """Following the trace hits the target peak force, at the planned contact point and orientation."""
    from tasks import TaskEnv

    env = TaskEnv(task, cfg)
    env.reset(seed=0)
    for _ in range(3):
        tr, kp = env.trace, env.trace.keypoints[-1]
        assert tr.keypoint_idx[-1] == tr.meta["contact_index"]
        # The keypoint is the head center at contact: the hammer face (head_long ahead of it) is on the struck point.
        np.testing.assert_allclose(kp.p + env.tool.head_long * kp.axis, tr.meta["contact_point"], atol=1e-6)
        assert kp.speed >= 0.5 * TaskCfg().strike_speed[0]
        if task == "nail":  # a vertical nail, struck straight down, raised to a training goal height
            np.testing.assert_allclose(kp.direction, [0, 0, -1], atol=1e-6)
            assert cfg.goal.height[0] <= kp.p[2] - env.rinfo.table_top <= cfg.goal.height[1]
        else:
            assert kp.direction[2] >= cfg.goal.min_dir_z
        m = _run(env)
        assert m["hit"] and m["success"], m
        assert abs(m["force_err"]) < 0.5 * env.task_cfg.force_tol, m  # what calibration guarantees for an instance
        assert m["contact_err"] < 0.01 and m["contact_theta_deg"] < 3.0, m
        env.reset()
    env.close()


def test_sweep_trace_cleans_everything(cfg):
    from tasks import TaskEnv

    env = TaskEnv("sweep", cfg)
    env.reset(seed=0)
    for _ in range(2):
        for kp in env.trace.keypoints:  # in the training goal distribution
            assert kp.speed >= 0.99 * cfg.goal.speed_min and kp.direction[2] >= cfg.goal.min_dir_z
        m = _run(env)
        assert m["cleaned"] >= 0.95 and m["keypoints_passed"] == m["keypoints"], m
        env.reset()
    env.close()


def test_sweep_force_gates_cleaning(cfg):
    """Pressing just hard enough for light dirt everywhere removes the light dirt and nothing heavier."""
    from tasks import TaskEnv

    env = TaskEnv("sweep", cfg)
    env.reset(seed=0)
    task = env.task
    light = task.tc.dirt_forces[0] * (1.0 + task.tc.force_margin)
    task._force_profile = lambda s, off: np.full(len(s), light)
    seen = set()
    for _ in range(3):
        env.start_trace(task.build_trace(env))
        m = _run(env)
        assert m["cleaned_l1"] >= 0.9 or np.isnan(m["cleaned_l1"]), m
        for lv in (2, 3):
            if not np.isnan(m[f"cleaned_l{lv}"]):
                seen.add(lv)
                assert m[f"cleaned_l{lv}"] == 0.0, m
        env.reset()
    assert seen
    env.close()


@pytest.mark.parametrize("task", ["plate", "nail"])
def test_hammer_trace_winds_up(cfg, task):
    """The strike starts with a windup: the hammer is raised back along an arc, then swung down it into the target,
    the head axis tangent to the arc at the hit; goals before the hit are training-like (fast enough)."""
    from tasks import TaskEnv
    from traces import PHASE

    env = TaskEnv(task, cfg)
    env.reset(seed=0)
    for _ in range(2):
        tr, task_ = env.trace, env.task
        k_hit = tr.meta["contact_index"]
        swing = np.flatnonzero(tr.phase == PHASE["swing"])
        assert len(swing) > 0 and swing[-1] == k_hit  # the hit is the last sample of the swing
        k_top = swing[0] - 1
        n = task_.normal
        assert (tr.p[k_top] - tr.meta["contact_point"]) @ n > 0.05  # raised back, away from the target
        assert tr.meta["windup_deg"] >= min(task_.tc.windup_angles) - 1
        np.testing.assert_allclose(mu.normalize(tr.v[k_hit]), -n, atol=1e-6)  # tangent to the arc at the hit
        for kp in tr.keypoints[:-1]:
            assert kp.speed >= 0.99 * cfg.goal.speed_min
        env.reset()
    env.close()


@pytest.mark.parametrize("task", ["plate", "nail"])
def test_hammer_is_held_like_a_hammer(cfg, task):
    """Every episode: the hand near the handle's end, the head beyond the index finger, the face along the fingers."""
    from tasks import TaskEnv

    env = TaskEnv(task, cfg)
    env.reset(seed=0)
    t, tc = env.tool, env.task_cfg
    for _ in range(3):
        d = env.data
        R, Rt = d.xmat[env.palm_id].reshape(3, 3), d.xmat[env.tool_id].reshape(3, 3)
        center, _ = env._grip_frame()
        grip = Rt.T @ (d.xpos[env.palm_id] + R @ center - d.xpos[env.tool_id])  # grip center in the tool frame
        assert abs(grip[2] - (-(0.5 * t.handle_len - tc.grasp_from_end))) < tc.grasp_slide_jitter + 0.005
        tips = d.xpos[env.tip_ids]
        assert (tips[0] - tips[-2]) @ Rt[:, 2] > 0  # ring -> index points along the handle towards the head
        fingers = d.site_xmat[0].reshape(3, 3)[:, 2]  # flange axis (attachment_site is site 0)
        face = Rt @ t.head_axis
        assert mu.angle_between(face, fingers - (fingers @ Rt[:, 2]) * Rt[:, 2]) < np.deg2rad(tc.grasp_roll_jitter_deg + 1)
        env.reset()
    env.close()
