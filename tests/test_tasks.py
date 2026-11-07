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


# Sweep tests use the reference dirt forces, so they don't depend on how TaskCfg.dirt_forces is tuned.
SWEEP_FORCES = (3.0, 8.0, 14.0)


def test_sweep_trace_cleans_everything(cfg):
    from config import TaskCfg
    from tasks import TaskEnv

    env = TaskEnv("sweep", cfg, TaskCfg(dirt_forces=SWEEP_FORCES))
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
    from config import TaskCfg
    from tasks import TaskEnv

    env = TaskEnv("sweep", cfg, TaskCfg(dirt_forces=SWEEP_FORCES))
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


def _pad_over_all_cells(env, rule):
    """Sweep env with every dirt cell moved under the pad's face, all heavy; pad force and sliding speed mocked."""
    task = env.task
    task.tc.sweep_rule = rule
    d = env.data
    R = d.xmat[task.pad_id].reshape(3, 3)
    face = d.xpos[task.pad_id] + R @ np.array([0.0, 0.0, 2 * task.tc.pad_half_thick])
    task.cells_world = np.tile(face, (len(task.cells_xy), 1))
    task.level[:] = len(task.tc.dirt_forces)
    task.required = np.full(len(task.cells_xy), task.tc.dirt_forces[-1])
    task.begin(env)
    return task


@pytest.mark.parametrize("rule, slide, expect_clean", [("scrub", 0.0, False), ("scrub", 0.3, True),
                                                      ("force", 0.0, True)])
def test_sweep_scrub_needs_sliding(cfg, rule, slide, expect_clean):
    """Scrub rule: pressing above the adhesion force in place removes nothing; sliding removes the dirt after
    about scrub_len x adhesion / force of travel. The force rule removes it on contact."""
    from config import TaskCfg
    from tasks import TaskEnv

    env = TaskEnv("sweep", cfg, TaskCfg(dirt_forces=SWEEP_FORCES))
    env.reset(seed=0)
    task = _pad_over_all_cells(env, rule)
    f = 2.0 * task.required[0]
    task.pad_force = lambda e: f
    task._pad_slide_speed = lambda e: slide
    dt = env.model.opt.timestep
    n_needed = task.tc.scrub_len * task.required[0] / (f * slide * dt) if slide > 0 else 1e9
    for _ in range(int(min(n_needed, 2000)) + 2):
        task.measure(env)
    assert task.cleaned.all() == expect_clean and task.cleaned.any() == expect_clean
    env.close()


def test_sweep_scrub_needs_the_adhesion_force(cfg):
    """Sliding below the adhesion force never removes the dirt, however far."""
    from config import TaskCfg
    from tasks import TaskEnv

    env = TaskEnv("sweep", cfg, TaskCfg(dirt_forces=SWEEP_FORCES))
    env.reset(seed=0)
    task = _pad_over_all_cells(env, "scrub")
    task.pad_force = lambda e: 0.9 * task.required[0]
    task._pad_slide_speed = lambda e: 0.3
    for _ in range(5000):  # 3 m of sliding
        task.measure(env)
    assert not task.cleaned.any() and task.scrub.max() == 0.0
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
def test_hammer_swing_straight_run_in(cfg, task):
    """With swing_straight, the head goes straight in along the strike line, orientation held and not rotating, for
    that distance before the hit and through the follow-through; the follower still hits the target force."""
    from tasks import TaskEnv

    tc = TaskCfg(swing_straight=0.05)
    env = TaskEnv(task, cfg, tc)
    env.reset(seed=0)
    for _ in range(2):
        tr, n = env.trace, env.task.normal
        k_hit = tr.meta["contact_index"]
        seg = np.flatnonzero(np.linalg.norm(tr.p - tr.p[k_hit], axis=1) < tc.swing_straight - 1e-3)
        seg = seg[seg >= k_hit - int(tc.swing_straight / (tr.meta["impact_speed"] * tr.dt)) - 1]
        k0, k1 = seg[0], k_hit + 2  # run-in start .. into the follow-through
        rel = tr.p[k0:k1] - tr.p[k_hit]
        np.testing.assert_allclose(rel - np.outer(rel @ n, n), 0.0, atol=1e-9)  # on the strike line
        np.testing.assert_allclose(tr.w[k0:k1], 0.0, atol=1e-12)
        for q in tr.quat[k0:k1]:
            assert np.linalg.norm(mu.quat_error(q, tr.quat[k_hit])) < 1e-9
        assert np.linalg.norm(tr.w[k0 - 3]) > 0.5  # still on the arc before the run-in
        m = _run(env)
        assert m["hit"] and m["success"], m
        env.reset()
    env.close()


def test_crowned_face_makes_plate_force_tilt_tolerant(cfg):
    """With a crowned face, the follower struck 2 deg off flat still hits the target force (a flat face lands on its
    rim at that tilt and loses up to ~20%); the crown is the struck point (contact point = face apex)."""
    import copy

    from tasks import TaskEnv

    env = TaskEnv("plate", cfg, TaskCfg(face_crown=0.002))
    assert env.task._face_offset() == pytest.approx(env.tool.head_long + 0.002)
    env.reset(seed=0)
    for _ in range(2):
        tr = env.trace
        for ax in (env.task.arc_axis, -env.task.arc_axis):
            r = mu.axis_angle_quat(ax, np.deg2rad(2.0))
            t2 = copy.deepcopy(tr)
            t2.quat = np.array([mu.quat_mul(r, q) for q in tr.quat])
            t2.w = tr.w @ mu.quat_to_mat(r).T
            m = env.rollout(t2)
            assert m["hit"] and abs(m["force_err"]) < 0.08, m
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


# ---------------------------------------------------------------------------------------------------------- slice
def _slice_env(cfg, seed=0):
    from tasks import TaskEnv

    env = TaskEnv("slice", cfg)
    env.reset(seed=seed)
    return env


def _shift_trace(env, fn):
    """Restarts the episode's trace with fn(k, p, quat) -> (p, quat) applied to every sample from the lowering onto
    the loaf on (the approach is left alone)."""
    import copy

    from traces import PHASE

    tr = copy.deepcopy(env.trace)
    k0 = int(np.argmax(tr.phase == PHASE["lower"]))
    for k in range(k0, len(tr)):
        tr.p[k], tr.quat[k] = fn(k, tr.p[k], tr.quat[k])
    env.start_trace(tr)


def test_slice_trace_cuts_through(cfg):
    """Following the trace saws through every kerf column on the line, without crushing or jamming, and the halves
    fall open."""
    env = _slice_env(cfg)
    for _ in range(2):
        tr = env.trace
        assert len(tr.keypoints) >= 8
        assert all(kp.speed >= cfg.goal.speed_min for kp in tr.keypoints)
        m = _run(env)
        assert m["success"], m
        assert m["cut_frac"] > 0.999 and not (m["crushed"] or m["jammed"] or m["off_line"] or m["board_hit"])
        assert m["line_offset"] < 0.5 * env.task_cfg.line_tol
        assert m["bind_deg_max"] < 0.5 * env.task_cfg.bind_deg
        assert m["slice_ratio_mean"] > 2 * env.task_cfg.slice_ratio_min
        assert np.all(env.data.qpos[env.task.hinge_qadr] > 0.2)  # the halves have opened
        env.reset()
    env.close()


def test_slice_push_without_slicing_crushes(cfg):
    """Pressing the edge straight down into the loaf (no motion along the blade) crushes it."""
    from traces import PHASE

    env = _slice_env(cfg)

    def push():
        if env.trace.phase[min(env.task_t, len(env.trace) - 1)] == PHASE["saw"]:
            a = np.zeros(6, np.float32)
            a[2] = -0.1 / cfg.head_v_max
            return a
        return env.trace_action()

    m = _run(env, push)
    assert m["crushed"] and m["fail_reason"] == "crushed" and not m["success"]
    assert env.model.geom_size[env.task.half_gids[0], 2] < 0.5 * env.task.half_size0[0, 2]  # squashed
    env.close()


def test_slice_twisted_blade_jams(cfg):
    """A blade turned 20 deg from the cut plane binds in the loaf: the knife jams and the arm stops."""
    env = _slice_env(cfg)
    twist = mu.axis_angle_quat([0, 0, 1], np.deg2rad(20.0))
    _shift_trace(env, lambda k, p, q: (p, mu.quat_mul(twist, q)))
    m = _run(env)
    assert m["jammed"] and m["fail_reason"] == "jammed" and not m["success"]
    head0 = env._head_state()[0]
    for _ in range(5):  # the knife stays stuck whatever the policy asks
        env.step(np.array([0, 0, 1.0, 0, 0, 0], np.float32))
    assert np.linalg.norm(env._head_state()[0] - head0) < 0.01
    env.close()


def test_slice_off_the_line(cfg):
    """A clean cut 1 cm beside the marked line goes through but fails: the halves stay shut."""
    env = _slice_env(cfg)
    n = env.task.n
    _shift_trace(env, lambda k, p, q: (p + 0.01 * n, q))
    m = _run(env)
    assert m["off_line"] and m["fail_reason"] == "off_line" and not m["success"]
    assert m["line_offset"] > env.task_cfg.line_tol and m["cut_frac"] > 0.99
    assert np.all(env.data.qpos[env.task.hinge_qadr] < 0.05)
    env.close()


def test_slice_obs_matches_training_env(cfg):
    """Pose + velocity and pose-only checkpoints both fit the slice env's observation."""
    import copy

    from env import PoseVelocityEnv
    from tasks import TaskEnv

    for goal_velocity in (True, False):
        c = copy.deepcopy(cfg)
        c.goal_velocity = goal_velocity
        e, t = PoseVelocityEnv(c), TaskEnv("slice", c)
        e.reset(seed=0)
        assert t.observation_space.shape == e.observation_space.shape
        e.close()
        t.close()


def test_slice_too_deep_hits_the_board(cfg):
    """Sawing 1.5 cm too deep chops into the cutting board: a failure even though the loaf is cut through."""
    from traces import PHASE

    env = _slice_env(cfg, seed=1)
    _shift_trace(env, lambda k, p, q: (p - (0.015 if env.trace.phase[k] == PHASE["saw"] else 0.0) * np.array(
        [0, 0, 1.0]), q))
    m = _run(env)
    assert m["board_hit"] and m["fail_reason"] == "board" and not m["success"]
    assert m["board_force"] > env.task_cfg.board_force_limit
    env.close()


# --------------------------------------------------------------------------------------------------- slice_tomato
def _tomato_env(cfg, seed=0):
    from tasks import TaskEnv

    env = TaskEnv("slice_tomato", cfg)
    env.reset(seed=seed)
    return env


def _saw_depth(env, fn):
    """Restarts the trace with the head heights from the first stroke on changed by fn(z, z_start, z_min) -> z
    (z_start, z_min: the heights at the start of the sawing and at its lowest), with velocities to match."""
    import copy

    from traces import PHASE

    tr = copy.deepcopy(env.trace)
    k0 = int(np.argmax(tr.phase == PHASE["saw"]))
    z = tr.p[k0:, 2]
    tr.p[k0:, 2] = fn(z, z[0], z.min())
    tr.v[k0:, 2] = np.gradient(tr.p[k0 - 1:, 2], tr.dt)[1:]
    env.start_trace(tr)


def test_tomato_trace_cuts_within_the_force_window(cfg):
    """Following the trace cuts the tomato through, pressing between the cut and the crush force, and it opens."""
    env = _tomato_env(cfg)
    tc = env.task_cfg
    for _ in range(2):
        m = _run(env)
        assert m["success"], m
        assert tc.tomato_cut_force < m["force_cut_mean"] < tc.tomato_crush_force
        assert m["force_max"] < tc.tomato_crush_force - 0.5
        assert np.all(env.data.qpos[env.task.hinge_qadr] > 0.2)
        env.reset()
    env.close()


def test_tomato_too_little_force_doesnt_cut(cfg):
    """An edge that only grazes the tomato (pressing under the cut force) leaves it whole; a trace that finishes a
    few mm short of the board doesn't cut it through."""
    env = _tomato_env(cfg)
    tc = env.task_cfg
    # The edge skims the top of the tomato (the follower dips a mm or so below its path, short of the cut force).
    head_z = env.task.c[2] + 2 * tc.tomato_radius + 0.5 * tc.blade_height  # the edge is this far below the head
    _saw_depth(env, lambda z, z0, zmin: np.maximum(z, head_z))
    m = _run(env)
    assert m["fail_reason"] == "not_through" and m["cut_frac"] == 0.0 and m["force_max"] < env.task_cfg.tomato_cut_force
    env.reset()
    _saw_depth(env, lambda z, z0, zmin: np.maximum(z, zmin + 0.005))
    m = _run(env)
    assert m["fail_reason"] == "not_through" and 0.5 < m["cut_frac"] < 1.0
    env.close()


def test_tomato_too_much_force_crushes(cfg):
    """Descending twice as steeply as the sliding cuts drives the force up to the crush force: the tomato bursts."""
    env = _tomato_env(cfg)
    _saw_depth(env, lambda z, z0, zmin: np.maximum(z0 - 2.0 * (z0 - z), zmin))
    m = _run(env)
    assert m["fail_reason"] == "crushed" and m["force_max"] >= env.task_cfg.tomato_crush_force
    assert env.model.geom_rgba[env.task.splat_gid, 3] == 1.0  # the splat shows, the halves are hidden
    assert np.all(env.model.geom_rgba[env.task.half_gids, 3] == 0.0)
    env.close()


def test_tomato_tolerates_position_errors_the_loaf_doesnt(cfg):
    """A cut 7 mm beside the middle still halves the tomato, though it is off the line for the loaf."""
    env = _tomato_env(cfg)
    n = env.task.n
    _shift_trace(env, lambda k, p, q: (p + 0.007 * n, q))
    m = _run(env)
    assert m["success"], m
    assert m["line_offset"] > env.task_cfg.line_tol  # the loaf's tolerance
    env.close()
