import numpy as np

import mathutils as mu
from goals import cone_sample, plane_crossing, segment_point_distance


def test_plane_crossing():
    d = np.array([1.0, 0, 0])
    assert np.isclose(plane_crossing([-0.1, 0.02, 0], [0.1, 0.02, 0], [0, 0, 0], d), 0.02)
    assert plane_crossing([0.1, 0, 0], [-0.1, 0, 0], [0, 0, 0], d) is None  # backward
    assert plane_crossing([-0.2, 0, 0], [-0.1, 0, 0], [0, 0, 0], d) is None  # not reached
    assert np.isclose(plane_crossing([-0.1, 0, 0], [0.0, 0, 0], [0, 0, 0], d), 0.0)  # landing on it counts


def test_segment_distance():
    assert np.isclose(segment_point_distance([0, 0, 0], [1, 0, 0], [0.5, 0.2, 0]), 0.2)
    assert np.isclose(segment_point_distance([0, 0, 0], [1, 0, 0], [2, 0, 0]), 1.0)
    assert np.isclose(segment_point_distance([0, 0, 0], [0, 0, 0], [0, 3, 4]), 5.0)


def test_goal_sampler_constraints(env):
    c = env.cfg.goal
    rng = np.random.default_rng(1)
    for level in (0.0, 1.0):
        env.set_curriculum(level)
        for _ in range(200):
            g = env.goal_sampler.sample(rng, 0.6, 90.0, env.tool.head_axis)
            assert np.isclose(np.linalg.norm(g.quat), 1.0)
            np.testing.assert_allclose(mu.rotate(g.quat, env.tool.head_axis), g.axis, atol=1e-9)
            assert mu.angle_between(g.v, g.axis) <= np.deg2rad(90.0) + 1e-9
            assert c.speed_min - 1e-9 <= g.speed <= 0.6 + 1e-9
            assert g.p[2] > env.rinfo.table_top + c.min_clearance
            assert env.reach.contains(g.p)
            assert env.reach.contains(g.p - c.lead_in * g.direction)


def test_cone_sample_stays_in_cone_and_covers_sphere():
    rng = np.random.default_rng(0)
    c = np.array([0.0, 0.0, 1.0])
    small = np.array([mu.angle_between(cone_sample(rng, c, 30.0), c) for _ in range(500)])
    assert small.max() <= np.deg2rad(30.0) + 1e-9
    full = np.array([cone_sample(rng, c, 180.0) for _ in range(4000)])
    assert abs(np.mean(full[:, 2])) < 0.05  # area-uniform on the sphere: z is uniform on [-1, 1]
    assert np.mean(full[:, 2] < -0.5) > 0.2


def test_quat_error_roundtrip():
    rng = np.random.default_rng(0)
    for angle in (0.0, 0.3, 2.0, np.pi - 1e-3):
        axis = mu.normalize(rng.normal(size=3))
        q = mu.normalize(rng.normal(size=4))
        target = mu.quat_mul(mu.axis_angle_quat(axis, angle), q)  # world-frame rotation applied to q
        for qq in (q, -q):  # sign of the quaternion doesn't matter
            np.testing.assert_allclose(mu.quat_error(target, qq), axis * angle, atol=1e-6)


def test_down_frac_samples_steep_strikes(env):
    """With down_frac = 1 most weld goals are steep downward strikes (vertical-nail-like). Not all: with some grasps
    no reachable configuration near the current pose points the head down, and the sampler falls back."""
    env.cfg.goal.down_frac = 1.0
    steep = 0
    for _ in range(20):
        env.reset(options={"kind": "regrip"})  # a new grasp each time
        steep += env.goal.direction[2] <= env.cfg.goal.down_max_z
    assert steep >= 8, steep
