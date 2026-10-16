import numpy as np

from goals import segment_point_distance


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
            g = env.goal_sampler.sample(rng, speed_max=0.6)
            assert np.isclose(np.linalg.norm(g.axis), 1.0)
            assert c.speed_min - 1e-9 <= g.speed <= 0.6 + 1e-9
            assert g.p[2] > env.rinfo.table_top + c.min_clearance
            assert env.reach.contains(g.p)
            assert env.reach.contains(g.p - c.lead_in * g.direction)
