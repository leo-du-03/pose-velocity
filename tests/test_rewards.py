import numpy as np

from config import RewardCfg
from rewards import pass_quality


def test_sideways_crossing_scores_below_straight_one():
    """With a steep sideways scale, the same total velocity error (0.5 x goal speed) scores well below
    along-the-line when it is sideways."""
    c, speed = RewardCfg(sigma_v_perp_frac=0.25), 0.3
    base = dict(d_perp=0.0, speed=speed, theta=0.0)
    perfect = pass_quality(c, v_along=speed, v_side=0.0, **base)
    too_slow = pass_quality(c, v_along=0.5 * speed, v_side=0.0, **base)
    sideways = pass_quality(c, v_along=speed, v_side=0.5 * speed, **base)
    assert np.isclose(perfect, 1.0)
    assert sideways < 0.3 * too_slow, (sideways, too_slow)


def test_pass_quality_improves_as_the_approach_angle_shrinks():
    """No plateau to get stuck on: every reduction of sideways velocity raises the score (default scale)."""
    c, speed = RewardCfg(), 0.3
    qs = [pass_quality(c, d_perp=0.02, v_along=0.6 * speed, v_side=f * speed, speed=speed, theta=0.3)
          for f in (0.8, 0.54, 0.3, 0.1, 0.0)]
    assert all(a < b for a, b in zip(qs, qs[1:])), qs
