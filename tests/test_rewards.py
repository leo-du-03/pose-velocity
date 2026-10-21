import numpy as np

from config import RewardCfg
from rewards import pass_quality


def test_pass_quality_is_one_for_a_perfect_pass_and_1_over_e_per_scale():
    c, speed = RewardCfg(), 0.3
    assert np.isclose(pass_quality(c, d_perp=0.0, dv=0.0, speed=speed, theta=0.0), 1.0)
    for kw in (dict(d_perp=c.sigma_p, dv=0.0, theta=0.0), dict(d_perp=0.0, dv=c.sigma_v_frac * speed, theta=0.0),
               dict(d_perp=0.0, dv=0.0, theta=np.deg2rad(c.sigma_theta_deg))):
        assert np.isclose(pass_quality(c, speed=speed, **kw), np.exp(-1.0)), kw


def test_parked_head_scores_low():
    """A head that drifts through the goal instead of striking it gives up the whole velocity factor."""
    c, speed = RewardCfg(), 0.3
    assert pass_quality(c, d_perp=0.0, dv=speed, speed=speed, theta=0.0) < 0.05
