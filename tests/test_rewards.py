import numpy as np

import mathutils as mu
from config import RewardCfg
from rewards import keypoint_distance, keypoint_offsets, step_terms, velocity_match

Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def test_keypoint_distance_covers_position_and_orientation():
    offs = keypoint_offsets(RewardCfg())
    p = np.array([0.4, 0.1, 0.3])
    assert np.isclose(keypoint_distance(p, Q0, p, Q0, offs), 0.0)
    shift = np.array([0.03, -0.04, 0.0])
    assert np.isclose(keypoint_distance(p + shift, Q0, p, Q0, offs), 0.05)
    # A pure rotation moves the keypoints too: about the head axis (z) less than about a cross axis (along the
    # 0.141 m long side).
    roll = keypoint_distance(p, mu.axis_angle_quat([0, 0, 1], np.deg2rad(20)), p, Q0, offs)
    pitch = keypoint_distance(p, mu.axis_angle_quat([1, 0, 0], np.deg2rad(20)), p, Q0, offs)
    assert 0.0 < roll < pitch


def test_velocity_match_is_one_at_the_goal_velocity_and_1_over_e_at_its_scale():
    c, speed = RewardCfg(), 0.4
    assert np.isclose(velocity_match(0.0, speed, c.sigma_v_frac), 1.0)
    assert np.isclose(velocity_match(c.sigma_v_frac * speed, speed, c.sigma_v_frac), np.exp(-1.0))
    # A head parked at the goal (dv = speed) gets almost nothing.
    assert velocity_match(speed, speed, c.sigma_v_frac) < 0.02


def test_step_terms_pay_progress_below_the_best_and_the_bonus_only_on_the_attempt():
    c = RewardCfg()
    kw = dict(dv=0.0, speed=0.4, qvel_l1=0.0)
    t, _ = step_terms(c, d=0.10, d_star=0.12, attempt=False, **kw)
    assert np.isclose(t["keypoint"], c.keypoint_rew_scale * 0.02) and t["bonus"] == 0.0
    t, _ = step_terms(c, d=0.14, d_star=0.12, attempt=False, **kw)  # moving away (a wind-up) costs nothing
    assert t["keypoint"] == 0.0
    t, q_v = step_terms(c, d=0.02, d_star=0.02, attempt=True, **kw)
    assert np.isclose(t["bonus"], c.reach_goal_bonus) and np.isclose(q_v, 1.0)
    t, _ = step_terms(c, d=0.02, d_star=0.02, attempt=True, dv=0.4, speed=0.4, qvel_l1=2.0)
    assert t["bonus"] < 0.02 * c.reach_goal_bonus
    assert np.isclose(t["smooth"], -c.arm_vel_penalty * 2.0)


def test_step_terms_bonus_frac_overrides_the_velocity_match():
    """Pose-only: the env passes bonus_frac (1 for "touch", 1 / success_steps per near step for "hold")."""
    c = RewardCfg()
    kw = dict(d=0.02, d_star=0.02, dv=0.4, speed=0.4, qvel_l1=0.0)  # wrong velocity: q_v would be ~0
    t, _ = step_terms(c, attempt=True, bonus_frac=1.0, **kw)
    assert np.isclose(t["bonus"], c.reach_goal_bonus)
    t, _ = step_terms(c, attempt=False, bonus_frac=0.1, **kw)
    assert np.isclose(t["bonus"], 0.1 * c.reach_goal_bonus)
    t, _ = step_terms(c, attempt=True, bonus_frac=0.0, **kw)
    assert t["bonus"] == 0.0
