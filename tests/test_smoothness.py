"""Smoothness-term ablations: RewardCfg.arm_vel_penalty = 0 and RewardCfg.smooth_norm."""
import dataclasses

import numpy as np
import pytest

from config import RewardCfg
from rewards import arm_vel_cost, step_terms


def test_arm_vel_cost_norms():
    q = np.array([3.0, -4.0, 0.0])
    assert np.isclose(arm_vel_cost(q, "l1"), 7.0)
    assert np.isclose(arm_vel_cost(q, "l2"), 5.0)
    assert np.isclose(arm_vel_cost(q, "l2sq"), 25.0)
    with pytest.raises(ValueError):
        arm_vel_cost(q, "linf")


def test_zero_penalty_removes_the_term():
    c = RewardCfg(arm_vel_penalty=0.0)
    t, _ = step_terms(c, d=0.1, d_star=0.1, attempt=False, dv=0.0, speed=0.4, qvel_cost=50.0)
    assert t["smooth"] == 0.0


def _rollout_smooth(cfg, reward, steps=15):
    """Smoothness reward summed over oracle steps, and the arm joint velocities it was paid on."""
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(dataclasses.replace(cfg, reward=reward))
    e.reset(seed=3)
    total, cost = 0.0, {n: 0.0 for n in ("l1", "l2", "l2sq")}
    for _ in range(steps):
        _, _, term, trunc, info = e.step(e.oracle_action())
        qv = e.data.qvel[e.arm_dadr]
        for n in cost:
            cost[n] += arm_vel_cost(qv, n)
        if term or trunc:
            break
    total = e.ep_terms.get("smooth", 0.0)
    motion = e._motion_info()
    e.close()
    return total, cost, motion


@pytest.mark.parametrize("norm", ["l1", "l2", "l2sq"])
def test_env_pays_the_configured_norm(cfg, norm):
    rc = dataclasses.replace(cfg.reward, smooth_norm=norm, arm_vel_penalty=0.05)
    total, cost, motion = _rollout_smooth(cfg, rc)
    assert cost[norm] > 0
    assert np.isclose(total, -0.05 * cost[norm], rtol=1e-6)
    assert motion["qvel_l1"] > 0 and motion["qvel_peak"] > 0


def test_env_without_the_term_pays_nothing_and_still_reports_motion(cfg):
    total, cost, motion = _rollout_smooth(cfg, dataclasses.replace(cfg.reward, arm_vel_penalty=0.0))
    assert total == 0.0 and cost["l1"] > 0
    assert motion["qvel_l1"] > 0 and motion["qacc"] > 0


def test_unknown_norm_is_rejected(cfg):
    from env import PoseVelocityEnv

    with pytest.raises(ValueError):
        PoseVelocityEnv(dataclasses.replace(cfg, reward=dataclasses.replace(cfg.reward, smooth_norm="l3")))


def test_old_reward_cfg_without_smooth_norm_defaults_to_l1(cfg):
    """Checkpoints pickled before smooth_norm existed lack the attribute; the env falls back to L1."""
    rc = dataclasses.replace(cfg.reward, arm_vel_penalty=0.05)
    del_ok = rc.__dict__.pop("smooth_norm", None)
    assert del_ok is not None
    total, cost, _ = _rollout_smooth(cfg, rc)
    assert np.isclose(total, -0.05 * cost["l1"], rtol=1e-6)
