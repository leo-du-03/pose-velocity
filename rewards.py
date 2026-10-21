"""Reward terms. Pure functions returning floats (summed by the env, logged per term).

Set up for PPO fine-tuning of a policy that already strikes (e.g. a bc.py clone), so the reward is just the task:
  pass      paid once, when the head crosses the goal plane near p_g (the attempt ends there):
            pass_scale * pass_quality, plus success_bonus if the pass is within the success tolerances
  contact   small per-step penalty for hitting the table / floor / pedestal
Physical grip adds a grasp potential (Phi' - Phi) and a drop penalty.
"""
from __future__ import annotations

import numpy as np

from config import RewardCfg


def pass_quality(cfg: RewardCfg, *, d_perp: float, dv: float, speed: float, theta: float) -> float:
    """Quality of one pass in [0, 1]: exp(-(d_perp/sigma_p + dv/(sigma_v_frac*speed) + theta/sigma_theta)).

    dv is the full velocity error |v_head - v_g|, so sideways motion costs as much as a wrong speed. Each factor is
    1/e at its scale; the scales default to the tightest success tolerances, so the reward keeps sharpening the strike
    inside the success region. Multiplicative, so it is high only when position, velocity and axis are all good.
    """
    s = max(speed, 1e-6)
    return float(np.exp(-d_perp / cfg.sigma_p - dv / (cfg.sigma_v_frac * s) - theta / np.deg2rad(cfg.sigma_theta_deg)))


def grasp_potential(cfg: RewardCfg, *, reach_dist: float, n_tip_contacts: int, n_tips: int, hand_contact: bool,
                    lift: float, holding: bool) -> float:
    """Phi(s) for physical grip: how far along the reach -> touch -> lift sequence the hand is."""
    # Reach saturates once held so that a held tool is never worse than a hovering hand.
    reach = cfg.w_reach if holding else cfg.w_reach * (1.0 - np.tanh(5.0 * reach_dist))
    contact = cfg.w_contact * n_tip_contacts / max(n_tips, 1)
    lift_term = cfg.w_lift * float(np.clip(lift / cfg.lift_scale, 0.0, 1.0)) if hand_contact else 0.0
    return float(reach + contact + lift_term)
