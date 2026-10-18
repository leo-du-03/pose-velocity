"""Reward terms. Pure functions returning floats or dicts of named terms (summed by the env, logged per term).

Five parts:
  pass      paid once, when the head crosses the goal plane near p_g (the attempt ends there)
  approach  paid for each new best distance to the lead-in point (plus axis error) during an attempt, so it points
            the way immediately; at a pass, (1 - pass quality) of it is clawed back, so rushing through the goal to
            get a fresh goal (and fresh approach reward) hands it back, while a time-out keeps it
  waypoint  a one-off bonus for reaching the lead-in point with the axis aligned; clawed back like approach
  track     paid for forward motion along the lead-in segment, times a quality factor for staying on the line at
            the goal velocity and axis (see track_quality); at most w_track per attempt
  contact   small per-step penalty for hitting the table / floor / pedestal
Physical grip adds a grasp potential (Phi' - Phi) and a drop penalty (a drop also claws back all approach reward).
"""
from __future__ import annotations

import numpy as np

from config import RewardCfg


def pass_quality(cfg: RewardCfg, *, d_perp: float, v_along: float, v_side: float, speed: float,
                 theta: float) -> float:
    """Quality of one pass in [0, 1]. Multiplicative, so it is high only when position, velocity and axis are all
    good at once (a head parked on the goal scores nothing).

    v_along / v_side: the head velocity's components along / across the goal direction. The speed error along it is
    scored leniently (a pass at 3x speed still scores something, so slowing down visibly pays); sideways velocity
    is scored steeply, so cutting across the plane at an angle scores low, and every degree of approach angle given
    up pays.
    """
    s = max(speed, 1e-6)
    return float(np.exp(-d_perp / cfg.sigma_p) * np.exp(-abs(v_along - speed) / (cfg.sigma_v_frac * s))
                 * np.exp(-abs(v_side) / (cfg.sigma_v_perp_frac * s)) * max(0.0, np.cos(theta)))


def track_quality(cfg: RewardCfg, *, d_line: float, dv_frac: float, theta: float) -> float:
    """How well the head is moving along the lead-in line: on the line, at the goal velocity, axis aligned.

    The track reward is this times the new distance covered, so it is paid per distance, not per step:
    crawling earns nothing extra (and lowers the velocity factor), rushing lowers the velocity factor, and
    backtracking only spends the attempt's budget again.
    """
    return float(np.exp(-d_line / cfg.track_sigma_line) * np.exp(-dv_frac / cfg.track_sigma_v_frac)
                 * max(0.0, np.cos(theta)))


def approach_distance(cfg: RewardCfg, *, d_lead: float, theta: float) -> float:
    """D: how far the head is from being at the lead-in point with its axis aligned (dimensionless)."""
    return float(d_lead / cfg.approach_scale + theta / cfg.approach_theta_scale)


def grasp_potential(cfg: RewardCfg, *, reach_dist: float, n_tip_contacts: int, n_tips: int, hand_contact: bool,
                    lift: float, holding: bool) -> float:
    """Phi(s) for physical grip: how far along the reach -> touch -> lift sequence the hand is."""
    # Reach saturates once held so that a held tool is never worse than a hovering hand.
    reach = cfg.w_reach if holding else cfg.w_reach * (1.0 - np.tanh(5.0 * reach_dist))
    contact = cfg.w_contact * n_tip_contacts / max(n_tips, 1)
    lift_term = cfg.w_lift * float(np.clip(lift / cfg.lift_scale, 0.0, 1.0)) if hand_contact else 0.0
    return float(reach + contact + lift_term)
