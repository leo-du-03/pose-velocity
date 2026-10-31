"""Reward terms. Pure functions (summed by the env, logged per term).

The SimToolReal reward (see strl_env.py), extended with a goal velocity. Per step:
  keypoint  keypoint_rew_scale * max(d* - d, 0): paid only for getting closer to the goal pose than ever before this
            episode, so a wind-up away from the goal costs nothing. d is the largest distance between 4 keypoints
            on the tool and the same keypoints at the goal pose, so it covers position and orientation.
  smooth    -arm_vel_penalty * L1 norm of the arm joint velocities
  bonus     paid once, on the attempt (the first step with d within tolerance; the episode ends there):
            reach_goal_bonus * velocity_match, so arriving slowly or sideways earns little
Physical grip adds a grasp potential (Phi' - Phi) and a drop penalty.
"""
from __future__ import annotations

import numpy as np

import mathutils as mu
from config import RewardCfg

KEYPOINT_CORNERS = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float64)


def keypoints(pos, quat, offsets) -> np.ndarray:
    """World positions (K, 3) of body-frame offsets (K, 3) at a body pose (pos, quat w x y z)."""
    return np.asarray(pos) + np.asarray(offsets) @ mu.quat_to_mat(quat).T


def keypoint_offsets(cfg: RewardCfg) -> np.ndarray:
    """Tool-frame keypoints: corners of one fixed box (the same for every tool), centered on the head."""
    return KEYPOINT_CORNERS * (0.5 * cfg.keypoint_scale * np.asarray(cfg.keypoint_size, dtype=np.float64))


def keypoint_distance(pos_a, quat_a, pos_b, quat_b, offsets) -> float:
    """Largest distance between matching keypoints at two poses."""
    return float(np.max(np.linalg.norm(keypoints(pos_a, quat_a, offsets) - keypoints(pos_b, quat_b, offsets),
                                       axis=1)))


def velocity_match(dv: float, speed: float, sigma_v_frac: float) -> float:
    """exp(-|v - v_g| / (sigma_v_frac * |v_g|)) in (0, 1]: 1/e at sigma_v_frac of the goal speed."""
    return float(np.exp(-dv / (sigma_v_frac * max(speed, 1e-6))))


def step_terms(cfg: RewardCfg, *, d: float, d_star: float, attempt: bool, dv: float, speed: float,
               qvel_l1: float) -> tuple[dict, float]:
    """Reward terms of one step and the velocity match q_v (computed for diagnostics even when not paid).

    d_star: the closest keypoint distance before this step.
    """
    q_v = velocity_match(dv, speed, cfg.sigma_v_frac)
    terms = {
        "keypoint": cfg.keypoint_rew_scale * max(d_star - d, 0.0),
        "smooth": -cfg.arm_vel_penalty * qvel_l1,
        "bonus": cfg.reach_goal_bonus * q_v if attempt else 0.0,
    }
    return terms, q_v


def grasp_potential(cfg: RewardCfg, *, reach_dist: float, n_tip_contacts: int, n_tips: int, hand_contact: bool,
                    lift: float, holding: bool) -> float:
    """Phi(s) for physical grip: how far along the reach -> touch -> lift sequence the hand is."""
    # Reach saturates once held so that a held tool is never worse than a hovering hand.
    reach = cfg.w_reach if holding else cfg.w_reach * (1.0 - np.tanh(5.0 * reach_dist))
    contact = cfg.w_contact * n_tip_contacts / max(n_tips, 1)
    lift_term = cfg.w_lift * float(np.clip(lift / cfg.lift_scale, 0.0, 1.0)) if hand_contact else 0.0
    return float(reach + contact + lift_term)
