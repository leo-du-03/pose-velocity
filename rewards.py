"""Reward terms. Pure functions returning dicts of named terms (summed by the env, logged per term)."""
from __future__ import annotations

import numpy as np

from config import RewardCfg


def phase_distance(dp: np.ndarray, dv: np.ndarray, theta: float, cfg: RewardCfg) -> float:
    """Dimensionless distance in (position, velocity, head-axis) space."""
    return float(np.linalg.norm(dp) / cfg.sigma_p + np.linalg.norm(dv) / cfg.sigma_v + theta / cfg.sigma_theta)


def grasp_potential(cfg: RewardCfg, *, reach_dist: float, n_tip_contacts: int, n_tips: int, hand_contact: bool,
                    lift: float, holding: bool) -> float:
    """Phi(s): how far along the reach -> touch -> lift sequence the hand is.

    Used as potential-based shaping (gamma*Phi(s') - Phi(s)), so improving it is rewarded once and
    merely maintaining it is not; losing it (e.g. a drop) is penalized by the same amount.
    """
    # Reach saturates once held so that a held tool is never worse than a hovering hand.
    reach = cfg.w_reach if holding else cfg.w_reach * (1.0 - np.tanh(5.0 * reach_dist))
    contact = cfg.w_contact * n_tip_contacts / max(n_tips, 1)
    lift_term = cfg.w_lift * float(np.clip(lift / cfg.lift_scale, 0.0, 1.0)) if hand_contact else 0.0
    return float(reach + contact + lift_term)


def goal_potential(cfg: RewardCfg, *, holding: bool, dp_norm: float, phase_d: float) -> float:
    """Phi_goal(s): closeness of the held tool's head to the goal.

    wide  ~ position only, informative from far away
    tight ~ position + velocity + axis together (phase distance D)
    Used as potential-based shaping, so hovering near the goal earns nothing; only approaching does.
    """
    if not holding:
        return 0.0
    return float(cfg.w_goal_wide * np.exp(-dp_norm / cfg.goal_wide_scale) + cfg.w_goal_tight * np.exp(-phase_d))


def step_reward(cfg: RewardCfg, *, phi: float, phi_prev: float, phi_goal: float, phi_goal_prev: float,
                progress: float, action: np.ndarray, prev_action: np.ndarray, arm_qvel: np.ndarray,
                robot_env_contact: bool) -> dict:
    terms = {}
    # 1-2. Reach / touch / lift, as potential-based shaping.
    terms["grasp_shaping"] = cfg.shaping_gamma * phi - phi_prev
    # 3. Goal attraction while holding, as potential-based shaping.
    terms["goal_shaping"] = cfg.shaping_gamma * phi_goal - phi_goal_prev
    # 4. Closest-approach progress (sums to D_start - D_min over an episode).
    terms["progress"] = cfg.w_progress * progress
    # 5. Regularizers.
    terms["action"] = -cfg.w_action * float(np.sum(action**2))
    terms["action_rate"] = -cfg.w_action_rate * float(np.sum((action - prev_action) ** 2))
    terms["arm_vel"] = -cfg.w_arm_vel * float(np.sum(arm_qvel**2))
    terms["table_contact"] = -cfg.w_table_contact if robot_env_contact else 0.0
    return terms


def terminal_reward(cfg: RewardCfg, *, success: bool, dropped: bool, time_frac_left: float, d_min: float) -> dict:
    terms = {
        "success": (cfg.success_bonus + cfg.success_time_bonus * time_frac_left) if success else 0.0,
        "drop": -cfg.drop_penalty if dropped else 0.0,
        # Graded credit for how close the attempt got (0 if the tool was never held).
        "closest": cfg.closest_approach_bonus * float(np.exp(-d_min)) if np.isfinite(d_min) else 0.0,
    }
    return terms
