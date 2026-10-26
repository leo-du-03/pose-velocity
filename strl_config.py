"""Settings for the SimToolReal-style baseline (strl_*.py) on the simplify-weld scene.

The baseline keeps the simplify-weld environment design (UR5e + LEAP hand, procedurally generated tools welded to the
palm, MuJoCo) and swaps in the SimToolReal training method (arXiv 2602.16863, Sec. III-B and Appendices A-D):
  * pose-only goals (no velocity), reached one after another within an episode;
  * a keypoint pose distance d = max_i |o_i - g_i| over 4 object-frame keypoints;
  * reward = smoothness penalty + 200 * (progress below the closest distance so far) + success bonus;
  * a multiplicative success-tolerance curriculum;
  * domain randomization: observation / action / object-pose delays, pose and joint-velocity noise, random wrenches;
  * an asymmetric critic with clean, privileged state.
Defaults follow the paper's Table I and the official code (isaacsimenvs/tasks/simtoolreal) where they apply.
Deviations forced by the weld setup or by CPU-scale training are marked "DEVIATION".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from config import EnvCfg, PPOCfg


@dataclass
class StrlCfg:
    # ---- episode / goals
    # Policy steps allowed per goal; the clock restarts on every success (official code zeroes episode_length_buf on a
    # goal hit). Paper: 600 steps at 60 Hz = 10 s; here 500 steps at 50 Hz = 10 s.
    goal_steps: int = 500
    max_consecutive_successes: int = 50
    # A goal counts as reached after success_steps steps within tolerance (cumulative, not necessarily consecutive,
    # as force_consecutive_near_goal_steps=False in the official config); the bonus is paid per near-goal step.
    success_steps: int = 10
    force_consecutive: bool = False
    # Subsequent goals: random walk from the previous goal (up to this translation per axis and rotation angle).
    delta_goal_distance: float = 0.1
    delta_rotation_deg: float = 90.0
    delta_goal_tries: int = 40
    ik_iters: int = 60
    # ---- keypoints (object frame: x along the handle towards the head, origin at the handle center)
    keypoint_scale: float = 1.5
    # Reward keypoints use one fixed size for every object, so the success metric is the same for all tools.
    reward_keypoint_size: Tuple[float, float, float] = (0.141, 0.03025, 0.0271)
    # ---- success tolerance curriculum (on the keypoint distance; effective threshold = tol * keypoint_scale)
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.01
    tolerance_increment: float = 0.9  # multiplicative, per curriculum step
    # Tighten when the mean number of goals reached per finished episode is at least this.
    tolerance_success_threshold: float = 3.0
    # DEVIATION: the official code checks every 3000 vectorized steps of 24576 envs; with ~12 CPU envs we check every
    # PPO iteration once this many episodes have finished since the last change.
    tolerance_min_episodes: int = 50
    # ---- reward (Table I)
    keypoint_rew_scale: float = 200.0
    reach_goal_bonus: float = 1000.0
    arm_vel_penalty: float = 0.03  # L1 norm of arm joint velocities
    # ---- termination
    table_force_limit: float = 100.0  # N, net normal force of robot/tool contacts with table, pedestal or floor
    # ---- control
    # Arm: joint-position-target deltas. The paper's arm pipeline (target += moving_average * dof_speed_scale * dt *
    # action, with moving_average 0.1, dof_speed_scale 1.5, dt 1/60) caps the joints at 0.15 rad/s.
    # DEVIATION: we use 0.03 rad per 50 Hz step (1.5 rad/s), the same limit as simplify-weld's joint action mode, so the
    # comparison is about the training method and goals, not about a 10x slower arm. Set to 0.003 for the paper's cap.
    arm_delta_scale: float = 0.03
    home_noise: float = 0.1  # rad, uniform, per arm joint at reset (Table I: robot joint pos range +-0.1 rad)
    # A new random tool (a recompiled model) every this many episodes; in between, the arm returns home and the same
    # tool is welded at a new random grasp. DEVIATION: in the official code each env keeps one object for the whole run
    # (diversity comes from 24576 envs); with ~12 envs we cycle tools, and reuse each for a few episodes to save the
    # model compile.
    new_tool_every: int = 4
    # ---- domain randomization (Appendix D / Table I)
    # DEVIATION: off by default. It exists for sim-to-real transfer; simplify-weld has none and both are compared in
    # simulation, so with DR the baseline would be handicapped. Enable with strl_train.py --dr.
    dr: bool = False
    obs_delay_max: int = 3  # policy steps
    action_delay_max: int = 3
    object_delay_max: int = 10
    object_pos_noise: float = 0.01  # m, Gaussian std
    object_rot_noise_deg: float = 5.0
    joint_vel_noise: float = 0.1  # rad/s, Gaussian std
    force_scale: float = 5.0  # N, Gaussian std per axis of a random force on the tool
    torque_scale: float = 0.5  # N m
    # Per-episode probability (log-uniform in this range) that a new random wrench fires on a given step.
    force_prob_range: Tuple[float, float] = (0.001, 0.1)
    torque_prob_range: Tuple[float, float] = (0.001, 0.1)


def make_env_cfg(scfg: StrlCfg) -> EnvCfg:
    """simplify-weld EnvCfg with the settings the baseline needs."""
    cfg = EnvCfg()
    cfg.grip_mode = "weld"
    cfg.action_mode = "joint"
    cfg.arm_action_scale = scfg.arm_delta_scale
    cfg.action_ema = 0.0  # targets are integrated from deltas; no extra action smoothing
    cfg.home_noise = scfg.home_noise
    return cfg


def make_ppo_cfg(**overrides) -> PPOCfg:
    """PPO settings, from the official SimToolRealPPO.yaml where they carry over to a small CPU batch."""
    cfg = PPOCfg(
        num_envs=12,
        rollout_steps=256,  # DEVIATION: official horizon 16 with 24576 envs; 256 x 12 envs = 3072 samples per update
        total_steps=100_000_000,
        epochs=2,  # mini_epochs
        minibatches=4,
        lr=1e-4,
        gamma=0.99,
        lam=0.95,
        clip=0.1,  # e_clip
        value_clip=0.2,
        ent_coef=0.0,
        max_grad_norm=1.0,
        adaptive_lr=True,  # lr_schedule: adaptive
        kl_target=0.016,
        kl_stop=0.08,
        actor_hidden=(1024, 1024, 512, 512),
        critic_hidden=(1024, 1024, 512, 512),
        # DEVIATION: the paper starts at std 1 (sigma_init 0), but its actions are scaled to at most 0.15 rad/s of joint
        # speed, so the exploration noise is ~0.15 rad/s. Our action scale allows 1.5 rad/s; std 0.1 keeps the same
        # joint-speed exploration noise. (At std 1 the arm random-walks ~10x more and PPO only learned to lift away.)
        init_log_std=-2.3,
        scale_rewards=True,
        save_every=50,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


__all__ = ["StrlCfg", "make_env_cfg", "make_ppo_cfg"]
