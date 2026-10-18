"""All tunable settings, grouped into dataclasses.

Every value has a sensible default; override fields in code or via train.py flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SceneCfg:
    # Optional explicit paths to Menagerie XMLs (otherwise fetched via robot_descriptions,
    # or taken from the POSEVEL_UR5E_XML / POSEVEL_LEAP_XML environment variables).
    ur5e_xml: Optional[str] = None
    leap_xml: Optional[str] = None
    arm_attach_site: str = "attachment_site"
    # Automatically rotate the hand so its fingers continue along the flange axis, flush on the flange.
    auto_align_hand: bool = True
    # Fine-tuning applied in the flange-site frame on top of the automatic alignment.
    # The flange axis is z, so (0, 0, angle) spins the hand about the wrist (which way the palm faces).
    hand_mount_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_mount_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # intrinsic xyz, radians
    # UR5e home pose (Menagerie "home" keyframe): forearm forward, wrist pointing down.
    arm_home: Tuple[float, ...] = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    timestep: float = 0.002
    impratio: float = 10.0
    gravcomp: bool = True  # gravity compensation on all robot bodies
    table_center_dist: float = 0.6  # distance of table center from robot base, along arm's home direction
    table_half_size: Tuple[float, float, float] = (0.3, 0.45, 0.02)  # (radial, lateral, thickness)
    pedestal_height: float = 0.75  # table top and robot base are both at z = 0; floor at -pedestal_height


@dataclass
class ToolCfg:
    handle_len: Tuple[float, float] = (0.15, 0.35)
    handle_rad: Tuple[float, float] = (0.015, 0.03)
    handle_aspect: Tuple[float, float] = (0.6, 1.0)  # box handles only
    head_long: Tuple[float, float] = (0.03, 0.07)  # half-extent along the head's long axis
    head_short: Tuple[float, float] = (0.015, 0.035)
    head_width: Tuple[float, float] = (0.015, 0.035)
    head_jitter: float = 0.01
    density: Tuple[float, float] = (300.0, 1500.0)
    mass: Tuple[float, float] = (0.1, 0.6)
    friction: Tuple[float, float] = (0.7, 1.2)


@dataclass
class GoalCfg:
    # Goal region, expressed in the table frame: u = distance from base along table axis,
    # w = lateral offset, z = height above the table top.
    radial: Tuple[float, float] = (0.35, 0.8)
    lateral: Tuple[float, float] = (-0.35, 0.35)
    height: Tuple[float, float] = (0.15, 0.5)
    min_base_dist: float = 0.3
    lead_in: float = 0.12  # p_g - lead_in * v_hat must be reachable
    follow_through: float = 0.08  # p_g + follow_through * v_hat must be reachable
    min_clearance: float = 0.08  # all three points must be this far above the table
    min_dir_z: float = -0.5  # reject velocity directions pointing steeply into the table
    speed_min: float = 0.1
    axis_mode: str = "velocity"  # "velocity": head axis = v_hat; "cone": random within axis_cone_deg of v_hat
    axis_cone_deg: float = 30.0
    reach_samples: int = 50000
    reach_voxel: float = 0.05


@dataclass
class RewardCfg:
    sigma_p: float = 0.05
    sigma_v: float = 0.2
    sigma_theta: float = 0.5
    # Grasp potential Phi = reach + contact + lift; rewarded as gamma*Phi' - Phi (paid once, not per step held).
    w_reach: float = 1.0
    w_contact: float = 2.0
    w_lift: float = 5.0
    lift_scale: float = 0.1
    shaping_gamma: float = 0.99  # keep equal to PPOCfg.gamma so the shaping doesn't change the optimal policy
    # Goal potential Phi_goal = wide + tight, also rewarded as gamma*Phi' - Phi (approaching pays, hovering doesn't).
    w_goal_wide: float = 5.0
    goal_wide_scale: float = 0.3
    w_goal_tight: float = 5.0
    w_progress: float = 3.0
    w_action: float = 1e-3
    w_action_rate: float = 2e-3
    w_arm_vel: float = 1e-4
    w_table_contact: float = 0.5
    success_bonus: float = 50.0
    success_time_bonus: float = 20.0
    drop_penalty: float = 20.0
    closest_approach_bonus: float = 10.0


@dataclass
class CurriculumCfg:
    # (start, end) values, linearly interpolated by the difficulty level in [0, 1].
    eps_p: Tuple[float, float] = (0.08, 0.03)
    # Velocity tolerance as a fraction of the goal speed: |v_head - v_g| < eps_v_frac * |v_g|.
    # Always < 1, so a stationary head can never count as matching the goal velocity.
    eps_v_frac: Tuple[float, float] = (0.6, 0.25)
    eps_theta_deg: Tuple[float, float] = (60.0, 20.0)
    speed_max: Tuple[float, float] = (0.1, 0.6)
    start_in_hand_prob: Tuple[float, float] = (0.5, 0.0)
    up_threshold: float = 0.6
    down_threshold: float = 0.2
    step: float = 0.05
    window: int = 200
    min_episodes: int = 100


@dataclass
class EnvCfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    tool: ToolCfg = field(default_factory=ToolCfg)
    goal: GoalCfg = field(default_factory=GoalCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    frame_skip: int = 10  # 0.002 s * 10 = 50 Hz control
    max_steps: int = 500  # 10 s episodes
    arm_action_scale: float = 0.03  # rad per control step
    hand_action_scale: float = 0.08
    action_ema: float = 0.0  # 0 disables action smoothing
    home_noise: float = 0.05
    spawn_radial: Tuple[float, float] = (0.45, 0.65)
    spawn_lateral: Tuple[float, float] = (-0.2, 0.2)
    settle_time: float = 0.3
    lift_threshold: float = 0.03
    grasp_min_parts: int = 2  # hand parts (4 fingers + palm) that must touch the tool to count as grasped
    drop_no_contact_steps: int = 10  # 0.2 s without hand-tool contact = drop
    success_requires_grasp: bool = True
    curriculum_level: float = 0.0
    seed: Optional[int] = None
    render_width: int = 640
    render_height: int = 480


@dataclass
class PPOCfg:
    num_envs: int = 16
    rollout_steps: int = 64
    total_steps: int = 200_000_000
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    vf_coef: float = 1.0
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    kl_target: float = 0.016
    # lr is adapted once per update from the KL the whole update produced: /1.5 if the update was stopped early
    # or ended above 2 x target, x1.5 if it ended below 0.5 x target.
    adaptive_lr: bool = True
    # Stop the update's remaining minibatch steps once KL to the rollout policy exceeds this multiple of kl_target.
    kl_stop_mult: float = 1.5
    actor_hidden: Tuple[int, ...] = (512, 256, 128)
    critic_hidden: Tuple[int, ...] = (512, 512, 256)
    init_log_std: float = -0.7
    scale_rewards: bool = True
    save_every: int = 50  # iterations
