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
    weld_solref: Tuple[float, float] = (0.01, 1.0)  # stiffness of the palm-tool weld in grip_mode "weld"


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
    lead_in: float = 0.12  # p_g - lead_in * v_hat must be reachable (with the goal orientation)
    follow_through: float = 0.08  # p_g + follow_through * v_hat must be reachable
    min_clearance: float = 0.08  # all three points must be this far above the table
    min_dir_z: float = -0.5  # reject velocity directions pointing steeply into the table
    speed_min: float = 0.3  # m/s; tolerances are relative to speed, so slow goals are the hardest under noise
    reach_samples: int = 50000
    reach_voxel: float = 0.05
    # Weld mode samples goals from arm configurations (see GoalSampler.sample_from_grasp): joints are drawn
    # around the home pose or the current pose with this std (rad), so goals stay on a reachable IK branch.
    goal_q_std: float = 0.5
    goal_q_home_prob: float = 0.0  # fraction of proposals centered on the home pose (else the current pose)
    goal_q_tries: int = 400
    # A crossing of the goal plane farther than this from p_g is not an attempt (it is ignored).
    pass_radius: float = 0.15


@dataclass
class RewardCfg:
    # Tuned for PPO fine-tuning of a policy that already strikes (bc.py): the reward is only the task, with no
    # shaping. Pass reward, paid once when the head crosses the goal plane: pass_scale * quality, with
    # quality = exp(-(d_perp/sigma_p + |v_head - v_g|/(sigma_v_frac*|v_g|) + theta/sigma_theta)) (rewards.pass_quality),
    # theta = full rotation angle between the tool and goal orientations (roll about the head axis included).
    # The scales are the tightest success tolerances (CurriculumCfg at level 1), so each factor is 1/e there.
    pass_scale: float = 20.0
    sigma_p: float = 0.03
    sigma_v_frac: float = 0.25
    sigma_theta_deg: float = 20.0
    success_bonus: float = 10.0  # added when the pass is within all success tolerances
    w_env_contact: float = 0.1  # per step; weld: robot or tool touching table/floor/pedestal; physical: arm links only
    # Physical grip only: grasp potential Phi = reach + contact + lift, paid as Phi' - Phi, and the drop penalty.
    w_reach: float = 1.0
    w_contact: float = 2.0
    w_lift: float = 5.0
    lift_scale: float = 0.1
    drop_penalty: float = 10.0


@dataclass
class CurriculumCfg:
    # (start, end) values, linearly interpolated by the difficulty level in [0, 1].
    eps_p: Tuple[float, float] = (0.08, 0.03)
    # Velocity tolerance as a fraction of the goal speed: |v_head - v_g| < eps_v_frac * |v_g|.
    # Always < 1, so a stationary head can never count as matching the goal velocity.
    eps_v_frac: Tuple[float, float] = (0.6, 0.25)
    eps_theta_deg: Tuple[float, float] = (60.0, 20.0)  # full rotation angle between tool and goal orientation
    # Max angle between the goal velocity and the head axis at the goal pose: small = strike out of the head's face,
    # 180 = any direction (swipes). Sampled uniformly within it.
    vel_cone_deg: Tuple[float, float] = (30.0, 180.0)
    speed_max: Tuple[float, float] = (0.3, 0.6)
    start_in_hand_prob: Tuple[float, float] = (0.5, 0.0)  # physical grip only
    up_threshold: float = 0.5
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
    # "weld": the tool is welded to the palm at a sampled grasp, the fingers are frozen and only the arm is
    # controlled (Stage A: learn the strike). "physical": the tool must be grasped and held by the hand.
    grip_mode: str = "weld"
    # "head_twist": arm actions are the tool head's linear + angular velocity, mapped to joints by damped
    # least squares. "joint": arm actions are changes to the joint-position targets.
    action_mode: str = "head_twist"
    head_v_max: float = 0.8  # m/s at action 1
    head_w_max: float = 2.5  # rad/s at action 1
    ik_damping: float = 0.05
    max_joint_speed: float = 3.0  # rad/s, per arm joint, head_twist only
    frame_skip: int = 10  # 0.002 s * 10 = 50 Hz control
    max_steps: int = 250  # 5 s per goal
    regrip_every: int = 8  # weld: new tool and grasp after this many goals
    regrip_rehome: bool = True  # weld: a regrip also returns the arm to home (limits drift into awkward postures)
    grasp_slide: float = 0.3  # weld: grasp point offset along the handle, as a fraction of its half length
    arm_action_scale: float = 0.03  # rad per control step, joint mode
    hand_action_scale: float = 0.08
    # 0 disables action smoothing; prev_action in the obs is the smoothed action. Smoothing also filters the
    # exploration noise, which otherwise swamps the velocity tolerance (see init_log_std).
    action_ema: float = 0.8
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
    num_envs: int = 32
    rollout_steps: int = 256
    total_steps: int = 200_000_000
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    lr_final: float = 3e-5  # linear anneal from lr to lr_final over the run (when adaptive_lr is off)
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0  # applied to the actor and the critic separately (they have separate optimizers)
    # Optional KL-adaptive lr, once per update: /1.5 if the update was stopped early or ended above 2 x target,
    # x1.5 if it ended below 0.5 x target. Floor 1e-5.
    adaptive_lr: bool = False
    kl_target: float = 0.016
    # Safety stop: skip the update's remaining minibatch steps once KL to the rollout policy exceeds this.
    kl_stop: float = 0.05
    actor_hidden: Tuple[int, ...] = (512, 256, 128)
    critic_hidden: Tuple[int, ...] = (512, 512, 256)
    # exp(-2) = 0.135. With head_twist actions (0.8 m/s at action 1) and action_ema 0.8 this is ~0.05 m/s of
    # velocity noise per axis; at -1.0 it was ~0.15 m/s, which kept even a perfect policy under 5% success.
    init_log_std: float = -2.0
    scale_rewards: bool = True
    save_every: int = 50  # iterations
