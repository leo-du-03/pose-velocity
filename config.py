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
    hold_solref: Tuple[float, float] = (0.05, 1.0)  # grip_mode "grasp": soft weld holding the floating tool in place


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
    speed_min: float = 0.3  # m/s; tolerances are relative to speed, so slow goals are the hardest under noise
    axis_mode: str = "velocity"  # "velocity": head axis = v_hat; "cone": random within axis_cone_deg of v_hat
    axis_cone_deg: float = 30.0
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
    # Pass reward, paid once when the head crosses the goal plane: pass_scale * quality (see rewards.pass_quality),
    # quality = exp(-d_perp/sigma_p) * exp(-|v_along - |v_g|| / (sigma_v_frac*|v_g|))
    #           * exp(-|v_sideways| / (sigma_v_perp_frac*|v_g|)) * max(0, cos theta),
    # where v_along / v_sideways are the head velocity's components along / across the goal direction.
    pass_scale: float = 20.0
    sigma_p: float = 0.05
    sigma_v_frac: float = 1.0  # along v_hat; a 3x-speed pass keeps ~10% of it, so the gradient to slow down exists
    # Across v_hat. 0.25 (steep) was tried against "swooping" in from the side, but mostly shrank the reward for good
    # passes under exploration noise; the waypoint gate (pass_requires_waypoint) handles swoops instead.
    sigma_v_perp_frac: float = 1.0
    success_bonus: float = 10.0  # added when the pass is within all success tolerances
    # Approach progress: w_approach * (improvement of the attempt's best D), D = d_lead/approach_scale +
    # theta/approach_theta_scale, d_lead = head distance to the lead-in point p_g - lead_in*v_hat. At most
    # w_approach * D(start) per attempt. At a pass, (1 - pass quality) of what the attempt earned is clawed back, so a
    # rushed pass to get the next goal sooner hands it back; a time-out keeps it.
    # Keep approach + waypoint (E per attempt) small next to pass_scale: finishing a strike beats timing out only when
    # pass_scale*q > (1 - q)*E, i.e. q > E / (pass_scale + E). With w_approach 1 and w_waypoint 3 (E ~ 6) that was
    # q > 0.23, so a learning policy's clumsy first strikes (q ~ 0.05-0.12) lost to waiting at the lead-in point.
    w_approach: float = 0.3
    approach_scale: float = 0.1
    approach_theta_scale: float = 0.5
    # Waypoint: a one-off bonus the first time in an attempt the head is within waypoint_radius of the lead-in point
    # with its axis within waypoint_theta_deg. It counts as approach reward, so the same clawback applies.
    w_waypoint: float = 1.0
    waypoint_radius: float = 0.03
    # Pay the pass reward and success bonus only if the attempt hit the waypoint first; a pass without it counts as
    # quality 0 (full clawback). Training reward only: success is still measured on every pass.
    pass_requires_waypoint: bool = True
    waypoint_theta_deg: float = 15.0
    # Track reward: paid per step for forward motion along the lead-in segment, out of a budget of lead_in metres
    # per attempt (so at most w_track per attempt), times quality
    # q = exp(-d_line/track_sigma_line) * exp(-dv_frac/track_sigma_v_frac) * max(0, cos theta).
    w_track: float = 10.0
    track_sigma_line: float = 0.03
    track_radius: float = 0.06  # only motion this close to the line counts (and spends the budget)
    track_sigma_v_frac: float = 0.5
    w_env_contact: float = 0.1  # per step; weld: robot or tool touching table/floor/pedestal; physical: arm links only
    # Physical grip only: grasp potential Phi = reach + contact + lift, paid as Phi' - Phi, and the drop penalty.
    w_reach: float = 1.0
    w_contact: float = 2.0
    w_lift: float = 5.0
    lift_scale: float = 0.1
    drop_penalty: float = 10.0  # physical: a drop; grasp: knocking the floating tool
    # Grasp grip only: w_grasp_approach * (improvement of the best D_grasp) before the weld, D_grasp = grip-point
    # distance to its grasp target / approach_scale + palm rotation error / approach_theta_scale (like approach);
    # w_grasp once when the weld turns on. A knock claws back the grasp-approach reward and costs drop_penalty.
    w_grasp_approach: float = 0.3
    w_grasp: float = 2.0


@dataclass
class CurriculumCfg:
    # (start, end) values, linearly interpolated by the difficulty level in [0, 1].
    eps_p: Tuple[float, float] = (0.08, 0.03)
    # Velocity tolerance as a fraction of the goal speed: |v_head - v_g| < eps_v_frac * |v_g|.
    # Always < 1, so a stationary head can never count as matching the goal velocity.
    eps_v_frac: Tuple[float, float] = (0.6, 0.25)
    eps_theta_deg: Tuple[float, float] = (60.0, 20.0)
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
    # "grasp": a new tool floats above the table (gravity compensated) until the hand closes on its handle, which
    # turns on the weld at the actual relative pose; from then on it is weld mode. One extra action: the grip.
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
    # Grasp grip only. The canonical grasp (env._canonical_grasp) puts the handle across the fingers, head on the
    # pinky side: grip point at (grasp_palm_xy, grasp_palm_face - handle half thickness - grasp_gap) in the palm frame,
    # grasp_slide_canon * half the handle length from the handle center towards its base.
    float_height: Tuple[float, float] = (0.15, 0.35)  # tool spawn height above the table
    float_tilt_deg: float = 30.0  # handle within this angle of horizontal
    float_start_prob: float = 1.0  # else a new tool starts already welded (random grasp, as in weld mode)
    float_tries: int = 60
    float_clearance: float = 0.12  # min distance of the spawned tool (as a segment) from the palm and fingertips
    float_behind: float = 0.04  # palm and fingertips start at least this far behind the grasp, along the approach
    grasp_palm_xy: Tuple[float, float] = (-0.03, -0.035)
    grasp_palm_face: float = -0.035  # palm surface, along palm z (the fingers curl towards -z)
    grasp_gap: float = 0.004
    grasp_slide_canon: float = 0.3
    grip_close_frac: float = 0.85  # grip action +1: hand joints this far from open to the top of their range
    grip_close_thresh: float = 0.5  # smoothed grip action above which the weld can turn on
    grip_open_thresh: float = -0.9  # smoothed grip action at or below which the hand counts as open (for knocks)
    grasp_handle_tol: float = 0.04  # weld needs the handle grip point this close to the palm grip point
    pregrasp_dist: float = 0.08  # the palm approaches the grasp along its -z from this far back
    # Knocked: the floating tool moved knock_pos (m) / knock_deg from its spawn pose while the grip was open, or
    # lost_dist even while closing (closing fingers shove a light tool a few cm before the weld turns on).
    knock_pos: float = 0.02
    knock_deg: float = 10.0
    lost_dist: float = 0.10  # moved this far even while closing: knocked
    grasp_extra_steps: int = 200  # added to max_steps for episodes that start with a floating tool
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
