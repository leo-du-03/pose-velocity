"""Task evaluation envs: hammer a plate, hammer a nail, sweep dirt off a board.

Each episode samples a task instance and builds a reference trace (traces.Trace) of head position, tool orientation
and velocity that achieves the task exactly when followed exactly:

* plate / nail  The target is the peak normal contact force between the hammer and the plate (or nail head) during
                the first `impact_window` after contact. That force depends on the arm, the weld and the contact
                softness, so each episode calibrates by simulation: from a snapshot, TraceFollower runs the strike
                at several impact speeds, the target force is drawn from the measured range, and the speed for it
                is interpolated and refined. trace.meta records the follower's confirmed force error.
* sweep         The brush has a spring-loaded bristle pad, so its force is set by how far the head is pressed down.
                Each dirt cell needs a minimum force (by dirt level); the trace's pressing depth along each stroke
                gives that force plus a margin over the cells under the brush, via a static depth -> force
                calibration of the pad on the board.

Trained policies only know single goals, so TaskEnv gives them the trace's keypoints one at a time (the impact for
hammering; waypoints along the strokes and the transfers between them for sweeping). It advances to the next keypoint
when the head crosses the current one's goal plane, and resets the per-goal clock each time so the observation looks
like training. Episodes end when the task is done, the task time runs out, or a keypoint isn't reached within
EnvCfg.max_steps.
"""
from __future__ import annotations

import copy
from typing import Optional

import mujoco
import numpy as np

import mathutils as mu
from config import EnvCfg, TaskCfg
from env import PoseVelocityEnv
from goals import plane_crossing
from tool_gen import PadParams, ToolParams, build_tool
from traces import Trace, TraceBuilder, TraceFollower, min_jerk, smoothstep

OBJ = mujoco.mjtObj
_GEOM = mujoco.mjtGeom
_UP = np.array([0.0, 0.0, 1.0])
_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
_PARK = np.array([0.0, 0.0, -5.0])  # where fixtures wait while an instance is sampled
_BLOCK_HALF = 0.025  # half thickness of the nail's wooden block
_PLATFORM_HALF = 0.3  # half height of the box the nail block stands on (in_dist); tall enough for any goal height


def hammer_tool() -> ToolParams:
    """Claw-less hammer: round handle, box head across it. The striking face is the head's +x end."""
    return build_tool("cylinder", 0.30, 0.015, 1.0, "box", "T", 0.06, 0.02, 0.02, density=1200.0)


def brush_tool(tc: TaskCfg) -> ToolParams:
    """Scrubbing brush: round handle, wide box head; the bristle pad is on the head face away from the handle."""
    t = build_tool("cylinder", 0.25, 0.015, 1.0, "box", "T", 0.07, 0.02, 0.035, density=500.0)
    t.pad = PadParams(half_size=np.array([t.head_long, t.head_width, tc.pad_half_thick]),
                      pos=t.head_pos + np.array([0.0, 0.0, t.head_short]), stiffness=tc.pad_stiffness,
                      damping=tc.pad_damping, travel=tc.pad_travel, friction=tc.pad_friction)
    return t


def mat_quat(R) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(-1))
    return q


def frame_quat(z, x_hint) -> np.ndarray:
    """Orientation whose z axis is z and whose x axis is as close as possible to x_hint."""
    z = mu.normalize(z)
    x = mu.normalize(np.asarray(x_hint, np.float64) - (np.dot(x_hint, z)) * z)
    if np.linalg.norm(x) < 1e-6:
        x = mu.any_perpendicular(z)
    return mat_quat(np.column_stack([x, np.cross(z, x), z]))


def arm_margin(env, q) -> float:
    """How well-conditioned arm configuration q is for head twists: the smallest singular value of the head
    Jacobian, or 0 if the arm collides with itself or with the scene there (scratch data)."""
    m, sd = env.model, env._scratch
    env._scratch_pose(q)
    if env._scratch_collides():
        return 0.0
    for i in range(sd.ncon):  # arm links touching each other (the closed fingers touch each other by design)
        b1, b2 = int(m.geom_bodyid[sd.contact.geom1[i]]), int(m.geom_bodyid[sd.contact.geom2[i]])
        if b1 in env.robot_body_set and b2 in env.robot_body_set and not (
                b1 in env.hand_body_set and b2 in env.hand_body_set):
            return 0.0
    mujoco.mj_comPos(m, sd)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, sd, jacp, jacr, sd.site_xpos[env.head_site], env.palm_id)
    J = np.vstack([jacp[:, env.arm_dadr], jacr[:, env.arm_dadr]])
    return float(np.linalg.svd(J, compute_uv=False)[-1])


# ====================================================================================================== env
class TaskEnv(PoseVelocityEnv):
    """PoseVelocityEnv with a fixed tool, task fixtures, and one task instance (with its trace) per episode.

    Weld grip and head_twist actions only. The reward is always 0; results are in info["task"] at the episode end.
    """

    _SNAP_ATTRS = ("targets", "_a_filt", "prev_action", "_prev_head", "t", "task_t", "goal", "kp_index", "kp_log",
                   "head_log", "d_best", "closest", "ep_return", "ep_terms", "_forced_reset_kind")

    def __init__(self, task="plate", cfg: Optional[EnvCfg] = None, task_cfg: Optional[TaskCfg] = None,
                 render_mode: Optional[str] = None):
        self.task_cfg = task_cfg or TaskCfg()
        if self.task_cfg.layout not in ("in_dist", "realistic"):
            raise ValueError(f"layout must be 'in_dist' or 'realistic', got {self.task_cfg.layout!r}")
        self.task = TASKS[task](self.task_cfg) if isinstance(task, str) else task
        cfg = copy.deepcopy(cfg or EnvCfg())
        if cfg.grip_mode != "weld" or cfg.action_mode != "head_twist":
            raise ValueError("task envs need grip_mode 'weld' and action_mode 'head_twist'")
        cfg.regrip_every = 1 if self.task_cfg.rehome_each_episode else 10**9
        cfg.regrip_rehome = True
        self.trace: Optional[Trace] = None
        self.task_t, self.kp_index, self.kp_log, self.head_log = 0, 0, [], []
        self.follower = TraceFollower(self.task_cfg.k_pos, self.task_cfg.k_rot)
        self.max_task_steps = int(round(self.task_cfg.max_time / (cfg.scene.timestep * cfg.frame_skip)))
        super().__init__(cfg, render_mode)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.cfg.frame_skip

    # ------------------------------------------------------------------ hooks into PoseVelocityEnv
    def _new_tool(self) -> ToolParams:
        return self.task.tool()

    def _add_fixtures(self, spec) -> None:
        self.task.add_fixtures(spec, self)

    def _index_model(self) -> None:
        super()._index_model()
        self.task.index(self)

    def _sample_grasp(self, center, axis):
        """The task's grasp of its tool if it has one (e.g. a hammer held like a person holds it), else random."""
        grasp = self.task.grasp(self, center, axis)
        return grasp if grasp is not None else super()._sample_grasp(center, axis)

    def _physics_step(self) -> None:
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self.task.measure(self)

    def _sample_goal(self) -> None:
        """Samples a task instance (and its trace); re-homes, then regrasps, if none is reachable from here."""
        self.goal_fallback = False
        for self.goal_recovery in range(8):
            if self.goal_recovery == 1:
                self._rehome_weld()
            elif self.goal_recovery >= 2:
                self._weld_tool()
                self._prev_head = self.data.site_xpos[self.head_site].copy()
            trace = self.task.sample(self)
            if trace is not None:
                break
        else:
            raise RuntimeError(f"no reachable {self.task.name} instance found")
        self.start_trace(trace)

    # ------------------------------------------------------------------ helpers for tasks
    def set_mocap(self, mocap_id: int, pos, quat) -> None:
        """Poses a fixture in the live and the scratch (goal sampling / IK) data."""
        for d in (self.data, self._scratch):
            d.mocap_pos[mocap_id] = pos
            d.mocap_quat[mocap_id] = quat
        mujoco.mj_forward(self.model, self.data)

    def start_trace(self, trace: Trace) -> None:
        self.trace = trace
        self.task_t, self.kp_index, self.kp_log, self.head_log = 0, 0, [], []
        self.t = 0
        self.goal = trace.keypoints[0]
        self._update_goal_markers()
        self.task.begin(self)

    def snapshot(self):
        st = np.empty(mujoco.mj_stateSize(self.model, _STATE))
        mujoco.mj_getState(self.model, self.data, st, _STATE)
        py = {k: copy.deepcopy(getattr(self, k)) for k in self._SNAP_ATTRS if hasattr(self, k)}
        return st, py, self.task.snapshot(self), self.trace

    def restore(self, snap) -> None:
        st, py, task_snap, trace = snap
        mujoco.mj_setState(self.model, self.data, st, _STATE)
        mujoco.mj_forward(self.model, self.data)
        for k, v in py.items():
            setattr(self, k, v)
        self.trace = trace
        self.task.restore(self, task_snap)

    def rollout(self, trace: Trace, policy=None, max_steps: Optional[int] = None) -> dict:
        """Runs `trace` from the current state (default policy: the reference follower) and returns the task
        metrics; the env is restored afterwards. Used for calibration."""
        snap = self.snapshot()
        render_mode, self.render_mode = self.render_mode, None
        try:
            self.start_trace(trace)
            policy = policy or self.follower
            info = {}
            for _ in range(max_steps or self.max_task_steps):
                _, _, term, trunc, info = self.step(policy(self))
                if term or trunc:
                    break
            return info.get("task") or self.task.metrics(self)
        finally:
            self.render_mode = render_mode
            self.restore(snap)

    def trace_action(self) -> np.ndarray:
        """The reference follower's action for this step."""
        return self.follower(self)

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        a = self._apply_action(action)
        prev_head = self._prev_head
        bad_before = int(self.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
        self._physics_step()
        self.t += 1
        self.task_t += 1

        if not self._physics_ok(bad_before):
            self.diverged = True
            self._forced_reset_kind = "full"
            m = self.task.metrics(self)
            m.update(self._trace_metrics(), diverged=True, success=False)
            info = {"priv": np.zeros(self.priv_dim, np.float32), "success": False, "diverged": True, "task": m}
            return self._last_obs, 0.0, False, True, info

        self._update_contacts()
        head, lin, ang, _ = self._head_state()
        self.head_log.append(head)
        kps = self.trace.keypoints
        if self.kp_index < len(kps):
            g = self.goal
            d_perp = plane_crossing(prev_head, head, g.p, g.direction)
            if d_perp is not None and d_perp < self.cfg.goal.pass_radius:
                self.kp_log.append({"d_perp": d_perp, "dv_frac": float(np.linalg.norm(lin - g.v) / max(g.speed, 1e-6)),
                                    "theta_deg": float(np.degrees(np.linalg.norm(self._rot_error()))),
                                    "step": self.task_t})
                self.kp_index += 1
                if self.kp_index < len(kps):
                    self.goal = kps[self.kp_index]
                    self.t = 0
                    self._update_goal_markers()

        terminated = bool(self.task.done(self))
        truncated = (not terminated) and (self.task_t >= self.max_task_steps or self.t >= self._kp_budget())
        self.prev_action = a
        self._prev_head = head
        obs = self._get_obs()
        self._last_obs = obs
        info = {"priv": self._get_priv(), "success": False, "keypoint": self.kp_index}
        if terminated or truncated:
            m = self.task.metrics(self)
            m.update(self._trace_metrics(), diverged=False, timeout=bool(truncated))
            info["task"] = m
            info["success"] = bool(m["success"])
            self.goals_with_grip += 1
            self._forced_reset_kind = "regrip" if self.goals_with_grip >= self.cfg.regrip_every else "soft"
        if self.render_mode == "human":
            self.render()
        return obs, 0.0, terminated, truncated, info

    def _kp_budget(self) -> int:
        """Steps allowed to reach the current keypoint: EnvCfg.max_steps (as in training), or more if the trace
        itself takes longer to get there from the previous keypoint (e.g. a slow windup before the first one)."""
        idx = self.trace.keypoint_idx
        if self.kp_index >= len(idx):
            return self.cfg.max_steps
        gap = idx[self.kp_index] - (idx[self.kp_index - 1] if self.kp_index > 0 else 0)
        return max(self.cfg.max_steps, int(1.5 * gap) + 50)

    def _trace_metrics(self) -> dict:
        """Keypoint pass errors, and how far the head strayed from the trace's path (ignoring timing)."""
        out = {"keypoints": len(self.trace.keypoints), "keypoints_passed": len(self.kp_log)}
        if self.kp_log:
            for k in ("d_perp", "dv_frac", "theta_deg"):
                out[f"kp_{k}"] = float(np.mean([e[k] for e in self.kp_log]))
        if self.head_log:
            path = self.trace.p[: max(1, min(len(self.trace), self.task_t + 1))]
            dev = [float(np.min(np.linalg.norm(path - h, axis=1))) for h in self.head_log]
            out["path_dev_mean"], out["path_dev_max"] = float(np.mean(dev)), float(np.max(dev))
        return out


# ====================================================================================================== tasks
class Task:
    name = "task"

    def __init__(self, tc: TaskCfg):
        self.tc = tc
        self._tool = None

    def tool(self) -> ToolParams:
        raise NotImplementedError

    def add_fixtures(self, spec, env: TaskEnv) -> None:
        raise NotImplementedError

    def index(self, env: TaskEnv) -> None:
        raise NotImplementedError

    def sample(self, env: TaskEnv) -> Optional[Trace]:
        """Places the fixtures for a new instance and returns its trace, or None if nothing reachable was found."""
        raise NotImplementedError

    def grasp(self, env: TaskEnv, center, axis):
        """Tool pose in the palm frame (pos, quat) for a new grasp, or None for the env's random grasp. center and
        axis: the grip center and the across-the-fingers direction (PoseVelocityEnv._grip_frame)."""
        return None

    def begin(self, env: TaskEnv) -> None:
        """Resets the measurements at the start of a trace."""

    def measure(self, env: TaskEnv) -> None:
        """Called after every physics substep."""

    def done(self, env: TaskEnv) -> bool:
        raise NotImplementedError

    def metrics(self, env: TaskEnv) -> dict:
        raise NotImplementedError

    def snapshot(self, env: TaskEnv):
        return None

    def restore(self, env: TaskEnv, snap) -> None:
        pass


def _fixture_geom(body, solref, **kw):
    return body.add_geom(solref=list(solref), **kw)


class HammerTask(Task):
    """Shared by plate and nail: hit the target with the hammer face at a calibrated speed for a target peak force."""

    vertical = False  # always strike straight down (a vertical target), in either layout
    target_names: tuple = ()  # geoms whose contact with the tool is the hit
    miss_names: tuple = ()  # geoms whose contact with the tool first is a miss

    def tool(self) -> ToolParams:
        if self._tool is None:
            self._tool = hammer_tool()
        return self._tool

    def grasp(self, env, center, axis):
        """A person's hammer grip: the handle across the fingers, the hand near the end of the handle
        (grasp_from_end), the head beyond the index finger and thumb, and the striking face pointing the way the
        knuckles do (the hand's finger-extension direction, the flange axis), so a swing of the wrist and elbow drives
        the face into the target. Jittered slightly (grasp_slide_jitter, grasp_roll_jitter_deg)."""
        tc, t, d, rng = self.tc, self.tool(), env.data, env.np_random
        R = d.xmat[env.palm_id].reshape(3, 3)
        tips = d.xpos[env.tip_ids]  # index, middle, ring, thumb (thumb last)
        across = R.T @ (tips[0] - tips[-2])  # ring -> index, in the palm frame
        h = mu.normalize(axis if across @ axis >= 0 else -axis)  # handle direction, towards the head
        sid = mujoco.mj_name2id(env.model, OBJ.mjOBJ_SITE, env.cfg.scene.arm_attach_site)
        fingers = R.T @ d.site_xmat[sid].reshape(3, 3)[:, 2]  # flange axis: the way the fingers extend
        f = mu.normalize(fingers - (fingers @ h) * h)
        f = mu.rotate(mu.axis_angle_quat(h, np.deg2rad(rng.uniform(-1, 1) * tc.grasp_roll_jitter_deg)), f)
        a = mu.normalize(t.head_axis - t.head_axis[2] * np.array([0.0, 0, 1]))  # head axis, across the handle
        tool_frame = np.column_stack([a, np.cross([0.0, 0, 1], a), [0.0, 0, 1]])
        palm_frame = np.column_stack([f, np.cross(h, f), h])
        quat = mat_quat(palm_frame @ tool_frame.T)
        # Tool origin is the handle center; put the grip center grasp_from_end from the handle's far (non-head) end.
        z_hand = -(0.5 * t.handle_len - tc.grasp_from_end) + rng.uniform(-1, 1) * tc.grasp_slide_jitter
        return np.asarray(center) - z_hand * h, quat

    def _add_post(self, spec, name: str) -> None:
        post = spec.worldbody.add_body(name=name, mocap=True, pos=list(_PARK))
        post.add_geom(name=name + "_geom", type=_GEOM.mjGEOM_CYLINDER, size=[0.012, 0.1, 0], contype=0, conaffinity=0,
                      group=2, rgba=[0.35, 0.35, 0.38, 1])

    def _index_post(self, env, name: str) -> None:
        m = env.model
        self.post_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, name)])
        self.post_gid = mujoco.mj_name2id(m, OBJ.mjOBJ_GEOM, name + "_geom")

    def _place_post(self, env, top) -> None:
        """Visual stand from the table up to `top` (in the realistic layout the fixture sits on the table)."""
        z0 = env.rinfo.table_top
        half = max(0.5 * (top[2] - z0), 1e-3)
        env.model.geom_size[self.post_gid][1] = half
        env.set_mocap(self.post_mocap, [top[0], top[1], z0 + half], [1.0, 0, 0, 0])

    def index(self, env) -> None:
        m = env.model
        gid = lambda n: mujoco.mj_name2id(m, OBJ.mjOBJ_GEOM, n)  # noqa: E731
        self.target_gids = {gid(n) for n in self.target_names}
        self.miss_gids = {gid(n) for n in self.miss_names}
        env.env_geoms = env.env_geoms | self.target_gids | self.miss_gids

    def place(self, env, contact, n) -> None:
        """Puts the fixture's struck point at `contact`, its outward normal along n (towards the hammer)."""
        raise NotImplementedError

    # ---------------------------------------------------------------- instance sampling
    def _face_offset(self) -> float:
        return self.tool().head_long

    def _pre_back(self, env, v: float) -> float:
        """Distance from the contact point back to where the strike starts from rest at impact speed v."""
        n_lead = int(np.ceil(env.cfg.goal.lead_in / (v * env.dt)))
        n_acc = max(1, int(round(self.tc.accel_time / env.dt)))
        return n_lead * v * env.dt + 0.5 * v * n_acc * env.dt

    def _clear_at(self, env, q, contact, n) -> bool:
        """Arm and tool at q don't touch the fixture (moved 5 mm away so the face itself doesn't count)."""
        self.place(env, contact - 0.005 * n, n)
        env._scratch_pose(q)
        clear = not env._scratch_collides()
        self.place(env, contact, n)
        return clear

    def _approach_clear(self, env, q_pre, pre, quat, contact, n) -> bool:
        """The straight approach from the current head pose to the pre-strike pose stays in front of the fixture and
        collision-free (checked by IK at a few points along it, walking back from the pre-strike configuration)."""
        head0, quat0 = env._head_state()[0], env._tool_quat()
        if (head0 - contact) @ n < 0.02:
            return False
        env._scratch_pose(env.data.qpos[env.arm_qadr])
        if env._scratch_collides():
            return False
        rv = mu.quat_error(quat0, quat)
        ang = np.linalg.norm(rv)
        axis = rv / ang if ang > 1e-9 else np.array([1.0, 0, 0])
        q = q_pre
        for s in (0.25, 0.5, 0.75):
            x = pre + s * (head0 - pre)
            if (x - contact) @ n < 0.02:
                return False
            q = env._ik_solve(q, x, mu.quat_mul(mu.axis_angle_quat(axis, s * ang), quat))
            if q is None:
                return False
        return True

    def _candidate(self, env, rng):
        """(head, quat, d, q) of a strike pose: the tool pose at a random nearby arm configuration (in_dist, unless the
        task strikes vertically), or a straight-down strike at strike_height solved by IK."""
        gc = env.cfg.goal
        if self.tc.layout == "in_dist" and not self.vertical:
            out = env._propose_goal_q(rng)
            if out is None:
                return None
            q, head, axis, quat, _ = out
            d = mu.normalize(axis)
            if d[2] < gc.min_dir_z or not env.goal_sampler.in_region(head):
                return None
            return head, quat, d, q
        d = -_UP
        contact = env.rinfo.table_point(rng.uniform(0.4, 0.65), rng.uniform(-0.2, 0.2), self.strike_height(env, rng))
        head = contact - self._face_offset() * d
        quat = mu.quat_mul(mu.axis_angle_quat(d, rng.uniform(-np.pi, np.pi)),
                           mu.quat_from_two_vectors(env.tool.head_axis, d))
        for q0 in (env.data.qpos[env.arm_qadr], np.asarray(env.cfg.scene.arm_home[: env.n_arm_act])):
            q = env._ik_solve(q0, head - 0.01 * d, quat, iters=100)
            if q is not None:
                return head, quat, d, q
        return None

    def surface_height(self) -> float:
        """Height of the struck point above the table in the realistic layout."""
        raise NotImplementedError

    def strike_height(self, env, rng) -> float:
        """Height of the struck point above the table for a straight-down strike."""
        return self.surface_height()

    def sample(self, env) -> Optional[Trace]:
        rng, gc = env.np_random, env.cfg.goal
        env._goal_q_std = gc.goal_q_std
        back = self._pre_back(env, self.tc.strike_speed[1])
        for _ in range(4 * gc.goal_q_tries):  # most proposals are cheap rejects (strike too steep, out of region)
            self.place(env, _PARK, _UP)  # out of the way of the proposal's own collision check
            cand = self._candidate(env, rng)
            if cand is None:
                continue
            head, quat, d, q = cand
            contact = head + self._face_offset() * d
            self.place(env, contact, -d)
            if not self._clear_at(env, q, contact, -d):
                continue
            self.contact, self.normal, self.head_goal, self.quat_goal = contact, -d, head, quat
            if self.tc.windup:
                if not self._plan_windup(env, q):
                    continue
            else:
                q_lead = env._ik_solve(q, head - gc.lead_in * d, quat)
                if q_lead is None or (head - back * d)[2] < env.rinfo.table_top + gc.min_clearance:
                    continue
                q_pre = env._ik_solve(q_lead, head - back * d, quat)
                if q_pre is None or not self._approach_clear(env, q_pre, head - back * d, quat, contact, -d):
                    continue
            trace = self._calibrate(env, rng)
            if trace is not None:
                return trace
        return None

    # ---------------------------------------------------------------- windup
    def arc_pose(self, phi: float):
        """Head position and tool orientation phi (rad) back along the swing arc from the contact pose: the tool
        rotated rigidly about the pivot by phi, raising the head back against the strike direction."""
        d, R = -self.normal, self.tc.windup_radius
        p = self.pivot + R * (np.cos(phi) * self.arc_u - np.sin(phi) * d)
        return p, mu.quat_mul(mu.axis_angle_quat(self.arc_axis, phi), self.quat_goal)

    def _plan_windup(self, env, q) -> bool:
        """Sets the swing arc for the current instance, with the largest reachable windup angle. The pivot is
        windup_radius from the head at contact, back along the handle (towards the hand), so the head axis is tangent
        to the arc at the hit."""
        tc, gc = self.tc, env.cfg.goal
        d = -self.normal
        u = mu.rotate(self.quat_goal, [0.0, 0.0, 1.0])  # tool +z: from the handle towards the head
        u = u - (u @ d) * d
        if np.linalg.norm(u) < 0.3:  # the handle points along the strike: no sensible swing plane
            return False
        self.arc_u = mu.normalize(u)
        self.arc_axis = mu.normalize(np.cross(d, self.arc_u))  # rotating by +phi about it raises the head back
        self.pivot = self.head_goal - tc.windup_radius * self.arc_u
        for deg in tc.windup_angles:
            if self._windup_path_ok(env, q, np.deg2rad(deg)):
                self.windup_deg = deg
                return True
        return False

    def _windup_path_ok(self, env, q_hit, phi_top: float) -> bool:
        """The swing arc, walked by IK in small steps from the hit configuration q_hit up to the top of the windup
        (it is executed in reverse, on the same branch), and the raise, a joint-space path from the current
        configuration to the top. Every point must be reachable, collision-free, well-conditioned (no singularity on
        the way), above the table and in front of the target. Sets self.q_top."""
        tc, gc = self.tc, env.cfg.goal
        min_z = env.rinfo.table_top + gc.min_clearance
        ok_at = lambda p: p[2] >= min_z and (p - self.contact) @ self.normal >= 0.02  # noqa: E731
        q = q_hit
        n_arc = int(np.ceil(phi_top * tc.windup_radius / 0.03)) + 1
        for phi in np.linspace(np.deg2rad(5.0), phi_top, n_arc):
            p, quat = self.arc_pose(phi)
            q = env._ik_solve(q, p, quat) if ok_at(p) else None
            if q is None or arm_margin(env, q) < tc.min_arm_margin:
                return False
        q0 = env.data.qpos[env.arm_qadr].copy()
        n = max(10, int(np.ceil(np.abs(q - q0).max() / 0.1)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:-1]:
            qq = q0 + s * (q - q0)
            if arm_margin(env, qq) < tc.min_arm_margin or not ok_at(env._scratch_pose(qq)[0]):
                return False
        self.q_top = q
        return True

    def _raise_samples(self, env, b: TraceBuilder, top, top_q) -> None:
        """Appends the raise: a minimum-jerk joint-space move from the current arm configuration to the top of the
        windup (like a person lifting a hammer: the whole arm moves smoothly), sampled as the tool pose it gives."""
        tc, dt = self.tc, env.dt
        q0 = env.data.qpos[env.arm_qadr].copy()
        q1 = env._ik_solve(self.q_top, top, top_q)  # the exact top (the swing profile rounds the windup angle)
        q1 = self.q_top if q1 is None else q1
        dq = q1 - q0
        dist = float(np.linalg.norm(top - b.end[0]))
        T = max(0.8, 1.875 * float(np.abs(dq).max()) / tc.windup_joint_speed, 1.875 * dist / tc.windup_raise_speed)
        n = max(2, int(round(T / dt)))
        sj, _ = min_jerk(np.arange(1, n + 1) / n)
        ps, qs = [], []
        for x in sj:
            head, _, quat, _ = env._scratch_pose(q0 + x * dq)
            ps.append(head)
            qs.append(quat)
        ps[-1], qs[-1] = top, top_q
        full_p = np.array([b.end[0]] + ps)
        full_q = [b.end[1]] + qs
        for k in range(1, len(full_q)):  # keep quaternion signs continuous
            if np.dot(full_q[k], full_q[k - 1]) < 0:
                full_q[k] = -full_q[k]
        vs = np.gradient(full_p, dt, axis=0)[1:]
        ws = [mu.quat_error(full_q[min(k + 1, n)], full_q[k - 1]) / ((min(k + 1, n) - (k - 1)) * dt)
              for k in range(1, n + 1)]
        vs[-1], ws[-1] = 0.0, np.zeros(3)
        b.samples(ps, full_q[1:], vs, ws, "windup")

    def _swing_profile(self, env, v: float):
        """Arc length from the top of the windup and head speed at each step of the swing, through the contact and
        the follow-through. Durations are whole steps; the windup angle is adjusted (slightly) so the contact lands on
        a sample. Returns (total arc to contact, arc lengths, speeds, steps to contact)."""
        tc, dt = self.tc, env.dt
        L_want = tc.windup_radius * np.deg2rad(self.windup_deg)
        n_a = max(1, int(round(2 * tc.swing_accel_frac * L_want / v / dt)))
        L_a = 0.5 * v * n_a * dt
        n_c = max(1, int(round((L_want - L_a) / (v * dt))))
        L = L_a + n_c * v * dt
        tau = np.arange(1, n_a + 1) / n_a
        sv, sx = smoothstep(tau)
        n_f = max(1, int(round(tc.impact_window / dt)))
        n_d = max(1, int(round(0.1 / dt)))
        tau_d = np.arange(1, n_d + 1) / n_d
        sv_d, sx_d = smoothstep(tau_d)
        s_f_end = L + n_f * v * dt
        s = np.concatenate([v * n_a * dt * sx, L_a + v * dt * np.arange(1, n_c + 1),
                            L + v * dt * np.arange(1, n_f + 1), s_f_end + v * n_d * dt * (tau_d - sx_d)])
        speed = np.concatenate([v * sv, np.full(n_c + n_f, v), v * (1.0 - sv_d)])
        return L, s, speed, n_a + n_c

    def _windup_trace(self, env, v: float, force: float) -> Trace:
        tc, gc, R = self.tc, env.cfg.goal, self.tc.windup_radius
        d = -self.normal
        L, s, speed, n_hit = self._swing_profile(env, v)
        phi_top = L / R
        p0, _, _, _ = env._head_state()
        b = TraceBuilder(env.dt, p0, env._tool_quat(), env.tool.head_axis)
        top, top_q = self.arc_pose(phi_top)
        k0 = b.index
        self._raise_samples(env, b, top, top_q)
        k_raise = (k0 + b.index) // 2
        b.hold(tc.windup_pause)
        k_top = b.index
        phi = (L - s) / R  # negative past the contact (follow-through along the arc)
        poses = [self.arc_pose(x) for x in phi]
        vel = speed[:, None] * (np.sin(phi)[:, None] * self.arc_u + np.cos(phi)[:, None] * d)
        w = -(speed / R)[:, None] * self.arc_axis
        ps, qs = [p for p, _ in poses], [q for _, q in poses]
        b.samples(ps[:n_hit], qs[:n_hit], vel[:n_hit], w[:n_hit], "swing")
        k_contact = b.index
        b.samples(ps[n_hit:], qs[n_hit:], vel[n_hit:], w[n_hit:], "contact", force)
        b.f[k_contact] = force
        back, back_q = self.arc_pose(np.deg2rad(20.0))
        b.move_to(back, back_q, 0.5, "retract")
        tr = b.build(task=self.name, impact_speed=v, force_target=force, contact_point=self.contact,
                     normal=self.normal, contact_index=k_contact, windup_deg=float(np.degrees(phi_top)),
                     windup_radius=R)
        # Keypoints for a policy: halfway up the windup, halfway down the swing, and the hit. The first two only if
        # they are fast enough to be training-like goals (GoalCfg.speed_min).
        if np.linalg.norm(tr.v[k_raise]) >= gc.speed_min and k_raise > k0:
            tr.add_keypoint(k_raise)
        k_mid = k_top + 1 + int(np.argmin(np.abs(phi[:n_hit] - 0.5 * phi_top)))
        fast = [k for k in range(k_mid, k_contact - 5) if np.linalg.norm(tr.v[k]) >= gc.speed_min]
        if fast:
            tr.add_keypoint(fast[0])
        tr.add_keypoint(k_contact)
        return tr

    # ---------------------------------------------------------------- trace + calibration
    def build_trace(self, env, v: float, force: float = 0.0) -> Trace:
        if self.tc.windup:
            return self._windup_trace(env, v, force)
        tc, gc = self.tc, env.cfg.goal
        d, head, quat = -self.normal, self.head_goal, self.quat_goal
        p0, _, _, _ = env._head_state()
        b = TraceBuilder(env.dt, p0, env._tool_quat(), env.tool.head_axis)
        pre = head - self._pre_back(env, v) * d
        b.move_to(pre, quat, b.move_time(pre, quat, 0.4, 1.5, 0.8), "approach")
        b.hold(0.2)
        b.ramp(d, 0.0, v, max(1, int(round(tc.accel_time / env.dt))) * env.dt)
        b.const(d, v, int(np.ceil(gc.lead_in / (v * env.dt))), "strike")
        k_contact = b.index
        # Follow through (the fixture stops the head; the trace keeps going so a follower doesn't brake before the
        # hit, and trained policies follow through too), then stop and back off.
        b.const(d, v, max(1, int(round(tc.impact_window / env.dt))), "contact", force)
        b.ramp(d, v, 0.0, 0.1, "contact")
        b.f[k_contact] = force
        b.move_to(head - 0.08 * d, quat, 0.5, "retract")
        tr = b.build(task=self.name, impact_speed=v, force_target=force, contact_point=self.contact,
                     normal=self.normal, contact_index=k_contact)
        tr.add_keypoint(k_contact)
        return tr

    def _calibrate(self, env, rng) -> Trace:
        """Impact speed -> peak force by follower rollouts; then a target force within that range and its speed."""
        tc = self.tc
        speeds = np.linspace(tc.strike_speed[0], tc.strike_speed[1], tc.calib_speeds)
        runs = [env.rollout(self.build_trace(env, v)) for v in speeds]
        forces = np.array([r["peak_force"] if r["hit"] else 0.0 for r in runs])
        hit = forces > 0.0  # the follower can miss (e.g. a fast strike it can't track); use the speeds that hit
        if hit.sum() < 2:
            return None
        mono = np.maximum.accumulate(forces[hit])  # peak force rises with speed; flatten measurement noise
        f_lo, f_hi = mono[0], mono[-1]
        if f_hi <= f_lo:
            return None
        target = float(rng.uniform(f_lo, f_hi))
        v = float(np.interp(target, mono, speeds[hit]))
        # Refine with the confirmed force (secant on the nearest calibration points, clipped to the speed range).
        pts = list(zip(speeds[hit], forces[hit]))
        best = None
        for _ in range(1 + tc.calib_refine):
            r = env.rollout(self.build_trace(env, v, target))
            f = r["peak_force"] if r["hit"] else 0.0
            err = (f - target) / target
            if best is None or abs(err) < abs(best[2]):
                best = (v, f, err, r)
            if abs(err) < 0.02:
                break
            pts.append((v, f))
            near = sorted(pts, key=lambda p: abs(p[1] - target))[:2]
            (v1, f1), (v2, f2) = near
            if abs(f2 - f1) > 1e-6 and v1 != v2:
                v = v1 + (target - f1) * (v2 - v1) / (f2 - f1)
            else:
                v = v * (target / max(f, 1e-6))
            v = float(np.clip(v, 0.5 * tc.strike_speed[0], env.cfg.head_v_max))
        v, f, err, r = best
        if abs(err) > 0.5 * tc.force_tol:  # not even the reference follower gets this one right: resample
            return None
        tr = self.build_trace(env, v, target)
        tr.meta.update(calib_speeds=speeds.tolist(), calib_forces=forces.tolist(), follower_force=f,
                       follower_force_err=err, follower_contact_speed=r.get("contact_speed", float("nan")))
        return tr

    # ---------------------------------------------------------------- measurement
    def begin(self, env) -> None:
        self.t_first = None
        self.first_target = False
        self.peak = 0.0
        self.impulse = 0.0
        self.at_contact: dict = {}
        self.t_end_kp = None

    def snapshot(self, env):
        return None

    def measure(self, env) -> None:
        m, d = env.model, env.data
        f6 = np.zeros(6)
        f_target, touched_miss = 0.0, False
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == env.tool_id:
                other = g2
            elif b2 == env.tool_id:
                other = g1
            else:
                continue
            if other in self.target_gids:
                mujoco.mj_contactForce(m, d, i, f6)
                f_target += abs(f6[0])
            elif other in self.miss_gids:
                touched_miss = True
        if self.t_first is None and (f_target > 0.0 or touched_miss):
            self.t_first = float(d.time)
            self.first_target = f_target > 0.0
            head, lin, _, axis = env._head_state()
            face = head + self._face_offset() * axis
            rel = face - self.contact
            self.at_contact = {
                "contact_speed": float(-lin @ self.normal),
                "contact_err": float(np.linalg.norm(rel - (rel @ self.normal) * self.normal)),
                "contact_theta_deg": float(np.degrees(np.linalg.norm(mu.quat_error(self.quat_goal, env._tool_quat())))),
            }
        if self.t_first is not None and d.time - self.t_first <= self.tc.impact_window + 1e-9:
            self.peak = max(self.peak, f_target)
            self.impulse += f_target * m.opt.timestep

    def done(self, env) -> bool:
        if self.t_first is not None:
            return env.data.time - self.t_first >= self.tc.impact_window
        if env.kp_index >= len(env.trace.keypoints):  # crossed the struck plane without touching: a miss
            self.t_end_kp = env.task_t if self.t_end_kp is None else self.t_end_kp
            return env.task_t - self.t_end_kp >= 10
        return False

    def metrics(self, env) -> dict:
        target = float(env.trace.meta.get("force_target", 0.0))
        hit = bool(self.t_first is not None and self.first_target and self._hit_ok())
        err = (self.peak - target) / target if target > 0 else float("nan")
        out = {"hit": hit, "touched": self.t_first is not None, "force_target": target, "peak_force": self.peak,
               "impulse": self.impulse, "force_err": float(err),
               "success": bool(hit and abs(err) < self.tc.force_tol),
               "follower_force_err": float(env.trace.meta.get("follower_force_err", float("nan")))}
        out.update(self.at_contact)
        n_windup = len(env.trace.keypoints) - 1  # goals before the hit (raise, swing)
        out["windup_keypoints"], out["windup_passed"] = n_windup, min(len(env.kp_log), n_windup)
        return out

    def _hit_ok(self) -> bool:
        return True


class PlateTask(HammerTask):
    name = "plate"
    target_names = ("plate",)

    def add_fixtures(self, spec, env) -> None:
        tc = self.tc
        plate = spec.worldbody.add_body(name="plate_body", mocap=True, pos=list(_PARK))
        _fixture_geom(plate, tc.fixture_solref, name="plate", type=_GEOM.mjGEOM_BOX, size=list(tc.plate_half_size),
                      pos=[0, 0, -tc.plate_half_size[2]], rgba=[0.55, 0.57, 0.62, 1])
        self._add_post(spec, "plate_post")

    def index(self, env) -> None:
        super().index(env)
        m = env.model
        self.mocap = int(m.body_mocapid[mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "plate_body")])
        self._index_post(env, "plate_post")

    def surface_height(self) -> float:
        return 2 * self.tc.plate_half_size[2]

    def place(self, env, contact, n) -> None:
        # The plate's top face (local z = 0) goes through the contact point, normal n.
        env.set_mocap(self.mocap, contact, frame_quat(n, _UP if abs(n[2]) < 0.9 else [1.0, 0, 0]))
        below = contact - mu.rotate(frame_quat(n, _UP), [0, 0, 2 * self.tc.plate_half_size[2]])
        self._place_post(env, below)


class NailTask(HammerTask):
    """A vertical nail in a wooden block, struck straight down. In the in_dist layout the block stands on a platform
    that raises the strike into the training goal heights (TaskCfg.nail_height); in the realistic layout the block
    sits on the table."""

    name = "nail"
    vertical = True
    target_names = ("nail_head", "nail_shaft")
    miss_names = ("nail_block", "nail_platform")

    def add_fixtures(self, spec, env) -> None:
        tc = self.tc
        block = spec.worldbody.add_body(name="nail_block_body", mocap=True, pos=list(_PARK))
        _fixture_geom(block, tc.fixture_solref, name="nail_block", type=_GEOM.mjGEOM_BOX,
                      size=[0.05, 0.05, _BLOCK_HALF], pos=[0, 0, -_BLOCK_HALF], rgba=[0.62, 0.45, 0.28, 1])
        nail = block.add_body(name="nail", gravcomp=1.0)  # else it creeps down against the soft frictionloss
        nail.add_joint(name="nail_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                       range=[-tc.nail_travel, 0.0], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                       frictionloss=tc.nail_hold, damping=tc.nail_damping)
        top = tc.nail_protrude
        shaft_len = top + tc.nail_travel
        _fixture_geom(nail, tc.fixture_solref, name="nail_shaft", type=_GEOM.mjGEOM_CYLINDER,
                      size=[0.003, 0.5 * shaft_len, 0], pos=[0, 0, top - 0.5 * shaft_len], mass=tc.nail_mass * 0.6,
                      rgba=[0.7, 0.7, 0.72, 1])
        _fixture_geom(nail, tc.fixture_solref, name="nail_head", type=_GEOM.mjGEOM_CYLINDER,
                      size=[tc.nail_head_radius, 0.002, 0], pos=[0, 0, top - 0.002], mass=tc.nail_mass * 0.4,
                      rgba=[0.75, 0.75, 0.78, 1])
        spec.add_exclude(bodyname1="nail_block_body", bodyname2="nail")
        spec.add_exclude(bodyname1="world", bodyname2="nail")  # the shaft may poke out of the block into the table
        # Platform under the block: a fixed-size box moved so its top meets the block's bottom (its lower part is
        # hidden in and under the table), so it collides correctly without resizing geoms at runtime.
        platform = spec.worldbody.add_body(name="nail_platform_body", mocap=True, pos=list(_PARK))
        _fixture_geom(platform, tc.fixture_solref, name="nail_platform", type=_GEOM.mjGEOM_BOX,
                      size=[0.07, 0.07, _PLATFORM_HALF], pos=[0, 0, -_PLATFORM_HALF], rgba=[0.4, 0.4, 0.43, 1])
        spec.add_exclude(bodyname1="nail_platform_body", bodyname2="nail")

    def index(self, env) -> None:
        super().index(env)
        m = env.model
        self.mocap = int(m.body_mocapid[mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "nail_block_body")])
        j = mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, "nail_slide")
        self.nail_qadr, self.nail_dadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.platform_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "nail_platform_body")])

    def surface_height(self) -> float:
        return 2 * _BLOCK_HALF + self.tc.nail_protrude

    def strike_height(self, env, rng) -> float:
        if self.tc.layout == "realistic":
            return self.surface_height()
        # The head center (face_offset above the nail head at contact) at a training goal height.
        return float(rng.uniform(*self.tc.nail_height)) - self._face_offset()

    def place(self, env, contact, n) -> None:
        # The block's top face sits nail_protrude below the nail head's top, which is the contact point.
        quat = frame_quat(n, _UP if abs(n[2]) < 0.9 else [1.0, 0, 0])
        env.set_mocap(self.mocap, contact - self.tc.nail_protrude * n, quat)
        bottom = contact - (self.tc.nail_protrude + 2 * _BLOCK_HALF) * n
        raised = bottom[2] > env.rinfo.table_top + 1e-3  # on the table (realistic) or parked: no platform
        env.set_mocap(self.platform_mocap, bottom if raised else _PARK, [1.0, 0, 0, 0])

    def begin(self, env) -> None:
        super().begin(env)
        d = env.data
        d.qpos[self.nail_qadr] = 0.0
        d.qvel[self.nail_dadr] = 0.0
        mujoco.mj_forward(env.model, d)

    def _hit_ok(self) -> bool:
        return self.at_contact.get("contact_err", np.inf) < self.tc.nail_tol

    def metrics(self, env) -> dict:
        out = super().metrics(env)
        out["nail_driven"] = float(-env.data.qpos[self.nail_qadr])
        return out


class SweepTask(Task):
    """Sweep a dirty patch with the brush. A dirt cell is removed when it is under the bristle pad while the pad
    pushes with at least that cell's required force (TaskCfg.dirt_forces, by dirt level)."""

    name = "sweep"

    def tool(self) -> ToolParams:
        if self._tool is None:
            self._tool = brush_tool(self.tc)
        return self._tool

    def _grid(self):
        tc = self.tc
        ns, nc = int(round(tc.dirt_size[0] / tc.dirt_cell)), int(round(tc.dirt_size[1] / tc.dirt_cell))
        xs = (np.arange(ns) + 0.5) * tc.dirt_cell - 0.5 * ns * tc.dirt_cell
        ys = (np.arange(nc) + 0.5) * tc.dirt_cell - 0.5 * nc * tc.dirt_cell
        return np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)  # dirt frame: x along strokes

    def board_half(self):
        tc = self.tc
        h = tc.board_height if tc.layout == "in_dist" else 0.004
        return np.array([0.24, 0.32, 0.5 * h])

    def add_fixtures(self, spec, env) -> None:
        tc = self.tc
        board = spec.worldbody.add_body(name="board_body", mocap=True, pos=list(_PARK))
        _fixture_geom(board, tc.fixture_solref, name="board", type=_GEOM.mjGEOM_BOX, size=list(self.board_half()),
                      pos=[0, 0, -self.board_half()[2]], friction=[tc.pad_friction, 0.005, 0.0001],
                      rgba=[0.86, 0.84, 0.78, 1])
        dirt = spec.worldbody.add_body(name="dirt_body", mocap=True, pos=list(_PARK))
        h = 0.45 * tc.dirt_cell
        for i, (x, y) in enumerate(self._grid()):
            dirt.add_geom(name=f"dirt{i}", type=_GEOM.mjGEOM_BOX, size=[h, h, 0.0008], pos=[x, y, 0.0008],
                          contype=0, conaffinity=0, group=2, rgba=[0.35, 0.22, 0.1, 0.0])

    def index(self, env) -> None:
        m = env.model
        name = lambda kind, n: mujoco.mj_name2id(m, kind, n)  # noqa: E731
        self.board_mocap = int(m.body_mocapid[name(OBJ.mjOBJ_BODY, "board_body")])
        self.dirt_mocap = int(m.body_mocapid[name(OBJ.mjOBJ_BODY, "dirt_body")])
        self.board_gid = name(OBJ.mjOBJ_GEOM, "board")
        self.cells_xy = self._grid()
        self.dirt_gids = np.array([name(OBJ.mjOBJ_GEOM, f"dirt{i}") for i in range(len(self.cells_xy))])
        self.pad_id = name(OBJ.mjOBJ_BODY, "tool_pad")
        j = name(OBJ.mjOBJ_JOINT, "tool_pad_slide")
        self.pad_qadr, self.pad_dadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        env.env_geoms = env.env_geoms | {self.board_gid}
        self.level = np.zeros(len(self.cells_xy), dtype=np.int64)
        self.depth_cal = None  # (depths, forces) of the pad on the board; per compiled model

    # ---------------------------------------------------------------- geometry
    def pad_force(self, env) -> float:
        return float(self.tc.pad_stiffness * max(0.0, -env.data.qpos[self.pad_qadr]))

    def touch_offset(self) -> float:
        """Head center to the pad's free face, along the tool's +z."""
        t = self.tool()
        return t.head_short + 2 * self.tc.pad_half_thick

    def _depth_calibration(self, env, top) -> tuple:
        """Pad force at several pressing depths, with the brush held exactly at the pose (weld off)."""
        snap = env.snapshot()
        m, d = env.model, env.data
        d.eq_active[env.grip_eq] = 0
        quat = frame_quat(-_UP, [1.0, 0, 0])
        R = mu.quat_to_mat(quat)
        depths = np.linspace(0.0, min(0.95 * self.tc.pad_travel, 2 * self.tc.pad_half_thick - 0.002), 8)
        forces = []
        for depth in depths:
            head = top + (self.touch_offset() - depth) * _UP
            body = head - R @ env.tool.head_pos
            d.qpos[self.pad_qadr], d.qvel[self.pad_dadr] = -depth, 0.0
            fs = []
            for k in range(250):
                env._set_tool_pose(body, quat)
                mujoco.mj_step(m, d)
                if k >= 200:
                    fs.append(self.pad_force(env))
            forces.append(float(np.mean(fs)))
        env.restore(snap)
        d.eq_active[env.grip_eq] = 1
        mujoco.mj_forward(m, d)
        forces = np.maximum.accumulate(np.array(forces))
        return depths, forces

    def depth_for(self, force) -> np.ndarray:
        depths, forces = self.depth_cal
        return np.interp(force, forces, depths)

    # ---------------------------------------------------------------- instance sampling
    def sample(self, env) -> Optional[Trace]:
        tc, rng, ri = self.tc, env.np_random, env.rinfo
        a = self.tool().head_long  # brush half-width across the stroke
        width = 2 * a
        spacing = width * (1.0 - tc.stroke_overlap)
        n_strokes = max(1, int(np.ceil((tc.dirt_size[1] - width) / spacing)) + 1)
        offsets = (np.arange(n_strokes) - 0.5 * (n_strokes - 1)) * spacing
        bh = self.board_half()
        for _ in range(40):
            u, w = rng.uniform(0.42, 0.6), rng.uniform(-0.12, 0.12)
            top = ri.table_point(u, w, 2 * bh[2])
            yaw0 = np.arctan2(ri.table_dir[1], ri.table_dir[0])
            env.set_mocap(self.board_mocap, top, mu.yaw_quat(yaw0))
            # Stroke direction and which way the head faces: the best-conditioned arm path wins.
            best = (self.tc.min_arm_margin, None)
            for phi in np.linspace(-np.pi / 2, np.pi / 2, 6, endpoint=False) + rng.uniform(0, np.pi / 6):
                e_s = mu.rotate(mu.yaw_quat(yaw0 + np.pi / 2 + phi), [1.0, 0, 0])  # along the strokes
                e_c = np.cross(_UP, e_s)
                for sign in (1.0, -1.0):
                    quat = frame_quat(-_UP, sign * e_c)  # tool z down; head long axis across the strokes
                    strokes = self._strokes(top, e_s, e_c, offsets)
                    margin = self._path_margin(env, strokes, quat)
                    if margin > best[0]:
                        best = (margin, (e_s, e_c, quat, strokes))
            if best[1] is None:
                continue
            e_s, e_c, quat, strokes = best[1]
            env.set_mocap(self.dirt_mocap, top, frame_quat(_UP, e_s))
            self.top, self.e_s, self.e_c, self.quat = top, e_s, e_c, quat
            self.offsets, self.strokes = offsets, strokes
            self._make_dirt(env, rng)
            if self.depth_cal is None:
                self.depth_cal = self._depth_calibration(env, top)
            return self.build_trace(env)
        return None

    def _strokes(self, top, e_s, e_c, offsets):
        """(start, end, direction) of each stroke at touch height; boustrophedon."""
        tc = self.tc
        half = 0.5 * tc.dirt_size[0]
        out = []
        for i, off in enumerate(offsets):
            dirn = e_s if i % 2 == 0 else -e_s
            mid = top + off * e_c + self.touch_offset() * _UP
            out.append((mid - (half + tc.run_up) * dirn, mid + (half + tc.overrun) * dirn, dirn))
        return out

    def _path_margin(self, env, strokes, quat) -> float:
        """Worst arm_margin along the strokes (chained IK every 3 cm, 5 mm above touch height so the pad doesn't
        count as a collision, plus the hover points); 0 if any point is unreachable."""
        q = env.data.qpos[env.arm_qadr].copy()
        lift = self.tc.hover * _UP
        worst = np.inf
        for start, end, dirn in strokes:
            n = max(2, int(np.ceil(np.linalg.norm(end - start) / 0.03)) + 1)
            pts = [start + lift] + [start + s * (end - start) + 0.005 * _UP for s in np.linspace(0, 1, n)] + [end + lift]
            for x in pts:
                q = env._ik_solve(q, x, quat, iters=100)
                if q is None:
                    return 0.0
                worst = min(worst, arm_margin(env, q))
                if worst <= self.tc.min_arm_margin:
                    return 0.0
        return worst

    def _make_dirt(self, env, rng) -> None:
        tc = self.tc
        xy = self.cells_xy
        level = np.zeros(len(xy), dtype=np.int64)
        half = 0.5 * np.array(tc.dirt_size)
        for _ in range(int(rng.integers(tc.dirt_blobs[0], tc.dirt_blobs[1] + 1))):
            c = rng.uniform(-half, half)
            r = rng.uniform(0.03, 0.07)
            lv = int(rng.integers(1, len(tc.dirt_forces) + 1))
            inside = np.linalg.norm(xy - c, axis=1) < r
            level[inside] = np.maximum(level[inside], lv)
        self.level = level
        R = mu.quat_to_mat(frame_quat(_UP, self.e_s))
        self.cells_world = self.top + (np.column_stack([xy, np.zeros(len(xy))]) @ R.T)
        self.required = np.array([0.0] + list(tc.dirt_forces))[level]
        self._paint(env, np.zeros(len(xy), dtype=bool))

    def _paint(self, env, cleaned) -> None:
        alpha = np.where(cleaned | (self.level == 0), 0.0, 0.3 + 0.6 * self.level / len(self.tc.dirt_forces))
        env.model.geom_rgba[self.dirt_gids, 3] = alpha

    # ---------------------------------------------------------------- trace
    def _force_profile(self, s, off) -> np.ndarray:
        """Force the trace presses with at stroke positions s (along the dirt frame x) for a stroke at offset off:
        required force of the cells under the brush plus the margin, at least a light touch, made smooth without
        dropping below the requirement (a moving max, then a moving average of the same width)."""
        tc, t = self.tc, self.tool()
        xy = self.cells_xy
        across = np.abs(xy[:, 1] - off) <= t.head_long
        req = np.zeros(len(s))
        for i, si in enumerate(s):
            under = across & (np.abs(xy[:, 0] - si) <= t.head_width + 0.5 * tc.dirt_cell)
            req[i] = self.required[under].max() if under.any() else 0.0
        f = np.maximum(req * (1.0 + tc.force_margin), 0.5 * min(tc.dirt_forces))
        h = max(1, int(round(0.03 / max(np.mean(np.diff(s)) if len(s) > 1 else 1.0, 1e-6))))
        pad = np.pad(f, h, mode="edge")
        fmax = np.array([pad[i:i + 2 * h + 1].max() for i in range(len(f))])
        pad = np.pad(fmax, h, mode="edge")
        return np.array([pad[i:i + 2 * h + 1].mean() for i in range(len(f))])

    def build_trace(self, env) -> Trace:
        tc = self.tc
        dt, vs = env.dt, tc.sweep_speed
        p0, _, _, _ = env._head_state()
        b = TraceBuilder(dt, p0, env._tool_quat(), env.tool.head_axis)
        hover = tc.hover * _UP
        quat = self.quat
        kps = []
        n_acc = max(1, int(round(tc.accel_time / dt)))
        half = 0.5 * tc.dirt_size[0]
        for i, (start, end, dirn) in enumerate(self.strokes):
            sgn = 1.0 if i % 2 == 0 else -1.0  # stroke direction in the dirt frame
            length = float(np.linalg.norm(end - start))
            # Arc length along the stroke: accelerate over the run-up, constant through the dirt, stop in the overrun.
            n_dec = max(1, int(round(2 * tc.overrun / vs / dt)))
            l_acc = 0.5 * vs * n_acc * dt
            l_dec = 0.5 * vs * n_dec * dt
            n_const = int(np.ceil((length - l_acc - l_dec) / (vs * dt)))
            tau_a = np.arange(1, n_acc + 1) / n_acc
            tau_d = np.arange(1, n_dec + 1) / n_dec
            sa, ss_a = smoothstep(tau_a)
            sd, ss_d = smoothstep(tau_d)
            arc = np.concatenate([vs * n_acc * dt * ss_a,
                                  l_acc + vs * dt * np.arange(1, n_const + 1),
                                  l_acc + vs * dt * n_const + vs * n_dec * dt * (tau_d - ss_d)])
            speed = np.concatenate([vs * sa, np.full(n_const, vs), vs * (1.0 - sd)])
            s_dirt = sgn * (arc - (tc.run_up + half))  # position in the dirt frame
            force = self._force_profile(s_dirt, self.offsets[i])
            depth = self.depth_for(force)
            ps = start + arc[:, None] * dirn - depth[:, None] * _UP
            ddepth = np.gradient(depth, arc) if len(arc) > 1 else np.zeros_like(arc)
            vel = speed[:, None] * (dirn[None, :] - ddepth[:, None] * _UP[None, :])
            above = start + hover
            if i == 0:
                b.move_to(above, quat, b.move_time(above, quat, 0.4, 1.5, 0.8), "approach")
            else:
                p_end, _ = b.end
                # Transfer at the sweep speed at its middle (a min-jerk peak is 1.875x the average), so the keypoint
                # there has a training-range speed.
                T = 1.875 * float(np.linalg.norm(above - p_end)) / vs
                k0 = b.index
                b.move_to(above, quat, T, "transfer")
                kps.append((k0 + b.index) // 2)
            b.hold(0.1)
            b.move_to(start - depth[0] * _UP, quat, 0.4, "lower")
            k0 = b.index
            b.path(ps, vel, "stroke", force)
            # Keypoints: through the dirt, every waypoint_spacing, all on the constant-speed part.
            for x in np.arange(-half, half + 1e-9, tc.waypoint_spacing):
                k = k0 + 1 + int(np.argmin(np.abs(s_dirt - sgn * x)))
                kps.append(k)
            b.move_to(b.end[0] + hover, quat, 0.3, "lift")
        tr = b.build(task=self.name, n_strokes=len(self.strokes), dirt_cells=int((self.level > 0).sum()))
        for k in sorted(kps):
            tr.add_keypoint(k)
        return tr

    # ---------------------------------------------------------------- measurement
    def begin(self, env) -> None:
        self.cleaned = np.zeros(len(self.cells_xy), dtype=bool)
        self.fmax = np.zeros(len(self.cells_xy))
        self._paint(env, self.cleaned)

    def snapshot(self, env):
        return None if not hasattr(self, "cleaned") else (self.cleaned.copy(), self.fmax.copy())

    def restore(self, env, snap) -> None:
        if snap is not None:
            self.cleaned, self.fmax = snap
            self._paint(env, self.cleaned)

    def measure(self, env) -> None:
        f = self.pad_force(env)
        if f <= 0.0:
            return
        d, t = env.data, self.tool()
        R = d.xmat[self.pad_id].reshape(3, 3)
        local = (self.cells_world - d.xpos[self.pad_id]) @ R  # cell centers in the pad frame
        face = 2 * self.tc.pad_half_thick
        under = ((np.abs(local[:, 0]) <= t.head_long) & (np.abs(local[:, 1]) <= t.head_width)
                 & (np.abs(local[:, 2] - face) < 0.01))
        if not under.any():
            return
        self.fmax[under] = np.maximum(self.fmax[under], f)
        new = under & ~self.cleaned & (self.level > 0) & (f >= self.required)
        if new.any():
            self.cleaned |= new
            self._paint(env, self.cleaned)

    def done(self, env) -> bool:
        return env.kp_index >= len(env.trace.keypoints)

    def metrics(self, env) -> dict:
        dirty = self.level > 0
        w = self.level[dirty].astype(np.float64)
        frac = float((w * self.cleaned[dirty]).sum() / max(w.sum(), 1e-9))
        out = {"cleaned": frac, "dirty_cells": int(dirty.sum()), "success": frac >= 0.95}
        for lv in range(1, len(self.tc.dirt_forces) + 1):
            sel = self.level == lv
            out[f"cleaned_l{lv}"] = float(self.cleaned[sel].mean()) if sel.any() else float("nan")
            out[f"dirt_l{lv}"] = int(sel.sum())  # cells of this level at the start
            out[f"removed_l{lv}"] = int((sel & self.cleaned).sum())
        missed = dirty & ~self.cleaned & (self.fmax > 0)
        out["force_deficit"] = (float(np.mean((self.required[missed] - self.fmax[missed]) / self.required[missed]))
                                if missed.any() else 0.0)
        out["untouched_cells"] = int((dirty & (self.fmax == 0)).sum())
        return out


TASKS = {"plate": PlateTask, "nail": NailTask, "sweep": SweepTask}
