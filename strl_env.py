"""StrlEnv: the SimToolReal training task (pose-goal reaching) on the simplify-weld scene.

Same robot, scene, tool generator and palm weld as PoseVelocityEnv (grip_mode "weld", action_mode "joint"); the task,
reward, observations and domain randomization follow SimToolReal (see strl_config.py).

Episode
-------
* Every episode starts from home (+-home_noise) with a new random tool and grasp.
* The first goal is a random reachable tool pose; after each success the next goal is a random walk from the previous
  one (<= delta_goal_distance per axis, <= delta_rotation_deg), and the per-goal clock restarts.
* A goal is reached after success_steps steps with d <= tol * keypoint_scale, where d is the largest distance between
  the 4 reward keypoints of the tool and of the goal.
* terminated: max_consecutive_successes reached, or the robot/tool pushes on the table/pedestal/floor harder than
  table_force_limit. truncated: goal_steps without reaching the current goal, or physics divergence.

Goal poses are tool-body poses (origin at the handle center). Every goal is reachable with the current grasp: it comes
with an arm configuration (Goal.q_ref) that puts the tool there.
"""
from __future__ import annotations

import functools
import warnings
from typing import Optional

import mujoco
import numpy as np
from gymnasium import spaces

import mathutils as mu
from config import EnvCfg
from env import PoseVelocityEnv
from goals import Goal
from strl_config import StrlCfg, make_env_cfg

warnings.filterwarnings("ignore", message=".*Attach conflict.*")

OBJ = mujoco.mjtObj
KEYPOINT_CORNERS = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float64)
# The tool body frame has z along the handle towards the head. SimToolReal's object frame has x along the handle
# towards the head: object x = tool z, object y = tool x, object z = tool y (a proper rotation).
R_TOOL_OBJ = np.column_stack([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(-1))
    return q


Q_TOOL_OBJ = _mat_to_quat(R_TOOL_OBJ)


def canon_quat(q) -> np.ndarray:
    """q and -q are the same rotation; pick w >= 0 so the observation is continuous for small rotations."""
    q = np.asarray(q, dtype=np.float64)
    return -q if q[0] < 0 else q


def keypoints(pos, quat_obj, offsets) -> np.ndarray:
    """World positions (K, 3) of object-frame offsets (K, 3) at an object pose."""
    return np.asarray(pos) + np.asarray(offsets) @ mu.quat_to_mat(quat_obj).T


class StrlEnv(PoseVelocityEnv):
    def __init__(self, cfg: Optional[EnvCfg] = None, scfg: Optional[StrlCfg] = None,
                 render_mode: Optional[str] = None):
        self.scfg = scfg or StrlCfg()
        cfg = cfg or make_env_cfg(self.scfg)
        if cfg.grip_mode != "weld" or cfg.action_mode != "joint":
            raise ValueError("StrlEnv needs grip_mode 'weld' and action_mode 'joint'")
        self.tol = float(self.scfg.success_tolerance)
        self.reward_offsets = KEYPOINT_CORNERS * (0.5 * self.scfg.keypoint_scale
                                                  * np.asarray(self.scfg.reward_keypoint_size))
        self._reset_goal_trackers()
        self.successes = 0
        self.abs_goal_fallbacks = 0
        self._last_reward = 0.0
        self._table_force = 0.0
        self._clear_dr_state()
        super().__init__(cfg, render_mode)  # calls our _sample_goal / _get_obs / _get_priv
        self._wrench_probs()
        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.priv_dim = int(self._get_priv().shape[0])

    # ------------------------------------------------------------------ state helpers
    def _reset_goal_trackers(self) -> None:
        self.d_star: Optional[float] = None  # closest keypoint distance reached for the current goal
        self.near_steps = 0
        self.t_goal = 0

    def _clear_dr_state(self) -> None:
        self._obs_q: list = []
        self._act_q: list = []
        self._obj_q: list = []

    def _wrench_probs(self) -> None:
        rng, s = self.np_random, self.scfg
        lu = lambda lo_hi: float(np.exp(rng.uniform(np.log(lo_hi[0]), np.log(lo_hi[1]))))  # noqa: E731
        self._force_prob, self._torque_prob = lu(s.force_prob_range), lu(s.torque_prob_range)

    def set_tolerance(self, tol: float) -> None:
        self.tol = float(tol)

    def handle_dims(self) -> np.ndarray:
        """Grasp-region bounding box (the handle) in the object frame: length, width, height."""
        t = self.tool
        h = 2.0 * t.handle_rad * (t.handle_aspect if t.handle_type == "box" else 1.0)
        return np.array([t.handle_len, 2.0 * t.handle_rad, h])

    def obs_offsets(self) -> np.ndarray:
        return KEYPOINT_CORNERS * (0.5 * self.scfg.keypoint_scale * self.handle_dims())

    def tool_pose(self):
        """Clean tool pose as (position of the handle center, object-frame quaternion)."""
        d = self.data
        return d.xpos[self.tool_id].copy(), mu.quat_mul(d.xquat[self.tool_id], Q_TOOL_OBJ)

    def goal_obj_quat(self) -> np.ndarray:
        return mu.quat_mul(self.goal.quat, Q_TOOL_OBJ)

    def keypoint_distance(self) -> float:
        pos, q = self.tool_pose()
        kp = keypoints(pos, q, self.reward_offsets)
        kg = keypoints(self.goal.p, self.goal_obj_quat(), self.reward_offsets)
        return float(np.max(np.linalg.norm(kp - kg, axis=1)))

    def near_threshold(self) -> float:
        return self.tol * self.scfg.keypoint_scale

    # ------------------------------------------------------------------ goals
    def _make_goal(self, pos, quat, q) -> Goal:
        quat = mu.normalize(quat)
        axis = mu.quat_to_mat(quat) @ self.tool.head_axis
        return Goal(p=np.asarray(pos, dtype=np.float64).copy(), v=np.zeros(3), quat=quat, axis=axis,
                    q_ref=np.asarray(q, dtype=np.float64).copy())

    def _sample_abs_goal(self) -> Goal:
        """A random collision-free arm configuration near the current pose; the goal is the tool pose there."""
        rng = self.np_random
        for std in (0.5, 0.3, 0.15):
            self._goal_q_std = std
            for _ in range(200):
                out = self._propose_goal_q(rng)
                if out is None:
                    continue
                q, _, _, quat, _ = out
                pos = self._scratch.xpos[self.tool_id].copy()
                if self.goal_sampler.in_region(pos):
                    return self._make_goal(pos, quat, q)
        self.abs_goal_fallbacks += 1
        pos, _ = self.tool_pose()
        return self._make_goal(pos, self.data.xquat[self.tool_id], self.data.qpos[self.arm_qadr])

    def _sample_delta_goal(self) -> Goal:
        """Random walk from the current goal, kept only if IK from the goal's arm configuration reaches it."""
        rng, s, g = self.np_random, self.scfg, self.goal
        for _ in range(s.delta_goal_tries):
            pos = g.p + rng.uniform(-s.delta_goal_distance, s.delta_goal_distance, size=3)
            if not self.goal_sampler.in_region(pos):
                continue
            axis = mu.normalize(rng.normal(size=3))
            angle = rng.uniform(-1.0, 1.0) * np.deg2rad(s.delta_rotation_deg)
            quat = mu.quat_mul(mu.axis_angle_quat(axis, angle), g.quat)
            head = pos + mu.rotate(quat, self.tool.head_pos)
            q = self._ik_solve(g.q_ref, head, quat, iters=s.ik_iters)
            if q is not None:
                return self._make_goal(pos, quat, q)
        return self._sample_abs_goal()

    def _sample_goal(self) -> None:
        # Called by PoseVelocityEnv.__init__ / reset: the first goal of an episode.
        self.goal = self._sample_abs_goal()
        self._update_goal_markers()

    def set_goal_pose(self, pos, quat, q_ref=None) -> None:
        """Sets the goal (tool-body pose); for tests and task evaluation."""
        q = self.data.qpos[self.arm_qadr] if q_ref is None else q_ref
        self.goal = self._make_goal(pos, quat, q)
        self._reset_goal_trackers()
        self._update_goal_markers()

    def _update_goal_markers(self) -> None:
        m, d, g = self.model, self.data, self.goal
        head = g.p + mu.rotate(g.quat, self.tool.head_pos)
        d.mocap_pos[self.goal_mocap] = head
        d.mocap_pos[self.goal_axis_mocap] = head
        d.mocap_quat[self.goal_axis_mocap] = mu.quat_from_two_vectors([0, 0, 1], g.axis)
        d.mocap_pos[self.goal_handle_mocap] = head
        d.mocap_quat[self.goal_handle_mocap] = g.quat
        d.mocap_pos[self.goal_vel_mocap] = head  # no goal velocity: shrink the arrow away
        m.geom_size[self.goal_vel_gid][1] = 1e-4
        m.geom_pos[self.goal_vel_gid][2] = 1e-4
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ measurements
    def _env_contact_force(self) -> float:
        """Net normal force of contacts between the robot or tool and the table, pedestal or floor."""
        m, d = self.model, self.data
        f6, total = np.zeros(6), 0.0
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
            if g1 in self.env_geoms or g2 in self.env_geoms:
                other = int(m.geom_bodyid[g2 if g1 in self.env_geoms else g1])
                if other in self.robot_body_set or other == self.tool_id:
                    mujoco.mj_contactForce(m, d, i, f6)
                    total += abs(float(f6[0]))
        return total

    def _body_vel(self, body_id: int) -> np.ndarray:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, OBJ.mjOBJ_BODY, body_id, v, 0)
        return np.concatenate([v[3:], v[:3]])  # linear, angular

    # ------------------------------------------------------------------ gym API
    def _rehome_regrasp(self) -> dict:
        """Same tool, no recompile: arm home (+-home_noise), a new random grasp, a new first goal."""
        self._set_home(noise=True)
        self.prev_action[:] = 0.0
        self._a_filt[:] = 0.0
        self._weld_tool()
        self._update_contacts()
        self._init_episode_state(keep_grasp=True)
        self.start_type = "rehome"
        self._forced_reset_kind = None
        self._sample_goal()
        return {"reset_kind": "rehome"}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        options = dict(options or {})
        n = getattr(self, "_episodes", 0)
        new_tool = (seed is not None or options.get("full") or self.diverged
                    or n % max(1, self.scfg.new_tool_every) == 0)
        self._episodes = n + 1
        if new_tool:
            options["full"] = True  # home, new tool, new grasp
            _, info = super().reset(seed=seed, options=options)
        else:
            info = self._rehome_regrasp()
        self._reset_goal_trackers()
        self.successes = 0
        self.d_min_ep = np.inf
        self._last_reward = 0.0
        self._table_force = 0.0
        self.ep_return = 0.0
        self.ep_terms = {}
        self._clear_dr_state()
        self._wrench_probs()
        self.data.xfrc_applied[self.tool_id] = 0.0
        obs = self._get_obs()
        self._last_obs = obs
        info["priv"] = self._get_priv()
        return obs, info

    def step(self, action):
        s, d = self.scfg, self.data
        rng = self.np_random
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if s.dr and s.action_delay_max > 0:
            self._act_q.insert(0, a)
            del self._act_q[s.action_delay_max + 1:]
            a = self._act_q[int(rng.integers(len(self._act_q)))]
        self._apply_action(a)

        d.xfrc_applied[self.tool_id] = 0.0
        if s.dr:
            if rng.random() < self._force_prob:
                d.xfrc_applied[self.tool_id, :3] = rng.normal(size=3) * s.force_scale
            if rng.random() < self._torque_prob:
                d.xfrc_applied[self.tool_id, 3:] = rng.normal(size=3) * s.torque_scale

        bad_before = int(d.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
        self._physics_step()
        self.t += 1
        self.t_goal += 1
        if not self._physics_ok(bad_before):
            self.diverged = True
            info = {"priv": np.zeros(self.priv_dim, np.float32), "success": False,
                    "episode": self._episode_info("diverged")}
            return self._last_obs, 0.0, False, True, info

        self._table_force = self._env_contact_force()
        dist = self.keypoint_distance()
        if self.d_star is None:
            self.d_star = dist
        kp_rew = s.keypoint_rew_scale * max(self.d_star - dist, 0.0)
        self.d_star = min(self.d_star, dist)
        near = dist <= self.near_threshold()
        self.near_steps = (self.near_steps + int(near)) * (int(near) if s.force_consecutive else 1)
        success = self.near_steps >= s.success_steps
        if s.force_consecutive:
            bonus = s.reach_goal_bonus if success else 0.0
        else:
            bonus = s.reach_goal_bonus / s.success_steps if near else 0.0
        smooth = -s.arm_vel_penalty * float(np.abs(d.qvel[self.arm_dadr]).sum())
        terms = {"keypoint": kp_rew, "bonus": bonus, "smooth": smooth}
        reward = float(sum(terms.values()))
        self._last_reward = reward
        self.ep_return += reward
        for k, v in terms.items():
            self.ep_terms[k] = self.ep_terms.get(k, 0.0) + float(v)

        self.d_last = dist
        self.d_min_ep = min(getattr(self, "d_min_ep", np.inf), dist)
        if success:
            self.successes += 1
            self.goal = self._sample_delta_goal()
            self._update_goal_markers()
            self._reset_goal_trackers()

        reason = None
        if self.successes >= s.max_consecutive_successes:
            reason = "max_successes"
        elif self._table_force > s.table_force_limit:
            reason = "table_force"
        terminated = reason is not None
        truncated = (not terminated) and self.t_goal >= s.goal_steps
        if truncated:
            reason = "timeout"

        self.prev_action = a
        obs = self._get_obs()
        self._last_obs = obs
        info = {"priv": self._get_priv(), "success": success, "near": near, "dist": dist}
        if terminated or truncated:
            info["episode"] = self._episode_info(reason)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _episode_info(self, reason: str) -> dict:
        return {"return": self.ep_return, "length": self.t, "successes": float(self.successes),
                "reason": reason, "tol": self.tol, "d_last": float(getattr(self, "d_last", np.nan)),
                "d_min": float(getattr(self, "d_min_ep", np.nan)),
                "terms": dict(self.ep_terms), "abs_goal_fallbacks": float(self.abs_goal_fallbacks)}

    # ------------------------------------------------------------------ observations
    def _get_obs(self) -> np.ndarray:
        s, d = self.scfg, self.data
        rng = self.np_random
        dr = s.dr
        palm = d.xpos[self.palm_id].copy()
        qvel = d.qvel[self.arm_dadr].copy()
        if dr and s.joint_vel_noise > 0:
            qvel = qvel + rng.normal(0.0, s.joint_vel_noise, size=qvel.shape)
        pos, q = self.tool_pose()
        if dr:
            self._obj_q.insert(0, (pos, q))
            del self._obj_q[s.object_delay_max + 1:]
            pos, q = self._obj_q[int(rng.integers(len(self._obj_q)))]
            pos = pos + rng.normal(0.0, s.object_pos_noise, size=3)
            ang = np.deg2rad(rng.uniform(0.0, s.object_rot_noise_deg))
            q = mu.quat_mul(mu.axis_angle_quat(mu.normalize(rng.normal(size=3)), ang), q)
        offs = self.obs_offsets()
        kp = keypoints(pos, q, offs)
        kg = keypoints(self.goal.p, self.goal_obj_quat(), offs)
        obs = np.concatenate([
            d.qpos[self.arm_qadr], qvel, self.targets[:self.n_arm_act],
            palm, canon_quat(d.xquat[self.palm_id]),
            (d.xpos[self.tip_ids] - palm).ravel(),
            canon_quat(q), (kp - palm).ravel(), (kp - kg).ravel(),
            self.handle_dims(),
        ]).astype(np.float32)
        if dr and s.obs_delay_max > 0:
            self._obs_q.insert(0, obs)
            del self._obs_q[s.obs_delay_max + 1:]
            obs = self._obs_q[int(rng.integers(len(self._obs_q)))]
        return obs

    def _get_priv(self) -> np.ndarray:
        m, d, s = self.model, self.data, self.scfg
        pos, q = self.tool_pose()
        offs = self.obs_offsets()
        kp = keypoints(pos, q, offs)
        kg = keypoints(self.goal.p, self.goal_obj_quat(), offs)
        palm = d.xpos[self.palm_id]
        d_star = self.d_star if self.d_star is not None else self.keypoint_distance()
        priv = np.concatenate([
            self._body_vel(self.palm_id), self._body_vel(self.tool_id),
            (kp - palm).ravel(), (kp - kg).ravel(), canon_quat(q),
            [min(d_star, 1.0), self.near_steps / s.success_steps, self.successes / s.max_consecutive_successes,
             self.t_goal / s.goal_steps, np.clip(self._last_reward / 100.0, -10.0, 10.0)],
            [m.body_mass[self.tool_id]], m.body_ipos[self.tool_id], [self.tool.friction],
            [min(self._table_force, 500.0) / 100.0],
        ])
        return priv.astype(np.float32)


def _build_strl_env(cfg: EnvCfg, scfg: StrlCfg, seed: int) -> StrlEnv:
    env = StrlEnv(cfg, scfg)
    env.reset(seed=seed)
    return env


def make_strl_env(cfg: EnvCfg, scfg: StrlCfg, seed: int):
    """Picklable env factory for SubprocVecEnv."""
    return functools.partial(_build_strl_env, cfg, scfg, seed)
