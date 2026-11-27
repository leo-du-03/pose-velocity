"""Goal sampling (position + velocity + tool orientation) and the pass-through test."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mujoco
import numpy as np

import mathutils as mu
from config import GoalCfg

_REACH_CACHE: dict = {}


@dataclass
class Goal:
    p: np.ndarray  # target head-center position (world)
    v: np.ndarray  # target head velocity (world)
    quat: np.ndarray  # target tool-body orientation (world, w x y z): the full orientation, roll included
    axis: np.ndarray  # head axis at that orientation (world, unit); derived from quat, for markers and diagnostics
    q_ref: Optional[np.ndarray] = None  # arm joints that put the tool at this pose (weld goals only)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))

    @property
    def direction(self) -> np.ndarray:
        return mu.normalize(self.v)


class ReachMap:
    """Voxel set of palm positions reachable by the arm (from random joint samples + FK)."""

    def __init__(self, model, arm_qadr, arm_range, palm_id, min_z, n_samples, voxel, seed=0):
        data = mujoco.MjData(model)
        rng = np.random.default_rng(seed)
        pts = np.empty((n_samples, 3))
        lo, hi = arm_range[:, 0], arm_range[:, 1]
        for i in range(n_samples):
            data.qpos[arm_qadr] = rng.uniform(lo, hi)
            mujoco.mj_kinematics(model, data)
            pts[i] = data.xpos[palm_id]
        pts = pts[pts[:, 2] > min_z]
        self.voxel = voxel
        self.keys = set(map(tuple, np.floor(pts / voxel).astype(np.int64)))

    def contains(self, p) -> bool:
        return tuple(np.floor(np.asarray(p) / self.voxel).astype(np.int64)) in self.keys

    @classmethod
    def cached(cls, model, arm_qadr, arm_range, palm_id, min_z, n_samples, voxel):
        key = (n_samples, voxel, round(min_z, 4))
        if key not in _REACH_CACHE:
            _REACH_CACHE[key] = cls(model, arm_qadr, arm_range, palm_id, min_z, n_samples, voxel)
        return _REACH_CACHE[key]


class GoalSampler:
    def __init__(self, cfg: GoalCfg, robot_info, reach: ReachMap):
        self.cfg = cfg
        self.ri = robot_info
        self.reach = reach

    def sample(self, rng: np.random.Generator, speed_max: float, vel_cone_deg: float,
               head_axis_tool: np.ndarray) -> Goal:
        """Grasp-agnostic goal from the reach map (physical grip, and the weld last resort). The velocity direction
        is sampled first; the head axis is then within vel_cone_deg of it, with a random roll about the axis.
        head_axis_tool: the tool's head axis in its own frame."""
        c, ri = self.cfg, self.ri
        min_z = ri.table_top + c.min_clearance
        for _ in range(500):
            p = ri.table_point(rng.uniform(*c.radial), rng.uniform(*c.lateral), rng.uniform(*c.height))
            if np.linalg.norm(p[:2]) < c.min_base_dist:
                continue
            d = mu.normalize(rng.normal(size=3))
            if d[2] < c.min_dir_z:
                continue
            pts = (p, p - c.lead_in * d, p + c.follow_through * d)
            if all(q[2] > min_z and self.reach.contains(q) for q in pts):
                return self._make(rng, p, d, speed_max, vel_cone_deg, head_axis_tool)
        # Fallback (should be rare): a horizontal pass above the table center.
        p = ri.table_point(float(np.mean(c.radial)), 0.0, float(np.mean(c.height)))
        return self._make(rng, p, ri.table_perp.copy(), speed_max, vel_cone_deg, head_axis_tool)

    def _make(self, rng, p, d, speed_max, vel_cone_deg, head_axis_tool) -> Goal:
        c = self.cfg
        speed = float(rng.uniform(c.speed_min, max(speed_max, c.speed_min)))
        axis = cone_sample(rng, d, vel_cone_deg)
        quat = mu.quat_mul(mu.axis_angle_quat(axis, rng.uniform(-np.pi, np.pi)),
                           mu.quat_from_two_vectors(head_axis_tool, axis))
        return Goal(p=np.asarray(p, dtype=np.float64), v=d * speed, quat=quat, axis=axis)

    def in_region(self, p) -> bool:
        """p is inside the goal region (table frame) and far enough from the base."""
        c, ri = self.cfg, self.ri
        u, w, z = float(p[:2] @ ri.table_dir[:2]), float(p[:2] @ ri.table_perp[:2]), float(p[2] - ri.table_top)
        return (c.radial[0] <= u <= c.radial[1] and c.lateral[0] <= w <= c.lateral[1]
                and c.height[0] <= z <= c.height[1] and np.linalg.norm(p[:2]) >= c.min_base_dist)

    def sample_from_grasp(self, rng: np.random.Generator, speed_max: float, vel_cone_deg: float,
                          propose: Callable[[np.random.Generator], Optional[tuple]],
                          reachable: Callable[[np.ndarray, np.ndarray, np.ndarray], bool]) -> Optional[Goal]:
        """Goal that the arm can reach with the current grasp, by construction.

        propose(rng) -> (q, head_pos, head_axis, tool_quat, palm_pos) for a collision-free arm configuration q, or
        None.
        reachable(q0, point, quat) -> whether IK from q0 puts the head at point with the tool at orientation quat,
        collision-free.
        The goal is the tool pose at q (head position and full orientation). The velocity direction is uniform
        within vel_cone_deg of the head axis: near it the goal is a strike out of the head's face, at 180 it can be
        any direction (a swipe). The lead-in and follow-through points along the velocity must clear the table and
        be reachable with the same orientation.
        A fraction down_frac of goals is a steep downward strike (v_dir z <= down_max_z, e.g. a vertical nail); they
        are rare otherwise. If none is found, the goal is drawn without that constraint.
        Returns None if no proposal is accepted within goal_q_tries.
        """
        c = self.cfg
        if c.down_frac > 0.0 and rng.random() < c.down_frac:
            goal = self._sample_from_grasp(rng, speed_max, vel_cone_deg, propose, reachable, c.down_max_z)
            if goal is not None:
                return goal
        return self._sample_from_grasp(rng, speed_max, vel_cone_deg, propose, reachable, None)

    def _sample_from_grasp(self, rng, speed_max, vel_cone_deg, propose, reachable, max_dir_z) -> Optional[Goal]:
        c = self.cfg
        min_z = self.ri.table_top + c.min_clearance
        for _ in range(c.goal_q_tries):
            out = propose(rng)
            if out is None:
                continue
            q, p, axis, quat, _ = out
            d = mu.normalize(axis)
            if not self.in_region(p):
                continue
            v_dir = cone_sample(rng, d, vel_cone_deg)
            if v_dir[2] < c.min_dir_z or (max_dir_z is not None and v_dir[2] > max_dir_z):
                continue
            points = (p - c.lead_in * v_dir, p + c.follow_through * v_dir)
            if not all(x[2] > min_z and reachable(q, x, quat) for x in points):
                continue
            speed = float(rng.uniform(c.speed_min, max(speed_max, c.speed_min)))
            return Goal(p=np.asarray(p, dtype=np.float64), v=v_dir * speed, quat=np.asarray(quat, dtype=np.float64),
                        axis=d.copy(), q_ref=np.asarray(q, dtype=np.float64).copy())
        return None


def cone_sample(rng: np.random.Generator, center, max_deg: float) -> np.ndarray:
    """Unit vector uniform (by area) within max_deg of `center`; uniform over the sphere at 180."""
    cos_a = float(rng.uniform(np.cos(np.deg2rad(min(max_deg, 180.0))), 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
    center = mu.normalize(center)
    perp = mu.rotate(mu.axis_angle_quat(center, rng.uniform(-np.pi, np.pi)), mu.any_perpendicular(center))
    return mu.normalize(cos_a * center + sin_a * perp)


def plane_crossing(p0, p1, p_g, d_hat) -> Optional[float]:
    """Distance from p_g at which the segment p0 -> p1 crosses the plane through p_g with normal d_hat,
    going forward (from behind the plane to on or in front of it); None if it doesn't cross that way."""
    p0, p1, p_g, d_hat = (np.asarray(x, dtype=np.float64) for x in (p0, p1, p_g, d_hat))
    s0, s1 = float((p0 - p_g) @ d_hat), float((p1 - p_g) @ d_hat)
    if not (s0 < 0.0 <= s1):
        return None
    t = s0 / (s0 - s1)
    return float(np.linalg.norm(p0 + t * (p1 - p0) - p_g))


def segment_point_distance(p0, p1, q) -> float:
    """Distance from point q to segment p0-p1 (closest approach of the head during one control step)."""
    p0, p1, q = (np.asarray(x, dtype=np.float64) for x in (p0, p1, q))
    seg = p1 - p0
    l2 = float(seg @ seg)
    t = 0.0 if l2 < 1e-12 else float(np.clip((q - p0) @ seg / l2, 0.0, 1.0))
    return float(np.linalg.norm(p0 + t * seg - q))
