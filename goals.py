"""Goal sampling (position + velocity + head axis) and the pass-through test."""
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
    axis: np.ndarray  # target head-axis direction (unit)
    q_ref: Optional[np.ndarray] = None  # arm joints that put the head at p with this axis (weld goals only)

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

    def sample(self, rng: np.random.Generator, speed_max: float) -> Goal:
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
                return self._make(rng, p, d, speed_max)
        # Fallback (should be rare): a horizontal pass above the table center.
        p = ri.table_point(float(np.mean(c.radial)), 0.0, float(np.mean(c.height)))
        return self._make(rng, p, ri.table_perp.copy(), speed_max)

    def _make(self, rng, p, d, speed_max) -> Goal:
        c = self.cfg
        speed = float(rng.uniform(c.speed_min, max(speed_max, c.speed_min)))
        axis = self._cone(rng, d) if c.axis_mode == "cone" else d.copy()
        return Goal(p=np.asarray(p, dtype=np.float64), v=d * speed, axis=axis)

    def _cone(self, rng, center) -> np.ndarray:
        """Random unit vector within axis_cone_deg of `center`."""
        ang = float(rng.uniform(0.0, np.deg2rad(self.cfg.axis_cone_deg)))
        perp = mu.rotate(mu.axis_angle_quat(center, rng.uniform(-np.pi, np.pi)), mu.any_perpendicular(center))
        return mu.normalize(np.cos(ang) * center + np.sin(ang) * perp)

    def in_region(self, p) -> bool:
        """p is inside the goal region (table frame) and far enough from the base."""
        c, ri = self.cfg, self.ri
        u, w, z = float(p[:2] @ ri.table_dir[:2]), float(p[:2] @ ri.table_perp[:2]), float(p[2] - ri.table_top)
        return (c.radial[0] <= u <= c.radial[1] and c.lateral[0] <= w <= c.lateral[1]
                and c.height[0] <= z <= c.height[1] and np.linalg.norm(p[:2]) >= c.min_base_dist)

    def sample_from_grasp(self, rng: np.random.Generator, speed_max: float,
                          propose: Callable[[np.random.Generator], Optional[tuple]],
                          reachable: Callable[[np.ndarray, np.ndarray, np.ndarray], bool]) -> Optional[Goal]:
        """Goal that the arm can reach with the current grasp, by construction.

        propose(rng) -> (q, head_pos, head_axis, palm_pos) for a collision-free arm configuration q, or None.
        reachable(q0, point, axis) -> whether IK from q0 puts the head at point with that axis, collision-free.
        The goal is the head pose at q; the velocity runs along the head axis (or, in "cone" mode, it is
        tilted within axis_cone_deg of the axis). The lead-in and follow-through points along the velocity
        must clear the table and be reachable with the same axis.
        Returns None if no proposal is accepted within goal_q_tries.
        """
        c = self.cfg
        min_z = self.ri.table_top + c.min_clearance
        for _ in range(c.goal_q_tries):
            out = propose(rng)
            if out is None:
                continue
            q, p, axis, _ = out
            d = mu.normalize(axis)
            if not self.in_region(p):
                continue
            v_dir = self._cone(rng, d) if c.axis_mode == "cone" else d
            if v_dir[2] < c.min_dir_z:
                continue
            points = (p - c.lead_in * v_dir, p + c.follow_through * v_dir)
            if not all(x[2] > min_z and reachable(q, x, d) for x in points):
                continue
            speed = float(rng.uniform(c.speed_min, max(speed_max, c.speed_min)))
            return Goal(p=np.asarray(p, dtype=np.float64), v=v_dir * speed, axis=d.copy(),
                        q_ref=np.asarray(q, dtype=np.float64).copy())
        return None


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
