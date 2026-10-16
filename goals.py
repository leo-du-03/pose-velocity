"""Goal sampling (position + velocity + head axis) and the pass-through test."""
from __future__ import annotations

from dataclasses import dataclass

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
        if c.axis_mode == "cone":
            ang = float(rng.uniform(0.0, np.deg2rad(c.axis_cone_deg)))
            perp = mu.rotate(mu.axis_angle_quat(d, rng.uniform(-np.pi, np.pi)), mu.any_perpendicular(d))
            axis = mu.normalize(np.cos(ang) * d + np.sin(ang) * perp)
        else:
            axis = d.copy()
        return Goal(p=np.asarray(p, dtype=np.float64), v=d * speed, axis=axis)


def segment_point_distance(p0, p1, q) -> float:
    """Distance from point q to segment p0-p1 (closest approach of the head during one control step)."""
    p0, p1, q = (np.asarray(x, dtype=np.float64) for x in (p0, p1, q))
    seg = p1 - p0
    l2 = float(seg @ seg)
    t = 0.0 if l2 < 1e-12 else float(np.clip((q - p0) @ seg / l2, 0.0, 1.0))
    return float(np.linalg.norm(p0 + t * seg - q))
