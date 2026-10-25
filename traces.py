"""Time-indexed reference traces of the tool head (position, tool orientation, velocity) and a follower for them.

A trace is sampled at the control rate (one sample per env step). Sample k is where the head should be after k
steps of the task episode. Velocities are analytic (from each segment's own profile), not finite differences, so an
impact keeps its full speed at the contact sample.

Trained policies don't see traces: they get the trace's keypoints one at a time as ordinary goals (see tasks.py).
TraceFollower is the reference for "follows the trace perfectly"; tasks calibrate their traces with it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

import numpy as np

import mathutils as mu
from goals import Goal

PHASES = ("approach", "hold", "accel", "strike", "contact", "retract", "lower", "stroke", "lift", "transfer", "windup",
          "swing")
PHASE = {name: i for i, name in enumerate(PHASES)}


def min_jerk(tau):
    """Position and d/dtau of the minimum-jerk profile on [0, 1]."""
    tau = np.asarray(tau, dtype=np.float64)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5, 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def smoothstep(tau):
    """Velocity ramp 3 tau^2 - 2 tau^3 and its integral over [0, tau]."""
    tau = np.asarray(tau, dtype=np.float64)
    return 3 * tau**2 - 2 * tau**3, tau**3 - 0.5 * tau**4


@dataclass
class Trace:
    dt: float
    p: np.ndarray  # (N, 3) head center, world
    quat: np.ndarray  # (N, 4) tool orientation, world (w x y z)
    v: np.ndarray  # (N, 3) head linear velocity
    w: np.ndarray  # (N, 3) tool angular velocity, world
    phase: np.ndarray  # (N,) index into PHASES
    force_target: np.ndarray  # (N,) N; nonzero where the tool should be pushing on something
    keypoints: List[Goal] = field(default_factory=list)  # what a trained policy is given, in order
    keypoint_idx: List[int] = field(default_factory=list)  # their sample indices
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.p)

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.p)) * self.dt

    @property
    def duration(self) -> float:
        return (len(self.p) - 1) * self.dt

    def add_keypoint(self, k: int) -> None:
        """Makes sample k a goal for the policy: pass through p[k] at v[k] in orientation quat[k]."""
        q = self.quat[k]
        self.keypoints.append(Goal(p=self.p[k].copy(), v=self.v[k].copy(), quat=q.copy(),
                                   axis=mu.rotate(q, self.meta["head_axis_tool"])))
        self.keypoint_idx.append(int(k))

    def save(self, path: str) -> None:
        meta = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.meta.items()}
        np.savez(path, dt=self.dt, t=self.t, p=self.p, quat=self.quat, v=self.v, w=self.w, phase=self.phase,
                 phase_names=np.array(PHASES), force_target=self.force_target,
                 keypoint_idx=np.array(self.keypoint_idx, dtype=np.int64), meta=json.dumps(meta, default=float))


class TraceBuilder:
    """Appends segments to a trace, starting from a pose at rest. Every segment starts where the last one ended;
    durations are rounded to whole control steps."""

    def __init__(self, dt: float, p0, q0, head_axis_tool):
        self.dt = dt
        self.p, self.q = [np.asarray(p0, np.float64)], [mu.normalize(q0)]
        self.v, self.w = [np.zeros(3)], [np.zeros(3)]
        self.ph, self.f = [PHASE["approach"]], [0.0]
        self.head_axis_tool = np.asarray(head_axis_tool, np.float64)

    @property
    def end(self):
        return self.p[-1].copy(), self.q[-1].copy()

    @property
    def index(self) -> int:
        """Index of the last sample (the current end of the trace)."""
        return len(self.p) - 1

    def _steps(self, T: float) -> int:
        return max(1, int(round(T / self.dt)))

    def _append(self, p, q, v, w, phase, force=0.0) -> None:
        n = len(p)
        self.p += list(p)
        self.q += list(q)
        self.v += list(v)
        self.w += list(w)
        self.ph += [PHASE[phase]] * n
        self.f += list(np.broadcast_to(np.asarray(force, np.float64), (n,)))

    def move_to(self, p1, q1, T: float, phase: str) -> None:
        """Minimum-jerk move (position along a line, orientation along the shortest arc), rest to rest."""
        p0, q0 = self.end
        p1, q1 = np.asarray(p1, np.float64), mu.normalize(q1)
        n = self._steps(T)
        T = n * self.dt
        tau = np.arange(1, n + 1) / n
        s, ds = min_jerk(tau)
        rv = mu.quat_error(q1, q0)
        ang = np.linalg.norm(rv)
        axis = rv / ang if ang > 1e-9 else np.array([1.0, 0, 0])
        ps = p0 + s[:, None] * (p1 - p0)
        qs = [mu.quat_mul(mu.axis_angle_quat(axis, si * ang), q0) for si in s]
        qs[-1] = q1
        self._append(ps, qs, (ds / T)[:, None] * (p1 - p0), (ds / T)[:, None] * rv, phase)

    def move_time(self, p1, q1, v_peak: float, w_peak: float, T_min: float) -> float:
        """Duration of a min-jerk move whose peak speeds stay within v_peak and w_peak."""
        p0, q0 = self.end
        dist = float(np.linalg.norm(np.asarray(p1) - p0))
        ang = float(np.linalg.norm(mu.quat_error(q1, q0)))
        return max(T_min, 1.875 * dist / v_peak, 1.875 * ang / w_peak)

    def hold(self, T: float, phase: str = "hold", force: float = 0.0) -> None:
        p0, q0 = self.end
        n = self._steps(T)
        self._append([p0] * n, [q0] * n, np.zeros((n, 3)), np.zeros((n, 3)), phase, force)

    def ramp(self, d, v0: float, v1: float, T: float, phase: str = "accel") -> None:
        """Speed along d goes smoothly from v0 to v1 (orientation held)."""
        p0, q0 = self.end
        d = mu.normalize(d)
        n = self._steps(T)
        T = n * self.dt
        tau = np.arange(1, n + 1) / n
        sv, sx = smoothstep(tau)
        x = v0 * T * tau + (v1 - v0) * T * sx
        speed = v0 + (v1 - v0) * sv
        self._append(p0 + x[:, None] * d, [q0] * n, speed[:, None] * d, np.zeros((n, 3)), phase)

    def const(self, d, speed: float, n: int, phase: str, force=0.0) -> None:
        """n steps at constant velocity speed * d (orientation held)."""
        p0, q0 = self.end
        d = mu.normalize(d)
        k = np.arange(1, n + 1)
        self._append(p0 + (k * self.dt * speed)[:, None] * d, [q0] * n, np.tile(speed * d, (n, 1)),
                     np.zeros((n, 3)), phase, force)

    def path(self, ps, vs, phase: str, force) -> None:
        """Explicit samples (orientation held); ps[0] follows the current end by one step."""
        _, q0 = self.end
        n = len(ps)
        self._append(ps, [q0] * n, vs, np.zeros((n, 3)), phase, force)

    def samples(self, ps, qs, vs, ws, phase: str, force=0.0) -> None:
        """Explicit samples of the full state; ps[0] follows the current end by one step."""
        self._append(list(ps), [mu.normalize(q) for q in qs], list(vs), list(ws), phase, force)

    def set_velocity(self, k: int, v) -> None:
        self.v[k] = np.asarray(v, np.float64)

    def build(self, **meta) -> Trace:
        meta.setdefault("head_axis_tool", self.head_axis_tool.tolist())
        return Trace(dt=self.dt, p=np.array(self.p), quat=np.array(self.q), v=np.array(self.v), w=np.array(self.w),
                     phase=np.array(self.ph, dtype=np.int64), force_target=np.array(self.f), meta=meta)


class TraceFollower:
    """Reference tracker for head_twist actions: the meaning of "follows the trace".

    At env step k it commands v = v_ff + k_pos (p[k] - head) and w = w_ff + k_rot * rot_err(quat[k], tool), with the
    feedforward averaged over the step (samples k and k+1). The env smooths actions with an EMA, so the follower
    inverts it: it sends the raw action whose smoothed value equals the command.
    """

    def __init__(self, k_pos: float = 8.0, k_rot: float = 6.0):
        self.k_pos, self.k_rot = k_pos, k_rot

    def __call__(self, env) -> np.ndarray:
        tr, cfg = env.trace, env.cfg
        k = min(env.task_t, len(tr) - 1)
        k1 = min(k + 1, len(tr) - 1)
        head = env._head_state()[0]
        v = 0.5 * (tr.v[k] + tr.v[k1]) + self.k_pos * (tr.p[k] - head)
        w = 0.5 * (tr.w[k] + tr.w[k1]) + self.k_rot * mu.quat_error(tr.quat[k], env._tool_quat())
        cmd = np.concatenate([v / cfg.head_v_max, w / cfg.head_w_max])
        ema = cfg.action_ema
        a = (cmd - ema * env._a_filt[:6]) / (1.0 - ema) if ema > 0 else cmd
        return np.clip(a, -1.0, 1.0).astype(np.float32)
