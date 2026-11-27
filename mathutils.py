"""Small rotation helpers. Quaternions use MuJoCo's (w, x, y, z) convention."""
from __future__ import annotations

import mujoco
import numpy as np


def normalize(v, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def quat_mul(a, b) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return out


def axis_angle_quat(axis, angle: float) -> np.ndarray:
    axis = normalize(axis)
    h = 0.5 * angle
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


def yaw_quat(yaw: float) -> np.ndarray:
    return axis_angle_quat([0.0, 0.0, 1.0], yaw)


def euler_xyz_quat(euler) -> np.ndarray:
    ex, ey, ez = euler
    q = axis_angle_quat([1.0, 0.0, 0.0], ex)
    q = quat_mul(q, axis_angle_quat([0.0, 1.0, 0.0], ey))
    return quat_mul(q, axis_angle_quat([0.0, 0.0, 1.0], ez))


def quat_to_mat(q) -> np.ndarray:
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def rotate(q, v) -> np.ndarray:
    return quat_to_mat(q) @ np.asarray(v, dtype=np.float64)


def any_perpendicular(v) -> np.ndarray:
    v = normalize(v)
    helper = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return normalize(np.cross(v, helper))


def quat_from_two_vectors(a, b) -> np.ndarray:
    """Shortest rotation taking direction a onto direction b."""
    a, b = normalize(a), normalize(b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-8:
        return axis_angle_quat(any_perpendicular(a), np.pi)
    q = np.concatenate([[1.0 + c], np.cross(a, b)])
    return q / np.linalg.norm(q)


def angle_between(a, b) -> float:
    return float(np.arccos(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0)))


def rot6d(mat) -> np.ndarray:
    """Continuous 6D rotation representation: first two columns of the rotation matrix."""
    return np.asarray(mat).reshape(3, 3)[:, :2].T.reshape(-1)


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_error(q_target, q) -> np.ndarray:
    """World-frame rotation vector (axis * angle, angle in [0, pi]) that rotates orientation q onto q_target."""
    e = quat_mul(q_target, quat_conj(q))
    if e[0] < 0.0:  # q and -q are the same rotation; take the short way round
        e = -e
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, e, 1.0)
    return out
