"""Procedural tools: a primitive handle plus a primitive head.

Tool frame: origin at the handle center, handle along +z. The head sits at the +z end in one
of three layouts:
  T      head long axis perpendicular to the handle, centered on it (hammer)
  L      like T but offset to one side (adze / L-key)
  inline head long axis along the handle (screwdriver / mallet-on-a-stick)
The head axis is the head's long direction (T/L) or the handle direction (inline); that is
the direction that must match the goal axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

import mathutils as mu
from config import ToolCfg

HANDLE_TYPES = ("cylinder", "capsule", "box")
HEAD_TYPES = ("box", "cylinder", "sphere", "ellipsoid", "capsule")
LAYOUTS = ("T", "L", "inline")
FEATURE_DIM = 25

_GEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}
_Y90 = mu.axis_angle_quat([0, 1, 0], np.pi / 2)  # local z -> x
_YM90 = mu.axis_angle_quat([0, 1, 0], -np.pi / 2)  # local x -> z


@dataclass
class ToolParams:
    handle_type: str
    handle_len: float
    handle_rad: float
    handle_aspect: float
    head_type: str
    layout: str
    head_long: float
    head_short: float
    head_width: float
    head_roll: float
    density: float
    friction: float
    handle_size: np.ndarray
    head_size: np.ndarray
    head_pos: np.ndarray  # head center in tool frame
    head_quat: np.ndarray
    head_axis: np.ndarray  # unit vector in tool frame
    mass: float
    rest_half_height: float  # height of tool origin above the table when lying flat
    pad: Optional["PadParams"] = None  # brushes only: compliant bristle pad (tasks.py)

    def features(self) -> np.ndarray:
        f = np.concatenate([
            _onehot(self.handle_type, HANDLE_TYPES),
            [self.handle_len, self.handle_rad, self.handle_aspect if self.handle_type == "box" else 1.0],
            _onehot(self.head_type, HEAD_TYPES),
            _onehot(self.layout, LAYOUTS),
            [self.head_long, self.head_short, self.head_width],
            [np.cos(self.head_roll), np.sin(self.head_roll)],
            self.head_pos,
            self.head_axis,
        ])
        return f.astype(np.float32)


def _onehot(value, options) -> np.ndarray:
    v = np.zeros(len(options))
    v[options.index(value)] = 1.0
    return v


def _handle_geom(kind: str, length: float, rad: float, aspect: float):
    """Returns (size, volume, vertical half-extent when lying flat)."""
    if kind == "cylinder":
        return np.array([rad, length / 2, 0.0]), np.pi * rad**2 * length, rad
    if kind == "capsule":
        h = max(length / 2 - rad, 0.01)
        return np.array([rad, h, 0.0]), np.pi * rad**2 * 2 * h + 4 / 3 * np.pi * rad**3, rad
    size = np.array([rad, rad * aspect, length / 2])
    return size, 8 * size[0] * size[1] * size[2], max(size[0], size[1])


def _head_geom(kind: str, a: float, b: float, c: float):
    """Head in its own frame: long axis x (half a), short axis z (half b), width y (half c).

    Returns (size, quat aligning the primitive to that frame, volume, long half, short half, width half).
    """
    if kind == "box":
        return np.array([a, c, b]), np.array([1.0, 0, 0, 0]), 8 * a * b * c, a, b, c
    if kind == "ellipsoid":
        return np.array([a, c, b]), np.array([1.0, 0, 0, 0]), 4 / 3 * np.pi * a * b * c, a, b, c
    if kind == "cylinder":
        return np.array([b, a, 0.0]), _Y90.copy(), np.pi * b**2 * 2 * a, a, b, b
    if kind == "capsule":
        h = max(a - b, 0.005)
        return np.array([b, h, 0.0]), _Y90.copy(), np.pi * b**2 * 2 * h + 4 / 3 * np.pi * b**3, h + b, b, b
    r = 1.3 * b  # sphere
    return np.array([r, 0.0, 0.0]), np.array([1.0, 0, 0, 0]), 4 / 3 * np.pi * r**3, r, r, r


@dataclass
class PadParams:
    """Compliant bristle pad under a brush head: a box on a spring-loaded slide joint along the tool's +z."""
    half_size: np.ndarray  # box half extents in the pad frame (x along the head's long axis, z along the tool's +z)
    pos: np.ndarray  # pad body origin in the tool frame (the head face farthest from the handle)
    stiffness: float  # N/m
    damping: float  # N s/m
    travel: float  # m of compression before the joint limit
    friction: float


def sample_tool(rng: np.random.Generator, cfg: ToolCfg) -> ToolParams:
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    handle_type = HANDLE_TYPES[rng.integers(len(HANDLE_TYPES))]
    head_type = HEAD_TYPES[rng.integers(len(HEAD_TYPES))]
    layout = LAYOUTS[rng.integers(len(LAYOUTS))]
    length, rad, aspect = u(cfg.handle_len), u(cfg.handle_rad), u(cfg.handle_aspect)
    a, b, c = u(cfg.head_long), u(cfg.head_short), u(cfg.head_width)
    roll = float(rng.uniform(-np.pi, np.pi))
    jx, jy = rng.uniform(-cfg.head_jitter, cfg.head_jitter, size=2)
    return build_tool(handle_type, length, rad, aspect, head_type, layout, a, b, c, roll, (jx, jy),
                      density=u(cfg.density), friction=u(cfg.friction), mass_range=cfg.mass,
                      head_jitter=cfg.head_jitter)


def build_tool(handle_type: str, length: float, rad: float, aspect: float, head_type: str, layout: str,
               a: float, b: float, c: float, roll: float = 0.0, jitter=(0.0, 0.0), density: float = 800.0,
               friction: float = 1.0, mass_range=(0.1, 0.6), head_jitter: float = 0.01) -> ToolParams:
    """A tool with the given shape (sample_tool draws these at random; tasks use fixed presets).

    a, b, c: the head's long, short and width half-extents. The mass is density * volume, clipped to mass_range.
    """
    jx, jy = jitter
    handle_size, handle_vol, handle_vert = _handle_geom(handle_type, length, rad, aspect)
    head_size, q_type, head_vol, long_h, short_h, width_h = _head_geom(head_type, a, b, c)

    roll_q = mu.yaw_quat(roll)
    top = length / 2
    if layout == "T":
        pos0 = np.array([jx, jy, top + 0.9 * short_h])
        layout_q = roll_q
        axis = mu.rotate(roll_q, [1.0, 0, 0])
    elif layout == "L":
        pos0 = np.array([0.9 * long_h + jx, jy, top + 0.9 * short_h])
        layout_q = roll_q
        axis = mu.rotate(roll_q, [1.0, 0, 0])
    else:
        pos0 = np.array([jx, jy, top + 0.9 * long_h])
        layout_q = mu.quat_mul(roll_q, _YM90)
        axis = np.array([0.0, 0, 1.0])
    head_pos = mu.rotate(roll_q, pos0)
    head_quat = mu.quat_mul(layout_q, q_type)

    volume = handle_vol + head_vol
    mass = float(np.clip(density * volume, *mass_range))
    density = mass / volume
    # When spawned flat (see env._spawn_on_table) the head's width axis is vertical.
    head_vert = width_h if head_type in ("box", "ellipsoid") else short_h
    rest = max(handle_vert, head_vert) + head_jitter + 0.002

    return ToolParams(
        handle_type=handle_type, handle_len=length, handle_rad=rad, handle_aspect=aspect,
        head_type=head_type, layout=layout, head_long=a, head_short=b, head_width=c, head_roll=roll,
        density=density, friction=friction,
        handle_size=handle_size, head_size=head_size, head_pos=head_pos, head_quat=head_quat,
        head_axis=mu.normalize(axis), mass=mass, rest_half_height=rest,
    )


def add_tool(spec: mujoco.MjSpec, p: ToolParams, park_pos=(0.0, 0.0, 3.0)) -> None:
    """Adds the tool as a free body named "tool" to the spec's worldbody."""
    body = spec.worldbody.add_body(name="tool", pos=list(park_pos))
    try:
        joint = body.add_freejoint()
    except AttributeError:
        joint = body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    joint.name = "tool_free"
    common = dict(density=p.density, friction=[p.friction, 0.01, 0.001], condim=4)
    body.add_geom(name="tool_handle", type=_GEOM[p.handle_type], size=list(p.handle_size),
                  rgba=[0.55, 0.35, 0.2, 1.0], **common)
    body.add_geom(name="tool_head", type=_GEOM[p.head_type], size=list(p.head_size),
                  pos=list(p.head_pos), quat=list(p.head_quat), rgba=[0.6, 0.62, 0.68, 1.0], **common)
    body.add_site(name="tool_head_center", pos=list(p.head_pos), size=[0.008, 0, 0], rgba=[1, 0.2, 0.2, 1])
    body.add_site(name="tool_handle_center", pos=[0, 0, 0], size=[0.006, 0, 0], rgba=[0.2, 0.2, 1, 1])
    if p.pad is not None:
        pd = p.pad
        pad = body.add_body(name="tool_pad", pos=list(pd.pos), gravcomp=1.0)
        pad.add_joint(name="tool_pad_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                      stiffness=[pd.stiffness, 0.0, 0.0], damping=pd.damping, range=[-pd.travel, 0.0],
                      limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        pad.add_geom(name="tool_pad", type=_GEOM["box"], size=list(pd.half_size), pos=[0, 0, pd.half_size[2]],
                     mass=0.02, friction=[pd.friction, 0.01, 0.001], condim=3, rgba=[0.85, 0.75, 0.4, 1.0])
