"""Scene construction: UR5e arm + LEAP hand (MuJoCo Menagerie), table, goal markers, tool.

The robot-only spec is built once (`SceneBuilder`), then copied and extended with a fresh
procedural tool whenever a new tool is needed.

Run `python scene.py --view` to print scene diagnostics and open the interactive viewer.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import List

import mujoco
import numpy as np

import mathutils as mu
from config import SceneCfg
from tool_gen import ToolParams, add_tool

HAND_PREFIX = "hand/"
_GEOM = mujoco.mjtGeom
# Plain ints: numpy ints from MjModel arrays don't compare equal to mjtJoint members inside `in`.
_ACTUATED_JOINT_TYPES = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
_TRN_JOINT = int(mujoco.mjtTrn.mjTRN_JOINT)


@dataclass
class RobotInfo:
    arm_joints: List[str]  # actuated arm joints, kinematic order
    hand_joints: List[str]  # actuated hand joints
    palm_body: str
    fingertip_bodies: List[str]  # thumb last
    finger_bodies: List[List[str]]  # per finger: all links from fingertip back to the palm (contacts count for that finger)
    robot_bodies: List[str]
    palm_home: np.ndarray
    table_dir: np.ndarray  # unit vector (3,), horizontal, from base toward the table
    table_perp: np.ndarray
    table_center: np.ndarray
    table_top: float
    table_half_size: np.ndarray
    floor_z: float

    def table_point(self, u: float, w: float, z_above_table: float) -> np.ndarray:
        return self.table_dir * u + self.table_perp * w + np.array([0.0, 0.0, self.table_top + z_above_table])


def resolve_model_paths(cfg: SceneCfg):
    ur5e = cfg.ur5e_xml or os.environ.get("POSEVEL_UR5E_XML")
    leap = cfg.leap_xml or os.environ.get("POSEVEL_LEAP_XML")
    if ur5e is None or leap is None:
        try:
            from robot_descriptions import leap_hand_mj_description, ur5e_mj_description
        except ImportError as e:
            raise ImportError(
                "Could not import robot_descriptions. `pip install robot_descriptions`, or set "
                "POSEVEL_UR5E_XML and POSEVEL_LEAP_XML to the Menagerie ur5e.xml / right_hand.xml paths."
            ) from e
        ur5e = ur5e or ur5e_mj_description.MJCF_PATH
        leap = leap or leap_hand_mj_description.MJCF_PATH
    return ur5e, leap


def _spec_list(spec, attr: str) -> list:
    try:
        return list(getattr(spec, attr))
    except Exception:
        return []


def _find_body(spec: mujoco.MjSpec, name: str):
    for getter in ("body", "find_body"):
        fn = getattr(spec, getter, None)
        if fn is None:
            continue
        try:
            body = fn(name)
        except Exception:
            body = None
        if body is not None:
            return body
    raise KeyError(f"body '{name}' not found in spec")


def _all_bodies(spec: mujoco.MjSpec) -> list:
    bodies = _spec_list(spec, "bodies")
    if not bodies:
        try:
            bodies = list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY))
        except Exception:
            bodies = []
    return bodies


def _load_spec(path: str) -> mujoco.MjSpec:
    """Loads an MJCF and makes asset paths absolute so it can be attached into another spec."""
    spec = mujoco.MjSpec.from_file(path)
    base = os.path.dirname(os.path.abspath(path))
    meshdir = getattr(spec, "meshdir", "") or ""
    texdir = getattr(spec, "texturedir", "") or ""
    for m in _spec_list(spec, "meshes"):
        if m.file and not os.path.isabs(m.file):
            m.file = os.path.join(base, meshdir, m.file)
    for t in _spec_list(spec, "textures"):
        if t.file and not os.path.isabs(t.file):
            t.file = os.path.join(base, texdir, t.file)
    # Keyframes would have the wrong qpos size once the specs are combined.
    for k in _spec_list(spec, "keys"):
        spec.delete(k)
    return spec


def _order_fingertips(m: mujoco.MjModel, d: mujoco.MjData, body_ids: List[int], palm: int, prefix: str = ""):
    """Leaf bodies of the hand tree (fingertips), non-thumb sorted by name, thumb last.

    `d` must have up-to-date kinematics (used by the thumb fallback).
    """
    n_children = {i: 0 for i in body_ids}
    for i in body_ids:
        parent = int(m.body_parentid[i])
        if parent in n_children:
            n_children[parent] += 1
    leaves = [i for i in body_ids if n_children[i] == 0 and i != palm]
    thumbs = [i for i in leaves if "th" in m.body(i).name[len(prefix):].lower()]
    if len(thumbs) != 1:  # fall back to the fingertip farthest from the others
        tips = d.xpos[leaves]
        dist = [np.linalg.norm(tips[k] - np.delete(tips, k, 0).mean(0)) for k in range(len(leaves))]
        thumbs = [leaves[int(np.argmax(dist))]]
    return sorted((i for i in leaves if i != thumbs[0]), key=lambda i: m.body(i).name) + thumbs


def _auto_hand_mount(leap_path: str):
    """Rotation + offset that put the hand's fingers along the flange's +z, flush against the flange.

    Computed from the hand model alone with its joints at qpos0 (fingers straight):
      rotation  maps the palm -> mean non-thumb fingertip direction onto +z
      offset    moves the rearmost hand geometry to z = 0 and centers the palm on the z axis
    Returns (quat, pos), expressed in the mount frame.
    """
    m = _load_spec(leap_path).compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    bodies = list(range(1, m.nbody))
    palm = bodies[0]
    tips = _order_fingertips(m, d, bodies, palm)
    finger_dir = d.xpos[tips[:-1]].mean(0) - d.xpos[palm]
    q = mu.quat_from_two_vectors(finger_dir, [0.0, 0.0, 1.0])
    rot = mu.quat_to_mat(q)

    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    all_pts, palm_pts = [], []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == 0:
            continue
        center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = d.geom_xpos[g] + (center + signs * half) @ d.geom_xmat[g].reshape(3, 3).T
        corners = corners @ rot.T
        all_pts.append(corners)
        if b == palm:
            palm_pts.append(corners)
    all_pts = np.concatenate(all_pts)
    palm_pts = np.concatenate(palm_pts) if palm_pts else all_pts
    xy_center = 0.5 * (palm_pts[:, :2].min(0) + palm_pts[:, :2].max(0))
    pos = np.array([-xy_center[0], -xy_center[1], -all_pts[:, 2].min()])
    return q, pos


def build_base_spec(cfg: SceneCfg):
    """Returns (spec, RobotInfo) for arm + hand + table + goal markers (no tool)."""
    ur5e_path, leap_path = resolve_model_paths(cfg)

    arm = _load_spec(ur5e_path)
    arm_model = arm.compile()
    arm_joint_names = [arm_model.joint(i).name for i in range(arm_model.njnt)]
    arm_body_names = [arm_model.body(i).name for i in range(1, arm_model.nbody)]
    sid = mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_SITE, cfg.arm_attach_site)
    if sid < 0:
        raise RuntimeError(f"UR5e model has no site '{cfg.arm_attach_site}'")
    site_body = arm_model.body(arm_model.site_bodyid[sid]).name
    site_pos = arm_model.site_pos[sid].copy()
    site_quat = arm_model.site_quat[sid].copy()

    hand = _load_spec(leap_path)
    for j in _spec_list(hand, "joints"):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            hand.delete(j)
    if cfg.auto_align_hand:
        auto_quat, auto_pos = _auto_hand_mount(leap_path)
    else:
        auto_quat, auto_pos = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
    # Site frame -> user fine-tuning -> automatic alignment.
    user_quat = mu.euler_xyz_quat(cfg.hand_mount_euler)
    mount_quat = mu.quat_mul(site_quat, mu.quat_mul(user_quat, auto_quat))
    offset = mu.rotate(user_quat, auto_pos) + np.asarray(cfg.hand_mount_pos, dtype=np.float64)
    mount_pos = site_pos + mu.rotate(site_quat, offset)
    frame = _find_body(arm, site_body).add_frame(pos=list(mount_pos), quat=list(mount_quat))
    arm.attach(hand, frame=frame, prefix=HAND_PREFIX)

    arm.option.timestep = cfg.timestep
    arm.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = cfg.impratio
    if cfg.gravcomp:
        for b in _all_bodies(arm):
            if b.name != "world":
                b.gravcomp = 1.0

    # Compile robot-only to discover joints, actuators, fingers and the home palm position.
    m = arm.compile()
    d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    actuated = {int(m.actuator_trnid[i, 0]) for i in range(m.nu)
                if int(m.actuator_trntype[i]) == _TRN_JOINT}
    arm_joints = [n for n in arm_joint_names
                  if jid(n) >= 0 and int(m.jnt_type[jid(n)]) in _ACTUATED_JOINT_TYPES and jid(n) in actuated]
    hand_joints = [m.joint(i).name for i in range(m.njnt)
                   if m.joint(i).name.startswith(HAND_PREFIX)
                   and int(m.jnt_type[i]) in _ACTUATED_JOINT_TYPES and i in actuated]
    if not arm_joints or not hand_joints:
        raise RuntimeError(f"Found {len(arm_joints)} actuated arm joints and {len(hand_joints)} hand joints; "
                           "check the Menagerie models.")

    hand_ids = [i for i in range(m.nbody) if m.body(i).name.startswith(HAND_PREFIX)]
    palm = min(hand_ids)

    qadr = [m.jnt_qposadr[jid(n)] for n in arm_joints]
    d.qpos[qadr] = np.asarray(cfg.arm_home[: len(qadr)])
    mujoco.mj_kinematics(m, d)
    palm_home = d.xpos[palm].copy()

    fingers = _order_fingertips(m, d, hand_ids, palm, prefix=HAND_PREFIX)
    finger_bodies = []
    for leaf in fingers:  # every link from the fingertip back to (not including) the palm
        group, b = [], leaf
        while b != palm and b in hand_ids:
            group.append(m.body(b).name)
            b = int(m.body_parentid[b])
        finger_bodies.append(group)

    # Table in front of the arm, along the direction the palm points to at home.
    xy = palm_home[:2]
    direction = xy / np.linalg.norm(xy) if np.linalg.norm(xy) > 0.15 else np.array([1.0, 0.0])
    table_dir = np.array([direction[0], direction[1], 0.0])
    table_perp = np.array([-direction[1], direction[0], 0.0])
    table_center = table_dir * cfg.table_center_dist
    hx, hy, hz = cfg.table_half_size
    ph = cfg.pedestal_height
    wb = arm.worldbody
    wb.add_geom(name="table", type=_GEOM.mjGEOM_BOX, size=[hx, hy, hz],
                pos=[table_center[0], table_center[1], -hz],
                quat=list(mu.yaw_quat(np.arctan2(direction[1], direction[0]))), rgba=[0.76, 0.7, 0.6, 1])
    wb.add_geom(name="pedestal", type=_GEOM.mjGEOM_CYLINDER, size=[0.09, ph / 2, 0], pos=[0, 0, -ph / 2],
                rgba=[0.3, 0.3, 0.33, 1])
    wb.add_geom(name="floor", type=_GEOM.mjGEOM_PLANE, size=[4, 4, 0.1], pos=[0, 0, -ph],
                rgba=[0.88, 0.88, 0.9, 1])
    try:
        wb.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
        wb.add_light(pos=[1.5, 1.5, 2], dir=[-1, -1, -1.3], diffuse=[0.3, 0.3, 0.3])
    except Exception:
        pass  # lights are cosmetic

    # Goal markers: mocap bodies with non-colliding geoms (geom group 2 is visible by default).
    nocol = dict(contype=0, conaffinity=0, group=2)
    goal = wb.add_body(name="goal", mocap=True)
    goal.add_geom(name="goal_sphere", type=_GEOM.mjGEOM_SPHERE, size=[0.015, 0, 0], rgba=[0.1, 0.9, 0.2, 0.6], **nocol)
    gv = wb.add_body(name="goal_vel", mocap=True)
    gv.add_geom(name="goal_vel_geom", type=_GEOM.mjGEOM_CAPSULE, size=[0.006, 0.05, 0], pos=[0, 0, 0.05],
                rgba=[0.95, 0.2, 0.1, 0.7], **nocol)
    ga = wb.add_body(name="goal_axis", mocap=True)
    ga.add_geom(name="goal_axis_geom", type=_GEOM.mjGEOM_CAPSULE, size=[0.003, 0.06, 0], pos=[0, 0, 0.06],
                rgba=[0.1, 0.4, 1.0, 0.8], **nocol)

    info = RobotInfo(
        arm_joints=arm_joints,
        hand_joints=hand_joints,
        palm_body=m.body(palm).name,
        fingertip_bodies=[m.body(i).name for i in fingers],
        finger_bodies=finger_bodies,
        robot_bodies=arm_body_names + [m.body(i).name for i in hand_ids],
        palm_home=palm_home,
        table_dir=table_dir,
        table_perp=table_perp,
        table_center=table_center,
        table_top=0.0,
        table_half_size=np.asarray(cfg.table_half_size),
        floor_z=-ph,
    )
    return arm, info


class SceneBuilder:
    """Builds the robot scene once; compiles a new MjModel for each tool."""

    def __init__(self, cfg: SceneCfg):
        self.cfg = cfg
        self.base_spec, self.info = build_base_spec(cfg)

    def compile(self, tool: ToolParams, weld: bool = False) -> mujoco.MjModel:
        """weld=True adds an equality "grip" (palm -> tool) and turns off hand-tool collisions.

        The weld's relative pose is set at runtime (eq_data[3:10], tool pose in the palm frame, see
        PoseVelocityEnv._weld_tool); its anchor eq_data[0:3] must stay zero.
        """
        spec = self._copy_base()
        add_tool(spec, tool, park_pos=(float(self.info.table_center[0]), float(self.info.table_center[1]), 3.0))
        if weld:
            eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="grip", name1=self.info.palm_body,
                                   name2="tool", objtype=mujoco.mjtObj.mjOBJ_BODY)
            eq.solref = list(self.cfg.weld_solref)
            for name in self.info.robot_bodies:
                if name.startswith(HAND_PREFIX):
                    spec.add_exclude(bodyname1=name, bodyname2="tool")
        m = spec.compile()
        if weld:
            m.eq_data[0, :] = 0.0
            m.eq_data[0, 6] = 1.0  # identity quaternion until the grasp is set
            m.eq_data[0, 10] = 1.0  # torquescale
        return m

    def _copy_base(self) -> mujoco.MjSpec:
        try:
            return self.base_spec.copy()
        except AttributeError:  # older bindings without MjSpec.copy: rebuild from files
            spec, _ = build_base_spec(self.cfg)
            return spec


def _main():
    import argparse

    from env import PoseVelocityEnv

    parser = argparse.ArgumentParser(description="Print scene diagnostics and optionally open the viewer.")
    parser.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.0, help="curriculum level (affects in-hand starts)")
    args = parser.parse_args()

    env = PoseVelocityEnv()
    env.set_curriculum(args.level)
    obs, info = env.reset(seed=args.seed)
    ri = env.rinfo
    m = env.model
    print(f"mujoco {mujoco.__version__}")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    print(f"arm joints  ({len(ri.arm_joints)}): {ri.arm_joints}")
    print(f"hand joints ({len(ri.hand_joints)}): {ri.hand_joints}")
    print(f"palm body: {ri.palm_body}")
    print(f"fingertips (thumb last): {ri.fingertip_bodies}")
    print(f"palm at home: {np.round(ri.palm_home, 3)}  table dir: {np.round(ri.table_dir, 3)}")
    print(f"action dim: {env.action_space.shape[0]}  obs dim: {obs.shape[0]}  priv dim: {env.priv_dim}")
    t = env.tool
    print(f"tool: handle={t.handle_type} L={t.handle_len:.3f} r={t.handle_rad:.3f} head={t.head_type} "
          f"layout={t.layout} mass={t.mass:.3f}kg")
    g = env.goal
    print(f"goal: p={np.round(g.p, 3)} v={np.round(g.v, 3)} axis={np.round(g.axis, 3)}")
    print(f"reset kind: {info['reset_kind']}  reach voxels: {len(env.reach.keys)}")
    if args.view:
        from mujoco import viewer  # `import mujoco.viewer` here would shadow the module-level mujoco

        print("Opening viewer: green sphere = goal point, red arrow = goal velocity, blue = goal head axis.")
        viewer.launch(env.model, env.data)


if __name__ == "__main__":
    _main()
