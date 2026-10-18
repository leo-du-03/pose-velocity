"""Grasp grip: a floating tool that has to be grasped (weld on grasp), then struck through a chain of goals."""
import mujoco
import numpy as np

import mathutils as mu


def _open(env, arm=None):
    a = np.zeros(env.action_space.shape)
    if arm is not None:
        a[:6] = arm
    a[6] = -1.0
    return a


def _teleport_to_grasp(env):
    """Puts the arm at the canonical grasp (hand open), as the oracle would arrive there."""
    target, _, palm_q = env.grasp_targets()
    palm_p = target - mu.rotate(palm_q, env.grip_point_local)
    q = env._ik_palm(env.data.qpos[env.arm_qadr], palm_p, palm_q)
    assert q is not None
    env.data.qpos[env.arm_qadr] = q
    env._hold_robot_pose()


def test_floating_tool_stays_put(grasp_env):
    env = grasp_env
    assert env.start_type == "float" and not env.welded and env.action_space.shape == (7,)
    p0 = env.data.xpos[env.tool_id].copy()
    for _ in range(100):  # 2 s with the arm still
        _, _, term, trunc, info = env.step(_open(env))
        assert not (term or trunc)
    assert np.linalg.norm(env.data.xpos[env.tool_id] - p0) < 1e-3
    assert not env.welded and not env.knocked


def test_knock_drops_the_tool(grasp_env):
    env = grasp_env
    p = env.data.qpos[env.tool_qadr:env.tool_qadr + 3]
    p += [0.05, 0.0, 0.0]  # as if knocked away
    mujoco.mj_forward(env.model, env.data)
    _, r, term, trunc, info = env.step(_open(env))
    assert term and info["dropped"] and info["episode"]["knocked"] == 1.0
    assert r <= -env.cfg.reward.drop_penalty
    assert env.model.body_gravcomp[env.tool_id] == 0.0 and env.data.eq_active[env.hold_eq] == 0
    assert info["episode"]["next_reset"] == "regrip"
    z0 = env.data.xpos[env.tool_id][2]
    env._settle(0.2)
    assert env.data.xpos[env.tool_id][2] < z0 - 0.05  # gravity is back on


def test_canonical_grasp_depends_only_on_the_tool(grasp_env):
    env = grasp_env
    pos, quat = env.data.xpos[env.tool_id].copy(), env.data.xquat[env.tool_id].copy()
    first = env._canonical_grasp(pos, quat)
    env.data.qpos[env.arm_qadr] += 0.3
    mujoco.mj_forward(env.model, env.data)
    for a, b in zip(first, env._canonical_grasp(pos, quat)):
        np.testing.assert_allclose(a, b)
    rel_q = first[3]
    np.testing.assert_allclose(mu.rotate(rel_q, [0, 0, 1]), [0, -1, 0], atol=1e-9)  # handle across the fingers


def test_scripted_grasp_welds_and_holds(grasp_env):
    env = grasp_env
    _teleport_to_grasp(env)
    a = np.zeros(7)
    a[6] = 1.0
    for _ in range(50):
        _, _, term, _, info = env.step(a)
        assert not term
        if env.welded:
            break
    assert env.welded and env.data.eq_active[env.grip_eq] == 1 and env.data.eq_active[env.hold_eq] == 0
    assert env.ep_terms["grasp"] >= env.cfg.reward.w_grasp
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        _, _, term, trunc, _ = env.step(np.concatenate([rng.uniform(-1, 1, 6), [rng.uniform(-1, 1)]]))
        d = env.data
        R = d.xmat[env.palm_id].reshape(3, 3)
        worst = max(worst, np.linalg.norm(R.T @ (d.xpos[env.tool_id] - d.xpos[env.palm_id]) - env.grasp_pos))
        if term or trunc:
            env.reset()
            assert env.welded and env.start_type == "soft"
    assert worst < 0.005, worst


def test_goals_chain_after_a_grasp(grasp_cfg):
    """After the grasp the tool stays in the hand for regrip_every goals (soft resets), then a new one floats."""
    from env import PoseVelocityEnv

    grasp_cfg.regrip_every = 3
    grasp_cfg.max_steps = 5
    env = PoseVelocityEnv(grasp_cfg)
    env.reset(seed=0)
    _teleport_to_grasp(env)
    a = np.zeros(7)
    a[6] = 1.0
    kinds = []
    for _ in range(3):
        for _ in range(env.ep_max_steps):
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        assert env.welded
        kinds.append(info["episode"]["next_reset"])
        _, rinfo = env.reset()
        assert rinfo["reset_kind"] == kinds[-1]
    assert kinds == ["soft", "soft", "regrip"]
    assert env.start_type == "float" and not env.welded and env.data.eq_active[env.grip_eq] == 0
    env.close()


def test_grasp_oracle_succeeds(grasp_cfg):
    """The scripted grasp + strike (oracle.GraspOracle) grasps most floating tools and succeeds on most goals."""
    from env import PoseVelocityEnv
    from oracle import GraspOracle

    grasp_cfg.regrip_every = 2
    env = PoseVelocityEnv(grasp_cfg)
    env.reset(seed=1)
    oracle = GraspOracle()
    eps = []
    for _ in range(16):
        while True:
            _, _, term, trunc, info = env.step(oracle(env))
            if term or trunc:
                break
        eps.append(info["episode"])
        env.reset()
    floats = [e for e in eps if e["start"] == "float"]
    assert len(floats) >= 6
    assert np.mean([e["grasp_step"] >= 0 for e in floats]) >= 0.75, floats
    assert np.mean([e["success"] for e in eps]) >= 0.6
    env.close()
