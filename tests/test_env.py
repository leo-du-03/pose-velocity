import mujoco
import numpy as np


def test_check_env(cfg):
    from gymnasium.utils.env_checker import check_env

    from env import PoseVelocityEnv

    env = PoseVelocityEnv(cfg)
    check_env(env, skip_render_check=True)
    env.close()


def test_random_rollout_is_finite(env):
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    episodes = 0
    for _ in range(1500):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        assert env.observation_space.contains(obs)
        assert info["priv"].shape == (env.priv_dim,)
        if term or trunc:
            episodes += 1
            assert "episode" in info
            obs, _ = env.reset()
    assert episodes >= 1


def test_soft_reset_keeps_state_and_changes_goal(env):
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(rng.uniform(-1, 1, env.action_space.shape))
    qpos = env.data.qpos.copy()
    goal_before = env.goal.p.copy()
    model_before = env.model
    env.reset(options={"kind": "soft"})
    assert env.model is model_before
    np.testing.assert_allclose(env.data.qpos, qpos)
    assert not np.allclose(env.goal.p, goal_before)


def test_respawn_keeps_arm_and_changes_tool(env):
    env.set_curriculum(1.0)  # table spawn only
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(rng.uniform(-1, 1, env.action_space.shape))
    arm_before = env.data.qpos[env.arm_qadr].copy()
    feat_before = env.tool.features()
    _, info = env.reset(options={"kind": "respawn"})
    assert info["reset_kind"] == "respawn"
    assert not np.allclose(env.tool.features(), feat_before)
    np.testing.assert_allclose(env.data.qpos[env.arm_qadr], arm_before, atol=0.05)


def test_episode_end_sets_next_reset_kind(env):
    env.set_curriculum(1.0)
    env.reset(options={"kind": "respawn"})
    for _ in range(env.cfg.max_steps):
        _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        if term or trunc:
            break
    assert trunc or term
    assert info["episode"]["next_reset"] in ("soft", "respawn")
    arm_before = env.data.qpos[env.arm_qadr].copy()
    _, rinfo = env.reset()
    assert rinfo["reset_kind"] == info["episode"]["next_reset"]
    np.testing.assert_allclose(env.data.qpos[env.arm_qadr], arm_before, atol=0.05)


def test_scripted_pass_through_succeeds(cfg):
    """Fly the tool head through the goal at the goal velocity (no gravity/contacts): success must fire."""
    from env import PoseVelocityEnv

    cfg.success_requires_grasp = False
    env = PoseVelocityEnv(cfg)
    env.set_curriculum(0.0)
    env.reset(seed=3)
    env.model.opt.gravity[:] = 0.0
    env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    g = env.goal
    dt = env.model.opt.timestep * cfg.frame_skip
    env.debug_set_head_state(g.p - g.v * 5 * dt, g.axis, g.v)
    success, total = False, 0.0
    for _ in range(10):
        _, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        total += r
        if info["success"]:
            success = True
            break
    assert success, f"seg_dist={info['seg_dist']:.3f}"
    assert total > env.cfg.reward.success_bonus * 0.9
    env.close()
