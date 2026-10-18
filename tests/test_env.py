import mujoco
import numpy as np
import pytest

import mathutils as mu


def test_check_env(cfg):
    from gymnasium.utils.env_checker import check_env

    from env import PoseVelocityEnv

    env = PoseVelocityEnv(cfg)
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_fixture", ["env", "physical_env"])
def test_random_rollout_is_finite(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
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


def test_respawn_keeps_arm_and_changes_tool(physical_env):
    env = physical_env
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


def test_episode_end_sets_next_reset_kind(physical_env):
    env = physical_env
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


def test_weld_regrips_after_n_goals(cfg):
    from env import PoseVelocityEnv

    cfg.regrip_every = 2
    cfg.max_steps = 5
    env = PoseVelocityEnv(cfg)
    env.reset(seed=0)
    kinds = []
    for _ in range(4):
        for _ in range(cfg.max_steps):
            _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape))
            if term or trunc:
                break
        kinds.append(info["episode"]["next_reset"])
        _, rinfo = env.reset()
        assert rinfo["reset_kind"] == kinds[-1]
    assert kinds == ["soft", "regrip", "soft", "regrip"]
    env.close()


def test_weld_holds(env):
    """Random arm motion must not move the tool relative to the palm (beyond the weld's softness)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
        d = env.data
        R = d.xmat[env.palm_id].reshape(3, 3)
        worst = max(worst, np.linalg.norm(R.T @ (d.xpos[env.tool_id] - d.xpos[env.palm_id]) - env.grasp_pos))
        if term or trunc:
            env.reset(options={"kind": "soft"})
    assert worst < 0.005, worst


def test_head_twist_tracks(cfg):
    """A constant linear-velocity command moves the head at roughly that velocity."""
    from env import PoseVelocityEnv

    cfg.max_steps = 1000
    env = PoseVelocityEnv(cfg)
    env.reset(seed=0)
    for axis in range(3):
        a = np.zeros(6)
        a[axis] = 0.25
        vs = []
        for k in range(30):
            env.step(a)
            if k >= 10:
                vs.append(env._head_state()[1])
        v = np.mean(vs, 0)
        target = a[:3] * cfg.head_v_max
        assert np.linalg.norm(v - target) < 0.3 * np.linalg.norm(target), (axis, v)
    env.close()


def test_goal_from_grasp_is_reachable(env):
    """Weld goals come from arm configurations: q_ref puts the head on the goal with the goal axis."""
    for _ in range(20):
        env._sample_goal()
        g = env.goal
        assert not env.goal_fallback and g.q_ref is not None
        head, axis, _ = env._scratch_pose(g.q_ref)
        assert np.linalg.norm(head - g.p) < 1e-3
        assert mu.angle_between(axis, g.axis) < np.deg2rad(1.0)
        assert env.goal_sampler.in_region(g.p)


@pytest.mark.parametrize("lateral, expect_pass", [(0.0, True), (2.0, False)])
def test_crossing_counts_only_near_goal(env, lateral, expect_pass):
    """A goal-plane crossing ends the attempt only within pass_radius of p_g (lateral in units of it)."""
    from goals import Goal

    head = env.data.site_xpos[env.head_site].copy()
    d = np.array([0.0, 0.0, 1.0])
    perp = mu.any_perpendicular(d)
    # Plane 1 cm behind the head; pretend the head came from 3 cm behind it.
    p = head - 0.01 * d + lateral * env.cfg.goal.pass_radius * perp
    env.goal = Goal(p=p, v=0.1 * d, axis=d)
    env._prev_head = head - 0.03 * d
    _, _, term, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["passed"] == expect_pass and term == expect_pass


def _run_attempt(env, policy):
    while True:
        _, _, term, trunc, info = env.step(policy(env))
        if term or trunc:
            return info["episode"]


@pytest.mark.parametrize("policy", ["oracle", "rush", "random"])
def test_approach_reward_is_bounded_and_clawed_back(env, policy):
    """Approach reward per attempt is at most w_approach * D(start); a pass hands back (1 - quality) of it."""
    rng = np.random.default_rng(0)
    pol = {"oracle": _speed_scaled_oracle(1.0), "rush": _speed_scaled_oracle(3.0),
           "random": lambda env: rng.uniform(-1, 1, env.action_space.shape)}[policy]
    for _ in range(5):
        bound = env.cfg.reward.w_approach * env.approach_best
        e = _run_attempt(env, pol)
        t = e["terms"]
        assert 0.0 <= t["approach"] <= bound + 1e-9, (t["approach"], bound)
        assert t["waypoint"] in (0.0, env.cfg.reward.w_waypoint)  # paid at most once per attempt
        earned = t["approach"] + t["waypoint"]
        if e["passed"]:
            paid = e["pass"]["paid_quality"]
            assert paid == (e["pass"]["quality"] if e["waypoint"] else 0.0)  # no waypoint, no pass reward
            assert np.isclose(t["clawback"], -(1.0 - paid) * earned)
        else:
            assert t["clawback"] == 0.0  # a time-out keeps the approach reward
        env.reset()


def test_pass_without_waypoint_pays_nothing(env):
    """A perfect crossing that didn't come through the lead-in point earns no pass reward and hands back its
    approach reward; success is still recorded (the gate is on the training reward only)."""
    from goals import Goal

    head = env.data.site_xpos[env.head_site].copy()
    _, lin, _, axis = env._head_state()
    d = np.array([0.0, 0.0, 1.0])
    env.goal = Goal(p=head - 0.01 * d, v=0.3 * d, axis=axis)  # plane 1 cm behind the head
    env._prev_head = head - 0.03 * d
    env.approach_earned, env.waypoint_hit = 2.0, False
    _, _, term, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["passed"] and term
    t = info["episode"]["terms"]
    assert t["pass"] == 0.0
    assert np.isclose(t["clawback"], -2.0 - t["approach"])


def test_oracle_strike_is_paid(env):
    """The lined-up strike goes through the waypoint, so its passes are paid."""
    pol = _speed_scaled_oracle(1.0)
    env.reset(seed=11)
    for _ in range(4):
        e = _run_attempt(env, pol)
        if e["success"]:
            assert e["waypoint"] == 1.0 and e["terms"]["pass"] > env.cfg.reward.success_bonus
        env.reset()


def test_oracle_collects_the_waypoint(env):
    """The scripted strike goes through the lead-in point aligned, so it earns the waypoint bonus every time."""
    pol = _speed_scaled_oracle(1.0)
    env.reset(seed=11)
    hits = []
    for _ in range(6):
        e = _run_attempt(env, pol)
        if e["success"]:
            hits.append(e["terms"]["waypoint"])
        env.reset()
    assert hits and all(h == env.cfg.reward.w_waypoint for h in hits), hits


def test_rushing_hands_back_the_approach_reward(env):
    """Net approach reward (after clawback) is much higher for a good strike than for rushing through the goal."""
    net = {}
    for mult in (1.0, 3.0):
        env.reset(seed=11)
        pol = _speed_scaled_oracle(mult)
        vals = []
        for _ in range(6):
            t = _run_attempt(env, pol)["terms"]
            vals.append(t["approach"] + t["clawback"])
            env.reset()
        net[mult] = float(np.mean(vals))
    assert net[3.0] < 0.4 * net[1.0], net


def _speed_scaled_oracle(mult):
    """The oracle, but striking at mult x the goal speed."""
    from evaluate import Oracle

    class Scaled(Oracle):
        def __call__(self, env):
            g = env.goal
            v0 = g.v.copy()
            g.v = v0 * mult
            try:
                return super().__call__(env)
            finally:
                g.v = v0

    return Scaled()


def test_track_reward_pays_for_the_right_traversal(env):
    """Traversing the lead-in line at the goal velocity earns most of w_track; crawling or rushing earns less, and
    no attempt ever earns more than w_track."""
    w = env.cfg.reward.w_track
    per_policy = {}
    for mult in (1.0, 0.3, 3.0):
        oracle = _speed_scaled_oracle(mult)
        env.reset(seed=11)
        got = []
        for _ in range(6):
            while True:
                _, _, term, trunc, info = env.step(oracle(env))
                if term or trunc:
                    break
            got.append(info["episode"]["terms"]["track"])
            env.reset()
        assert max(got) <= w + 1e-6, got
        per_policy[mult] = float(np.mean(got))
    assert per_policy[1.0] > 0.4 * w, per_policy  # the oracle starts from rest at the lead-in point
    assert per_policy[0.3] < per_policy[1.0] and per_policy[3.0] < per_policy[1.0], per_policy


def test_awkward_posture_recovers_with_reachable_goal(env):
    """If no goal can be sampled near the current pose, the arm is re-homed (and if needed regripped) instead of
    falling back to a grasp-agnostic, possibly unreachable goal."""
    home = np.asarray(env.cfg.scene.arm_home)
    propose = env._propose_goal_q

    def only_near_home(rng):  # simulate a posture where nothing nearby is reachable
        if np.linalg.norm(env.data.qpos[env.arm_qadr] - home) > 0.2:
            return None
        return propose(rng)

    env._propose_goal_q = only_near_home
    env.data.qpos[env.arm_qadr] = home + np.array([0.8, -0.6, 0.7, 0.5, -0.4, 0.3])
    mujoco.mj_forward(env.model, env.data)
    env._sample_goal()
    assert env.goal_recovery == 1 and not env.goal_fallback
    assert np.linalg.norm(env.data.qpos[env.arm_qadr] - home) < 0.2
    head, axis, _ = env._scratch_pose(env.goal.q_ref)
    assert np.linalg.norm(head - env.goal.p) < 1e-3
    # The re-home must not register as head motion (no spurious goal-plane crossing on the next step).
    np.testing.assert_allclose(env._prev_head, env.data.site_xpos[env.head_site])


def test_goal_sampling_rarely_needs_recovery(env):
    rng = np.random.default_rng(0)
    rec, fb = [], []
    for _ in range(40):
        for _ in range(25):
            env.step(rng.uniform(-1, 1, env.action_space.shape))
        env._sample_goal()
        rec.append(env.goal_recovery)
        fb.append(env.goal_fallback)
    assert not any(fb)
    assert np.mean(np.array(rec) > 0) < 0.3, rec




def test_scripted_pass_through_succeeds(physical_cfg):
    """Fly the tool head through the goal at the goal velocity (no gravity/contacts): success must fire."""
    from env import PoseVelocityEnv

    cfg = physical_cfg
    cfg.success_requires_grasp = False
    cfg.reward.pass_requires_waypoint = False  # the head is teleported onto the line, not flown through the lead-in
    env = PoseVelocityEnv(cfg)
    env.set_curriculum(0.0)
    env.reset(seed=3)
    env.model.opt.gravity[:] = 0.0
    env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    g = env.goal
    dt = env.model.opt.timestep * cfg.frame_skip
    env.debug_set_head_state(g.p - g.v * 5 * dt, g.axis, g.v)
    success = False
    for _ in range(10):
        _, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        if info["success"]:
            success = True
            break
    assert success, f"d_perp={info['d_perp']}"
    assert term
    # Only the pass term: teleporting the tool out of a scripted grasp also (correctly) costs grasp/approach.
    assert env.ep_terms["pass"] > 0.95 * (env.cfg.reward.pass_scale + env.cfg.reward.success_bonus)
    env.close()


def test_oracle_succeeds(cfg):
    """The scripted strike (evaluate.Oracle) succeeds on most weld goals: they are feasible."""
    from env import PoseVelocityEnv
    from evaluate import Oracle

    env = PoseVelocityEnv(cfg)
    env.reset(seed=1)
    oracle = Oracle()
    successes = 0
    for _ in range(12):
        while True:
            _, _, term, trunc, info = env.step(oracle(env))
            if term or trunc:
                break
        successes += info["episode"]["success"]
        env.reset()
    assert successes >= 8, successes
    env.close()
