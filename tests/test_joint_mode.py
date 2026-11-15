"""Joint actions (EnvCfg.action_mode "joint") with oracle labels for behavior cloning: the env converts the oracle's
head twists into joint-target changes (PoseVelocityEnv.joint_action_for_twist)."""
import dataclasses

import numpy as np


def _env(cfg, mode, scale=0.1, seed=0):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(dataclasses.replace(cfg, action_mode=mode, arm_action_scale=scale))
    e.set_curriculum(1.0)
    e.reset(seed=seed)
    return e


def test_joint_obs_adds_target_offsets(cfg):
    """Joint mode observes the servo targets' lead over the joints (6 extra values, right after prev_action);
    head_twist observations are unchanged, so existing checkpoints still load."""
    h, j = _env(cfg, "head_twist"), _env(cfg, "joint")
    assert j.observation_space.shape[0] == h.observation_space.shape[0] + 6
    for _ in range(5):
        obs, *_ = j.step(j.oracle_action())
    start = 12 + j.n_act  # arm qpos, qvel, prev_action
    np.testing.assert_allclose(obs[start:start + 6], j.targets[:6] - j.data.qpos[j.arm_qadr], atol=1e-6)
    assert np.abs(obs[start:start + 6]).max() > 1e-3  # the arm is moving, so the targets lead it
    h.close()
    j.close()


def test_joint_labels_set_head_twist_targets(cfg):
    """From the same state, the first joint label puts the servo targets exactly where head_twist mode puts them
    (no smoothing history yet, and a scale large enough that nothing clips)."""
    h, j = _env(cfg, "head_twist", seed=3), _env(cfg, "joint", scale=1.0, seed=3)
    h.step(h.oracle_action())
    j.step(j.oracle_action())
    np.testing.assert_allclose(j.targets[:6], h.targets[:6], atol=1e-6)
    h.close()
    j.close()


def test_commanded_twist_resets_with_the_arm(cfg):
    """A full reset sets the targets to the joints: no command left over for the labels to smooth from."""
    j = _env(cfg, "joint")
    for _ in range(10):
        j.step(j.oracle_action())
    assert np.abs(j.prev_twist()).max() > 1e-3
    j.reset(options={"kind": "full"})
    np.testing.assert_allclose(j.prev_twist(), 0.0, atol=1e-9)
    j.close()


def test_joint_oracle_succeeds(cfg):
    """The converted oracle strikes as well as in head_twist mode (~90% at level 1 over 100 goals with
    arm_action_scale 0.1; ~75% at the default 0.03, where about half the labels clip)."""
    j = _env(cfg, "joint", scale=0.1, seed=1)
    successes = 0
    for _ in range(12):
        while True:
            a = j.oracle_action()
            assert a.shape == (6,) and np.all(np.abs(a) <= 1.0)
            _, _, term, trunc, info = j.step(a)
            if term or trunc:
                break
        successes += info["episode"]["success"]
        j.reset()
    assert successes >= 8, successes
    j.close()


# ---------------------------------------------------------------- joint_vel: joint-velocity actions


def _jv(cfg, seed=0, **kw):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(dataclasses.replace(cfg, action_mode="joint_vel", **kw))
    e.set_curriculum(1.0)
    e.reset(seed=seed)
    return e


def test_joint_vel_obs_adds_kin_features(cfg):
    """joint_vel adds 30 Jacobian-projected goal errors (5 joint-space 6-vectors), none with kin_obs off; head_twist
    and joint observations are unchanged."""
    h, j = _env(cfg, "head_twist"), _env(cfg, "joint")
    jv, jv0 = _jv(cfg), _jv(cfg, kin_obs=False)
    n = h.observation_space.shape[0]
    assert j.observation_space.shape[0] == n + 6
    assert jv.observation_space.shape[0] == n + 30
    assert jv0.observation_space.shape[0] == n
    for e in (h, j, jv, jv0):
        e.close()


def test_joint_vel_labels_set_head_twist_targets(cfg):
    """From the same state, the first joint_vel label (no smoothing history yet) puts the servo targets exactly where
    head_twist mode puts them, when nothing clips at joint_v_max."""
    big = (10.0,) * 6
    h, jv = _env(cfg, "head_twist", seed=3), _jv(cfg, seed=3, joint_v_max=big)
    h.step(h.oracle_action())
    jv.step(jv.oracle_action())
    np.testing.assert_allclose(jv.targets[:6], h.targets[:6], atol=1e-6)
    h.close()
    jv.close()


def test_joint_vel_prev_twist_is_commanded_twist(cfg):
    """prev_twist (read by the oracle's commit gate) is the head twist of the smoothed joint-velocity command."""
    jv = _jv(cfg, seed=2)
    for _ in range(8):
        jv.step(jv.oracle_action())
    dq = (jv.targets[:6] - jv._ctrl_ref[1]) / jv.arm_lead
    twist = jv._head_jacobian() @ dq
    expect = np.concatenate([twist[:3] / jv.cfg.head_v_max, twist[3:] / jv.cfg.head_w_max])
    np.testing.assert_allclose(jv.prev_twist(), np.clip(expect, -1, 1), atol=1e-9)
    assert np.abs(jv.prev_twist()).max() > 1e-3
    jv.close()


def test_joint_vel_kin_features_match_label(cfg):
    """The kinematic features are P(x) for each part of the oracle's twist, so a linear combination reproduces the
    oracle's joint velocity: in the approach phase, k_pos * P(lead - head) (uncapped) + k_rot * P(rot_err)."""
    jv = _jv(cfg, seed=4)
    o = jv.oracle_action()  # builds the oracle
    head, lin = jv._head_state()[0], jv._head_state()[1]
    f = jv._kin_features(head, lin).reshape(5, 6)
    orc = jv._oracle
    twist = np.concatenate([orc.k_pos * (jv.lead_point() - head), orc.k_rot * jv._rot_error()])
    if np.linalg.norm(twist[:3]) <= orc.v_approach and orc.alpha == 0.0:
        np.testing.assert_allclose(orc.k_pos * f[0] + orc.k_rot * f[4], jv._dls_pinv() @ twist / jv._joint_v_max(),
                                   atol=1e-9)
    # In general, P applied to the oracle's twist is the unclipped label.
    a = jv._oracle(jv)
    tw = np.concatenate([a[:3] * jv.cfg.head_v_max, a[3:] * jv.cfg.head_w_max])
    raw = jv._dls_pinv() @ tw / jv._joint_v_max()
    np.testing.assert_allclose(np.clip(raw, -1, 1), o, atol=1e-5)
    jv.close()


def test_joint_vel_oracle_succeeds(cfg):
    """The converted oracle strikes as well as in head_twist mode (~85-90% at level 1)."""
    jv = _jv(cfg, seed=1)
    successes = 0
    for _ in range(12):
        while True:
            a = jv.oracle_action()
            assert a.shape == (6,) and np.all(np.abs(a) <= 1.0)
            _, _, term, trunc, info = jv.step(a)
            if term or trunc:
                break
        successes += info["episode"]["success"]
        jv.reset()
    assert successes >= 8, successes
    jv.close()
