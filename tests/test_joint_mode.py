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
