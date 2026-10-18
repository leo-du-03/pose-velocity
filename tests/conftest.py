import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EnvCfg  # noqa: E402


@pytest.fixture
def cfg():
    c = EnvCfg()
    c.goal.reach_samples = 20000  # faster for tests
    c.goal.reach_voxel = 0.06
    return c


@pytest.fixture
def env(cfg):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(cfg)
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture
def physical_cfg(cfg):
    cfg.grip_mode = "physical"
    cfg.action_mode = "joint"
    return cfg


@pytest.fixture
def physical_env(physical_cfg):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(physical_cfg)
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture
def grasp_cfg(cfg):
    cfg.grip_mode = "grasp"
    return cfg


@pytest.fixture
def grasp_env(grasp_cfg):
    from env import PoseVelocityEnv

    e = PoseVelocityEnv(grasp_cfg)
    e.reset(seed=0)
    yield e
    e.close()
