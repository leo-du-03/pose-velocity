"""Measure simulation throughput of the simplify-weld env on this machine.

Put this file in the root of the pose-velocity repo (simplify-weld branch) and run:
    python bench_sps.py
It uses weld grip and joint-delta arm actions (the SimToolReal-style action space).
Takes about 1-3 minutes. Close other heavy apps first.
"""
from __future__ import annotations

import os
import time

import numpy as np

from config import EnvCfg
from env import PoseVelocityEnv, make_env
from vec_env import SubprocVecEnv

STEPS_SINGLE = 2000
STEPS_VEC = 500


def make_cfg() -> EnvCfg:
    cfg = EnvCfg()
    cfg.grip_mode = "weld"
    cfg.action_mode = "joint"
    return cfg


def bench_single() -> float:
    env = PoseVelocityEnv(make_cfg())
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    n = env.action_space.shape[0]
    t0 = time.perf_counter()
    for _ in range(STEPS_SINGLE):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, n).astype(np.float32))
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    env.close()
    return STEPS_SINGLE / dt


def bench_vec(num_envs: int) -> float:
    cfg = make_cfg()
    venv = SubprocVecEnv([make_env(cfg, 1000 + i) for i in range(num_envs)])
    venv.reset(seeds=[1000 + i for i in range(num_envs)])
    n = venv.action_space.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(20):  # warm-up
        venv.step(rng.uniform(-1, 1, (num_envs, n)).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(STEPS_VEC):
        venv.step(rng.uniform(-1, 1, (num_envs, n)).astype(np.float32))
    dt = time.perf_counter() - t0
    venv.close()
    return STEPS_VEC * num_envs / dt


def main():
    cores = os.cpu_count() or 1
    print(f"logical cores: {cores}")
    print(f"single env: {bench_single():8.0f} steps/s")
    for n in sorted({4, 8, max(1, cores - 2), cores}):
        print(f"{n:3d} envs:   {bench_vec(n):8.0f} steps/s")
    print("\nHours for N million steps at the best rate: hours = N * 1e6 / best_sps / 3600")


if __name__ == "__main__":
    main()
