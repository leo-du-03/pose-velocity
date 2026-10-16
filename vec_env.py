"""Minimal vector envs with same-step autoreset that also carry privileged critic features.

Both classes expose:
  reset(seeds)  -> obs (N, D), priv (N, P)
  step(actions) -> obs, priv, reward (N,), terminated (N,), truncated (N,), episode_infos (list, None if not done)
  call(name, *args) -> list of per-env return values
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _step_and_autoreset(env, action):
    obs, rew, term, trunc, info = env.step(action)
    priv = info["priv"]
    ep = info.get("episode")
    if term or trunc:
        obs, rinfo = env.reset()
        priv = rinfo["priv"]
        if ep is not None:
            ep["reset_kind"] = rinfo["reset_kind"]
    return obs, priv, rew, term, trunc, ep


def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "step":
                remote.send(_step_and_autoreset(env, payload))
            elif cmd == "reset":
                obs, info = env.reset(seed=payload)
                remote.send((obs, info["priv"]))
            elif cmd == "call":
                name, args = payload
                remote.send(getattr(env, name)(*args))
            elif cmd == "spaces":
                remote.send((env.observation_space, env.action_space, env.priv_dim))
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        pass


class SerialVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        e = self.envs[0]
        self.observation_space, self.action_space, self.priv_dim = e.observation_space, e.action_space, e.priv_dim

    def reset(self, seeds=None):
        seeds = seeds if seeds is not None else [None] * self.num_envs
        out = [env.reset(seed=s) for env, s in zip(self.envs, seeds)]
        return np.stack([o for o, _ in out]), np.stack([i["priv"] for _, i in out])

    def step(self, actions):
        out = [_step_and_autoreset(env, a) for env, a in zip(self.envs, actions)]
        return _collate(out)

    def call(self, name, *args):
        return [getattr(env, name)(*args) for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()


class SubprocVecEnv:
    def __init__(self, env_fns, start_method: str = "spawn"):
        ctx = mp.get_context(start_method)
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.procs = []
        for work_remote, remote, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(work_remote, remote, fn), daemon=True)
            p.start()
            work_remote.close()
            self.procs.append(p)
        self.remotes[0].send(("spaces", None))
        self.observation_space, self.action_space, self.priv_dim = self.remotes[0].recv()

    def reset(self, seeds=None):
        seeds = seeds if seeds is not None else [None] * self.num_envs
        for r, s in zip(self.remotes, seeds):
            r.send(("reset", s))
        out = [r.recv() for r in self.remotes]
        return np.stack([o for o, _ in out]), np.stack([p for _, p in out])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        return _collate([r.recv() for r in self.remotes])

    def call(self, name, *args):
        for r in self.remotes:
            r.send(("call", (name, args)))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)


def _collate(out):
    obs, priv, rew, term, trunc, eps = zip(*out)
    return (np.stack(obs), np.stack(priv), np.asarray(rew, dtype=np.float32),
            np.asarray(term, dtype=bool), np.asarray(trunc, dtype=bool), list(eps))
