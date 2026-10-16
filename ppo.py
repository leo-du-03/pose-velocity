"""Asymmetric actor-critic PPO (CleanRL-style), device-agnostic (CPU or CUDA)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import PPOCfg


class RunningMeanStd(nn.Module):
    """Running observation normalizer; stored as buffers so it is saved in checkpoints."""

    def __init__(self, shape: int, clip: float = 10.0, min_var: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip, self.min_var = clip, min_var

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + n
        m2 = self.var * self.count + b_var * n + delta**2 * self.count * n / tot
        self.mean.add_(delta * n / tot)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var.clamp_min(self.min_var)), -self.clip, self.clip)


def mlp(inp: int, hidden: Sequence[int], out: int, out_gain: float) -> nn.Sequential:
    layers: list = [nn.LayerNorm(inp)]
    last = inp
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        last = h
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, act_dim: int, cfg: PPOCfg):
        super().__init__()
        self.obs_rms = RunningMeanStd(obs_dim)
        self.priv_rms = RunningMeanStd(priv_dim)
        self.actor = mlp(obs_dim, cfg.actor_hidden, act_dim, out_gain=0.01)
        self.critic = mlp(obs_dim + priv_dim, cfg.critic_hidden, 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))

    def normalize(self, obs: torch.Tensor, priv: torch.Tensor):
        return self.obs_rms(obs), self.priv_rms(priv)

    def dist(self, nobs: torch.Tensor) -> Normal:
        mean = self.actor(nobs)
        return Normal(mean, torch.exp(self.log_std.clamp(-5.0, 1.0)).expand_as(mean))

    def value(self, nobs: torch.Tensor, npriv: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([nobs, npriv], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_rms(obs)).clamp(-1.0, 1.0)


class ReturnScaler:
    """Scales rewards by the running std of the discounted return (per-env return tracking)."""

    def __init__(self, num_envs: int, gamma: float):
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def __call__(self, rew: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + rew
        b_mean, b_var, n = self.ret.mean(), self.ret.var(), len(self.ret)
        delta = b_mean - self.mean
        tot = self.count + n
        self.var = (self.var * self.count + b_var * n + delta**2 * self.count * n / tot) / tot
        self.mean += delta * n / tot
        self.count = tot
        self.ret[done] = 0.0
        return rew / np.sqrt(self.var + 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class PPO:
    def __init__(self, agent: ActorCritic, cfg: PPOCfg):
        self.agent, self.cfg = agent, cfg
        self.opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
        self.lr = cfg.lr

    def update(self, b: dict) -> dict:
        """b: flattened rollout tensors (obs, priv already normalized)."""
        cfg = self.cfg
        n = b["obs"].shape[0]
        mb_size = max(1, n // cfg.minibatches)
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=b["obs"].device)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]
                dist = self.agent.dist(b["obs"][idx])
                logp = dist.log_prob(b["actions"][idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = logp - b["logp"][idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip).float().mean()

                adv = b["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()

                v = self.agent.value(b["obs"][idx], b["priv"][idx])
                v_old, ret = b["values"][idx], b["returns"][idx]
                v_clipped = v_old + (v - v_old).clamp(-cfg.value_clip, cfg.value_clip)
                v_loss = 0.5 * torch.max((v - ret) ** 2, (v_clipped - ret) ** 2).mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.opt.step()

                if cfg.adaptive_lr:
                    if kl > 2.0 * cfg.kl_target:
                        self.lr = max(self.lr / 1.5, 1e-6)
                    elif kl < 0.5 * cfg.kl_target:
                        self.lr = min(self.lr * 1.5, 1e-2)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr

                for k, val in (("pg_loss", pg_loss), ("v_loss", v_loss), ("entropy", entropy), ("kl", kl),
                               ("clipfrac", clipfrac)):
                    stats[k].append(float(val))
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["lr"] = self.lr
        out["std"] = float(torch.exp(self.agent.log_std).mean())
        return out


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: (T, N) tensors. dones[t] = 1 cuts the return after step t (no bootstrap)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values
