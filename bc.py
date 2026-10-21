"""Behavior-clone the scripted oracle (oracle.py) into the PPO actor, as a starting point for PPO fine-tuning.

Uses DAgger rather than plain behavior cloning. Each round rolls out a mixture: on each step, every training env
executes the oracle's action with probability beta (beta_decay ** round, so round 0 is pure oracle) and the learner's
otherwise, plus Gaussian noise at PPO's exploration std. Every visited state is labelled with the oracle's action and
added to an aggregated dataset, and the actor is regressed onto it. Plain cloning only sees the oracle's own states,
so the learner's small errors take it to states it never saw labels for; DAgger labels the states the learner
actually visits.

The oracle is stateful (approach the lead-in point, then strike) and reads privileged env state, so each env worker
runs its own oracle (PoseVelocityEnv.oracle_action). Weld grip and head_twist actions only.

The envs are spread evenly over --levels: the curriculum tolerances are part of the observation, so the cloned
policy has to have seen the levels PPO will run at.

The checkpoint works with evaluate.py (--ckpt) and train.py (--init-from):
  python bc.py --name bc
  python evaluate.py --ckpt runs/bc/bc_latest.pt --episodes 100
  python train.py --init-from runs/bc/bc_latest.pt --name strike_bc --lr 1e-4
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time
from collections import deque

import numpy as np

from config import EnvCfg, PPOCfg
from env import make_env
from train import CKPT_FORMAT, resolve_device
from vec_env import SerialVecEnv, SubprocVecEnv

# torch / ppo are imported inside main(), as in train.py: env workers re-import this module under "spawn".


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=PPOCfg.num_envs)
    p.add_argument("--eval-envs", type=int, default=4,
                   help="envs driven by the deterministic learner alone (no oracle, no noise); their success is "
                        "reported as 'eval succ'. Their states are still labelled and trained on")
    p.add_argument("--rollout-steps", type=int, default=256, help="steps per env per DAgger round")
    p.add_argument("--rounds", type=int, default=120, help="DAgger rounds")
    p.add_argument("--beta-decay", type=float, default=0.8,
                   help="probability of executing the oracle's action in round i is beta_decay ** i")
    p.add_argument("--noise-std", type=float, default=None,
                   help="std of the Gaussian noise added to executed actions in the training envs "
                        "(default: exp(PPOCfg.init_log_std), PPO's exploration noise)")
    p.add_argument("--grad-steps", type=int, default=500, help="actor gradient steps per round")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=1e-4, help="lr is annealed linearly to this over the rounds")
    p.add_argument("--buffer-size", type=int, default=2_000_000,
                   help="aggregated dataset size; the oldest samples are dropped beyond this")
    p.add_argument("--recent-rounds", type=int, default=10,
                   help="the newest this-many rounds of data form the 'recent' pool (see --recent-frac)")
    p.add_argument("--recent-frac", type=float, default=0.5,
                   help="fraction of each batch drawn from the recent pool, the rest from the whole dataset. Early "
                        "data is mostly the oracle's own (easy) states and otherwise dominates the loss")
    p.add_argument("--levels", type=float, nargs=2, metavar=("LO", "HI"), default=(0.0, 1.0),
                   help="curriculum levels the envs are spread over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--serial", action="store_true", help="run envs in the main process (debugging)")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--name", default=None)
    return p.parse_args()


def main():
    import torch
    import torch.nn as nn

    from ppo import ActorCritic

    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    env_cfg, ppo_cfg = EnvCfg(), PPOCfg(num_envs=args.num_envs)
    if env_cfg.grip_mode != "weld" or env_cfg.action_mode != "head_twist":
        raise SystemExit("the oracle needs grip_mode 'weld' and action_mode 'head_twist'")
    K, N, T = args.eval_envs, args.num_envs, args.rollout_steps
    if not 0 <= K < N:
        raise SystemExit(f"--eval-envs must be in [0, num_envs), got {K}")
    n_train = N - K
    noise_std = args.noise_std if args.noise_std is not None else float(np.exp(ppo_cfg.init_log_std))

    run_name = args.name or time.strftime("bc_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir)
        writer.add_text("args", "  \n".join(f"{k}: {v}" for k, v in vars(args).items()))
    except Exception:
        writer = None

    lo, hi = args.levels
    spread = lambda k: list(np.linspace(lo, hi, k)) if k > 1 else [hi] * k  # noqa: E731
    levels = spread(n_train) + spread(K)
    fns = [make_env(dataclasses.replace(env_cfg, curriculum_level=float(lv)), args.seed * 1000 + i)
           for i, lv in enumerate(levels)]
    venv = SerialVecEnv(fns) if (args.serial or N == 1) else SubprocVecEnv(fns)
    obs_dim, act_dim, priv_dim = venv.observation_space.shape[0], venv.action_space.shape[0], venv.priv_dim
    print(f"device={device}  num_envs={N} ({K} eval)  levels {lo:.2f}-{hi:.2f}  noise std {noise_std:.3f}  "
          f"run_dir={run_dir}")

    agent = ActorCritic(obs_dim, priv_dim, act_dim, ppo_cfg).to(device)
    opt = torch.optim.Adam(agent.actor.parameters(), lr=args.lr)
    f32 = dict(device=device, dtype=torch.float32)

    cap = args.buffer_size
    data_obs = np.zeros((cap, obs_dim), np.float32)
    data_act = np.zeros((cap, act_dim), np.float32)
    size, ptr = 0, 0
    ep_stats, eval_stats = deque(maxlen=200), deque(maxlen=100)

    def mse(o: np.ndarray, a: np.ndarray) -> float:
        if len(o) == 0:
            return float("nan")
        with torch.no_grad():
            pred = agent.actor(agent.obs_rms(torch.as_tensor(o, **f32)))
            return float(((pred - torch.as_tensor(a, **f32)) ** 2).mean())

    obs_np, priv_np = venv.reset(seeds=[args.seed * 1000 + i for i in range(N)])
    t_start = time.time()
    for rnd in range(args.rounds):
        t_round = time.time()
        beta = args.beta_decay ** rnd
        lr = args.lr + (args.lr_final - args.lr) * rnd / max(1, args.rounds - 1)
        for group in opt.param_groups:
            group["lr"] = lr
        r_obs, r_priv, r_act = [], [], []
        for _ in range(T):
            expert = np.stack(venv.call("oracle_action"))
            with torch.no_grad():
                learner = agent.act_deterministic(torch.as_tensor(obs_np, **f32)).cpu().numpy()
            use_expert = rng.random(N) < beta
            use_expert[n_train:] = False
            act = np.where(use_expert[:, None], expert, learner)
            act[:n_train] += rng.normal(0.0, noise_std, (n_train, act_dim))
            r_obs.append(obs_np)
            r_priv.append(priv_np)
            r_act.append(expert)
            obs_np, priv_np, _, _, _, eps = venv.step(np.clip(act, -1.0, 1.0))
            for i, e in enumerate(eps):
                if e is not None:
                    (eval_stats if i >= n_train else ep_stats).append(e)

        new_obs = np.concatenate(r_obs)
        new_act = np.concatenate(r_act)
        # Loss on this round's states before training on them: how well the policy matches the oracle where it
        # actually goes. The training loss is averaged over all the data and hides this. Rows are step-major.
        is_eval = (np.arange(len(new_obs)) % N) >= n_train
        loss_new, loss_new_eval = mse(new_obs, new_act), mse(new_obs[is_eval], new_act[is_eval])
        idx = (ptr + np.arange(len(new_obs))) % cap
        data_obs[idx], data_act[idx] = new_obs, new_act
        ptr, size = (ptr + len(new_obs)) % cap, min(cap, size + len(new_obs))
        # The critic is not trained here (train.py warms it up), but its priv normalizer is fitted.
        agent.obs_rms.update(torch.as_tensor(new_obs, **f32))
        agent.priv_rms.update(torch.as_tensor(np.concatenate(r_priv), **f32))

        losses = []
        n_recent = int(round(args.recent_frac * args.batch_size))
        pool = min(size, args.recent_rounds * N * T)
        for _ in range(args.grad_steps):
            b = np.concatenate([(ptr - 1 - rng.integers(0, pool, n_recent)) % cap,
                                rng.integers(0, size, args.batch_size - n_recent)])
            o, a = torch.as_tensor(data_obs[b], **f32), torch.as_tensor(data_act[b], **f32)
            loss = ((agent.actor(agent.obs_rms(o)) - a) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), ppo_cfg.max_grad_norm)
            opt.step()
            losses.append(float(loss.detach()))

        mean = lambda stats, k: float(np.mean([e[k] for e in stats])) if stats else float("nan")  # noqa: E731
        side = [e["pass"]["v_side_frac"] for e in eval_stats if e.get("pass")]
        log = {"bc/loss": float(np.mean(losses)), "bc/loss_new": loss_new, "bc/loss_new_eval": loss_new_eval,
               "bc/lr": lr, "bc/beta": beta, "bc/dataset": size,
               "ep/success": mean(ep_stats, "success"),
               "eval/success": mean(eval_stats, "success"), "eval/passed": mean(eval_stats, "passed"),
               "eval/pass/v_side_frac": float(np.mean(side)) if side else float("nan")}
        if writer is not None:
            for k, v in log.items():
                writer.add_scalar(k, v, rnd)
        print(f"round {rnd:3d} | beta {beta:.3f} | data {size:8d} | loss {log['bc/loss']:.5f} new {loss_new:.5f} | "
              f"mix succ {log['ep/success']:.3f} | eval succ {log['eval/success']:.3f} "
              f"pass {log['eval/passed']:.3f} "
              f"side {log['eval/pass/v_side_frac']:.2f} (last {len(eval_stats)}) | {time.time() - t_round:.0f}s")

        # Same keys train.py and evaluate.py read from a PPO checkpoint (no optimizer: --init-from starts fresh).
        torch.save({"format": CKPT_FORMAT, "agent": agent.state_dict(), "env_cfg": env_cfg, "ppo_cfg": ppo_cfg,
                    "obs_dim": obs_dim, "priv_dim": priv_dim, "act_dim": act_dim, "level": hi,
                    "bc_round": rnd, "bc_args": vars(args)}, os.path.join(run_dir, "bc_latest.pt"))

    venv.close()
    if writer is not None:
        writer.close()
    print(f"saved {os.path.join(run_dir, 'bc_latest.pt')}  done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
