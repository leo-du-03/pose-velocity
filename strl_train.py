"""Train the SimToolReal-style baseline (StrlEnv) with asymmetric PPO.

Simulation runs in CPU worker processes; the networks and PPO update run on --device.

Examples:
  python strl_train.py --num-envs 4 --total-steps 30000 --name strl_smoke     # smoke test, about a minute
  python strl_train.py --name strl_seed0 --seed 0                            # real run
  python strl_train.py --resume runs/strl_seed0/ckpt_latest.pt
  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from collections import Counter, defaultdict, deque

import numpy as np

from strl_config import StrlCfg, make_env_cfg, make_ppo_cfg
from strl_env import make_strl_env
from vec_env import SerialVecEnv, SubprocVecEnv

warnings.filterwarnings("ignore", message=".*Attach conflict.*")


def parse_args():
    d = make_ppo_cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-envs", type=int, default=d.num_envs)
    p.add_argument("--rollout-steps", type=int, default=d.rollout_steps)
    p.add_argument("--total-steps", type=int, default=d.total_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--torch-threads", type=int, default=6, help="CPU threads for the PPO update")
    p.add_argument("--lr-max", type=float, default=3e-4, help="upper bound for the KL-adaptive learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=None, help="initial success tolerance (default StrlCfg)")
    p.add_argument("--fixed-tol", action="store_true", help="disable the tolerance curriculum")
    p.add_argument("--tol-window", type=int, default=None,
                   help="episodes averaged by the tolerance curriculum (overrides StrlCfg / a resumed checkpoint)")
    p.add_argument("--dr", action="store_true", help="enable domain randomization (off by default)")
    p.add_argument("--serial", action="store_true", help="run envs in the main process (debugging)")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=d.save_every)
    return p.parse_args()


def resolve_device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def save_checkpoint(path, agent, ppo, scaler, env_cfg, scfg, ppo_cfg, tol, ep_window, global_step, it, best):
    import torch

    torch.save({
        "kind": "strl", "best": best, "agent": agent.state_dict(), "optimizer": ppo.state_dict(), "lr": ppo.lr,
        "return_scaler": scaler.state_dict(), "env_cfg": env_cfg, "strl_cfg": scfg, "ppo_cfg": ppo_cfg,
        "obs_dim": agent.obs_rms.mean.shape[0], "priv_dim": agent.priv_rms.mean.shape[0],
        "act_dim": agent.log_std.shape[0], "tol": tol, "ep_window": list(ep_window),
        "global_step": global_step, "iteration": it,
    }, path)


def main():
    import torch

    from ppo import PPO, ActorCritic, ReturnScaler, compute_gae

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cpu":
        # Rollouts use one thread (idle torch threads spin and take cores from the simulation workers); the update uses
        # several, since the workers wait while it runs (the vec env is synchronous).
        torch.set_num_threads(1)

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        scfg, env_cfg = ckpt["strl_cfg"], ckpt["env_cfg"]
    else:
        scfg = StrlCfg()
        scfg.dr = bool(args.dr)
        env_cfg = make_env_cfg(scfg)
    if args.tol_window is not None:
        scfg.tolerance_min_episodes = args.tol_window
    print(f"tolerance curriculum: mean of the last {scfg.tolerance_min_episodes} episodes "
          f">= {scfg.tolerance_success_threshold} goals")
    ppo_cfg = make_ppo_cfg(num_envs=args.num_envs, rollout_steps=args.rollout_steps, total_steps=args.total_steps,
                           lr=args.lr, epochs=args.epochs, save_every=args.save_every)

    run_name = args.name or time.strftime("strl_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir)
    except Exception:  # tensorboard missing: stdout only
        writer = None

    N, T = args.num_envs, args.rollout_steps
    seeds = [args.seed * 1000 + i for i in range(N)]
    fns = [make_strl_env(env_cfg, scfg, s) for s in seeds]
    venv = SerialVecEnv(fns) if (args.serial or N == 1) else SubprocVecEnv(fns)
    obs_dim, act_dim, priv_dim = venv.observation_space.shape[0], venv.action_space.shape[0], venv.priv_dim
    print(f"device={device} num_envs={N} rollout={T} dr={scfg.dr} run_dir={run_dir}")
    print(f"obs_dim={obs_dim} priv_dim={priv_dim} act_dim={act_dim}")

    agent = ActorCritic(obs_dim, priv_dim, act_dim, ppo_cfg).to(device)
    ppo = PPO(agent, ppo_cfg)
    scaler = ReturnScaler(N, ppo_cfg.gamma)
    tol = scfg.success_tolerance if args.tol is None else args.tol
    global_step, start_iter, best = 0, 0, None
    ep_window: deque = deque(maxlen=200)  # finished episodes (for logging and the best checkpoint)
    if ckpt is not None:
        agent.load_state_dict(ckpt["agent"])
        ppo.load_state_dict(ckpt["optimizer"])
        ppo.set_lr(ckpt["lr"])
        scaler.load_state_dict(ckpt["return_scaler"])
        tol = ckpt["tol"] if args.tol is None else args.tol
        global_step, start_iter, best = ckpt["global_step"], ckpt["iteration"] + 1, ckpt.get("best")
        ep_window.extend(ckpt.get("ep_window", []))
        print(f"resumed from {args.resume} at step {global_step}, tol {tol:.4f}")
    venv.call("set_tolerance", tol)
    curr_eps: list = []  # successes of episodes finished since the last tolerance change

    obs_np, priv_np = venv.reset(seeds=seeds)
    f32 = dict(device=device, dtype=torch.float32)
    buf = {k: torch.zeros((T, N, dim), **f32) for k, dim in
           (("obs", obs_dim), ("priv", priv_dim), ("raw_obs", obs_dim), ("raw_priv", priv_dim),
            ("actions", act_dim))}
    for k in ("logp", "values", "rewards", "dones"):
        buf[k] = torch.zeros((T, N), **f32)
    num_iters = ppo_cfg.total_steps // (T * N)
    t_start = time.time()

    for it in range(start_iter, num_iters):
        t_iter = time.time()
        for t in range(T):
            o, p = torch.as_tensor(obs_np, **f32), torch.as_tensor(priv_np, **f32)
            with torch.no_grad():
                no, npv = agent.normalize(o, p)
                dist = agent.dist(no)
                action = dist.sample()
                buf["logp"][t] = dist.log_prob(action).sum(-1)
                buf["values"][t] = agent.value(no, npv)
            buf["obs"][t], buf["priv"][t], buf["raw_obs"][t], buf["raw_priv"][t] = no, npv, o, p
            buf["actions"][t] = action

            obs_np, priv_np, rew, term, trunc, eps = venv.step(action.clamp(-1, 1).cpu().numpy())
            cut = term | trunc
            r = scaler(rew.astype(np.float64), cut) if ppo_cfg.scale_rewards else rew.astype(np.float64)
            # value_bootstrap (rl_games): on a time-out add gamma * V(s_t), since the return would have continued.
            r = r + ppo_cfg.gamma * buf["values"][t].cpu().numpy() * (trunc & ~term)
            buf["rewards"][t] = torch.as_tensor(r, **f32)
            buf["dones"][t] = torch.as_tensor(cut, **f32)
            for e in eps:
                if e is not None:
                    ep_window.append(e)
                    curr_eps.append(e["successes"])
        global_step += T * N
        t_roll = time.time() - t_iter

        with torch.no_grad():
            no, npv = agent.normalize(torch.as_tensor(obs_np, **f32), torch.as_tensor(priv_np, **f32))
            last_value = agent.value(no, npv)
        adv, returns = compute_gae(buf["rewards"], buf["values"], buf["dones"], last_value,
                                   ppo_cfg.gamma, ppo_cfg.lam)
        batch = {k: buf[k].reshape(T * N, *buf[k].shape[2:]) for k in ("obs", "priv", "actions", "logp", "values")}
        batch["adv"], batch["returns"] = adv.reshape(-1), returns.reshape(-1)
        if device.type == "cpu":
            torch.set_num_threads(args.torch_threads)
        stats = ppo.update(batch)
        if ppo.lr > args.lr_max:
            ppo.set_lr(args.lr_max)
            stats["lr"] = ppo.lr
        if device.type == "cpu":
            torch.set_num_threads(1)
        agent.obs_rms.update(buf["raw_obs"])
        agent.priv_rms.update(buf["raw_priv"])
        t_upd = time.time() - t_iter - t_roll

        # Success-tolerance curriculum: tighten when episodes reach enough goals on average.
        if not args.fixed_tol and len(curr_eps) >= scfg.tolerance_min_episodes:
            if np.mean(curr_eps) >= scfg.tolerance_success_threshold and tol > scfg.target_success_tolerance:
                tol = max(tol * scfg.tolerance_increment, scfg.target_success_tolerance)
                venv.call("set_tolerance", tol)
                print(f"  tolerance -> {tol:.4f} m (threshold {tol * scfg.keypoint_scale * 100:.1f} cm)")
            curr_eps = []

        sps = T * N / (time.time() - t_iter)
        log = dict(stats)
        log.update({"sps": sps, "tol": tol, "time/rollout_s": t_roll, "time/update_s": t_upd})
        if ep_window:
            for k in ("return", "length", "successes", "d_last", "d_min"):
                vals = [e[k] for e in ep_window if np.isfinite(e[k])]
                if vals:
                    log[f"ep/{k}"] = float(np.mean(vals))
            reasons = Counter(e["reason"] for e in ep_window)
            # Each episode reaches its goals in a row and ends on the first failure (unless it hits the cap), so the
            # per-goal success rate is reached / (reached + failed attempts).
            reached = sum(e["successes"] for e in ep_window)
            failed = len(ep_window) - reasons.get("max_successes", 0)
            log["ep/goal_success"] = reached / max(1.0, reached + failed)
            for k in ("timeout", "max_successes", "table_force", "diverged"):
                log[f"end/{k}"] = reasons.get(k, 0) / len(ep_window)
            terms = defaultdict(list)
            for e in ep_window:
                for k, v in e["terms"].items():
                    terms[k].append(v)
            for k, v in terms.items():
                log[f"rew/{k}"] = float(np.mean(v))
        if writer is not None:
            for k, v in log.items():
                writer.add_scalar(k, v, global_step)
        print(f"it {it:5d} step {global_step:10d} | sps {sps:6.0f} | tol {tol * 100:5.2f}cm | "
              f"goals/ep {log.get('ep/successes', float('nan')):5.2f} goal succ {log.get('ep/goal_success', float('nan')):4.0%} | len {log.get('ep/length', float('nan')):5.0f} | "
              f"d_min {log.get('ep/d_min', float('nan')) * 100:5.1f}cm last {log.get('ep/d_last', float('nan')) * 100:5.1f}cm | "
              f"kl {stats['final_kl']:.4f} ({stats['steps']}) lr {stats['lr']:.1e} std {stats['std']:.3f}")
        if it % 10 == 0 and ep_window:
            print(f"  ends: timeout {log['end/timeout']:.2f} max {log['end/max_successes']:.2f} "
                  f"table {log['end/table_force']:.2f} diverged {log['end/diverged']:.2f} | "
                  f"rew kp {log.get('rew/keypoint', 0):.1f} bonus {log.get('rew/bonus', 0):.1f} "
                  f"smooth {log.get('rew/smooth', 0):.1f} | time: rollout {t_roll:.2f}s update {t_upd:.2f}s")

        # Best checkpoint: smallest tolerance first, then goals per episode (window of 200 finished episodes).
        if len(ep_window) == ep_window.maxlen:
            score = (-round(tol, 6), log["ep/successes"])
            if best is None or score > tuple(best):
                best = score
                save_checkpoint(os.path.join(run_dir, "ckpt_best.pt"), agent, ppo, scaler, env_cfg, scfg, ppo_cfg,
                                tol, ep_window, global_step, it, best)
        if (it + 1) % ppo_cfg.save_every == 0 or it == num_iters - 1:
            save_checkpoint(os.path.join(run_dir, "ckpt_latest.pt"), agent, ppo, scaler, env_cfg, scfg, ppo_cfg,
                            tol, ep_window, global_step, it, best)
            print(f"  saved {run_dir}/ckpt_latest.pt")

    venv.close()
    if writer is not None:
        writer.close()
    print(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
