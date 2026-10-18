"""Train the asymmetric PPO agent on PoseVelocityEnv.

Simulation always runs on CPU worker processes; the networks and PPO update run on --device.

Examples:
  python train.py --device cpu --num-envs 4 --total-steps 20000      # smoke test
  python train.py --device auto --num-envs 16                         # real run
  python train.py --resume runs/<name>/ckpt_latest.pt
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict, deque

import numpy as np

from config import EnvCfg, PPOCfg
from env import make_env
from vec_env import SerialVecEnv, SubprocVecEnv

# torch / ppo are imported inside the functions that use them: env workers are started with "spawn",
# which re-imports this module in every worker, and a top-level torch import costs ~350 MB per worker.


class Curriculum:
    def __init__(self, cfg, level: float):
        self.cfg, self.level = cfg, level
        self.window = deque(maxlen=cfg.window)

    def record(self, success: float) -> None:
        self.window.append(success)

    def update(self) -> bool:
        if len(self.window) < self.cfg.min_episodes:
            return False
        rate = float(np.mean(self.window))
        old = self.level
        if rate > self.cfg.up_threshold:
            self.level = min(1.0, self.level + self.cfg.step)
        elif rate < self.cfg.down_threshold:
            self.level = max(0.0, self.level - self.cfg.step)
        if self.level != old:
            self.window.clear()
            return True
        return False


def episode_diagnostics(ep_stats, cc, level: float) -> dict:
    """Breakdowns that show where the policy is stuck.

    by_start/<type>/*  episodes split by how they started: "soft" (continuing with the tool held after the
                       previous goal), "in_hand" (scripted grasp at respawn) or "table" (must pick the tool up)
    closest/*          error components at each episode's closest approach (min phase distance) while holding,
                       and the fraction of episodes whose closest approach met each tolerance individually
    """
    out = {}
    n = len(ep_stats)
    for start in ("soft", "in_hand", "table"):
        eps = [e for e in ep_stats if e.get("start") == start]
        out[f"by_start/{start}/frac"] = len(eps) / n
        if eps:
            for k in ("success", "dropped", "ever_grasped"):
                out[f"by_start/{start}/{k}"] = float(np.mean([e[k] for e in eps]))

    closest = [e["closest"] for e in ep_stats if e.get("closest")]
    if closest:
        lerp = lambda pair: pair[0] + (pair[1] - pair[0]) * level  # noqa: E731
        eps_p, eps_v_frac, eps_th = lerp(cc.eps_p), lerp(cc.eps_v_frac), lerp(cc.eps_theta_deg)
        dp = np.array([c["dp"] for c in closest])
        dv_frac = np.array([c["dv_frac"] for c in closest])
        th = np.array([c["theta_deg"] for c in closest])
        out["closest/dp_cm"] = float(np.mean(dp) * 100)
        out["closest/dv_mps"] = float(np.mean([c["dv"] for c in closest]))
        out["closest/dv_frac"] = float(np.mean(dv_frac))
        out["closest/theta_deg"] = float(np.mean(th))
        out["closest/ok_pos"] = float(np.mean(dp < eps_p))
        out["closest/ok_vel"] = float(np.mean(dv_frac < eps_v_frac))
        out["closest/ok_axis"] = float(np.mean(th < eps_th))
    return out


def resolve_device(name: str) -> torch.device:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=PPOCfg.num_envs)
    p.add_argument("--rollout-steps", type=int, default=PPOCfg.rollout_steps)
    p.add_argument("--total-steps", type=int, default=PPOCfg.total_steps)
    p.add_argument("--lr", type=float, default=PPOCfg.lr)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", type=float, default=0.0, help="initial curriculum level")
    p.add_argument("--fixed-level", action="store_true", help="disable automatic curriculum")
    p.add_argument("--serial", action="store_true", help="run envs in the main process (debugging)")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=PPOCfg.save_every)
    # Overrides applied on top of the config (also on --resume, where the checkpoint's config is used otherwise).
    p.add_argument("--eps-p", type=float, nargs=2, metavar=("LEVEL0", "LEVEL1"), default=None,
                   help="success position tolerance in meters at curriculum level 0 and 1, e.g. --eps-p 0.08 0.01")
    p.add_argument("--sigma-p", type=float, default=None,
                   help="position scale (m) of the phase distance D used by the goal reward, e.g. 0.02")
    return p.parse_args()


def save_checkpoint(path, agent, ppo, scaler, env_cfg, ppo_cfg, curriculum, ep_stats, global_step, iteration,
                    best):
    import torch

    torch.save({
        "best": best,
        "agent": agent.state_dict(), "optimizer": ppo.opt.state_dict(), "lr": ppo.lr,
        "return_scaler": scaler.state_dict(), "env_cfg": env_cfg, "ppo_cfg": ppo_cfg,
        "obs_dim": agent.obs_rms.mean.shape[0], "priv_dim": agent.priv_rms.mean.shape[0],
        "act_dim": agent.log_std.shape[0], "level": curriculum.level,
        # Recent-episode windows: without them a resumed run reports (and lets the curriculum react to)
        # stats from only the first few, mostly-dropped episodes after the synchronized full reset.
        "curriculum_window": list(curriculum.window), "ep_stats": list(ep_stats),
        "global_step": global_step, "iteration": iteration,
    }, path)


def main():
    import torch

    from ppo import PPO, ActorCritic, ReturnScaler, compute_gae

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    env_cfg, ppo_cfg = EnvCfg(), PPOCfg(num_envs=args.num_envs, rollout_steps=args.rollout_steps,
                                        total_steps=args.total_steps, lr=args.lr, save_every=args.save_every)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        env_cfg = ckpt["env_cfg"]
    if args.eps_p is not None:
        env_cfg.curriculum.eps_p = tuple(args.eps_p)
    if args.sigma_p is not None:
        env_cfg.reward.sigma_p = args.sigma_p
    print(f"eps_p (level 0 -> 1) = {env_cfg.curriculum.eps_p}  sigma_p = {env_cfg.reward.sigma_p}")

    if abs(env_cfg.reward.shaping_gamma - ppo_cfg.gamma) > 1e-9:
        print(f"warning: RewardCfg.shaping_gamma={env_cfg.reward.shaping_gamma} != PPOCfg.gamma={ppo_cfg.gamma}; "
              "grasp shaping will no longer be policy-invariant")

    run_name = args.name or time.strftime("posevel_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir)
    except Exception:
        writer = None
        print("tensorboard not available; logging to stdout only")

    print(f"device={device}  num_envs={args.num_envs}  run_dir={run_dir}")
    fns = [make_env(env_cfg, args.seed * 1000 + i) for i in range(args.num_envs)]
    venv = SerialVecEnv(fns) if (args.serial or args.num_envs == 1) else SubprocVecEnv(fns)
    obs_dim = venv.observation_space.shape[0]
    act_dim = venv.action_space.shape[0]
    priv_dim = venv.priv_dim
    print(f"obs_dim={obs_dim} priv_dim={priv_dim} act_dim={act_dim}")

    agent = ActorCritic(obs_dim, priv_dim, act_dim, ppo_cfg).to(device)
    ppo = PPO(agent, ppo_cfg)
    scaler = ReturnScaler(args.num_envs, ppo_cfg.gamma)
    curriculum = Curriculum(env_cfg.curriculum, args.level)
    global_step, start_iter = 0, 0
    # ckpt_best.pt is ranked by (curriculum level, success over the last ep_stats.maxlen episodes): a policy
    # that has advanced the curriculum beats any at a lower level. Only full windows count.
    best_path = os.path.join(run_dir, "ckpt_best.pt")
    best = None
    if ckpt is not None and os.path.exists(best_path):
        best = ckpt.get("best")  # only meaningful if that best checkpoint is in this run dir
    if ckpt is not None:
        agent.load_state_dict(ckpt["agent"])
        ppo.opt.load_state_dict(ckpt["optimizer"])
        ppo.lr = ckpt["lr"]
        scaler.load_state_dict(ckpt["return_scaler"])
        curriculum.level = ckpt["level"]
        global_step, start_iter = ckpt["global_step"], ckpt["iteration"] + 1
        print(f"resumed from {args.resume} at step {global_step}, level {curriculum.level:.2f}")
    venv.call("set_curriculum", curriculum.level)

    obs_np, priv_np = venv.reset(seeds=[args.seed * 1000 + i for i in range(args.num_envs)])
    T, N = ppo_cfg.rollout_steps, args.num_envs
    f32 = dict(device=device, dtype=torch.float32)
    buf = {
        "obs": torch.zeros((T, N, obs_dim), **f32), "priv": torch.zeros((T, N, priv_dim), **f32),
        "raw_obs": torch.zeros((T, N, obs_dim), **f32), "raw_priv": torch.zeros((T, N, priv_dim), **f32),
        "actions": torch.zeros((T, N, act_dim), **f32), "logp": torch.zeros((T, N), **f32),
        "values": torch.zeros((T, N), **f32), "rewards": torch.zeros((T, N), **f32),
        "dones": torch.zeros((T, N), **f32),
    }
    ep_stats = deque(maxlen=200)
    if ckpt is not None:
        # Older checkpoints predate these keys; they resume with empty windows as before.
        curriculum.window.extend(ckpt.get("curriculum_window", []))
        ep_stats.extend(ckpt.get("ep_stats", []))
    # Episodes finished since the last level change; ckpt_best only considers windows entirely at one level.
    eps_at_level = len(ep_stats)
    num_iters = ppo_cfg.total_steps // (T * N)
    t_start = time.time()

    for it in range(start_iter, num_iters):
        t_iter = time.time()
        for t in range(T):
            o = torch.as_tensor(obs_np, **f32)
            p = torch.as_tensor(priv_np, **f32)
            with torch.no_grad():
                no, npv = agent.normalize(o, p)
                dist = agent.dist(no)
                action = dist.sample()
                buf["logp"][t] = dist.log_prob(action).sum(-1)
                buf["values"][t] = agent.value(no, npv)
            buf["obs"][t], buf["priv"][t], buf["raw_obs"][t], buf["raw_priv"][t] = no, npv, o, p
            buf["actions"][t] = action

            obs_np, priv_np, rew, term, trunc, eps = venv.step(action.clamp(-1, 1).cpu().numpy())
            # After a soft reset (success or time-out with the tool still held) the physical state carries
            # on and the post-reset obs is the true next state, so keep bootstrapping through the boundary.
            # Only drops / respawns / divergence cut the return.
            continues = np.array([e is not None and e.get("reset_kind") == "soft" for e in eps])
            cut = (term | trunc) & ~continues
            r = scaler(rew.astype(np.float64), cut) if ppo_cfg.scale_rewards else rew
            buf["rewards"][t] = torch.as_tensor(r, **f32)
            buf["dones"][t] = torch.as_tensor(cut, **f32)
            for e in eps:
                if e is not None:
                    ep_stats.append(e)
                    curriculum.record(e["success"])
                    eps_at_level += 1
        global_step += T * N

        with torch.no_grad():
            no, npv = agent.normalize(torch.as_tensor(obs_np, **f32), torch.as_tensor(priv_np, **f32))
            last_value = agent.value(no, npv)
        adv, returns = compute_gae(buf["rewards"], buf["values"], buf["dones"], last_value,
                                   ppo_cfg.gamma, ppo_cfg.lam)
        batch = {k: buf[k].reshape(T * N, *buf[k].shape[2:]) for k in ("obs", "priv", "actions", "logp", "values")}
        batch["adv"], batch["returns"] = adv.reshape(-1), returns.reshape(-1)
        train_stats = ppo.update(batch)
        # Update normalizers after the update so stored (normalized) rollouts stay consistent.
        agent.obs_rms.update(buf["raw_obs"])
        agent.priv_rms.update(buf["raw_priv"])

        if not args.fixed_level and curriculum.update():
            venv.call("set_curriculum", curriculum.level)
            eps_at_level = 0
            print(f"  curriculum level -> {curriculum.level:.2f}")

        sps = T * N / (time.time() - t_iter)
        log = dict(train_stats)
        log["sps"] = sps
        log["level"] = curriculum.level
        if ep_stats:
            for k in ("return", "length", "success", "dropped", "ever_grasped", "diverged"):
                log[f"ep/{k}"] = float(np.mean([e[k] for e in ep_stats]))
            d_min = [e["d_min"] for e in ep_stats if np.isfinite(e["d_min"])]
            if d_min:
                log["ep/d_min"] = float(np.mean(d_min))
            terms = defaultdict(list)
            for e in ep_stats:
                for k, v in e["terms"].items():
                    terms[k].append(v)
            for k, v in terms.items():
                log[f"rew/{k}"] = float(np.mean(v))
            log.update(episode_diagnostics(ep_stats, env_cfg.curriculum, curriculum.level))
        if writer is not None:
            for k, v in log.items():
                writer.add_scalar(k, v, global_step)
        print(f"it {it:5d} step {global_step:10d} | sps {sps:7.0f} | lvl {curriculum.level:.2f} | "
              f"ret {log.get('ep/return', float('nan')):8.2f} | succ {log.get('ep/success', float('nan')):.3f} | "
              f"drop {log.get('ep/dropped', float('nan')):.3f} | grasp {log.get('ep/ever_grasped', float('nan')):.3f} | "
              f"kl {train_stats['final_kl']:.4f} ({train_stats['steps']} steps) lr {train_stats['lr']:.1e} "
              f"std {train_stats['std']:.3f}")
        if it % 10 == 0 and "closest/dp_cm" in log:
            g = lambda k: log.get(k, float("nan"))  # noqa: E731
            print(f"  succ by start: soft {g('by_start/soft/success'):.2f} in_hand {g('by_start/in_hand/success'):.2f} "
                  f"table {g('by_start/table/success'):.2f} (table grasp {g('by_start/table/ever_grasped'):.2f}) | "
                  f"closest: {g('closest/dp_cm'):.1f}cm {g('closest/dv_frac'):.2f}xv {g('closest/theta_deg'):.0f}deg | "
                  f"ok pos {g('closest/ok_pos'):.2f} vel {g('closest/ok_vel'):.2f} axis {g('closest/ok_axis'):.2f}")

        if eps_at_level >= ep_stats.maxlen:
            score = (curriculum.level, log["ep/success"])
            if best is None or score > tuple(best):
                best = score
                save_checkpoint(best_path, agent, ppo, scaler, env_cfg, ppo_cfg, curriculum, ep_stats,
                                global_step, it, best)
                print(f"  new best: level {score[0]:.2f} success {score[1]:.3f} -> saved {best_path}")

        if (it + 1) % ppo_cfg.save_every == 0 or it == num_iters - 1:
            path = os.path.join(run_dir, "ckpt_latest.pt")
            save_checkpoint(path, agent, ppo, scaler, env_cfg, ppo_cfg, curriculum, ep_stats, global_step, it, best)
            print(f"  saved {path}")

    if num_iters <= start_iter:
        print("total-steps too small for a single iteration; nothing trained")
    venv.close()
    if writer is not None:
        writer.close()
    print(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
