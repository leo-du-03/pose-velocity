"""Average reward earned up to the attempt: the first step within the keypoint tolerance (d_kp <= eps_kp).

That step is where touch and pose+velocity episodes end, while hold episodes keep going. Cutting every episode
there compares the policies on the same footing (rew/keypoint in training also counts hold's steps after it).
Episodes that never reach the tolerance are left out of the average and counted separately.

Examples:
  python eval_attempt_reward.py runs/strike_bc_pose_hold/ckpt_latest.pt runs/strike_bc_pose_touch/ckpt_latest.pt \
      runs/strike_vertical_kpvel_1cm/ckpt_latest.pt
  python eval_attempt_reward.py runs/*/ckpt_latest.pt --term all --episodes 500 --level 1.0
  python eval_attempt_reward.py runs/strike_bc_pose_hold/ckpt_latest.pt --term bonus --window step
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from collections import Counter

import numpy as np
import torch

from config import RewardCfg
from env import PoseVelocityEnv
from ppo import ActorCritic

TERMS = ("keypoint", "smooth", "bonus", "grasp", "drop", "total")  # grasp / drop: physical grip only


def load(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["env_cfg"]
    # Use the checkpoint's reward settings, on top of today's defaults (older checkpoints lack newer fields).
    rc = RewardCfg()
    old = getattr(cfg, "reward", None)
    if old is not None:
        for f in dataclasses.fields(RewardCfg):
            if hasattr(old, f.name):
                setattr(rc, f.name, getattr(old, f.name))
    cfg.reward = rc
    # Checkpoints saved before ik_fail_tol existed were trained without IK-failure terminations.
    if "ik_fail_tol" not in vars(cfg):
        cfg.ik_fail_tol = 0.0
    agent = ActorCritic(ck["obs_dim"], ck["priv_dim"], ck["act_dim"], ck["ppo_cfg"]).to(device)
    agent.load_state_dict(ck["agent"])
    agent.eval()
    return cfg, agent, ck["level"]


def evaluate(path: str, args) -> dict:
    cfg, agent, level = load(path, args.device)
    level = level if args.level is None else args.level
    if args.ik_fail_tol is not None:
        cfg.ik_fail_tol = args.ik_fail_tol
    env = PoseVelocityEnv(cfg)
    env.set_curriculum(level)
    obs, _ = env.reset(seed=args.seed)

    at_attempt = []  # per episode: {term: value} at the attempt
    no_attempt = Counter()  # why episodes ended without an attempt
    for _ in range(args.episodes):
        done, snap = False, None
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32, device=args.device)[None]
            with torch.no_grad():
                if args.stochastic:
                    action = agent.dist(agent.obs_rms(o)).sample().clamp(-1, 1)[0].cpu().numpy()
                else:
                    action = agent.act_deterministic(o)[0].cpu().numpy()
            before = dict(env.ep_terms)
            had_attempt = bool(env.pass_info)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            if snap is None and not had_attempt and env.pass_info:  # this step was the attempt
                after = dict(env.ep_terms)
                if args.window == "step":
                    snap = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
                else:
                    snap = after
                snap["total"] = float(sum(v for k, v in snap.items() if k != "total"))
        if snap is None:
            e = info["episode"]
            no_attempt[next((k for k in ("ik_failed", "pushed", "dropped", "diverged") if e.get(k)), "timeout")] += 1
        else:
            at_attempt.append(snap)
        obs, _ = env.reset()
    env.close()

    mode = "pose+velocity" if cfg.goal_velocity else f"pose-only {cfg.pose_success}"
    return {"path": path, "mode": mode, "level": level, "eps_kp": float(env.eps_kp()),
            "episodes": args.episodes, "no_attempt": no_attempt, "values": at_attempt}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="checkpoint files (ckpt_latest.pt / ckpt_best.pt)")
    p.add_argument("--term", default="keypoint", choices=TERMS + ("all",),
                   help="reward component to report; 'total' is their sum, 'all' prints every component")
    p.add_argument("--window", default="cumulative", choices=["cumulative", "step"],
                   help="cumulative: summed from the episode start through the attempt step; "
                        "step: the attempt step alone")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--level", type=float, default=None, help="curriculum level (default: from each checkpoint)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--seed", type=int, default=0, help="same seed for every checkpoint")
    p.add_argument("--ik-fail-tol", type=float, default=None,
                   help="override EnvCfg.ik_fail_tol (default: from the checkpoint; 0 for checkpoints saved before it existed)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    terms = [t for t in TERMS if t not in ("grasp", "drop")] if args.term == "all" else [args.term]
    results = [evaluate(c, args) for c in args.ckpts]
    if args.term == "all" and any("grasp" in v for r in results for v in r["values"]):
        terms = list(TERMS)

    name_w = max(len(os.path.basename(os.path.dirname(r["path"])) or r["path"]) for r in results)
    print(f"\nreward up to the attempt ({args.window}), mean +- std err over episodes with an attempt")
    header = f"{'run':{name_w}s}  {'mode':18s} {'level':>5s} {'eps_kp':>7s} {'attempts':>9s}  " + \
             "  ".join(f"{t:>17s}" for t in terms)
    print(header)
    print("-" * len(header))
    for r in results:
        name = os.path.basename(os.path.dirname(r["path"])) or r["path"]
        n = len(r["values"])
        cells = []
        for t in terms:
            v = np.array([e.get(t, 0.0) for e in r["values"]])
            cells.append(f"{v.mean():8.2f} +- {v.std(ddof=1) / np.sqrt(n):5.2f}" if n > 1 else f"{'n/a':>17s}")
        print(f"{name:{name_w}s}  {r['mode']:18s} {r['level']:5.2f} {r['eps_kp'] * 100:5.2f}cm "
              f"{n:4d}/{r['episodes']:<4d}  " + "  ".join(cells))
    for r in results:
        if r["no_attempt"]:
            name = os.path.basename(os.path.dirname(r["path"])) or r["path"]
            print(f"{name}: no attempt: " + ", ".join(f"{k} {v}" for k, v in r["no_attempt"].most_common()))


if __name__ == "__main__":
    main()
