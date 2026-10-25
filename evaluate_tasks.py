"""Evaluate a policy on the tasks in tasks.py: hammer a plate, hammer a nail, sweep dirt off a board.

Examples:
  python evaluate_tasks.py --task plate --policy trace --episodes 20     # reference follower: the trace's own accuracy
  python evaluate_tasks.py --task sweep --policy trace --episodes 2 --video sweep.mp4
  python evaluate_tasks.py --task nail --ckpt runs/<name>/ckpt_best.pt --episodes 50
  python evaluate_tasks.py --task plate --ckpt runs/<name>/ckpt_best.pt --layout realistic   # out of distribution
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from config import EnvCfg, RewardCfg, TaskCfg
from tasks import TASKS, TaskEnv

# Per-task columns of the episode line: (metric key, label, format, scale)
COLUMNS = {
    "plate": [("force_target", "F*", "{:6.1f}N", 1), ("peak_force", "F", "{:6.1f}N", 1),
              ("force_err", "err", "{:+6.1%}", 1), ("contact_speed", "v", "{:4.2f}m/s", 1),
              ("contact_err", "pos", "{:4.1f}cm", 100), ("contact_theta_deg", "rot", "{:3.0f}deg", 1)],
    "nail": [("force_target", "F*", "{:6.1f}N", 1), ("peak_force", "F", "{:6.1f}N", 1),
             ("force_err", "err", "{:+6.1%}", 1), ("contact_err", "off-axis", "{:4.1f}cm", 100),
             ("nail_driven", "driven", "{:4.1f}mm", 1000), ("hit", "hit", "{}", None)],
    "sweep": [("cleaned", "cleaned", "{:5.1%}", 1), ("cleaned_l1", "light", "{:5.1%}", 1),
              ("cleaned_l2", "medium", "{:5.1%}", 1), ("cleaned_l3", "heavy", "{:5.1%}", 1),
              ("force_deficit", "deficit", "{:5.1%}", 1), ("untouched_cells", "untouched", "{}", None)],
}
SUMMARY = {"plate": ["force_err", "contact_err", "contact_speed"], "nail": ["force_err", "contact_err", "nail_driven"],
           "sweep": ["cleaned", "cleaned_l1", "cleaned_l2", "cleaned_l3", "force_deficit"]}


def _fmt(m: dict, key: str, fmt: str, scale) -> str:
    v = m.get(key)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return fmt.format(v * scale if scale is not None else v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy", default=None, choices=["ckpt", "trace", "zero"],
                   help="defaults to ckpt if --ckpt is given, else trace (the reference follower)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--layout", default="in_dist", choices=["in_dist", "realistic"])
    p.add_argument("--level", type=float, default=None,
                   help="curriculum level, sets the tolerances in the observation (default: from checkpoint, else 1)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--render", default="none", choices=["none", "human", "rgb_array"])
    p.add_argument("--video", default=None, help="write an mp4 (implies --render rgb_array)")
    p.add_argument("--realtime", action="store_true", help="slow down to real time in the viewer")
    p.add_argument("--save-traces", default=None, metavar="DIR", help="write each episode's trace to DIR/ep<k>.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    policy = args.policy or ("ckpt" if args.ckpt else "trace")
    cfg, agent, level = EnvCfg(), None, 1.0
    if policy == "ckpt":
        if not args.ckpt:
            p.error("--policy ckpt needs --ckpt")
        import torch

        from ppo import ActorCritic

        ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        cfg, level = ck["env_cfg"], ck["level"]
        cfg.reward = RewardCfg()  # rewards don't affect evaluation; older checkpoints' RewardCfg lacks current fields
        agent = ActorCritic(ck["obs_dim"], ck["priv_dim"], ck["act_dim"], ck["ppo_cfg"]).to(args.device)
        agent.load_state_dict(ck["agent"])
        agent.eval()
    if args.level is not None:
        level = args.level

    render_mode = "rgb_array" if args.video else (None if args.render == "none" else args.render)
    env = TaskEnv(args.task, cfg, TaskCfg(layout=args.layout), render_mode=render_mode)
    env.set_curriculum(level)
    if agent is not None and env.observation_space.shape[0] != ck["obs_dim"]:
        raise SystemExit(f"checkpoint obs dim {ck['obs_dim']} != env obs dim {env.observation_space.shape[0]}")
    if args.save_traces:
        os.makedirs(args.save_traces, exist_ok=True)
    frames, dt = [], env.dt

    obs, _ = env.reset(seed=args.seed)
    results = []
    for ep in range(args.episodes):
        if args.save_traces:
            env.trace.save(os.path.join(args.save_traces, f"ep{ep}.npz"))
        if render_mode == "human":
            env.render()
        done = False
        while not done:
            t0 = time.time()
            if policy == "trace":
                action = env.trace_action()
            elif policy == "zero":
                action = np.zeros(env.action_space.shape)
            else:
                import torch

                o = torch.as_tensor(obs, dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    if args.stochastic:
                        action = agent.dist(agent.obs_rms(o)).sample().clamp(-1, 1)[0].cpu().numpy()
                    else:
                        action = agent.act_deterministic(o)[0].cpu().numpy()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if render_mode == "rgb_array":
                frames.append(env.render())
            if args.realtime and render_mode == "human":
                time.sleep(max(0.0, dt - (time.time() - t0)))
        m = info["task"]
        results.append(m)
        cols = "  ".join(f"{label} {_fmt(m, key, fmt, scale)}" for key, label, fmt, scale in COLUMNS[args.task])
        end = "timeout" if m.get("timeout") else ("diverged" if m.get("diverged") else "done")
        if args.task == "sweep":
            kps = f"keypoints {m['keypoints_passed']}/{m['keypoints']}  "
        else:  # the hit itself rarely counts as a goal-plane crossing (the target stops the head)
            kps = f"windup goals {m['windup_passed']}/{m['windup_keypoints']}  "
        print(f"episode {ep:3d}: success {str(bool(m['success'])):5s}  {cols}  "
              f"{kps}path {_fmt(m, 'path_dev_mean', '{:4.1f}cm', 100)}  [{end}]")
        obs, _ = env.reset()

    print(f"\n{args.task} ({args.layout}, policy {policy}, level {level:.2f}): "
          f"success rate {np.mean([r['success'] for r in results]):.2f} over {len(results)} episodes")
    for key in SUMMARY[args.task] + ["path_dev_mean"]:
        vals = np.array([r.get(key, np.nan) for r in results], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            print(f"  {key:18s} mean {vals.mean():+.4f}  std {vals.std():.4f}  |mean| {np.abs(vals).mean():.4f}")
    if args.task == "sweep":
        cell_cm2 = (env.task_cfg.dirt_cell * 100) ** 2
        print(f"  dirt per episode by level (cells of {cell_cm2:.0f} cm^2), and the share removed over all runs:")
        n_lv = len(env.task_cfg.dirt_forces)
        names = ("light", "medium", "heavy") if n_lv == 3 else tuple(f"level {i}" for i in range(1, n_lv + 1))
        for lv, name in enumerate(names, start=1):
            dirt = np.array([r[f"dirt_l{lv}"] for r in results], dtype=np.float64)
            removed = np.array([r[f"removed_l{lv}"] for r in results], dtype=np.float64)
            done = f"{removed.sum() / dirt.sum():6.1%}" if dirt.sum() > 0 else "     -"
            force = env.task_cfg.dirt_forces[lv - 1]
            print(f"    {name:6s} ({force:4.1f} N)  mean {dirt.mean():5.1f} cells ({dirt.mean() * cell_cm2:5.0f} cm^2)  "
                  f"std {dirt.std():5.1f}  removed {done}")
    if args.task in ("plate", "nail"):
        ff = np.array([r.get("follower_force_err", np.nan) for r in results], dtype=np.float64)
        print(f"  trace accuracy (reference follower force error): |mean| {np.nanmean(np.abs(ff)):.4f}")
    if args.video and frames:
        import imageio

        imageio.mimsave(args.video, frames, fps=int(round(1.0 / dt)))
        print(f"wrote {args.video} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
