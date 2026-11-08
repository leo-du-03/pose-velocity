"""Evaluate a policy on the tasks in tasks.py: hammer a plate, hammer a nail, sweep dirt off a board, slice a loaf or a
cherry tomato.

Examples:
  python evaluate_tasks.py --task plate --policy trace --episodes 20     # reference follower: the trace's own accuracy
  python evaluate_tasks.py --task sweep --policy trace --episodes 2 --video sweep.mp4
  python evaluate_tasks.py --task nail --ckpt runs/<name>/ckpt_best.pt --episodes 50
  python evaluate_tasks.py --task plate --ckpt runs/<name>/ckpt_best.pt --layout realistic   # out of distribution
  python evaluate_tasks.py --task slice --ckpt runs/<name>/ckpt_best.pt --episodes 20 --video slice.mp4
  python evaluate_tasks.py --task slice_tomato --ckpt runs/<name>/ckpt_best.pt --episodes 20
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
              ("force_err", "err", "{:+6.1%}", 1), ("contact_speed", "v", "{:5.2f}m/s", 1),
              ("follower_contact_speed", "v*", "{:4.2f}m/s", 1), ("speed_err", "v err", "{:+6.1%}", 1),
              ("contact_err", "pos", "{:4.1f}cm", 100), ("contact_theta_deg", "rot", "{:3.0f}deg", 1)],
    "nail": [("force_target", "F*", "{:6.1f}N", 1), ("peak_force", "F", "{:6.1f}N", 1),
             ("force_err", "err", "{:+6.1%}", 1), ("contact_speed", "v", "{:4.2f}m/s", 1),
             ("follower_contact_speed", "v*", "{:4.2f}m/s", 1), ("speed_err", "v err", "{:+6.1%}", 1),
             ("contact_err", "off-axis", "{:4.1f}cm", 100),
             ("nail_driven", "driven", "{:4.1f}mm", 1000), ("hit", "hit", "{}", None)],
    "sweep": [("cleaned", "cleaned", "{:5.1%}", 1), ("cleaned_l1", "light", "{:5.1%}", 1),
              ("cleaned_l2", "medium", "{:5.1%}", 1), ("cleaned_l3", "heavy", "{:5.1%}", 1),
              ("force_deficit", "deficit", "{:5.1%}", 1), ("untouched_cells", "untouched", "{}", None),
              ("unscrubbed_cells", "unscrubbed", "{}", None)],
    "slice": [("cut_frac", "cut", "{:5.1%}", 1), ("line_offset", "off-line", "{:4.1f}mm", 1000),
              ("bind_deg_max", "twist", "{:4.1f}deg", 1), ("bind_offset_max", "drift", "{:4.1f}mm", 1000),
              ("crush_depth_max", "pressed", "{:4.1f}mm", 1000), ("slice_ratio_mean", "slice ratio", "{:4.1f}", 1),
              ("board_force", "board", "{:5.1f}N", 1), ("fail_reason", "fail", "{}", None)],
    "slice_tomato": [("cut_frac", "cut", "{:5.1%}", 1), ("force_cut_mean", "force", "{:4.1f}N", 1),
                     ("force_max", "peak", "{:4.1f}N", 1), ("line_offset", "off-center", "{:4.1f}mm", 1000),
                     ("bind_deg_max", "twist", "{:4.1f}deg", 1), ("board_force", "board", "{:5.1f}N", 1),
                     ("fail_reason", "fail", "{}", None)],
}
# Slice failure modes, in the order the summary lists them (metrics fail_reason).
SLICE_FAILS = {
    "slice": (("crushed", "crushed (pushed without slicing: velocity)"),
              ("jammed", "jammed (blade twisted or moved sideways in the cut: orientation / position)"),
              ("off_line", "off the line (cut too far from the mark: position)"),
              ("board", "chopped into the board (too deep: position)"),
              ("not_through", "not through (too shallow, or ran out of time)")),
    "slice_tomato": (("crushed", "crushed (too much force: pressed down too steeply for the sliding, or not sliding)"),
                     ("not_through", "not through (too little force: the edge didn't press deep enough)"),
                     ("jammed", "jammed (blade twisted or moved sideways in the cut: orientation / position)"),
                     ("off_line", "off center (cut too far from the middle: position)"),
                     ("board", "chopped into the board (too deep: position)")),
}
# Metrics summarized over episodes, each with the unit to print it in (metrics are stored as fractions and meters)
SUMMARY = {"plate": [("force_err", "%"), ("force_diff", "N"), ("speed_err", "%"), ("speed_diff", "m/s"),
                     ("contact_speed", "m/s"), ("contact_err", "cm")],
           "nail": [("force_err", "%"), ("force_diff", "N"), ("speed_err", "%"), ("speed_diff", "m/s"),
                    ("contact_speed", "m/s"), ("contact_err", "cm"), ("nail_driven", "mm")],
           "sweep": [("cleaned", "%"), ("cleaned_l1", "%"), ("cleaned_l2", "%"), ("cleaned_l3", "%"),
                     ("force_deficit", "%")],
           "slice": [("cut_frac", "%"), ("line_offset", "mm"), ("bind_deg_max", "deg"), ("bind_offset_max", "mm"),
                     ("crush_depth_max", "mm"), ("slice_ratio_mean", ""), ("board_force", "N")],
           "slice_tomato": [("cut_frac", "%"), ("force_cut_mean", "N"), ("force_max", "N"), ("line_offset", "mm"),
                            ("bind_deg_max", "deg"), ("board_force", "N")]}
UNITS = {"%": 100, "cm": 100, "mm": 1000, "m/s": 1, "deg": 1, "N": 1, "": 1}


def _add_diffs(m: dict) -> None:
    """Hammer tasks: peak force minus target (N), and contact speed minus the reference follower's (m/s)."""
    if "force_target" in m and m["force_target"] > 0:
        m["force_diff"] = float(m["peak_force"] - m["force_target"])
    v, v_ref = m.get("contact_speed", np.nan), m.get("follower_contact_speed", np.nan)
    if np.isfinite(v) and np.isfinite(v_ref) and v_ref > 0:  # same episodes as speed_err
        m["speed_diff"] = float(v - v_ref)


def _fmt(m: dict, key: str, fmt: str, scale) -> str:
    v = m.get(key)
    if isinstance(v, str):
        return v
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
    p.add_argument("--swing-straight", type=float, default=None, metavar="M",
                   help="plate/nail: straight run-in (m) at the end of the swing, no rotation at impact "
                        "(TaskCfg.swing_straight; default from config)")
    p.add_argument("--face-crown", type=float, default=None, metavar="M",
                   help="plate/nail: crowned hammer face, bulging this much at its center (TaskCfg.face_crown)")
    p.add_argument("--sweep-rule", default=None, choices=["scrub", "force"],
                   help="sweep: how dirt comes off (TaskCfg.sweep_rule; default from config)")
    p.add_argument("--scrub-len", type=float, default=None, metavar="M",
                   help="sweep, scrub rule: sliding distance (m) needed at the adhesion force (TaskCfg.scrub_len)")
    p.add_argument("--dirt-forces", type=float, nargs="+", default=None, metavar="N",
                   help="sweep: adhesion force (N) of each dirt level, e.g. --dirt-forces 3 8 14 (TaskCfg.dirt_forces)")
    p.add_argument("--saw-speed", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="slice: range of the sawing speed along the blade, m/s (TaskCfg.saw_speed)")
    p.add_argument("--crush-depth", type=float, default=None, metavar="M",
                   help="slice: how far (m) the edge may press into uncut loaf without slicing (TaskCfg.crush_depth)")
    p.add_argument("--bind-deg", type=float, default=None, metavar="DEG",
                   help="slice: blade twist from the cut plane inside the loaf that jams it (TaskCfg.bind_deg)")
    p.add_argument("--bind-slack", type=float, default=None, metavar="M",
                   help="slice: sideways motion (m) of the blade inside the cut that jams it (TaskCfg.bind_slack)")
    p.add_argument("--line-tol", type=float, default=None, metavar="M",
                   help="slice: how far (m) from the marked line the cut may stray (TaskCfg.line_tol)")
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
    task_cfg = TaskCfg(layout=args.layout)
    if args.swing_straight is not None:
        task_cfg.swing_straight = args.swing_straight
    if args.face_crown is not None:
        task_cfg.face_crown = args.face_crown
    if args.sweep_rule is not None:
        task_cfg.sweep_rule = args.sweep_rule
    if args.scrub_len is not None:
        task_cfg.scrub_len = args.scrub_len
    if args.dirt_forces is not None:
        task_cfg.dirt_forces = tuple(args.dirt_forces)
    if args.saw_speed is not None:
        task_cfg.saw_speed = tuple(args.saw_speed)
    if args.crush_depth is not None:
        task_cfg.crush_depth = args.crush_depth
    if args.bind_deg is not None:
        task_cfg.bind_deg = args.bind_deg
    if args.bind_slack is not None:
        task_cfg.bind_slack = args.bind_slack
    if args.line_tol is not None:
        task_cfg.line_tol = args.line_tol
    env = TaskEnv(args.task, cfg, task_cfg, render_mode=render_mode)
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
        _add_diffs(m)
        results.append(m)
        cols = "  ".join(f"{label} {_fmt(m, key, fmt, scale)}" for key, label, fmt, scale in COLUMNS[args.task])
        end = "timeout" if m.get("timeout") else ("diverged" if m.get("diverged") else "done")
        if args.task in ("sweep", "slice", "slice_tomato"):
            kps = f"keypoints {m['keypoints_passed']}/{m['keypoints']}  "
        else:  # the hit itself rarely counts as a goal-plane crossing (the target stops the head)
            kps = f"windup goals {m['windup_passed']}/{m['windup_keypoints']}  "
        print(f"episode {ep:3d}: success {str(bool(m['success'])):5s}  {cols}  "
              f"{kps}path {_fmt(m, 'path_dev_mean', '{:4.1f}cm', 100)}  [{end}]")
        obs, _ = env.reset()

    print(f"\n{args.task} ({args.layout}, policy {policy}, level {level:.2f}): "
          f"success rate {np.mean([r['success'] for r in results]):.2f} over {len(results)} episodes")
    # mean|x| is the typical size of a signed error; mean is its bias. n < episodes when a metric is undefined
    # for some episodes (e.g. no contact).
    for key, unit in SUMMARY[args.task] + [("path_dev_mean", "cm")]:
        vals = np.array([r.get(key, np.nan) for r in results], dtype=np.float64)
        vals = vals[np.isfinite(vals)] * UNITS[unit]
        if len(vals):
            print(f"  {key:18s} mean {vals.mean():+8.2f}{unit:3s}  median {np.median(vals):+8.2f}{unit:3s}  "
                  f"std {vals.std():7.2f}{unit:3s}  mean|x| {np.abs(vals).mean():7.2f}{unit:3s}  (n={len(vals)})")
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
    if args.task in SLICE_FAILS:
        print("  failures:")
        for reason, label in SLICE_FAILS[args.task]:
            n = sum(r["fail_reason"] == reason for r in results)
            print(f"    {n:3d}  {label}")
        kp = np.array([r["keypoints_passed"] / max(r["keypoints"], 1) for r in results])
        print(f"  keypoints passed: mean {kp.mean():.1%}")
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
