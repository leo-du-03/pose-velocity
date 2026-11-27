"""Roll out a policy in PoseVelocityEnv, optionally in the viewer or recorded to video.

Examples:
  python evaluate.py --policy random --render human            # watch random actions live
  python evaluate.py --policy zero --episodes 3 --video zero.mp4
  python evaluate.py --policy oracle --episodes 50             # scripted strike: checks goals are feasible
  python evaluate.py --ckpt runs/<name>/ckpt_latest.pt --render human --level 1.0
  python evaluate.py --ckpt runs/<name>/ckpt_best.pt --level 1 --episodes 500 --dump results/abl/<name>.json
  python evaluate.py --policy oracle --pose-only touch --pose-chain 4 --level 1 --episodes 200

The summary scores every policy by the same rule at the attempt (the first step within the keypoint tolerance), with
the evaluated level's tolerances: pose (head distance < eps_p, rotation < eps_theta), velocity
(|v - v_g| < eps_v_frac |v_g|) and both. Pose-only policies never observe v_g, but it is still sampled, so their
velocity at the attempt is what a pose goal gives for free.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from config import GOAL_MODES, EnvCfg, RewardCfg, TrajCfg, describe_goal


def _lerp(pair, level: float) -> float:
    return float(pair[0] + (pair[1] - pair[0]) * level)


def episode_row(e: dict, cfg: EnvCfg, level: float, eps_v_frac=None) -> dict:
    """One episode's outcome, scored by the same pose / velocity rule whatever the policy's goal."""
    cc, ps = cfg.curriculum, e["pass"]
    row = {k: float(e[k]) for k in ("success", "passed", "pushed", "ik_failed", "dropped", "diverged")}
    row["length"] = int(e["length"])
    row["attempt"] = bool(ps)
    row.update({k: float(v) for k, v in e.get("motion", {}).items()})
    row.update({k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in ps.items()})
    if ps:
        pose_ok = ps["dp"] < _lerp(cc.eps_p, level) and ps["theta_deg"] < _lerp(cc.eps_theta_deg, level)
        vel_ok = ps["dv_frac"] < _lerp(eps_v_frac if eps_v_frac is not None else cc.eps_v_frac, level)
        row.update(pose_ok=bool(pose_ok), vel_ok=bool(vel_ok), joint_ok=bool(pose_ok and vel_ok))
    else:
        row.update(pose_ok=False, vel_ok=False, joint_ok=False)
    return row


def print_summary(rows: list) -> None:
    att = [r for r in rows if r["attempt"]]
    print(f"attempt rate {len(att) / max(len(rows), 1):.3f}   over all episodes: "
          + "  ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in ("pose_ok", "vel_ok", "joint_ok")))
    if not att:
        return
    print("at the attempt (median [25%, 75%]):")
    for k, scale, unit in (("dp", 100, "cm"), ("theta_deg", 1, "deg"), ("dv_frac", 1, "x|v_g|"),
                           ("v_along_frac", 1, "x|v_g|"), ("v_side_frac", 1, "x|v_g|"), ("length", 1, "steps")):
        q = np.percentile([r[k] * scale for r in att], [50, 25, 75])
        print(f"  {k:13s} {q[0]:7.3f} [{q[1]:7.3f}, {q[2]:7.3f}] {unit}")
    if "qvel_l1" in rows[0]:
        print("motion, per-episode step means (median [25%, 75%] over all episodes):")
        for k, unit in (("qvel_l1", "rad/s"), ("qvel_peak", "rad/s (max)"), ("qacc", "rad/s^2"), ("da", "action")):
            q = np.percentile([r[k] for r in rows], [50, 25, 75])
            print(f"  {k:13s} {q[0]:7.3f} [{q[1]:7.3f}, {q[2]:7.3f}] {unit}")
from env import PoseVelocityEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy", default=None, choices=["ckpt", "random", "zero", "oracle"],
                   help="defaults to ckpt if --ckpt is given, else random")
    p.add_argument("--grip-mode", default=None, choices=["weld", "physical"],
                   help="override the grip mode (default: from checkpoint, else config.py)")
    p.add_argument("--action-mode", default=None, choices=["head_twist", "joint", "joint_vel"],
                   help="override the action mode (default: from checkpoint, else config.py)")
    p.add_argument("--arm-action-scale", type=float, default=None,
                   help="override EnvCfg.arm_action_scale (rad per step at action 1, joint mode)")
    p.add_argument("--joint-v-max", type=float, nargs=6, default=None, metavar="V",
                   help="override EnvCfg.joint_v_max (rad/s per arm joint at action 1, joint_vel mode)")
    p.add_argument("--no-kin-obs", action="store_true",
                   help="joint_vel: leave the Jacobian-projected goal errors out of the observation (EnvCfg.kin_obs)")
    p.add_argument("--pose-only", default=None, choices=["hold", "touch"],
                   help="without --ckpt: pose-only baseline (no goal velocity) with this success rule "
                        "(a checkpoint carries its own setting)")
    p.add_argument("--pose-chain", type=int, default=None, metavar="N",
                   help="pose-only touch (also a checkpoint's): timed pose sub-goals along the strike line "
                        "(EnvCfg.pose_chain)")
    p.add_argument("--vel-cone", type=float, nargs=2, default=None, metavar=("LEVEL0", "LEVEL1"),
                   help="override CurriculumCfg.vel_cone_deg, e.g. 30 180 to test a strikes-only checkpoint on swipes")
    p.add_argument("--goal-mode", default=None, choices=GOAL_MODES,
                   help="without --ckpt: 'points' or 'trajectory' (a reference into each goal; --policy oracle then "
                        "follows it). A checkpoint carries its own setting")
    p.add_argument("--score-eps-v-frac", type=float, nargs=2, default=None, metavar=("LEVEL0", "LEVEL1"),
                   help="velocity tolerance (fraction of goal speed) for vel_ok / joint_ok in the summary and dump, so "
                        "checkpoints trained with different ones are scored alike; the env and its observation keep "
                        "the checkpoint's")
    p.add_argument("--dump", default=None, metavar="PATH",
                   help="write every episode's outcome and attempt diagnostics to this JSON file")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", default="none", choices=["none", "human", "rgb_array"])
    p.add_argument("--video", default=None, help="write an mp4 (implies --render rgb_array)")
    p.add_argument("--level", type=float, default=None, help="curriculum level (default: from checkpoint, else 0)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--realtime", action="store_true", help="slow down to real time in the viewer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--table-start", action="store_true",
                   help="physical grip: every episode starts with a new tool on the table (no in-hand starts, "
                        "no soft resets); the arm still stays where the previous episode left it")
    args = p.parse_args()

    policy = args.policy or ("ckpt" if args.ckpt else "random")
    cfg, agent, level = EnvCfg(), None, 0.0
    if policy == "ckpt":
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
    if args.grip_mode is not None:
        cfg.grip_mode = args.grip_mode
    if args.action_mode is not None:
        cfg.action_mode = args.action_mode
    if args.arm_action_scale is not None:
        cfg.arm_action_scale = args.arm_action_scale
    if args.joint_v_max is not None:
        cfg.joint_v_max = tuple(args.joint_v_max)
    if args.no_kin_obs:
        cfg.kin_obs = False
    if args.table_start:
        if cfg.grip_mode != "physical":
            p.error("--table-start needs --grip-mode physical")
        cfg.curriculum.start_in_hand_prob = (0.0, 0.0)
    if args.pose_only is not None:
        if policy == "ckpt":
            p.error("--pose-only is taken from the checkpoint; drop it with --ckpt")
        cfg.goal_velocity, cfg.pose_success = False, args.pose_only
    if args.pose_chain is not None:
        cfg.pose_chain = args.pose_chain  # the observation layout doesn't change, so a checkpoint can take it
    if args.vel_cone is not None:
        cfg.curriculum.vel_cone_deg = tuple(args.vel_cone)
    if args.goal_mode is not None:
        if policy == "ckpt" and args.goal_mode != getattr(cfg, "goal_mode", "points"):
            p.error("--goal-mode is taken from the checkpoint (it changes the observation); drop it with --ckpt")
        cfg.goal_mode = args.goal_mode
        if not hasattr(cfg, "traj"):
            cfg.traj = TrajCfg()

    render_mode = "rgb_array" if args.video else (None if args.render == "none" else args.render)
    env = PoseVelocityEnv(cfg, render_mode=render_mode)
    env.set_curriculum(level)
    rng = np.random.default_rng(args.seed)
    frames = []
    dt = env.model.opt.timestep * cfg.frame_skip

    reset_opts = {"kind": "respawn"} if args.table_start else None
    obs, info = env.reset(seed=args.seed)
    results = []
    for ep in range(args.episodes):
        done, ep_ret = False, 0.0
        start_kind = info["reset_kind"]
        if render_mode == "human":
            env.render()
        while not done:
            t0 = time.time()
            if policy == "random":
                action = rng.uniform(-1, 1, env.action_space.shape)
            elif policy == "zero":
                action = np.zeros(env.action_space.shape)
            elif policy == "oracle":
                action = env.oracle_action()  # converted to joint actions in joint mode
            else:
                import torch

                o = torch.as_tensor(obs, dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    if args.stochastic:
                        action = agent.dist(agent.obs_rms(o)).sample().clamp(-1, 1)[0].cpu().numpy()
                    else:
                        action = agent.act_deterministic(o)[0].cpu().numpy()
            obs, rew, term, trunc, info = env.step(action)
            ep_ret += rew
            done = term or trunc
            if render_mode == "rgb_array":
                frames.append(env.render())
            if args.realtime and render_mode == "human":
                time.sleep(max(0.0, dt - (time.time() - t0)))
        e = info["episode"]
        results.append(e)
        ps = e["pass"]
        pass_txt = (f"attempt kp {ps['d_kp'] * 100:4.1f}cm head {ps['dp'] * 100:4.1f}cm v {ps['dv_frac']:.2f}xv "
                    f"(q_v {ps['q_v']:.2f}) rot {ps['theta_deg']:3.0f}deg" if ps else "no attempt")
        if env.pose_hold:
            pass_txt += f"  near steps {e['near_steps']}"
        if env.traj_mode:
            pass_txt += (f"  track {e['track_err'] * 100:4.1f}cm{'  LOST' if e['lost'] else ''}"
                         f"{'  ref-fallback' if e['ref_fallback'] else ''}")
        print(f"episode {ep}: start {start_kind:7s}  return {ep_ret:8.2f}  len {e['length']:4d}  "
              f"success {bool(e['success'])}  dropped {bool(e['dropped'])}  {pass_txt}  "
              f"d_min {e['d_min'] * 100:.1f}cm")
        obs, info = env.reset(options=reset_opts)

    print(f"\nsuccess rate {np.mean([r['success'] for r in results]):.2f}  "
          f"pass rate {np.mean([r['passed'] for r in results]):.2f}  "
          f"drop rate {np.mean([r['dropped'] for r in results]):.2f}  "
          f"mean return {np.mean([r['return'] for r in results]):.2f}")
    rows = [episode_row(r, cfg, level, args.score_eps_v_frac) for r in results]
    print_summary(rows)
    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        meta = {"ckpt": args.ckpt, "policy": policy, "level": level, "seed": args.seed, "goal": describe_goal(cfg),
                "action_mode": cfg.action_mode, "vel_cone_deg": list(cfg.curriculum.vel_cone_deg)}
        with open(args.dump, "w") as f:
            json.dump({"meta": meta, "episodes": rows}, f)
        print(f"wrote {args.dump}")
    if args.video and frames:
        import imageio

        imageio.mimsave(args.video, frames, fps=int(round(1.0 / dt)))
        print(f"wrote {args.video} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
