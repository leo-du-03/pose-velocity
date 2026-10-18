"""Roll out a policy in PoseVelocityEnv, optionally in the viewer or recorded to video.

Examples:
  python evaluate.py --policy random --render human            # watch random actions live
  python evaluate.py --policy zero --episodes 3 --video zero.mp4
  python evaluate.py --policy oracle --episodes 50             # scripted strike: checks goals are feasible
  python evaluate.py --policy oracle --grip-mode grasp --episodes 50   # scripted grasp of a floating tool + strike
  python evaluate.py --ckpt runs/<name>/ckpt_latest.pt --render human --level 1.0
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from config import EnvCfg
from env import PoseVelocityEnv
from oracle import make_oracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy", default=None, choices=["ckpt", "random", "zero", "oracle"],
                   help="defaults to ckpt if --ckpt is given, else random")
    p.add_argument("--grip-mode", default=None, choices=["weld", "physical", "grasp"],
                   help="override the grip mode (default: from checkpoint, else config.py)")
    p.add_argument("--action-mode", default=None, choices=["head_twist", "joint"],
                   help="override the action mode (default: from checkpoint, else config.py)")
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
        agent = ActorCritic(ck["obs_dim"], ck["priv_dim"], ck["act_dim"], ck["ppo_cfg"]).to(args.device)
        agent.load_state_dict(ck["agent"])
        agent.eval()
    if args.level is not None:
        level = args.level
    if args.grip_mode is not None:
        cfg.grip_mode = args.grip_mode
    if args.action_mode is not None:
        cfg.action_mode = args.action_mode
    if args.table_start:
        if cfg.grip_mode != "physical":
            p.error("--table-start needs --grip-mode physical")
        cfg.curriculum.start_in_hand_prob = (0.0, 0.0)
    oracle = make_oracle(cfg) if policy == "oracle" else None

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
                action = oracle(env)
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
        pass_txt = (f"pass d {ps['d_perp'] * 100:4.1f}cm v {ps['dv_frac']:.2f}xv axis {ps['theta_deg']:3.0f}deg"
                    if ps else "no pass")
        if e["start"] == "float":
            pass_txt += (f"  grasped at step {int(e['grasp_step'])}" if e["grasp_step"] >= 0
                         else "  knocked" if e["knocked"] else "  not grasped")
        print(f"episode {ep}: start {start_kind:7s}  return {ep_ret:8.2f}  len {e['length']:4d}  "
              f"success {bool(e['success'])}  dropped {bool(e['dropped'])}  {pass_txt}  "
              f"d_min {e['d_min'] * 100:.1f}cm")
        obs, info = env.reset(options=reset_opts)

    print(f"\nsuccess rate {np.mean([r['success'] for r in results]):.2f}  "
          f"pass rate {np.mean([r['passed'] for r in results]):.2f}  "
          f"drop rate {np.mean([r['dropped'] for r in results]):.2f}  "
          f"mean return {np.mean([r['return'] for r in results]):.2f}")
    floats = [r for r in results if r["start"] == "float"]
    if floats:
        print(f"floating starts {len(floats)}: grasp rate {np.mean([r['grasp_step'] >= 0 for r in floats]):.2f}  "
              f"knock rate {np.mean([r['knocked'] for r in floats]):.2f}  "
              f"success {np.mean([r['success'] for r in floats]):.2f}")
    if args.video and frames:
        import imageio

        imageio.mimsave(args.video, frames, fps=int(round(1.0 / dt)))
        print(f"wrote {args.video} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
