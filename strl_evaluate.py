"""Roll out a StrlEnv policy (deterministic: the policy mean), optionally in the viewer or recorded to video.

Examples:
  python strl_evaluate.py --policy zero --episodes 2 --video strl_zero.mp4
  python strl_evaluate.py --ckpt runs/strl_seed0/ckpt_best.pt --episodes 50
  python strl_evaluate.py --ckpt runs/strl_seed0/ckpt_best.pt --tol 0.01 --episodes 50
  mjpython strl_evaluate.py --ckpt runs/strl_seed0/ckpt_best.pt --render human --realtime --episodes 3
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter

import numpy as np

from strl_config import StrlCfg, make_env_cfg
from strl_env import StrlEnv

warnings.filterwarnings("ignore", message=".*Attach conflict.*")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy", default=None, choices=["ckpt", "random", "zero", "qref"],
                   help="qref: scripted check that drives the joints to each goal's stored arm configuration")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--tol", type=float, default=None, help="success tolerance (default: the checkpoint's)")
    p.add_argument("--no-dr", action="store_true", help="evaluate without domain randomization")
    p.add_argument("--dr", action="store_true", help="evaluate with domain randomization")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--render", default="none", choices=["none", "human", "rgb_array"])
    p.add_argument("--video", default=None, help="write an mp4 (implies --render rgb_array)")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    policy = args.policy or ("ckpt" if args.ckpt else "random")
    agent, scfg, env_cfg, tol = None, StrlCfg(), None, None
    if policy == "ckpt":
        import torch

        from ppo import ActorCritic

        ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        if ck.get("kind") != "strl":
            raise SystemExit("not a strl_train.py checkpoint")
        scfg, env_cfg, tol = ck["strl_cfg"], ck["env_cfg"], ck["tol"]
        agent = ActorCritic(ck["obs_dim"], ck["priv_dim"], ck["act_dim"], ck["ppo_cfg"]).to(args.device)
        agent.load_state_dict(ck["agent"])
        agent.eval()
    if args.no_dr:
        scfg.dr = False
    if args.dr:
        scfg.dr = True
    env_cfg = env_cfg or make_env_cfg(scfg)
    render_mode = "rgb_array" if args.video else (None if args.render == "none" else args.render)
    env = StrlEnv(env_cfg, scfg, render_mode=render_mode)
    tol = args.tol if args.tol is not None else (tol if tol is not None else scfg.success_tolerance)
    env.set_tolerance(tol)
    rng = np.random.default_rng(args.seed)
    dt = env.model.opt.timestep * env_cfg.frame_skip
    print(f"policy={policy} tol={tol * 100:.2f} cm (keypoint threshold {tol * scfg.keypoint_scale * 100:.2f} cm) "
          f"dr={scfg.dr}")

    frames, results = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            t0 = time.time()
            if policy == "ckpt":
                import torch

                o = torch.as_tensor(obs, dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    if args.stochastic:
                        action = agent.dist(agent.obs_rms(o)).sample()[0].clamp(-1, 1).cpu().numpy()
                    else:
                        action = agent.act_deterministic(o)[0].cpu().numpy()
            elif policy == "qref":
                # Close a tenth of the remaining joint-target gap per step (the action is a target delta).
                gap = env.goal.q_ref - env.targets[:env.n_arm_act]
                action = np.clip(gap / (10.0 * env.act_scale[:env.n_arm_act]), -1.0, 1.0).astype(np.float32)
            elif policy == "zero":
                action = np.zeros(env.action_space.shape, np.float32)
            else:
                action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if render_mode == "rgb_array":
                frames.append(env.render())
            if args.realtime and render_mode == "human":
                time.sleep(max(0.0, dt - (time.time() - t0)))
        e = info["episode"]
        results.append(e)
        print(f"ep {ep:3d}: goals reached {int(e['successes']):3d} | steps {e['length']:5d} | end {e['reason']:13s} | "
              f"last distance {e['d_last'] * 100:5.1f} cm")
    env.close()

    goals = np.array([e["successes"] for e in results])
    print(f"\ngoals per episode: mean {goals.mean():.2f}  median {np.median(goals):.1f}  max {goals.max():.0f}")
    ends = Counter(e["reason"] for e in results)
    print("episode ends:", dict(ends))
    reached, failed = goals.sum(), len(results) - ends.get("max_successes", 0)
    attempts = max(1.0, reached + failed)
    print(f"per-goal success {reached / attempts:.0%} | failed by timeout {ends.get('timeout', 0) / attempts:.0%}, "
          f"table {ends.get('table_force', 0) / attempts:.0%}")
    if args.video and frames:
        import imageio

        imageio.mimsave(args.video, frames, fps=int(round(1.0 / dt)))
        print(f"wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
