"""Per-goal failure diagnosis for a StrlEnv policy.

For every goal it records what kind of goal it was (the episode's first, absolute goal or a random-walk delta goal),
how far it was when it appeared (keypoint distance, position, rotation, arm joint change), how close it is to the
table, the tool, and how it ended (reached / timeout near / timeout far / table contact, and which body touched).
Then it prints success rates broken down by each of these, so you can see which goals the policy cannot do.

Success metric (--metric):
  keypoint  the baseline's own: max keypoint distance < tol x 1.5 for success_steps steps (SimToolReal).
  weld      simplify-weld's pose criteria: tool-head position error < --eps-p AND full rotation angle between tool
            and goal orientation < --eps-theta, for --hold steps. (simplify-weld also checks velocity, which a pose-only
            goal does not have.) Level 1 there is 3 cm / 20 deg, level 0 is 8 cm / 60 deg.
  In weld mode the policy still sees the same goal; only the test for "reached" (which then moves on to the next goal)
  changes.

Examples:
  python strl_diagnose.py --ckpt runs/strl_a01_seed0/ckpt_best.pt --metric weld --eps-p 0.03 --eps-theta 20
  python strl_diagnose.py --ckpt checkpoints/strl_seed0/ckpt_best.pt --tol 0.03 --episodes 100 --csv diag_old_3cm.csv
  python strl_diagnose.py --policy qref --tol 0.03 --episodes 100 --csv diag_qref_3cm.csv
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import Counter, defaultdict

import mujoco
import numpy as np

import mathutils as mu
from strl_config import StrlCfg, make_env_cfg
from strl_env import StrlEnv

warnings.filterwarnings("ignore", message=".*Attach conflict.*")

FAR = 0.10  # m: a timeout with the last keypoint distance at or above this counts as "far" (could not get there)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy", default=None, choices=["ckpt", "qref"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--tol", type=float, default=None, help="success tolerance (default: the checkpoint's)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--csv", default=None, help="write one row per goal")
    p.add_argument("--metric", default="keypoint", choices=["keypoint", "weld"])
    p.add_argument("--eps-p", type=float, default=0.03, help="weld metric: head position tolerance (m)")
    p.add_argument("--eps-theta", type=float, default=20.0, help="weld metric: orientation tolerance (deg)")
    p.add_argument("--hold", type=int, default=None, help="weld metric: steps within tolerance (default success_steps)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def rot_angle_deg(q1, q2) -> float:
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))))


def goal_features(env: StrlEnv, index: int, fallbacks_before: int) -> dict:
    d, g, t = env.data, env.goal, env.tool
    tool_pos = d.xpos[env.tool_id].copy()
    head_goal = g.p + mu.rotate(g.quat, t.head_pos)
    top = env.rinfo.table_top
    kind = "first" if index == 0 else ("delta_fallback_abs" if env.abs_goal_fallbacks > fallbacks_before else "delta")
    return {
        "index": index,
        "kind": kind,
        "kp_dist_cm": 100 * env.keypoint_distance(),
        "pos_dist_cm": 100 * float(np.linalg.norm(g.p - tool_pos)),
        "rot_deg": rot_angle_deg(d.xquat[env.tool_id], g.quat),
        "joint_max_deg": float(np.degrees(np.max(np.abs(g.q_ref - d.qpos[env.arm_qadr])))),
        "goal_clear_cm": 100 * float(min(g.p[2], head_goal[2]) - top),
        "tool": f"{t.handle_type}/{t.head_type}",
    }


def roll_swing_deg(q_tool, q_goal):
    """Split the tool-vs-goal rotation into roll about the handle (tool z) and the remaining swing, in degrees."""
    w, x, y, z = mu.quat_mul(np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]]), q_tool)
    n = np.hypot(w, z)
    if n < 1e-9:
        return 0.0, float(np.degrees(2.0 * np.arccos(min(1.0, abs(w)))))
    roll = float(np.degrees(2.0 * np.arccos(min(1.0, abs(w) / n))))
    sw = np.array([w, x, y, z]) / 1.0
    tw_conj = np.array([w / n, 0.0, 0.0, -z / n])
    swing_w = mu.quat_mul(sw, tw_conj)[0]
    return roll, float(np.degrees(2.0 * np.arccos(min(1.0, abs(swing_w)))))


def head_errors(env: StrlEnv):
    """(head position error m, full rotation angle deg) between the tool and the goal."""
    d, g, t = env.data, env.goal, env.tool
    head = d.xpos[env.tool_id] + mu.rotate(d.xquat[env.tool_id], t.head_pos)
    head_goal = g.p + mu.rotate(g.quat, t.head_pos)
    return float(np.linalg.norm(head - head_goal)), rot_angle_deg(d.xquat[env.tool_id], g.quat)


def table_bodies(env: StrlEnv) -> str:
    m, d = env.model, env.data
    names = set()
    for i in range(d.ncon):
        g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
        if g1 in env.env_geoms or g2 in env.env_geoms:
            other = int(m.geom_bodyid[g2 if g1 in env.env_geoms else g1])
            if other in env.robot_body_set or other == env.tool_id:
                names.add("tool" if other == env.tool_id else (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other)
                                                               or str(other)))
    return "+".join(sorted(names)) or "?"


def outcome(row: dict) -> str:
    if row["end"] == "reached":
        return "reached"
    if row["end"] == "table_force":
        return "table"
    if row["end"] == "timeout":
        return "timeout_far" if row["d_last_cm"] >= 100 * FAR else "timeout_near"
    return row["end"]


def breakdown(rows, key, bins=None, title=None):
    groups = defaultdict(list)
    for r in rows:
        v = r[key]
        if bins is not None:
            lab = None
            for lo, hi in zip(bins[:-1], bins[1:]):
                if lo <= v < hi:
                    lab = f"{lo:g}-{hi:g}"
                    break
            v = lab or f">={bins[-1]:g}"
        groups[v].append(r)
    print(f"\n{title or key}:")
    print(f"  {'group':22s} {'goals':>6s} {'reached':>8s} {'tmo_near':>9s} {'tmo_far':>8s} {'table':>6s}")
    order = sorted(groups, key=lambda k: (bins is not None and k.startswith(">="), str(k)))
    if bins is not None:
        labs = [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])] + [f">={bins[-1]:g}"]
        order = [k for k in labs if k in groups]
    for k in order:
        rs = groups[k]
        c = Counter(r["outcome"] for r in rs)
        n = len(rs)
        print(f"  {str(k):22s} {n:6d} {c['reached'] / n:8.0%} {c['timeout_near'] / n:9.0%} "
              f"{c['timeout_far'] / n:8.0%} {c['table'] / n:6.0%}")


def main():
    args = parse_args()
    policy = args.policy or ("ckpt" if args.ckpt else "qref")
    agent, scfg, env_cfg, tol = None, StrlCfg(), None, None
    if policy == "ckpt":
        import torch

        from ppo import ActorCritic

        ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        scfg, env_cfg, tol = ck["strl_cfg"], ck["env_cfg"], ck["tol"]
        agent = ActorCritic(ck["obs_dim"], ck["priv_dim"], ck["act_dim"], ck["ppo_cfg"]).to(args.device)
        agent.load_state_dict(ck["agent"])
        agent.eval()
    scfg.dr = False
    env_cfg = env_cfg or make_env_cfg(scfg)
    env = StrlEnv(env_cfg, scfg)
    tol = args.tol if args.tol is not None else (tol if tol is not None else scfg.success_tolerance)
    weld = args.metric == "weld"
    hold = args.hold or scfg.success_steps
    if weld:
        env.set_tolerance(1e-9)  # the env's own keypoint success never fires; we test success here
        print(f"policy={policy} metric=weld eps_p={args.eps_p * 100:.1f} cm eps_theta={args.eps_theta:.0f} deg "
              f"hold={hold} episodes={args.episodes}")
    else:
        env.set_tolerance(tol)
        print(f"policy={policy} tol={tol * 100:.2f} cm episodes={args.episodes}")

    def new_goal_row(index, fb):
        r = goal_features(env, index, fb)
        r.update(episode=ep, steps=0, d_min_cm=np.inf, best_r=np.inf, best_pos_cm=np.nan, best_ang_deg=np.nan,
                 best_roll_deg=np.nan, best_swing_deg=np.nan, near=0)
        return r

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        cur = new_goal_row(0, env.abs_goal_fallbacks)
        done = False
        while not done:
            if policy == "ckpt":
                import torch

                o = torch.as_tensor(obs, dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    action = agent.act_deterministic(o)[0].cpu().numpy()
            else:
                gap = env.goal.q_ref - env.targets[:env.n_arm_act]
                action = np.clip(gap / (10.0 * env.act_scale[:env.n_arm_act]), -1.0, 1.0).astype(np.float32)
            fb = env.abs_goal_fallbacks
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            cur["steps"] += 1
            if "dist" in info:
                cur["d_min_cm"] = min(cur["d_min_cm"], 100 * info["dist"])
            pos_err, ang_err = head_errors(env)
            r_norm = max(pos_err / args.eps_p, ang_err / args.eps_theta)
            if r_norm < cur["best_r"]:
                roll, swing = roll_swing_deg(env.data.xquat[env.tool_id], env.goal.quat)
                cur.update(best_r=r_norm, best_pos_cm=100 * pos_err, best_ang_deg=ang_err, best_roll_deg=roll,
                           best_swing_deg=swing)
            if weld and not (term and info["episode"]["reason"] == "table_force"):
                cur["near"] += int(r_norm < 1.0)
                if cur["near"] >= hold:
                    cur.update(end="reached", d_last_cm=100 * info.get("dist", np.nan), contact="")
                    rows.append(cur)
                    env.successes += 1
                    if env.successes >= scfg.max_consecutive_successes:
                        break
                    fb = env.abs_goal_fallbacks
                    env.goal = env._sample_delta_goal()
                    env._update_goal_markers()
                    env._reset_goal_trackers()
                    obs = env._get_obs()
                    cur = new_goal_row(cur["index"] + 1, fb)
                    done = False  # a timeout on this step belonged to the goal just reached
                    continue
            if not weld and info.get("success"):
                cur.update(end="reached", d_last_cm=100 * info["dist"], contact="")
                rows.append(cur)
                cur = new_goal_row(cur["index"] + 1, fb)
            if done:
                e = info["episode"]
                if e["reason"] == "max_successes":
                    break
                cur.update(end=e["reason"], d_last_cm=100 * e["d_last"],
                           contact=table_bodies(env) if e["reason"] == "table_force" else "")
                rows.append(cur)
        if (ep + 1) % 10 == 0:
            print(f"  {ep + 1} episodes, {len(rows)} goals")
    env.close()
    for r in rows:
        r["outcome"] = outcome(r)

    c = Counter(r["outcome"] for r in rows)
    n = len(rows)
    print(f"\n{n} goals: " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
    breakdown(rows, "kind", title="goal kind (first = absolute goal at episode start)")
    breakdown(rows, "rot_deg", [0, 30, 60, 90, 120, 150], "rotation needed (deg)")
    breakdown(rows, "pos_dist_cm", [0, 5, 10, 15, 20, 30], "position distance (cm)")
    breakdown(rows, "joint_max_deg", [0, 10, 20, 40, 60, 90], "largest arm joint change to the goal's IK solution (deg)")
    breakdown(rows, "goal_clear_cm", [0, 5, 10, 15, 20, 30], "goal height above the table (lower of handle / head, cm)")
    breakdown(rows, "tool", title="tool (handle / head)")
    failed = [r for r in rows if r["outcome"] in ("timeout_near", "timeout_far")]
    if failed:
        lim_p = sum(1 for r in failed if r["best_pos_cm"] / (100 * args.eps_p) >= r["best_ang_deg"] / args.eps_theta)
        print(f"\ntimeouts, at their closest (by max(pos/{args.eps_p * 100:.0f}cm, angle/{args.eps_theta:.0f}deg)): "
              f"limited by position {lim_p / len(failed):.0%}, by angle {1 - lim_p / len(failed):.0%}; "
              f"median head error {np.median([r['best_pos_cm'] for r in failed]):.1f} cm, "
              f"median angle {np.median([r['best_ang_deg'] for r in failed]):.0f} deg")
        ang = [r for r in failed if r["best_ang_deg"] >= args.eps_theta]
        if ang:
            roll_dom = sum(1 for r in ang if r["best_roll_deg"] >= r["best_swing_deg"])
            print(f"  timeouts with angle >= {args.eps_theta:.0f} deg at their closest: {len(ang)}; "
                  f"median roll about the handle {np.median([r['best_roll_deg'] for r in ang]):.0f} deg, "
                  f"median remaining swing {np.median([r['best_swing_deg'] for r in ang]):.0f} deg; "
                  f"roll is the larger part in {roll_dom / len(ang):.0%}")
    tb = Counter(r["contact"] for r in rows if r["outcome"] == "table")
    if tb:
        print("\ntable contacts by body:", dict(tb.most_common()))

    if args.csv:
        keys = ["episode", "index", "kind", "outcome", "end", "steps", "kp_dist_cm", "pos_dist_cm", "rot_deg",
                "joint_max_deg", "goal_clear_cm", "tool", "d_min_cm", "d_last_cm", "best_pos_cm", "best_ang_deg", "best_roll_deg",
                "best_swing_deg", "contact"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{r[k]:.2f}" if isinstance(r[k], float) else r[k]) for k in keys})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
