"""Tables for the goal-velocity ablations from `evaluate.py --dump` files.

Runs named <condition>_s<seed> are grouped into one condition and reported as the mean +- std over seeds of each
run's own rate or median. The score is the same for every policy (see evaluate.py): at the attempt, pose_ok (head
distance and rotation within the level's tolerances), vel_ok (|v - v_g| < eps_v_frac |v_g|) and joint_ok (both);
episodes without an attempt count as failures.

Examples:
  python analyze_ablation.py results/abl/
  python analyze_ablation.py results/abl/A_*.json --bins
  python analyze_ablation.py results/abl/ --vel-tol 0.25     # rescore vel_ok / joint_ok at 25% from the dumps
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

RATES = ("attempt", "pose_ok", "vel_ok", "joint_ok", "success", "pushed")
MEDIANS = (("dp", 100.0), ("theta_deg", 1.0), ("dv_frac", 1.0), ("v_side_frac", 1.0), ("length", 1.0),
           ("qvel_l1", 1.0), ("qvel_peak", 1.0), ("qacc", 1.0), ("da", 1.0))
SWIPE_BINS = ((0, 30), (30, 90), (90, 181))


def load(paths: list, vel_tol=None) -> dict:
    """vel_tol: rescore vel_ok / joint_ok with |v - v_g| < vel_tol |v_g| (pose_ok is kept as dumped)."""
    runs = {}
    for p in paths:
        with open(p) as f:
            eps = json.load(f)["episodes"]
        if vel_tol is not None:
            for e in eps:
                e["vel_ok"] = bool(e["attempt"] and e["dv_frac"] < vel_tol)
                e["joint_ok"] = bool(e["pose_ok"] and e["vel_ok"])
        runs[os.path.splitext(os.path.basename(p))[0]] = eps
    return runs


def run_stats(eps: list) -> dict:
    out = {k: float(np.mean([e[k] for e in eps])) for k in RATES}
    att = [e for e in eps if e["attempt"]]
    for k, scale in MEDIANS:
        vals = [e[k] * scale for e in att if k in e]
        out[k] = float(np.median(vals)) if vals else float("nan")
    return out


def fmt(vals: list) -> str:
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "     -     "
    return f"{np.mean(vals):6.3f}" + (f"+-{np.std(vals):.3f}" if len(vals) > 1 else "      ")


def table(groups: dict, select=lambda e: True, title: str = "") -> None:
    cols = RATES + tuple(k for k, _ in MEDIANS)
    if title:
        print(f"\n== {title}")
    print(f"{'condition':28s} {'n':>2s} " + " ".join(f"{c:>12s}" for c in cols))
    for name, runs in groups.items():
        stats = [run_stats(sel) for sel in ([e for e in eps if select(e)] for eps in runs) if sel]
        if not stats:
            continue
        print(f"{name:28s} {len(stats):2d} " + " ".join(f"{fmt([s[c] for s in stats]):>12s}" for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="dump files, or directories of them")
    p.add_argument("--vel-tol", type=float, default=None, metavar="F",
                   help="rescore vel_ok / joint_ok at this velocity tolerance (fraction of |v_g|) instead of the one "
                        "evaluate.py scored with")
    p.add_argument("--bins", action="store_true", help="also break down by swipe angle, goal speed and downward strikes")
    args = p.parse_args()
    files = []
    for x in args.paths:
        files += sorted(glob.glob(os.path.join(x, "*.json"))) if os.path.isdir(x) else [x]
    runs = load(files, args.vel_tol)
    groups = defaultdict(list)
    for name, eps in sorted(runs.items()):
        groups[re.sub(r"_s\d+$", "", name)].append(eps)
    print("dp in cm, theta in deg, dv / v_side as fractions of |v_g|, length in steps; medians over attempts. Motion "
          "(episode step means): qvel_l1 rad/s, qvel_peak rad/s (max), qacc rad/s^2, da action change")
    table(groups, title="all goals")
    if not args.bins:
        return
    for lo, hi in SWIPE_BINS:
        table(groups, lambda e, lo=lo, hi=hi: e["attempt"] and lo <= e["swipe_deg"] < hi,
              f"swipe angle {lo}-{min(hi, 180)} deg (attempts only)")
    speeds = [e["speed"] for g in groups.values() for eps in g for e in eps if e["attempt"]]
    if speeds:
        t1, t2 = np.percentile(speeds, [100 / 3, 200 / 3])
        for lo, hi, lab in ((0, t1, "slow"), (t1, t2, "mid"), (t2, np.inf, "fast")):
            table(groups, lambda e, lo=lo, hi=hi: e["attempt"] and lo <= e["speed"] < hi,
                  f"goal speed {lab} ({lo:.2f}-{hi:.2f} m/s, attempts only)")
    for down in (True, False):
        table(groups, lambda e, down=down: e["attempt"] and e["down"] == down,
              f"{'steep downward' if down else 'other'} goals (attempts only)")


if __name__ == "__main__":
    main()
