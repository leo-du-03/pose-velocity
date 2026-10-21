"""Scripted strike policy using privileged goal knowledge (feasibility checks and behavior-cloning labels)."""
from __future__ import annotations

import numpy as np


def _ramp(x: float, x0: float, x1: float) -> float:
    """0 at x0, 1 at x1, linear in between (x0 may be above x1)."""
    return float(np.clip((x - x0) / (x1 - x0), 0.0, 1.0))


class Oracle:
    """Scripted strike for head_twist actions, using privileged goal knowledge.

    1. Fly the head to the lead-in point p_g - lead_in * v_hat while turning the tool onto the goal orientation.
    2. Once there and aligned, move at v_g along v_hat (correcting lateral error), holding the goal orientation,
       until the pass ends the episode. v_hat need not be along the head axis (a swipe).

    markov=True (default) makes the action a continuous function of the observation, which behavior cloning needs.
    The linear velocity is a blend (1 - alpha) * approach + alpha * strike, where the strike weight alpha is a product
    of linear ramps, one per condition:
      * progress s past the lead-in point: 0 at s = -2 lead_tol, 1 at s = 0;
      * not past the goal plane: 1 on the near side, 0 at 2 cm past it;
      * inside a cone around the strike line (radius c = strike_lat at the lead-in point, widening by
        strike_lat_slope per metre towards the goal): 1 within c/2, 0 at c;
      * aligned: 1 within strike_axis/2, 0 at strike_axis;
      * committed (once more than lead_tol past the lead-in point, ramped in over another lead_tol): the smoothed
        commanded velocity (the observation's prev_action) along v_hat is at least commit_frac of the goal speed
        (ramped up over commit_len), with a soft band of commit_band * goal speed below that.
    `striking` reports alpha > 0.5 (diagnostics only).

    An earlier Markov version switched hard between the two velocities at the edge of that region. Labels on either
    side of the edge differ by about the goal velocity, so a clone regressed onto them averaged the two: it hesitated
    at the lead-in point (about 4x the oracle's time there) and struck at ~65% of the goal speed with extra sideways
    velocity, and DAgger could not fix it (more labels at the edge only pin the average down). commit_frac 0.5 made
    that worse: a clone that accelerated a little slower than the oracle was labelled "go back" and cycled at the
    lead-in point.

    With markov=False the phase is a hidden latched flag, so at the lead-in point the same observation is labelled
    "hold" before the flag flips and "go at v_g" after it, and a clone averages the two and stalls there.
    """

    def __init__(self, k_pos: float = 4.0, v_approach: float = 0.4, k_rot: float = 4.0, k_lat: float = 4.0,
                 lead_tol: float = 0.015, axis_tol_deg: float = 8.0, markov: bool = True,
                 strike_lat: float = 0.015, strike_lat_slope: float = 0.25, strike_axis_deg: float = 10.0,
                 commit_frac: float = 0.0, commit_len: float = 0.03, commit_band: float = 0.2,
                 past_goal: float = 0.02):
        self.k_pos, self.v_approach, self.k_rot, self.k_lat = k_pos, v_approach, k_rot, k_lat
        self.lead_tol, self.axis_tol = lead_tol, np.deg2rad(axis_tol_deg)
        self.markov = markov
        self.strike_lat, self.strike_lat_slope, self.strike_axis = strike_lat, strike_lat_slope, np.deg2rad(strike_axis_deg)
        self.commit_frac, self.commit_len, self.commit_band = commit_frac, commit_len, commit_band
        self.past_goal = past_goal
        self.goal, self.striking, self.alpha = None, False, 0.0

    def strike_weight(self, env, head, rot_err: float) -> float:
        """Blend weight in [0, 1] of the strike velocity (markov=True); continuous in the observation."""
        g = env.goal
        rel = head - g.p
        along = float(rel @ g.direction)
        s = along + env.cfg.goal.lead_in  # progress past the lead-in point
        lat = float(np.linalg.norm(rel - along * g.direction))
        cone = self.strike_lat + self.strike_lat_slope * max(s, 0.0)
        v_cmd = float(env.prev_action[:3] @ g.direction) * env.cfg.head_v_max
        need = self.commit_frac * g.speed * float(np.clip((s - self.lead_tol) / self.commit_len, 0.0, 1.0))
        committed = _ramp(v_cmd, need - self.commit_band * g.speed, need)
        commit_gate = _ramp(s, self.lead_tol, 2 * self.lead_tol)  # no commitment needed near the lead-in point
        return (_ramp(s, -2 * self.lead_tol, 0.0)
                * _ramp(along, self.past_goal, 0.0)
                * _ramp(lat, cone, 0.5 * cone)
                * _ramp(rot_err, self.strike_axis, 0.5 * self.strike_axis)
                * (1.0 - commit_gate * (1.0 - committed)))

    def __call__(self, env) -> np.ndarray:
        if env.cfg.action_mode != "head_twist":
            raise ValueError("the oracle needs action_mode 'head_twist'")
        g = env.goal
        if self.goal is not g:
            self.goal, self.striking = g, False
        head = env._head_state()[0]
        d = g.direction
        rot = env._rot_error()
        w = self.k_rot * rot

        err = env.lead_point() - head
        v_app = self.k_pos * err
        n = np.linalg.norm(v_app)
        if n > self.v_approach:
            v_app *= self.v_approach / n
        rel = head - g.p
        v_strike = g.v - self.k_lat * (rel - (rel @ d) * d)

        if self.markov:
            self.alpha = self.strike_weight(env, head, float(np.linalg.norm(rot)))
            self.striking = self.alpha > 0.5
        else:  # the latch flips after this step's approach action, as before
            self.alpha = float(self.striking)
            if not self.striking:
                self.striking = bool(np.linalg.norm(err) < self.lead_tol and np.linalg.norm(rot) < self.axis_tol)
        v = (1.0 - self.alpha) * v_app + self.alpha * v_strike
        a = np.concatenate([v / env.cfg.head_v_max, w / env.cfg.head_w_max])
        return np.clip(a, -1.0, 1.0)
