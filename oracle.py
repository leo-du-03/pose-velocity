"""Scripted strike policy using privileged goal knowledge (feasibility checks and behavior-cloning labels)."""
from __future__ import annotations

import numpy as np

import mathutils as mu


class Oracle:
    """Scripted strike for head_twist actions, using privileged goal knowledge.

    1. Fly the head to the lead-in point p_g - lead_in * v_hat while turning the head axis onto the goal axis.
    2. Once there and aligned, move at v_g along v_hat (correcting lateral error) until the pass ends the episode.

    markov=True (default) decides the phase from the observation alone, which behavior cloning needs. With
    markov=False the phase is a hidden latched flag, so at the lead-in point the same observation is labelled "hold"
    before the flag flips and "go at v_g" after it, and a clone averages the two and stalls there. The Markov version
    strikes while the head is between the lead-in point and the goal plane, inside a cone around the strike line
    (strike_lat at the lead-in point, widening by strike_lat_slope per metre towards the goal), aligned within
    strike_axis_deg, and committed: the smoothed commanded velocity (the observation's prev_action) along v_hat is at
    least commit_frac of the goal speed, a requirement that ramps up from 0 at lead_tol past the lead-in point over
    commit_len. The
    commitment plays the latch's role but is observable: the ramp leaves room to accelerate from rest, and a head
    still moving backwards (coming back to the lead-in point from the goal side) does not qualify.
    """

    def __init__(self, k_pos: float = 4.0, v_approach: float = 0.4, k_rot: float = 4.0, k_lat: float = 4.0,
                 lead_tol: float = 0.015, axis_tol_deg: float = 8.0, markov: bool = True,
                 strike_lat: float = 0.015, strike_lat_slope: float = 0.25, strike_axis_deg: float = 10.0,
                 commit_frac: float = 0.5, commit_len: float = 0.03):
        self.k_pos, self.v_approach, self.k_rot, self.k_lat = k_pos, v_approach, k_rot, k_lat
        self.lead_tol, self.axis_tol = lead_tol, np.deg2rad(axis_tol_deg)
        self.markov = markov
        self.strike_lat, self.strike_lat_slope, self.strike_axis = strike_lat, strike_lat_slope, np.deg2rad(strike_axis_deg)
        self.commit_frac, self.commit_len = commit_frac, commit_len
        self.goal, self.striking = None, False

    def _in_strike_region(self, env, head, axis) -> bool:
        g = env.goal
        rel = head - g.p
        along = float(rel @ g.direction)
        s = along + env.cfg.goal.lead_in  # progress past the lead-in point
        lat = float(np.linalg.norm(rel - along * g.direction))
        v_cmd = float(env.prev_action[:3] @ g.direction) * env.cfg.head_v_max
        need = self.commit_frac * g.speed * min(1.0, (s - self.lead_tol) / self.commit_len)
        return (-self.lead_tol < s and along < 0.0 and mu.angle_between(axis, g.axis) < self.strike_axis
                and lat < self.strike_lat + self.strike_lat_slope * max(s, 0.0)
                and (s <= self.lead_tol or v_cmd >= need))

    def __call__(self, env) -> np.ndarray:
        if env.cfg.action_mode != "head_twist":
            raise ValueError("the oracle needs action_mode 'head_twist'")
        g = env.goal
        if self.goal is not g:
            self.goal, self.striking = g, False
        head, _, _, axis = env._head_state()
        d = g.direction
        w = self.k_rot * np.cross(axis, g.axis)
        if self.markov:
            self.striking = self._in_strike_region(env, head, axis)
        if not self.striking:
            err = env.lead_point() - head
            v = self.k_pos * err
            n = np.linalg.norm(v)
            if n > self.v_approach:
                v *= self.v_approach / n
            if not self.markov:
                self.striking = (np.linalg.norm(err) < self.lead_tol
                                 and mu.angle_between(axis, g.axis) < self.axis_tol)
        else:
            rel = head - g.p
            v = g.v - self.k_lat * (rel - (rel @ d) * d)
        a = np.concatenate([v / env.cfg.head_v_max, w / env.cfg.head_w_max])
        return np.clip(a, -1.0, 1.0)
