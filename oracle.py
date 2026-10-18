"""Scripted strike policy using privileged goal knowledge (feasibility checks and behavior-cloning labels)."""
from __future__ import annotations

from typing import Optional

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


class GraspOracle:
    """Grasp grip: grasp the floating tool, then strike with the welded tool (Oracle; grip action held at +1).

    Before the weld it drives the palm grip point (the head_twist reference point until the weld) into the
    canonical grasp (env.grasp_targets), which depends only on the observed tool pose, so the labels are a function
    of the observation. Along the approach line n (the palm's -z at the grasp), with s the signed distance to the
    grasp (-pregrasp_dist at the pregrasp point):
      1. Outside the approach corridor, fly to a far pregrasp point (far_extra behind the pregrasp point) while
         turning the palm onto the grasp orientation. The palm, fingertips and grip point are pushed away from the
         tool (as a segment) when closer than clear_dist, and turning slows down there, so the hand does not sweep it.
      2. Inside the corridor (behind the pregrasp point, lateral error under corridor_lat widening by corridor_slope
         per metre, palm rotation under corridor_deg), fly along it to the pregrasp point.
      3. Inside the funnel (between the pregrasp point and the grasp, lateral error under funnel_lat widening by
         funnel_slope per metre back from the grasp, rotation under funnel_deg), slide in to the grasp.
      4. Close the grip at the grasp (within close_tol and close_deg). The closing stays committed while the observed
         smoothed grip (prev_action) is already closing and the grip point is within hold_tol, the same role the
         strike's commit condition plays; the weld turns on as the fingers close.
    """

    def __init__(self, strike: Optional[Oracle] = None, k_pos: float = 4.0, v_approach: float = 0.35,
                 v_insert: float = 0.12, k_rot: float = 3.0, far_extra: float = 0.1, corridor_lat: float = 0.03,
                 corridor_slope: float = 0.3, corridor_deg: float = 15.0, funnel_lat: float = 0.012,
                 funnel_slope: float = 0.3, funnel_deg: float = 10.0, close_tol: float = 0.008,
                 close_deg: float = 6.0, hold_tol: float = 0.03, clear_dist: float = 0.07, k_clear: float = 6.0):
        self.strike = strike or Oracle()
        self.k_pos, self.v_approach, self.v_insert, self.k_rot = k_pos, v_approach, v_insert, k_rot
        self.far_extra = far_extra
        self.corridor_lat, self.corridor_slope, self.corridor_rot = corridor_lat, corridor_slope, np.deg2rad(corridor_deg)
        self.funnel_lat, self.funnel_slope, self.funnel_rot = funnel_lat, funnel_slope, np.deg2rad(funnel_deg)
        self.close_tol, self.close_rot, self.hold_tol = close_tol, np.deg2rad(close_deg), hold_tol
        self.clear_dist, self.k_clear = clear_dist, k_clear

    def _clearance(self, env):
        """(smallest distance of the palm / fingertips / grip point to the tool segment, unit push away from it)."""
        base, head = env.tool_segment()
        seg = head - base
        best, push = np.inf, np.zeros(3)
        for h in env.hand_points():
            t = float(np.clip((h - base) @ seg / max(seg @ seg, 1e-12), 0.0, 1.0))
            away = h - (base + t * seg)
            dist = float(np.linalg.norm(away))
            if dist < best:
                best, push = dist, mu.normalize(away)
        return best, push

    def __call__(self, env) -> np.ndarray:
        if env.welded:
            return np.concatenate([self.strike(env), [1.0]])
        cfg = env.cfg
        L = cfg.pregrasp_dist
        gp = env._grip_point()
        target, pre, palm_q = env.grasp_targets()
        rot = mu.rotation_error(env.data.xquat[env.palm_id], palm_q)
        rot_err = float(np.linalg.norm(rot))
        n = mu.normalize(target - pre)
        rel = gp - target
        along = float(rel @ n)
        lat = float(np.linalg.norm(rel - along * n))
        dist = float(np.linalg.norm(rel))
        closing = env.prev_action[env.n_arm_act] > 0.0
        grip = 1.0 if ((dist < self.close_tol and rot_err < self.close_rot) or (closing and dist < self.hold_tol)) else -1.0
        in_funnel = (-L - 0.01 < along < 0.01 and rot_err < self.funnel_rot
                     and lat < self.funnel_lat + self.funnel_slope * max(-along, 0.0))
        in_corridor = (along <= -L and rot_err < self.corridor_rot
                       and lat < self.corridor_lat + self.corridor_slope * (-along - L))
        w = self.k_rot * rot
        if in_funnel or grip > 0:
            goal, v_max = target, self.v_insert
        elif in_corridor:
            goal, v_max = pre, self.v_approach
        else:
            goal, v_max = target - (L + self.far_extra) * n, self.v_approach
        v = self.k_pos * (goal - gp)
        speed = np.linalg.norm(v)
        if speed > v_max:
            v *= v_max / speed
        if not (in_funnel or in_corridor or grip > 0):
            clear, push = self._clearance(env)
            if clear < self.clear_dist:
                inward = float(v @ push)
                if inward < 0.0:
                    v = v - inward * push  # slide around the tool instead of into it
                v = v + self.k_clear * (self.clear_dist - clear) * push
                w = w * float(np.clip((clear - 0.02) / (self.clear_dist - 0.02), 0.1, 1.0))
        a = np.concatenate([v / cfg.head_v_max, w / cfg.head_w_max])
        return np.concatenate([np.clip(a, -1.0, 1.0), [grip]])


def make_oracle(cfg):
    """The scripted policy for the env config: GraspOracle for grasp grip, else the strike Oracle."""
    return GraspOracle() if cfg.grip_mode == "grasp" else Oracle()
