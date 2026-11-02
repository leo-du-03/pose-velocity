"""The oracle's labels must vary gradually across neighbouring states: a clone regressed onto a sharp switch
averages the two sides and stalls there (runs/bc_swipe12, runs/bc_swipe12_smooth)."""
from types import SimpleNamespace

import numpy as np

from goals import Goal
from oracle import Oracle

LEAD_IN, V_MAX = 0.12, 0.8


class StubEnv:
    """Just what Oracle reads: a goal along +x at 0.45 m/s, the tool already at the goal orientation."""

    def __init__(self):
        self.cfg = SimpleNamespace(action_mode="head_twist", head_v_max=V_MAX, head_w_max=2.5,
                                   goal=SimpleNamespace(lead_in=LEAD_IN))
        self.goal = Goal(p=np.zeros(3), v=np.array([0.45, 0.0, 0.0]), quat=np.array([1.0, 0, 0, 0]),
                         axis=np.array([0.0, 0, 1]))
        self.prev_action = np.zeros(6)
        self.head = np.zeros(3)

    def _head_state(self):
        return self.head, None, None, None

    def _rot_error(self):
        return np.zeros(3)

    def lead_point(self):
        return self.goal.p - LEAD_IN * self.goal.direction


def _forward_label(env, oracle, s, lat):
    env.head = np.array([s - LEAD_IN, lat, 0.0])
    return float(oracle(env)[0])


def test_labels_change_gradually_across_the_strike_cone():
    """Sweeping the head sideways off the strike line just past the lead-in point, the forward label changes by less
    than 0.04 (action units) per mm. The first smooth oracle faded over ~7 mm (~0.08 per mm) and the clone stalled."""
    env, oracle = StubEnv(), Oracle()
    env.prev_action[0] = 0.45 / V_MAX  # already moving forward at the goal speed: committed
    lats = np.arange(0.0, 0.08, 0.001)
    for s in (0.0, 0.015, 0.03, 0.05):
        fwd = np.array([_forward_label(env, oracle, s, lat) for lat in lats])
        assert np.abs(np.diff(fwd)).max() < 0.04, (s, np.abs(np.diff(fwd)).max())
        cone = oracle.strike_lat + oracle.strike_lat_slope * s
        inside = lats <= cone
        np.testing.assert_allclose(fwd[inside], 0.45 / V_MAX, atol=1e-6)  # full strike anywhere inside the cone
