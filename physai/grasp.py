"""Friction-cone grasp checks and a scripted pick state machine (lesson 16 · Build it).

Plain NumPy. Checked against MuJoCo's contact forces, SciPy's Wilson interval and the lesson's SO-101
pick experiment in experiments/16-grasping-and-pick-place/build.py (and in tests/).
"""
import math

import numpy as np


def cone_ratio(mu, normal_forces, mass, g=9.81):
    """Friction-cone margin of an antipodal grasp: mu (N1 + N2) / (m g). Below 1 the object slides out."""
    return mu * np.sum(normal_forces) / (mass * g)


def max_mass(mu, normal_forces, g=9.81):
    """The heaviest object this squeeze can hold by friction alone."""
    return mu * np.sum(normal_forces) / g


def cone_use(f_contact, mu):
    """How much of its friction cone a contact uses: |tangential| / (mu * normal), from a contact-frame
    force (normal, tangent1, tangent2), the order mj_contactForce returns. Above 1 it slides."""
    f = np.asarray(f_contact, float)
    return np.hypot(f[1], f[2]) / (mu * f[0])


def squeeze_from_cap(tau_max, r_jaw):
    """Squeeze of a stalled parallel jaw: the moving jaw presses tau/r, the fixed jaw pushes back as hard,
    so N1 + N2 = 2 tau / r (the force is not split between the jaws)."""
    return 2 * tau_max / r_jaw


def minjerk(u):
    """Minimum-jerk blend 0 -> 1 over u in [0, 1]: zero velocity and acceleration at both ends."""
    u = min(1.0, max(0.0, u))
    return u ** 3 * (10 - 15 * u + 6 * u * u)


class PickMachine:
    """settle -> pre -> down -> close -> lift -> hold -> done. Timed states end after their duration;
    close ends on a sensed condition: both jaws touch and the jaw is still for `still_time` (or a timeout)."""

    NEXT = dict(settle="pre", pre="down", down="close", close="lift", lift="hold", hold="done")
    MOVES = dict(settle=("home", "home"), pre=("home", "pre"), down=("pre", "down"),
                 lift=("down", "lift"), hold=("lift", "lift"))

    def __init__(self, durations, close_timeout=4.0, still_speed=0.05, still_time=0.1, t0=0.0):
        self.dur, self.timeout = durations, close_timeout
        self.still_speed, self.still_time = still_speed, still_time
        self.state, self.t_state, self.still_since, self.close_s = "settle", t0, None, float("nan")

    def command(self, now, poses, jaw_open, jaw_closed):
        """Arm joint targets and jaw target for this step: min-jerk between poses; in close, a jaw ramp."""
        t = now - self.t_state
        if self.state == "close":
            u = minjerk(t / self.dur["close_ramp"])
            return poses["down"], jaw_open + u * (jaw_closed - jaw_open)
        a, b = self.MOVES[self.state]
        q = poses[a] + minjerk(t / self.dur[self.state]) * (poses[b] - poses[a])
        return q, jaw_open if self.state in ("settle", "pre", "down") else jaw_closed

    def update(self, now, both_touch=False, jaw_speed=0.0):
        """Call after every physics step with what the sensors say; returns the (possibly new) state."""
        t = now - self.t_state
        if self.state == "close":
            settled = both_touch and abs(jaw_speed) < self.still_speed and t > self.dur["close_ramp"]
            if not settled:
                self.still_since = None
            elif self.still_since is None:
                self.still_since = now
            if self.still_since is not None and now - self.still_since >= self.still_time:
                self.close_s = t                           # the grip formed: lift
                self.state, self.t_state = "lift", now
            elif t > self.timeout:
                self.state, self.t_state = "lift", now     # lift anyway; the success test will tell
        elif t >= self.dur[self.state] - 1e-9:
            self.state, self.t_state = self.NEXT[self.state], now
        return self.state


def pick_outcome(lift, touches_both, closed, min_lift=0.04):
    """Physical success test: lifted at least min_lift (m) and still in both jaws; else why it failed."""
    if lift >= min_lift and touches_both:
        return "held"
    return "dropped" if closed else "not closed"


def wilson(k, n, z=1.96):
    """Wilson 95 % interval for a success rate of k out of n trials."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)
