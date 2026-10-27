"""Safety layers between a command and the motors (lesson 15 · Build it).

Plain NumPy. Each layer is checked against MuJoCo's position actuator and against the lesson's SO-101
experiment in experiments/15-safety/build.py (and in tests/).
"""
import numpy as np


def clamp_goal(target, q_now, q_min=None, q_max=None, delta=None):
    """Position limit, then relative-target limit: the goal stays in [q_min, q_max] and within delta of q_now."""
    goal = target
    if q_min is not None or q_max is not None:
        goal = np.clip(goal, q_min, q_max)                  # a fence around the safe range
    if delta is not None:
        goal = np.clip(goal, q_now - delta, q_now + delta)  # at most delta ahead of where the joint is now
    return goal


def servo_torque(goal, q, qd, kp, kv, tau_max):
    """A position servo with a torque cap: kp (goal - q) - kv qd, clipped to +-tau_max."""
    return np.clip(kp * (goal - q) - kv * qd, -tau_max, tau_max)


class Watchdog:
    """Expires when no fresh command has arrived for longer than `timeout` seconds."""

    def __init__(self, timeout):
        self.timeout, self.last = timeout, None

    def feed(self, t):
        """Call whenever a fresh command arrives at time t."""
        self.last = t

    def expired(self, t):
        return self.last is not None and t - self.last > self.timeout


class EStop:
    """A latch: one press stops the robot until someone resets it after the cause is gone."""

    def __init__(self):
        self.tripped = False

    def update(self, pressed):
        """Call every tick with the button (or fault) state; returns True while the stop is active."""
        self.tripped = self.tripped or bool(pressed)
        return self.tripped

    def reset(self, pressed):
        """Clear the latch, but only if the button is released."""
        if not pressed:
            self.tripped = False
        return self.tripped
