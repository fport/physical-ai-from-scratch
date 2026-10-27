"""Sim-to-real tools: delay, system identification, domain randomization, hardware budgets (lessons 37-40).

Plain NumPy. Lesson 37: a delay line, a delayed / sampled / predictive PD loop and the phase-margin delay
budget. Lesson 38: an output-error fit with a hand-written Levenberg-Marquardt loop. Lesson 39: a
domain-randomization sampler and the cross-entropy method. Lesson 40: servo ticks to radians, a
gravity-torque budget and preflight checks. Each piece is checked in experiments/<lesson>/build.py.
"""
from collections import deque

import numpy as np


# ---------------------------------------------------------------- lesson 37 · latency
class DelayLine:
    """Returns the value pushed n ticks ago (the first value until the line has filled)."""

    def __init__(self, n, x0):
        self.buf = deque([x0] * (n + 1), maxlen=n + 1)

    def __call__(self, x):
        self.buf.append(x)
        return self.buf[0]


def pd_delay_margin(inertia, kp, kd):
    """Crossover w_c (rad/s), phase margin (rad) and delay margin PM / w_c (s) of PD on an inertia."""
    w_c = np.sqrt((kd**2 + np.sqrt(kd**4 + 4 * inertia**2 * kp**2)) / (2 * inertia**2))   # |L(j w_c)| = 1
    pm = np.arctan2(kd * w_c, kp)                  # the PD numerator's phase lead at w_c
    return w_c, pm, pm / w_c


class DelayedPD:
    """PD on one joint that sees it nd ticks late, recomputes every `period` s (None: every tick) and holds
    the command in between; late=True applies each command one controller tick after computing it."""

    def __init__(self, kp, kd, target, dt, nd=0, period=None, late=False):
        self.kp, self.kd, self.target, self.dt, self.nd = kp, kd, target, dt, nd
        self.period, self.late = period, late
        self.k, self.next_tick, self.u, self.queued, self.line, self.sent = 0, 0.0, 0.0, 0.0, None, []

    def see(self, q, qd):
        """What the controller sees at this tick: the reading from nd ticks ago."""
        if self.line is None:
            self.line = DelayLine(self.nd, (q, qd))
        return self.line((q, qd))

    def __call__(self, q, qd):
        q_m, qd_m = self.see(q, qd)
        if self.period is None or self.k * self.dt >= self.next_tick - 1e-12:   # a controller tick
            self.next_tick += self.period or 0.0
            u = self.kp * (self.target - q_m) - self.kd * qd_m
            self.u, self.queued = (self.queued, u) if self.late else (u, u)
        self.k += 1
        self.sent.append(self.u)                   # torque applied during this tick
        return self.u


class PredictivePD(DelayedPD):
    """DelayedPD that integrates the model I qdd = tau over the assumed delay (model_nd ticks) from the delayed
    reading, with the torques it already sent, and controls that predicted present state."""

    def __init__(self, kp, kd, target, dt, nd, model_inertia, model_nd):
        super().__init__(kp, kd, target, dt, nd)
        self.model_inertia, self.model_nd = model_inertia, model_nd

    def see(self, q, qd):
        q_m, qd_m = super().see(q, qd)
        j = max(self.k - self.nd, 0)               # tick of the reading
        for i in range(j, min(j + self.model_nd, self.k)):
            qd_m += self.sent[i] / self.model_inertia * self.dt   # same semi-implicit Euler as the plant
            q_m += qd_m * self.dt
        return q_m, qd_m


def overshoot(q, target):
    """Overshoot past a positive step target in percent; None if the run diverged (|q| > 50)."""
    if not np.all(np.abs(q) < 50):
        return None
    return max(0.0, (q.max() - target) / target * 100)


def is_stable(q, target):
    """Stable if the run stayed bounded and the late error (last quarter) is below 90% of the error in the
    second quarter, or below 1 mrad: a decaying response, not a sustained or growing oscillation."""
    if overshoot(q, target) is None:
        return False
    e, n = np.abs(q - target), len(q)
    late, early = e[3 * n // 4:].max(), e[n // 4:n // 2].max()
    return late < 0.9 * early or late < 1e-3


# ---------------------------------------------------------------- lesson 38 · system identification
def residual_jacobian(resid, x, r0=None, rel_step=1e-3):
    """Forward-difference Jacobian of a residual vector: one extra simulation per parameter."""
    r0 = resid(x) if r0 is None else r0
    J = np.empty((len(r0), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = rel_step * max(1.0, abs(x[i]))
        J[:, i] = (resid(x + dx) - r0) / dx[i]
    return J


def levenberg_marquardt(resid, x0, lb, ub, iters=100, lam=1e-3, ftol=1e-10):
    """Minimize |resid(x)|^2 inside the box [lb, ub]: Gauss-Newton steps, damped by lam and clipped to the box.
    Returns the solution, its residual, the Jacobian at the last linearization and the number of evaluations."""
    x = np.clip(np.asarray(x0, float), lb, ub)
    r = resid(x)
    cost, nfev = r @ r, 1
    for _ in range(iters):
        J = residual_jacobian(resid, x, r)
        nfev += len(x)
        A, g = J.T @ J, J.T @ r
        while True:                                # raise the damping until the step lowers the cost
            x_new = np.clip(x + np.linalg.solve(A + lam * np.diag(np.diag(A)), -g), lb, ub)
            r_new = resid(x_new)
            nfev += 1
            if r_new @ r_new < cost or lam > 1e10:
                break
            lam *= 10
        if r_new @ r_new >= cost:                  # no step helps: we are at the minimum
            break
        done = cost - r_new @ r_new < ftol * cost
        x, r, cost, lam = x_new, r_new, r_new @ r_new, lam / 10
        if done:
            break
    return x, r, J, nfev


# ---------------------------------------------------------------- lesson 39 · domain randomization
def dr_sample(rng, n, w, delay0=0.005, delay_per_w=0.040):
    """n random plants of width w: mass, damping and friction scales log-uniform in [2^-w, 2^w], delay uniform in
    [delay0, delay0 + delay_per_w * w] seconds. w = 0 returns n nominal plants and draws nothing from rng."""
    if w == 0:
        return dict(mass=np.ones(n), damping=np.ones(n), friction=np.ones(n), delay=np.full(n, delay0))
    s = lambda: 2.0 ** rng.uniform(-w, w, n)  # noqa: E731
    return dict(mass=s(), damping=s(), friction=s(), delay=rng.uniform(delay0, delay0 + delay_per_w * w, n))


def inside_support(plants, w, delay0=0.005, delay_per_w=0.040):
    """For every plant and parameter: can the width-w training distribution draw this value?"""
    lo, hi, eps = 2.0 ** -w, 2.0 ** w, 1e-12
    out = {k: (plants[k] >= lo - eps) & (plants[k] <= hi + eps) for k in ("mass", "damping", "friction")}
    out["delay"] = (plants["delay"] >= delay0 - eps) & (plants["delay"] <= delay0 + delay_per_w * w + eps)
    return out


def expected_cost(rollout, gains, plants):
    """Mean cost of every controller (row of gains) over the same sampled plants (common random numbers).
    rollout(gain_rows, plant_rows) runs one episode per row and returns its cost."""
    n, k = len(gains), len(plants["delay"])
    tiled = {key: np.tile(v, n) for key, v in plants.items()}       # plants 0..k-1 for controller 0, then 1, ...
    return rollout(np.repeat(gains, k, axis=0), tiled).reshape(n, k).mean(axis=1)


def cem(cost, mu, sd, lo, hi, rng, iters=12, pop=24, n_elite=6, sd_extra=0.02):
    """Cross-entropy method: draw pop candidates from N(mu, sd) clipped to [lo, hi], keep the n_elite cheapest,
    refit mu and sd to them (plus sd_extra, so the search never collapses), repeat. cost(x, rng) -> one per row."""
    for _ in range(iters):
        x = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), lo, hi)
        elite = x[np.argsort(cost(x, rng))[:n_elite]]
        mu, sd = elite.mean(axis=0), elite.std(axis=0) + sd_extra
    return mu


# ---------------------------------------------------------------- lesson 40 · hardware bookkeeping
def ticks_to_deg(raw, range_min, range_max, resolution=4096):
    """LeRobot's degrees mode: 0 at the middle of the calibrated range, 360 deg per resolution - 1 ticks."""
    return (raw - (range_min + range_max) / 2) * 360 / (resolution - 1)


def deg_to_ticks(deg, range_min, range_max, resolution=4096):
    """Back to an encoder position; int() truncates toward zero, as LeRobot's _unnormalize does."""
    return int(deg * (resolution - 1) / 360 + (range_min + range_max) / 2)


def fit_calibration(deg, q_model):
    """Sign / scale and offset with q_model = a * radians(deg) + b, least squares over poses where both are known."""
    a, b = np.polyfit(np.radians(deg), q_model, 1)
    return a, b


def gravity_torques(m, qpos, frames, g=(0.0, 0.0, -9.81)):
    """Joint torques gravity puts on a hinge chain at rest: for hinge i, the sum over the bodies it carries of
    (a_i x (c_b - o_i)) . (m_b g). frames: world 4x4 pose of every body (lesson 07's frame_tree)."""
    tau = np.zeros(m.njnt)
    for j in range(m.njnt):
        if m.jnt_type[j] != 3:                     # hinges only
            continue
        T = frames[m.jnt_bodyid[j]]
        a, o = T[:3, :3] @ m.jnt_axis[j], T[:3, :3] @ m.jnt_pos[j] + T[:3, 3]   # axis and anchor in the world
        for b in range(1, m.nbody):
            p = b
            while p not in (0, m.jnt_bodyid[j]):   # is body b on the far side of joint j?
                p = m.body_parentid[p]
            if p == m.jnt_bodyid[j]:
                c = frames[b][:3, :3] @ m.body_ipos[b] + frames[b][:3, 3]          # centre of mass
                tau[j] += np.cross(a, c - o) @ (m.body_mass[b] * np.asarray(g))
    return tau


def max_payload(tau0, per_kg, limit):
    """Smallest payload (kg) at which some |tau0 + p * per_kg| reaches `limit`; rows are poses, columns joints."""
    p = np.inf
    for t, k in zip(np.ravel(tau0), np.ravel(per_kg)):
        if k != 0:
            for target in (limit, -limit):
                root = (target - t) / k
                if root >= 0:
                    p = min(p, root)
    return p
