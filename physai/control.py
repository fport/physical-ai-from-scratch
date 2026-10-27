"""Joint controllers and a one-joint simulator (lesson 12 · Build it).

Plain NumPy: the gravity torque of a single link, a semi-implicit Euler step (the integrator
MuJoCo's default `Euler` option uses), the PD law and a PID controller with an integrator clamp,
gravity feedforward and a torque limit. Each piece is checked against MuJoCo in build.py.
"""
from dataclasses import dataclass

import numpy as np


def gravity_torque(q, m, l, g=9.81):
    """Torque gravity puts on a hinge with a point mass m at distance l (q = 0 horizontal, q > 0 down)."""
    return m * g * l * np.cos(q)


def step_joint(q, qd, tau, inertia, dt, load=0.0):
    """One semi-implicit Euler step: new speed from the torques, then new angle from the new speed."""
    qdd = (tau + load) / inertia                   # motor torque plus load torque (e.g. gravity)
    qd = qd + dt * qdd
    q = q + dt * qd                                # uses the updated speed: this is the "semi"
    return q, qd


def simulate_joint(torque_fn, q0, inertia, dt, steps, load_fn=None):
    """Run a torque controller on one joint from rest at q0; returns the angle after every step."""
    q, qd, qs = q0, 0.0, []
    for _ in range(steps):
        load = load_fn(q) if load_fn else 0.0
        q, qd = step_joint(q, qd, torque_fn(q, qd), inertia, dt, load)
        qs.append(q)
    return np.array(qs)


def pd_torque(q, qd, q_target, kp, kd):
    """PD law: a spring toward q_target (stiffness kp) and a damper against the joint speed (kd)."""
    return kp * (q_target - q) - kd * qd


def pd_sag(kp, mgl, iters=200):
    """Steady error e of a PD joint holding a link at horizontal: the fixed point of kp e = mgl cos(e)."""
    e = 0.0
    for _ in range(iters):
        e = mgl * np.cos(e) / kp
    return e


@dataclass
class PID:
    """PD + integral + feedforward, with an anti-windup clamp and the motor's torque limit."""
    kp: float
    kd: float
    ki: float = 0.0
    i_max: float = np.inf      # anti-windup: the integral term may give at most this torque (N m)
    tau_max: float = np.inf    # motor torque limit (N m)
    integ: float = 0.0         # running sum of the error (rad s)

    def __call__(self, q, qd, q_target, dt, tau_ff=0.0):
        self.integ += (q_target - q) * dt
        if self.ki:                                # clamp the sum, not just the output
            lim = self.i_max / self.ki
            self.integ = min(max(self.integ, -lim), lim)
        tau = pd_torque(q, qd, q_target, self.kp, self.kd) + self.ki * self.integ + tau_ff
        return min(max(tau, -self.tau_max), self.tau_max)


# ---- Task-space control (lesson 14 · Build it) ----

def orientation_error(R, R_des):
    """Rotation (rad, world axes) that turns R toward R_des: half the sum of column cross products."""
    return 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))


def task_inertia(J, M):
    """The arm's inertia as felt at the tool: Lambda = (J M^-1 J^T)^-1."""
    return np.linalg.inv(J @ np.linalg.solve(M, J.T))


def impedance_wrench(e, v, K, D, Kr, Dr):
    """A spring-damper at the tool: force K e - D v (N) and moment Kr e_rot - Dr w (N m)."""
    return np.r_[K * e[:3] - D * v[:3], Kr * e[3:] - Dr * v[3:]]


def osc_wrench(Lam, e, v, kp, kd):
    """Operational space: the wrench that gives every direction the unit-mass response kp e - kd v."""
    return Lam @ (kp * e - kd * v)


def null_space_projector(J, M, Lam):
    """N = I - J^T Jbar^T with Jbar = M^-1 J^T Lambda: N tau0 never accelerates the tool."""
    Jbar = np.linalg.solve(M, J.T) @ Lam
    return np.eye(M.shape[0]) - J.T @ Jbar.T


def task_torque(J, F, bias, lim, N=None, tau0=None):
    """Joint torques J^T F + bias (+ N tau0), clipped joint by joint at the limits; also returns 'clipped'."""
    tau = J.T @ F + bias
    if N is not None:
        tau = tau + N @ tau0
    return np.clip(tau, -lim, lim), bool(np.any(np.abs(tau) > lim))
