"""Lesson 38 · System identification of the SO-101, sim-to-sim.

A "real" SO-101 is the Menagerie model with six hidden, perturbed parameters:
link mass scale, armature, joint damping, friction loss, servo gain (kp scale) and command delay.
It is driven by an excitation signal and logged (joint positions at 100 Hz, optional noise).
We then fit the same six parameters of a nominal model to the log with scipy least_squares
and check the fit on a held-out validation trajectory.

  A. Sweep the amplitude of a multisine excitation (noise 2 mrad) -> parameter error,
     validation error before/after, share of time a servo is torque-saturated.
  B. Sweep the measurement noise at 0.1 rad -> parameter error, validation error.
  Break it: fit without the delay parameter (wrong model structure).
  Exercise 3: refit the 0.4 rad row with the torque limit as a seventh parameter.

Run:  python experiment.py      (deterministic; well under a minute of CPU)
"""
import os
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

DT = 0.005                 # physics step (the Menagerie default)
LOG_DT = 0.01              # logging / command rate (100 Hz)
T_EXP = 4.0                # length of each excitation
Q0 = np.array([0.0, -0.6, 0.6, 0.6, 0.0, 0.0])
ARM = 5                    # identify from the five arm joints (gripper held at 0)
NAMES = ["mass scale", "armature", "damping", "frictionloss", "kp scale", "delay [ms]"]
NOMINAL = np.array([1.0, 0.028, 0.60, 0.052, 1.0, 0.0])
REAL = np.array([1.20, 0.040, 0.90, 0.100, 0.70, 0.015])   # hidden "real robot" (invented for this test)

BASE = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
BASE.opt.timestep = DT
MOVING = [BASE.body(n).id for n in ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                                    "camera_mount", "moving_jaw_so101_v1"]]
MASS0, INERTIA0 = BASE.body_mass.copy(), BASE.body_inertia.copy()
GAIN0, BIAS0 = BASE.actuator_gainprm.copy(), BASE.actuator_biasprm.copy()
FORCE0 = BASE.actuator_forcerange.copy()
DATA = mujoco.MjData(BASE)
N_STEPS = int(round(T_EXP / DT))
T_LOG = np.arange(0, T_EXP, LOG_DT)
T_PHYS = np.arange(N_STEPS) * DT
LOG_EVERY = int(round(LOG_DT / DT))


def set_params(theta):
    m = BASE
    s_m, arm, damp, fric, s_k = theta[:5]
    s_f = theta[6] if len(theta) > 6 else 1.0      # optional 7th parameter: torque-limit scale (exercise 3)
    m.body_mass[:] = MASS0
    m.body_inertia[:] = INERTIA0
    m.body_mass[MOVING] = MASS0[MOVING] * s_m
    m.body_inertia[MOVING] = INERTIA0[MOVING] * s_m
    m.dof_armature[:] = arm
    m.dof_damping[:] = damp
    m.dof_frictionloss[:] = fric
    m.actuator_gainprm[:, 0] = GAIN0[:, 0] * s_k
    m.actuator_biasprm[:, 1] = BIAS0[:, 1] * s_k
    m.actuator_forcerange[:] = FORCE0 * s_f


def simulate(theta, cmd):
    """cmd: (len(T_LOG), 6) goal positions sent at 100 Hz. Returns logged q (len(T_LOG), ARM)."""
    set_params(theta)
    delay = theta[5]
    # the servo receives the command stream delayed by `delay` (linear interpolation keeps it smooth in delay)
    u = np.stack([np.interp(T_PHYS - delay, T_LOG, cmd[:, j], left=cmd[0, j]) for j in range(6)], axis=1)
    d = DATA
    mujoco.mj_resetData(BASE, d)
    d.qpos[:6] = Q0
    out = np.zeros((len(T_LOG), ARM))
    for k in range(N_STEPS):
        if k % LOG_EVERY == 0:
            out[k // LOG_EVERY] = d.qpos[:ARM]
        d.ctrl[:6] = u[k]
        mujoco.mj_step(BASE, d)
    return out


SHAPE = np.array([1.0, 0.8, 0.8, 1.0, 1.3, 0.0])     # relative amplitude per joint


def excitation(kind, amp=0.1):
    """Goal positions at 100 Hz. amp: peak amplitude [rad] of the excitation (per-joint SHAPE applied)."""
    t = T_LOG[:, None]
    a = amp * SHAPE
    if kind == "multisine":
        rng = np.random.default_rng(1)
        freqs = np.array([0.5, 1.0, 1.7, 2.6, 3.5])
        sig = np.zeros((len(T_LOG), 6))
        for j in range(6):
            ph = rng.uniform(0, 2 * np.pi, len(freqs))
            s = np.sin(2 * np.pi * freqs * t + ph).sum(axis=1)
            sig[:, j] = a[j] * s / np.abs(s).max()
    elif kind == "validation":      # smoothed random goals, never used for fitting
        rng = np.random.default_rng(100)
        tk = np.linspace(0, T_EXP, 9)
        knots = rng.uniform(-1, 1, (9, 6)) * 0.25 * SHAPE
        sig = np.stack([np.interp(T_LOG, tk, knots[:, j]) for j in range(6)], axis=1)
        kernel = np.ones(15) / 15
        sig = np.stack([np.convolve(sig[:, j], kernel, mode="same") for j in range(6)], axis=1)
    else:
        raise ValueError(kind)
    sig[:, 5] = 0.0
    return Q0 + sig


def saturated_fraction(theta, cmd):
    """Fraction of physics steps in which at least one arm servo is at its torque limit."""
    set_params(theta)
    u = np.stack([np.interp(T_PHYS - theta[5], T_LOG, cmd[:, j], left=cmd[0, j]) for j in range(6)], axis=1)
    d = DATA
    mujoco.mj_resetData(BASE, d)
    d.qpos[:6] = Q0
    lim = BASE.actuator_forcerange[0, 1] - 1e-6
    n = 0
    for k in range(N_STEPS):
        d.ctrl[:6] = u[k]
        mujoco.mj_step(BASE, d)
        n += bool((np.abs(d.actuator_force[:ARM]) >= lim).any())
    return n / N_STEPS


# unknowns in normalized coordinates: log-ratio to nominal for the five positive ones, delay in 10 ms units
def to_theta(x, fixed_delay=None):
    theta = NOMINAL.copy()
    theta[:5] = NOMINAL[:5] * np.exp(x[:5])
    theta[5] = 0.01 * x[5] if fixed_delay is None else fixed_delay
    return theta


def fit(cmd, q_log, with_delay=True):
    nx = 6 if with_delay else 5

    def resid(x):
        xx = np.append(x, 0.0) if not with_delay else x
        return 1e3 * (simulate(to_theta(xx, None if with_delay else 0.0), cmd) - q_log).ravel()   # mrad

    lb = np.array([-1.5] * 5 + [0.0])[:nx]
    ub = np.array([1.5] * 5 + [5.0])[:nx]
    x0 = np.array([0.0] * 5 + [0.5])[:nx]          # start inside the bounds (delay 5 ms)
    r = least_squares(resid, x0, bounds=(lb, ub), diff_step=1e-3, x_scale=1.0,
                      xtol=1e-8, ftol=1e-10, gtol=1e-10, max_nfev=150)
    xx = np.append(r.x, 0.0) if not with_delay else r.x
    theta = to_theta(xx, None if with_delay else 0.0)
    # parameter correlation from the Jacobian at the solution
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        corr = np.full((nx, nx), np.nan)
    return theta, r.nfev, corr


def rms_mrad(a, b):
    return 1e3 * np.sqrt(np.mean((a - b) ** 2))


def param_err(theta):
    e = np.zeros(6)
    e[:5] = (theta[:5] - REAL[:5]) / REAL[:5] * 100
    e[5] = (theta[5] - REAL[5]) * 1e3            # delay error in ms
    return e


def report(label, theta, nfev, val_cmd, val_real, extra=""):
    pe = param_err(theta)
    v_nom = rms_mrad(simulate(NOMINAL, val_cmd), val_real)
    v_fit = rms_mrad(simulate(theta, val_cmd), val_real)
    cols = " | ".join(f"{p:>+7.1f}" for p in pe[:5]) + f" | {pe[5]:>+6.1f}"
    print(f"    {label:<14} | {cols} | {v_nom:>8.2f} | {v_fit:>8.2f} | {nfev:>4}{extra}")
    return pe, v_fit


def header():
    print(f"    {'':<14} | {'mass%':>7} | {'arm%':>7} | {'damp%':>7} | {'fric%':>7} | {'kp%':>7} | "
          f"{'dly ms':>6} | {'val nom':>8} | {'val fit':>8} | nfev")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · "
          f"scipy {__import__('scipy').__version__}")
    print()
    print("Hidden 'real' parameters vs nominal model (shared by all STS3215 joints):")
    for n, a, b in zip(NAMES, NOMINAL, REAL):
        bb = b * 1e3 if "delay" in n else b
        aa = a * 1e3 if "delay" in n else a
        print(f"    {n:<14} nominal {aa:>7.3f}   real {bb:>7.3f}")
    print(f"Excitation {T_EXP:.0f} s, positions logged at {1 / LOG_DT:.0f} Hz, physics {DT * 1e3:.0f} ms.")
    print("Table columns: parameter error after the fit (percent of the real value; delay in ms),")
    print("validation RMS error [mrad] of the nominal and the fitted model on a held-out trajectory.")
    print()

    val_cmd = excitation("validation")
    val_real = simulate(REAL, val_cmd)
    print(f"Validation trajectory: real-vs-nominal RMS {rms_mrad(simulate(NOMINAL, val_cmd), val_real):.2f} mrad")
    print()

    sigma_a = 0.002
    print(f"[A] Multisine excitation (0.5-3.5 Hz), sweep its amplitude; noise sigma {sigma_a * 1e3:.0f} mrad")
    header()
    rng = np.random.default_rng(0)
    for amp in [0.005, 0.02, 0.05, 0.1, 0.2, 0.4]:
        cmd = excitation("multisine", amp)
        q_clean = simulate(REAL, cmd)
        q_log = q_clean + rng.normal(0, sigma_a, q_clean.shape)
        theta, nfev, corr = fit(cmd, q_log)
        sat = saturated_fraction(REAL, cmd)
        report(f"amp {amp:.3f} rad", theta, nfev, val_cmd, val_real, f" | sat {sat * 100:>5.1f}%")
    print()

    amp_b = 0.1
    print(f"[B] Multisine amplitude {amp_b} rad, sweep the measurement noise (white, per joint, per sample)")
    header()
    cmd = excitation("multisine", amp_b)
    q_clean = simulate(REAL, cmd)
    rng = np.random.default_rng(1)
    for sigma in [0.0, 0.002, 0.01, 0.03]:
        q_log = q_clean + rng.normal(0, sigma, q_clean.shape)
        theta, nfev, _ = fit(cmd, q_log)
        report(f"sigma {sigma * 1e3:.0f} mrad", theta, nfev, val_cmd, val_real)
    print()

    print(f"[Break it] Multisine {amp_b} rad, no noise, model without a delay parameter (delay fixed at 0)")
    header()
    theta, nfev, _ = fit(cmd, q_clean, with_delay=False)
    report("no delay term", theta, nfev, val_cmd, val_real)
    print(f"    fitted values: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(NAMES[:5], theta[:5])))
    print()
    exercise_forcerange(val_cmd, val_real)


def exercise_forcerange(val_cmd, val_real):
    """Exercise 3: the 0.4 rad row of part A again (same noise draw), with the torque limit as a 7th parameter."""
    print("[Exercise 3] Part A, amp 0.400 rad, noise 2 mrad (same draw), torque-limit scale as a 7th parameter")
    rng = np.random.default_rng(0)                  # replay part A's noise sequence up to the 0.4 rad row
    for amp in [0.005, 0.02, 0.05, 0.1, 0.2, 0.4]:
        cmd = excitation("multisine", amp)
        noise = rng.normal(0, 0.002, (len(T_LOG), ARM))
    q_log = simulate(REAL, cmd) + noise

    def resid(x):
        theta = np.append(to_theta(x[:6]), np.exp(x[6]))
        return 1e3 * (simulate(theta, cmd) - q_log).ravel()

    lb, ub = np.array([-1.5] * 5 + [0.0, -1.0]), np.array([1.5] * 5 + [5.0, 1.0])
    x0 = np.array([0.0] * 5 + [0.5, 0.0])
    r = least_squares(resid, x0, bounds=(lb, ub), diff_step=1e-3, x_scale=1.0,
                      xtol=1e-8, ftol=1e-10, gtol=1e-10, max_nfev=150)
    theta = np.append(to_theta(r.x[:6]), np.exp(r.x[6]))
    pe = param_err(theta[:6])
    v_fit = rms_mrad(simulate(theta, val_cmd), val_real)
    set_params(NOMINAL)                              # restore the nominal torque limit
    header()
    cols = " | ".join(f"{p:>+7.1f}" for p in pe[:5]) + f" | {pe[5]:>+6.1f}"
    print(f"    {'7 parameters':<14} | {cols} | {rms_mrad(simulate(NOMINAL, val_cmd), val_real):>8.2f} | "
          f"{v_fit:>8.2f} | {r.nfev:>4} | torque-limit scale {theta[6]:.3f} (real 1.000)")


if __name__ == "__main__":
    main()
