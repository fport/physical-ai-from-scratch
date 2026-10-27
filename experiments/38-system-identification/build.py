"""Lesson 38 · Build it: an output-error fit of the SO-101's parameters with a hand-written Levenberg-Marquardt loop.

The black box is the Experiment's: a MuJoCo replay of logged commands with six parameters (mass scale, armature,
damping, friction loss, kp scale, delay), fitted in log-ratios to nominal (delay in 10 ms units).

Step 1  Fingerprints: the forward-difference Jacobian of the residual at two excitation amplitudes.
Step 2  Levenberg-Marquardt by hand on the noise-free 0.1 rad log: recover the hidden parameters.
Step 3  The same fit with 2 mrad of noise (the Experiment's part B draw); validation error.
Step 4  Why some parameters wander: singular values of the Jacobian and the sloppy direction.
Step 5  A missing mechanism: fit without the delay parameter.
Use it  scipy.optimize.least_squares on the same residual.

The functions live in physai/sim2real.py. Run:  uv run python experiments/38-system-identification/build.py  (~6 s)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np
import scipy
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim2real import levenberg_marquardt, residual_jacobian  # noqa: E402

spec = importlib.util.spec_from_file_location("exp38", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)                       # the Experiment's simulator, excitation and parameters

NAMES = ["mass", "armature", "damping", "friction", "kp", "delay"]
LB, UB, X0 = np.array([-1.5] * 5 + [0.0]), np.array([1.5] * 5 + [5.0]), np.array([0.0] * 5 + [0.5])
VAL_CMD = exp.excitation("validation")
VAL_REAL = exp.simulate(exp.REAL, VAL_CMD)


def residual(cmd, q_log, fixed_delay=None):
    """Logged minus simulated joint positions (mrad) for log-ratio parameters x."""
    def r(x):
        theta = exp.to_theta(np.append(x, 0.0) if fixed_delay is not None else x, fixed_delay)
        return 1e3 * (exp.simulate(theta, cmd) - q_log).ravel()
    return r


def row(label, theta, nfev):
    e = exp.param_err(theta)
    val = exp.rms_mrad(exp.simulate(theta, VAL_CMD), VAL_REAL)
    cols = " ".join(f"{v:+8.1f}" for v in e[:5])
    print(f"{label:<22s} {cols} {e[5]:+8.1f} {val:9.2f} {nfev:5d}")
    return val


def header():
    print(f"{'':<22s} " + " ".join(f"{n[:7] + '%':>8s}" for n in NAMES[:5]) + f" {'dly ms':>8s} {'val mrad':>9s} {'evals':>5s}")


print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")
print(f"validation trajectory, nominal vs real: {exp.rms_mrad(exp.simulate(exp.NOMINAL, VAL_CMD), VAL_REAL):.2f} mrad RMS")

print("\n== Step 1. Fingerprints: |column| of the residual Jacobian at nominal (mrad per unit of x) ==")
print(f"{'amplitude':<10s} " + " ".join(f"{n:>9s}" for n in NAMES))
for amp in (0.02, 0.1):
    cmd = exp.excitation("multisine", amp)
    q_log = exp.simulate(exp.REAL, cmd)
    J = residual_jacobian(residual(cmd, q_log), X0)
    print(f"{amp:<10.2f} " + " ".join(f"{v:9.1f}" for v in np.linalg.norm(J, axis=0)))
print(f"noise floor of the same log, 2 mrad x sqrt({J.shape[0]} residuals) = {2.0 * np.sqrt(J.shape[0]):.1f} mrad")

CMD = exp.excitation("multisine", 0.1)
Q_CLEAN = exp.simulate(exp.REAL, CMD)

print("\n== Step 2. Levenberg-Marquardt by hand, noise-free log at 0.1 rad ==")
header()
x, r, J, nfev = levenberg_marquardt(residual(CMD, Q_CLEAN), X0, LB, UB, iters=15)
row("ours", exp.to_theta(x), nfev)
print(f"final cost |r|^2 = {r @ r:.1e} mrad^2")

print("\n== Step 3. The same fit with 2 mrad of noise (the Experiment's part B draw) ==")
rng = np.random.default_rng(1)                     # part B: sigma 0, then 2 mrad, from this generator
rng.normal(0, 0.0, Q_CLEAN.shape)
Q_NOISY = Q_CLEAN + rng.normal(0, 0.002, Q_CLEAN.shape)
res_noisy = residual(CMD, Q_NOISY)
header()
x2, r2, J2, nfev2 = levenberg_marquardt(res_noisy, X0, LB, UB)
row("ours", exp.to_theta(x2), nfev2)
print(f"final cost |r|^2 = {r2 @ r2:.2f} mrad^2 over {len(r2)} residuals (2 mrad noise alone: {len(r2) * 4:.0f})")

print("\n== Step 4. The sloppy direction: singular values of J at the noisy solution ==")
J2 = residual_jacobian(res_noisy, x2, r2)
U, s, Vt = np.linalg.svd(J2, full_matrices=False)
print("singular values: " + " ".join(f"{v:.1f}" for v in s) + f"  (ratio {s[0] / s[-1]:.0f})")
print("weakest direction: " + " ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, Vt[-1])))
sd = 2.0 * np.sqrt(np.diag(np.linalg.inv(J2.T @ J2)))   # 2 mrad noise -> standard deviation of x
print("predicted 1-sigma spread: " + " ".join(f"{n} {100 * v:.1f}%" for n, v in zip(NAMES[:5], sd[:5]))
      + f" delay {10 * sd[5]:.2f} ms")

print("\n== Step 5. A missing mechanism: no delay parameter, noise-free log ==")
header()
x5, _, _, nfev5 = levenberg_marquardt(residual(CMD, Q_CLEAN, fixed_delay=0.0), X0[:5], LB[:5], UB[:5], iters=30)
theta5 = exp.to_theta(np.append(x5, 0.0), 0.0)
res5 = residual(CMD, Q_CLEAN, fixed_delay=0.0)
row("ours, no delay", theta5, nfev5)
print(f"{'':<22s} cost |r|^2 = {res5(x5) @ res5(x5):.1f} mrad^2")
print("fitted: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(NAMES[:5], theta5[:5])))

print("\n== Use it. scipy.optimize.least_squares on the same residuals ==")
header()
for label, res, n in (("noise-free", residual(CMD, Q_CLEAN), 6), ("2 mrad noise", res_noisy, 6), ("no delay", res5, 5)):
    calls = []                                     # count every simulation, Jacobian columns included
    counted = lambda x, res=res: calls.append(1) or res(x)  # noqa: E731
    s_ = least_squares(counted, X0[:n], bounds=(LB[:n], UB[:n]), diff_step=1e-3, xtol=1e-8, ftol=1e-10, gtol=1e-10,
                       max_nfev=150)
    theta = exp.to_theta(s_.x) if n == 6 else exp.to_theta(np.append(s_.x, 0.0), 0.0)
    row(f"SciPy, {label}", theta, len(calls))
    print(f"{'':<22s} cost |r|^2 = {2 * s_.cost:.2f} mrad^2, nfev (without Jacobian calls) {s_.nfev}")
