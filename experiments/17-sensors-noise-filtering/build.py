"""Lesson 17 · Build it: an encoder model, two smoothing filters, velocity and delay, checked at every step.

Step 1  The encoder: rounding error variance vs step^2 / 12, and the reading vs the Experiment's encoder.
Step 2  The moving average: window mean, noise left vs 1/sqrt(n), delay on a ramp vs (n - 1) / 2.
Step 3  The first-order low-pass: vs scipy.signal.lfilter, noise left and delay on a ramp vs the formulas.
Step 4  Velocity by central differences: vs np.gradient, raw velocity noise vs sqrt(2) sigma / (2 dt).
Step 5  Delay and error of every filter on the lesson's signal vs the Experiment table.

The functions live in physai/estimation.py (the lesson shows them step by step).
The signal (MuJoCo shoulder_pan) is taken from experiment.py, so both tell one story.
Run:  uv run python experiments/17-sensors-noise-filtering/build.py      (~3 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np
import scipy
from scipy.signal import lfilter, lfilter_zi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.estimation import (central_difference, delay_samples, encoder_reading, low_pass,  # noqa: E402
                               moving_average)

spec = importlib.util.spec_from_file_location("filter_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
DT, K0 = exp.DT, 500                            # the Experiment scores after the first 500 samples

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")

q, v, _, _, _ = exp.simulate(exp.build_model())
step = 2 * np.pi / exp.COUNTS


def rms(e):
    return float(np.sqrt(np.mean(e**2)))


print(f"\n== Step 1. The encoder: {exp.COUNTS} counts/rev, sigma {exp.SIGMA * 1000:.0f} mrad ==")
angles = np.random.default_rng(1).uniform(-np.pi, np.pi, 1_000_000)
round_err = encoder_reading(angles, exp.COUNTS, 0.0, np.random.default_rng(1)) - angles
print(f"step = {step * 1000:.3f} mrad, max |rounding error| = {np.abs(round_err).max() * 1000:.3f} mrad "
      f"(step / 2 = {step / 2 * 1000:.3f})")
print(f"rounding error std {round_err.std() * 1000:.4f} mrad vs step / sqrt(12) = {step / np.sqrt(12) * 1000:.4f} mrad")
z = encoder_reading(q, exp.COUNTS, exp.SIGMA, np.random.default_rng(exp.SEED))
z_exp = exp.encoder(q, np.random.default_rng(exp.SEED))
print(f"lesson signal: max |ours - Exp. encoder| = {np.abs(z - z_exp).max():.1e} rad")
print(f"expected reading error sqrt(sigma^2 + step^2/12) = {np.sqrt(exp.SIGMA**2 + step**2 / 12) * 1000:.2f} mrad, "
      f"measured {rms(z[K0:] - q[K0:]) * 1000:.2f} mrad")

print("\n== Step 2. Moving average: window mean, noise left, delay on a ramp ==")
x = np.random.default_rng(2).normal(size=200)
loop = np.array([x[max(0, k - 4):k + 1].mean() for k in range(len(x))])
print(f"n = 5: max |ours - loop over windows| = {np.abs(moving_average(x, 5) - loop).max():.1e}")
white = np.random.default_rng(3).normal(0, 1, 200_000)
ramp = np.arange(2000.0)
print("    n  noise std left  1/sqrt(n)  ramp lag samples  (n-1)/2")
for n in (5, 17, 65):
    lag = ramp[-1] - moving_average(ramp, n)[-1]
    print(f"{n:5d} {moving_average(white, n)[n:].std():15.4f} {1 / np.sqrt(n):10.4f} {lag:17.1f} {(n - 1) / 2:8.1f}")

print("\n== Step 3. First-order low-pass: vs scipy.signal.lfilter, noise left, delay on a ramp ==")
print("fc Hz   alpha  max |ours - lfilter|  noise std left  sqrt(a/(2-a))  ramp lag samples  tau/dt")
for fc in (50, 10, 2):
    a = DT / (1 / (2 * np.pi * fc) + DT)
    b_, a_ = [a], [1.0, a - 1.0]                       # y[k] = a x[k] + (1 - a) y[k-1]
    ref, _ = lfilter(b_, a_, z, zi=lfilter_zi(b_, a_) * z[0])
    y = low_pass(z, fc, DT)
    noise = low_pass(white, fc, DT)[2000:].std()
    lag = ramp[-1] - low_pass(ramp, fc, DT)[-1]
    tau_dt = 1 / (2 * np.pi * fc) / DT
    print(f"{fc:5d} {a:7.4f} {np.abs(y - ref).max():21.1e} {noise:15.4f} {np.sqrt(a / (2 - a)):14.4f} "
          f"{lag:17.2f} {tau_dt:7.2f}")

print("\n== Step 4. Velocity by central differences ==")
vel = central_difference(z, DT)
print(f"max |ours - np.gradient| = {np.abs(vel - np.gradient(z, DT)).max():.1e} rad/s")
pred = np.sqrt(2) * np.sqrt(exp.SIGMA**2 + step**2 / 12) / (2 * DT)
print(f"raw velocity error: predicted sqrt(2) * 10.01 mrad / (2 dt) = {pred:.2f} rad/s, "
      f"measured {rms(vel[K0:] - v[K0:]):.2f} rad/s (true velocity RMS {rms(v):.2f})")

print("\n== Step 5. Every filter on the lesson's signal vs the Experiment ==")
print("      filter  delay ms  theory ms  pos RMS mrad  vel RMS rad/s  max |ours - Exp.|")
for kind, knob in [("MA", n) for n in (1, 3, 5, 9, 17, 33, 65, 129)] + [("LPF", f) for f in (50, 20, 10, 2, 0.5)]:
    if kind == "MA":
        y, y_exp, theory, name = moving_average(z, knob), exp.moving_average(z_exp, knob), (knob - 1) / 2 * DT, f"MA n={knob}"
    else:
        y, y_exp, theory, name = low_pass(z, knob, DT), exp.low_pass(z_exp, knob), 1 / (2 * np.pi * knob), f"LPF {knob:g} Hz"
    ours = (1000 * DT * delay_samples(y, q), 1000 * rms(y[K0:] - q[K0:]), rms(central_difference(y, DT)[K0:] - v[K0:]))
    p, dl, ve = exp.score(y_exp, q, v)
    diff = max(abs(ours[0] - dl), abs(ours[1] - p), abs(ours[2] - ve))
    print(f"{name:>12} {ours[0]:9.1f} {1000 * theory:10.1f} {ours[1]:13.2f} {ours[2]:14.2f} {diff:18.1e}")
v_rms = rms(v[K0:])
print(f"lag error of MA n=17: speed x delay = {v_rms:.2f} rad/s x 16 ms = {v_rms * 16:.1f} mrad")
