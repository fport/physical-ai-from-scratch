"""Lesson 37 · Build it: a delayed control loop and its delay budget from scratch, checked at every step.

The joint is the Experiment's: I = 0.0901 kg m^2, no gravity, PD Kp 20, zeta 0.7, physics and control at 1 kHz.

Step 1  A delay line: the reading from n ticks ago.
Step 2  The delay budget: crossover, phase margin, PM / w_c; checked on the delayed characteristic equation.
Step 3  The delayed PD loop by hand vs the Experiment's MuJoCo loop; overshoot table and the stability boundary.
Step 4  Control rate: sample and hold, and one tick of compute, vs part A at T/2 and 3T/2.
Step 5  Prediction over the delay with a model, exact and wrong.

The functions live in physai/sim2real.py; the joint step is physai.control.step_joint (lesson 12).
Run:  uv run python experiments/37-latency/build.py      (~5 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.control import simulate_joint  # noqa: E402
from physai.sim2real import DelayedPD, DelayLine, PredictivePD, is_stable, overshoot, pd_delay_margin  # noqa: E402

spec = importlib.util.spec_from_file_location("exp37", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)                       # the Experiment's MuJoCo joint and loop (reference only)

I, DT, KP, TARGET = 0.0901, 0.001, 20.0, 0.5
KD = 2 * 0.7 * np.sqrt(KP * I)


def run(ctrl, seconds=4.0):
    """Our loop: the controller object on lesson 12's semi-implicit Euler joint."""
    with np.errstate(over="ignore", invalid="ignore"):
        return simulate_joint(ctrl, 0.0, I, DT, int(round(seconds / DT)))


def fmt(o):
    return "diverges" if o is None else f"{o:.2f}"


print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. A delay line of 3 ticks ==")
line = DelayLine(3, 0)
out = [line(k) for k in range(1, 9)]
print(f"pushed  {list(range(1, 9))}")
print(f"read    {out}")
print(f"equals the history shifted by 3 (first value until filled): {out == [0, 0, 0, 1, 2, 3, 4, 5]}")

print("\n== Step 2. The delay budget of PD on an inertia ==")
w_c, pm, d_max = pd_delay_margin(I, KP, KD)
L = (KP + KD * 1j * w_c) / (I * (1j * w_c) ** 2)
s = 1j * w_c
char = I * s ** 2 + (KP + KD * s) * np.exp(-s * d_max)       # I s^2 + (kp + kd s) e^{-s d} = 0 on the boundary
roots = np.roots([I ** 2, 0, -KD ** 2, 0, -KP ** 2])         # I^2 w^4 - kd^2 w^2 - kp^2 = 0
print(f"Kd = {KD:.3f}: w_c = {w_c:.2f} rad/s, PM = {np.degrees(pm):.1f} deg, d_max = PM / w_c = {d_max * 1e3:.1f} ms")
print(f"|L(j w_c)| = {abs(L):.12f}, phase of L = {np.degrees(np.angle(L)):.1f} deg = -180 + PM")
print(f"np.roots of the crossover quartic: w = {roots.real.max():.4f} rad/s (ours {w_c:.4f})")
print(f"|characteristic equation| at s = j w_c, d = d_max: {abs(char):.1e} (a pole sits on the imaginary axis)")
print(f"oscillation period at the boundary 2 pi / w_c = {2 * np.pi / w_c * 1e3:.0f} ms")

print("\n== Step 3. Delayed PD by hand vs the Experiment's MuJoCo loop ==")
print(f"{'delay ms':>8s} {'ours %':>9s} {'stable':>7s} {'max |q_ours - q_mujoco|':>24s}")
worst = 0.0
for ms in (0, 10, 20, 30, 40, 50, 60):
    ours = run(DelayedPD(KP, KD, TARGET, DT, nd=ms))
    ref = exp.run_joint(ms / 1000)
    n = np.isfinite(ref).sum()                     # the Experiment stops a diverged run at |q| > 50
    diff = np.abs(ours[:n] - ref[:n]).max()
    worst = max(worst, diff)
    print(f"{ms:8d} {fmt(overshoot(ours, TARGET)):>9s} {'yes' if is_stable(ours, TARGET) else 'NO':>7s} {diff:24.1e}")
lo, hi = 0, 120                                    # bisection on whole milliseconds, 10 s runs
while hi - lo > 1:
    mid = (lo + hi) // 2
    if is_stable(run(DelayedPD(KP, KD, TARGET, DT, nd=mid), 10.0), TARGET):
        lo = mid
    else:
        hi = mid
print(f"stability boundary: stable at {lo} ms, unstable at {hi} ms (predicted d_max {d_max * 1e3:.1f} ms)")

print("\n== Step 4. Control rate: sample and hold, and one tick of compute ==")
print(f"{'rate Hz':>7s} {'T ms':>6s} {'hold %':>9s} {'+1 tick %':>10s} {'delay T/2 %':>12s} {'delay 3T/2 %':>13s}")
for f in (1000, 100, 50, 30, 20, 10):
    T = 1.0 / f
    hold = run(DelayedPD(KP, KD, TARGET, DT, period=T))
    late = run(DelayedPD(KP, KD, TARGET, DT, period=T, late=True))
    half = run(DelayedPD(KP, KD, TARGET, DT, nd=round(T * 500)))
    three = run(DelayedPD(KP, KD, TARGET, DT, nd=round(T * 1500)))
    print(f"{f:7d} {T * 1e3:6.1f} {fmt(overshoot(hold, TARGET)):>9s} {fmt(overshoot(late, TARGET)):>10s} "
          f"{fmt(overshoot(half, TARGET)):>12s} {fmt(overshoot(three, TARGET)):>13s}")
same = np.abs(run(DelayedPD(KP, KD, TARGET, DT, period=1 / 30)) - exp.run_joint(rate_hz=30)).max()
print(f"30 Hz hold, ours vs the Experiment's MuJoCo loop: max diff {same:.1e} rad")

print("\n== Step 5. Predict the present state over a 40 ms delay ==")
print(f"{'controller':<34s} {'overshoot %':>11s}")
rows = [("no delay", DelayedPD(KP, KD, TARGET, DT)),
        ("delay, no prediction", DelayedPD(KP, KD, TARGET, DT, nd=40)),
        ("prediction, exact model", PredictivePD(KP, KD, TARGET, DT, 40, I, 40)),
        ("prediction, inertia +50%", PredictivePD(KP, KD, TARGET, DT, 40, 1.5 * I, 40)),
        ("prediction, inertia -30%", PredictivePD(KP, KD, TARGET, DT, 40, 0.7 * I, 40)),
        ("prediction, assumed delay 20 ms", PredictivePD(KP, KD, TARGET, DT, 40, I, 20))]
q = {}
for name, c in rows:
    q[name] = run(c)
    print(f"{name:<34s} {fmt(overshoot(q[name], TARGET)):>11s}")
print(f"exact prediction vs no delay: max |q difference| = {np.abs(q['prediction, exact model'] - q['no delay']).max():.1e} rad")
ref = exp.run_joint(0.040, predictor=dict(inertia=1.5 * I, delay=0.040))
print(f"inertia +50%, ours vs the Experiment's MuJoCo loop: max diff {np.abs(q['prediction, inertia +50%'] - ref).max():.1e} rad")

print("\n== Use it. MuJoCo's own delay buffer, SciPy's frequency response ==")


def mujoco_delayed(act="", sen=""):
    """The joint with MuJoCo's history buffers; the PD reads jointpos / jointvel sensors after every mj_step."""
    m = mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 0"/>
  <worldbody><body><joint name="j" axis="0 1 0"/>
    <inertial pos="0.3 0 0" mass="1" diaginertia="1e-4 1e-4 1e-4"/></body></worldbody>
  <actuator><motor joint="j" {act}/></actuator>
  <sensor><jointpos joint="j" {sen}/><jointvel joint="j" {sen}/></sensor>
</mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qs = []
    for _ in range(4000):
        d.ctrl[0] = KP * (TARGET - d.sensordata[0]) - KD * d.sensordata[1]
        mujoco.mj_step(m, d)
        qs.append(d.qpos[0])
    return np.array(qs)


print(f"{'MuJoCo model':<44s} {'overshoot %':>11s} {'ours, delay':>12s} {'ours %':>7s}")
for label, act, sen, nd in (("no delay attribute", "", "", 1),
                            ('sensors delay="0.039" nsample="39"', "", 'delay="0.039" nsample="39"', 40),
                            ('motor delay="0.039" nsample="39"', 'delay="0.039" nsample="39"', "", 40)):
    q_mj = mujoco_delayed(act, sen)
    ours = overshoot(run(DelayedPD(KP, KD, TARGET, DT, nd=nd)), TARGET)
    print(f"{label:<44s} {overshoot(q_mj, TARGET):11.2f} {f'{nd} ms':>12s} {ours:7.2f}")
same = np.abs(mujoco_delayed("", 'delay="0.039" nsample="39"') - run(DelayedPD(KP, KD, TARGET, DT, nd=40))).max()
print(f"sensor delay 39 ms + reading sensordata after mj_step vs ours at 40 ms: max diff {same:.1e} rad")

from scipy import optimize, signal  # noqa: E402

loop = signal.TransferFunction([KD, KP], [I, 0, 0])
gain_minus_one = lambda w: abs(signal.freqresp(loop, [w])[1][0]) - 1  # noqa: E731
w_sp = optimize.brentq(gain_minus_one, 1.0, 1000.0)
pm_sp = 180 + np.degrees(np.angle(signal.freqresp(loop, [w_sp])[1][0]))
print(f"SciPy freqresp + brentq: w_c = {w_sp:.4f} rad/s, PM = {pm_sp:.2f} deg, d_max = {np.radians(pm_sp) / w_sp * 1e3:.2f} ms "
      f"(ours {w_c:.4f}, {np.degrees(pm):.2f}, {d_max * 1e3:.2f})")
