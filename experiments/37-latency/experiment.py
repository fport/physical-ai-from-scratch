"""Lesson 37 · Control rate and end-to-end latency, measured in simulation.

  A. One joint (I = 0.0901 kg m^2, no gravity), PD torque control at 1 kHz, step 0 -> 0.5 rad.
     A pure delay d is injected between the joint and the controller (sensing + compute + bus).
     Sweep d -> overshoot, settling; find the delay where the loop goes unstable and compare it
     with the phase-margin estimate d_max = PM / w_c.
  B. Same joint, no added delay, but the controller runs at f Hz (sample, compute, hold).
     Sweep f, with and without one tick of compute delay -> overshoot; compare with part A at the
     equivalent delays T/2 and 3T/2.
  C. Mitigation at d = 40 ms: predict the current state from the delayed one with a model
     (integrate the dynamics over the delay using the torques already sent). Exact and wrong models,
     and pure extrapolation q + qd*d that ignores the torques (exercise 3).
  D. SO-101 (Menagerie), shoulder_pan servo. An outer "policy" loop at 30 Hz nudges the servo
     target toward a goal from delayed position readings (integral action, gain K).
     Sweep the delay -> overshoot; compare the boundary with K * d_eff = pi/2.
  Break it: fixed d = 30 ms, raise Kp at constant damping ratio -> the loop goes unstable.

Run:  python experiment.py      (deterministic; well under a minute of CPU)
"""
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

DT = 0.001
I_LINK = 0.0901            # 1 kg at 0.30 m (+1e-4 own inertia), as in lesson 12
KP, ZETA = 20.0, 0.7
TARGET = 0.5


def kd_for(kp, zeta=ZETA, inertia=I_LINK):
    return 2 * zeta * math.sqrt(kp * inertia)


def build_joint():
    spec = mujoco.MjSpec()
    spec.option.timestep = DT
    spec.option.gravity = [0, 0, 0]
    b = spec.worldbody.add_body(name="link")
    b.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02], pos=[0.3, 0, 0], mass=0)
    b.mass, b.ipos, b.inertia = 1.0, [0.3, 0, 0], [1e-4, 1e-4, 1e-4]
    b.explicitinertial = True
    a = spec.add_actuator(name="motor", target="hinge")
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.gainprm[0] = 1.0
    return spec.compile()


JOINT = build_joint()


def analytic_delay_margin(kp, kd, inertia=I_LINK):
    """Gain crossover and phase margin of L(s) = (kp + kd s) / (I s^2); d_max = PM / w_c."""
    # |kp + j kd w| = I w^2  ->  I^2 w^4 - kd^2 w^2 - kp^2 = 0
    w2 = (kd ** 2 + math.sqrt(kd ** 4 + 4 * inertia ** 2 * kp ** 2)) / (2 * inertia ** 2)
    wc = math.sqrt(w2)
    pm = math.atan2(kd * wc, kp)
    return wc, pm, pm / wc


def run_joint(delay_s=0.0, rate_hz=None, compute_tick=False, kp=KP, kd=None, T=4.0,
              predictor=None):
    """Step response. delay_s: pure measurement delay. rate_hz: controller sample rate (None = every step).
    predictor: None or dict(inertia=..., delay=...) used to predict the present state."""
    m, d = JOINT, mujoco.MjData(JOINT)
    kd = kd_for(kp) if kd is None else kd
    n = int(T / DT)
    nd = int(round(delay_s / DT))
    hist_q, hist_v, hist_u = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    q_log = np.zeros(n)
    u_hold, u_next, next_tick = 0.0, None, 0.0
    for k in range(n):
        hist_q[k], hist_v[k] = d.qpos[0], d.qvel[0]
        t = k * DT
        tick = rate_hz is None or t >= next_tick - 1e-12
        if tick:
            if rate_hz is not None:
                next_tick += 1.0 / rate_hz
            j = max(k - nd, 0)
            q_m, v_m = hist_q[j], hist_v[j]          # what the controller sees
            if predictor is not None:                # integrate the model over the assumed delay,
                npd = int(round(predictor["delay"] / DT))  # starting from the delayed reading,
                for i in range(j, min(j + npd, k)):        # using the torques already sent
                    if predictor["inertia"] is not None:     # inertia None: ignore the torques (q + qd*d)
                        v_m += hist_u[i] / predictor["inertia"] * DT
                    q_m += v_m * DT
            u = kp * (TARGET - q_m) - kd * v_m
            if compute_tick:                         # result is ready one tick later
                u_hold, u_next = (u_next if u_next is not None else 0.0), u
            else:
                u_hold = u
        hist_u[k] = u_hold
        d.ctrl[0] = u_hold
        mujoco.mj_step(m, d)
        q_log[k] = d.qpos[0]
        if abs(d.qpos[0]) > 50:                      # diverged: stop, mark the rest as NaN
            q_log[k:] = np.nan
            break
    return q_log


def metrics(q, T=None):
    if np.isnan(q).any():
        return None, None, False
    e = q - TARGET
    over = max(0.0, (q.max() - TARGET) / TARGET * 100)
    band = 0.02 * TARGET
    outside = np.nonzero(np.abs(e) > band)[0]
    settle = None if len(outside) == 0 else (outside[-1] + 1) * DT
    if settle is not None and settle >= len(q) * DT - 1e-9:
        settle = None
    n = len(q)
    late = np.abs(e[int(0.75 * n):]).max()
    early = np.abs(e[int(0.25 * n):int(0.5 * n)]).max()
    stable = late < 0.9 * early or late < 1e-3
    return over, settle, stable


def is_unstable(delay_s, kp=KP):
    q = run_joint(delay_s, kp=kp, T=10.0)
    return not metrics(q)[2]


def osc_period(q):
    e = q - TARGET
    tail = e[len(e) // 2:]
    zc = np.nonzero(np.diff(np.sign(tail)) != 0)[0]
    return 2 * np.mean(np.diff(zc)) * DT if len(zc) > 3 else float("nan")


def fmt_over(o):
    return "diverges" if o is None else f"{o:.2f}"


def fmt_settle(s):
    return "never" if s is None else f"{s:.3f}"


def part_a():
    kd = kd_for(KP)
    wc, pm, dmax = analytic_delay_margin(KP, kd)
    print(f"[A] One joint (I = {I_LINK} kg m^2), PD at 1 kHz, Kp {KP:.0f}, Kd {kd:.3f} (zeta {ZETA}), "
          f"step 0 -> {TARGET} rad, 4 s")
    print(f"    analytic: crossover w_c = {wc:.2f} rad/s, phase margin {math.degrees(pm):.1f} deg, "
          f"delay margin PM/w_c = {dmax * 1e3:.1f} ms")
    print(f"    {'delay ms':>8} | {'overshoot %':>11} | {'settle 2% s':>11} | stable")
    for dl in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]:
        o, s, st = metrics(run_joint(dl / 1000))
        print(f"    {dl:>8} | {fmt_over(o):>11} | {fmt_settle(s):>11} | {'yes' if st else 'NO'}")
    lo, hi = 0, 120                              # integer ms bisection on the 10 s criterion
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if is_unstable(mid / 1000):
            hi = mid
        else:
            lo = mid
    per = osc_period(run_joint(hi / 1000, T=10.0))
    print(f"    measured boundary: stable at {lo} ms, unstable at {hi} ms "
          f"(oscillation period at {hi} ms: {per * 1e3:.0f} ms, 2*pi/w_c = {2 * math.pi / wc * 1e3:.0f} ms)")
    print()
    return dmax


def part_b():
    print("[B] Same joint, no added delay, controller at f Hz (zero-order hold); physics at 1 kHz")
    print(f"    {'rate Hz':>7} | {'T ms':>6} | {'overshoot %':>11} | stable | "
          f"{'+1 tick compute: overshoot %':>28} | stable | {'A at T/2':>8} | {'A at 3T/2':>9}")
    for f in [1000, 500, 200, 100, 50, 30, 20, 10]:
        Tms = 1000 / f
        o1, _, s1 = metrics(run_joint(rate_hz=f))
        o2, _, s2 = metrics(run_joint(rate_hz=f, compute_tick=True))
        oa1, _, _ = metrics(run_joint(round(Tms / 2) / 1000))
        oa2, _, sa2 = metrics(run_joint(round(1.5 * Tms) / 1000))
        print(f"    {f:>7} | {Tms:>6.1f} | {fmt_over(o1):>11} | {'yes' if s1 else 'NO':>6} | {fmt_over(o2):>28} | "
              f"{'yes' if s2 else 'NO':>6} | {fmt_over(oa1):>8} | {fmt_over(oa2):>9}")
    print()


def part_c():
    dl = 0.040
    print(f"[C] Delay {dl * 1e3:.0f} ms; predict the present state with a model before applying PD")
    print(f"    {'controller':<44} | {'overshoot %':>11} | {'settle 2% s':>11}")
    rows = [("no delay (reference)", dict(delay_s=0.0)),
            ("delay, no prediction", dict(delay_s=dl)),
            ("delay + prediction, exact model", dict(delay_s=dl, predictor=dict(inertia=I_LINK, delay=dl))),
            ("delay + prediction, inertia +50%", dict(delay_s=dl, predictor=dict(inertia=1.5 * I_LINK, delay=dl))),
            ("delay + prediction, inertia -30%", dict(delay_s=dl, predictor=dict(inertia=0.7 * I_LINK, delay=dl))),
            ("delay + prediction, assumed delay 20 ms", dict(delay_s=dl, predictor=dict(inertia=I_LINK, delay=0.020))),
            ("delay + extrapolation q + qd*d (no torques)", dict(delay_s=dl, predictor=dict(inertia=None, delay=dl)))]
    for name, kw in rows:
        o, s, _ = metrics(run_joint(**kw))
        print(f"    {name:<44} | {fmt_over(o):>11} | {fmt_settle(s):>11}")
    print()


def so101_model():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    m.opt.timestep = 0.002
    return m


def servo_step_times(amp):
    """Plain servo step on shoulder_pan: time to 50% and peak overshoot."""
    m = so101_model()
    d = mujoco.MjData(m)
    q = np.zeros(500)
    for k in range(500):
        d.ctrl[0] = amp
        mujoco.mj_step(m, d)
        q[k] = d.qpos[0]
    return (np.argmax(q >= 0.5 * amp) + 1) * m.opt.timestep, (q.max() - amp) / amp * 100


def run_so101_outer(delay_s, K, goal, T=4.0, rate=30.0):
    m = so101_model()
    d = mujoco.MjData(m)
    pan = m.actuator("shoulder_pan").id
    n = int(T / m.opt.timestep)
    nd = int(round(delay_s / m.opt.timestep))
    hist = np.zeros(n + 1)
    q = np.zeros(n)
    target, next_tick = 0.0, 0.0
    for k in range(n):
        hist[k] = d.qpos[pan]
        t = k * m.opt.timestep
        if t >= next_tick - 1e-12:
            next_tick += 1.0 / rate
            q_seen = hist[max(k - nd, 0)]
            target += K * (1.0 / rate) * (goal - q_seen)
        d.ctrl[pan] = target
        mujoco.mj_step(m, d)
        q[k] = d.qpos[pan]
    over = max(0.0, (q.max() - goal) / goal * 100)
    late = np.abs(q[int(0.75 * n):] - goal).max() / goal
    return over, late


def discrete_boundary_ticks(KT):
    """Ideal servo (q = target): q[k+1] = q[k] + KT (g - q[k-n]). Smallest n with a root outside |z| = 1."""
    for n in range(0, 50):
        c = np.zeros(n + 2)
        c[0], c[1] = 1.0, -1.0
        c[-1] += KT
        if np.abs(np.roots(c)).max() > 1.0:
            return n


def part_d():
    K, rate = 10.0, 30.0
    T = 1.0 / rate
    print(f"[D] SO-101 shoulder_pan servo (Menagerie, 2 ms step). Outer loop at {rate:.0f} Hz: "
          f"target += K*T*(goal - q_delayed), K = {K:.0f} 1/s, 4 s")
    for amp in [0.02, 0.1, 0.5]:
        t50, ov = servo_step_times(amp)
        print(f"    servo alone, step {amp:.2f} rad: 50% after {t50 * 1e3:.0f} ms, overshoot {ov:.1f}%")
    nb = discrete_boundary_ticks(K * T)
    print(f"    estimates for an ideal servo: continuous K*(d + T/2) = pi/2 -> d = {(math.pi / 2 / K - T / 2) * 1e3:.0f} ms; "
          f"discrete -> unstable from {nb} ticks = {nb * T * 1e3:.0f} ms")
    print(f"    {'delay ms':>8} | {'goal 0.1: overshoot %':>21} | {'late |e|/goal':>13} | "
          f"{'goal 0.5: overshoot %':>21} | {'late |e|/goal':>13}")
    for dl in [0, 17, 33, 50, 67, 83, 100, 117, 133]:
        o1, l1 = run_so101_outer(dl / 1000, K, 0.1)
        o2, l2 = run_so101_outer(dl / 1000, K, 0.5)
        print(f"    {dl:>8} | {o1:>21.1f} | {l1:>13.3f} | {o2:>21.1f} | {l2:>13.3f}")
    print()


def break_it():
    dl = 0.030
    print(f"[Break it] Fixed delay {dl * 1e3:.0f} ms, raise Kp at zeta = {ZETA} (Kd scaled with it)")
    print(f"    {'Kp':>5} | {'Kd':>6} | {'w_c rad/s':>9} | {'delay margin ms':>15} | {'overshoot %':>11} | stable")
    for kp in [5, 10, 20, 40, 80, 160]:
        kd = kd_for(kp)
        wc, _, dmax = analytic_delay_margin(kp, kd)
        q = run_joint(dl, kp=kp, T=10.0)
        o, _, st = metrics(q)
        print(f"    {kp:>5} | {kd:>6.3f} | {wc:>9.2f} | {dmax * 1e3:>15.1f} | {fmt_over(o):>11} | {'yes' if st else 'NO'}")
    print()


if __name__ == "__main__":
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    print()
    part_a()
    part_b()
    part_c()
    part_d()
    break_it()
