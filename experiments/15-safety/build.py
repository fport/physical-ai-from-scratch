"""Lesson 15 · Build it: the safety layers from scratch, checked against MuJoCo and the Experiment.

Step 1  The goal clamp (position limit + relative-target limit) and its guarantees.
Step 2  The capped position servo vs MuJoCo's actuator_force (ctrlrange and forcerange included).
Step 3  Part A again, with our layers in the loop: the same table; the steady push vs tau_max / r.
Step 4  A watchdog object: part D's timeout rows.
Step 5  An E-stop latch: what a latch guarantees, then part E's two ways to stop.

The functions live in physai/safety.py (the lesson shows them step by step).
The model, the post and the force measurement come from experiment.py, so both tell one story.
Run:  uv run python experiments/15-safety/build.py      (~3 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.safety import EStop, Watchdog, clamp_goal, servo_torque  # noqa: E402

spec = importlib.util.spec_from_file_location("safety_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
DT, TICK, POSE = exp.DT, exp.TICK, exp.POSE

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")


def run_layers(command, T=2.0, cap=None, q_lim=None, delta=None, timeout=None, stop=None):
    """The Experiment's loop with our layers: command(t) -> (pan target or None, is_velocity).
    stop = (kind, t): the E-stop is pressed at t; kind 'hold' (goal := present) or 'torque off'."""
    m = exp.build(cap)
    d = mujoco.MjData(m)
    d.qpos[:6], d.ctrl[:6] = POSE, POSE
    mujoco.mj_forward(m, d)
    post, f6, pan = m.geom("post").id, np.zeros(6), m.actuator("shoulder_pan").id
    gz0 = d.site("gripperframe").xpos[2]
    wd, es = Watchdog(timeout if timeout is not None else np.inf), EStop()
    target, v_cmd, q_last, t_last, silent = 0.0, 0.0, None, None, False
    out = dict(peak_force=0.0, peak_speed=0.0, peak_tau=0.0, impact_speed=None, t_stop=None, max_travel=0.0)
    forces, step = [], 0
    while d.time < T - 1e-9:
        if step % TICK == 0:
            t = d.time
            if es.update(stop is not None and t >= stop[1]):
                if not silent:                                      # first tick of the stop
                    silent, q_last, t_last = True, d.qpos[0], t
                    if stop[0] == "hold":
                        target, v_cmd = d.qpos[0], 0.0
                    else:
                        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
            else:
                msg, is_vel = command(t)
                if msg is not None:
                    wd.feed(t)
                    q_last, t_last = d.qpos[0], t
                    v_cmd, target = (msg, target) if is_vel else (v_cmd, msg)
                else:
                    silent = True
                    if wd.expired(t) and v_cmd != 0.0:
                        v_cmd, target = 0.0, d.qpos[0]              # stale command: stop and hold here
                if is_vel:
                    target += v_cmd * TICK * DT                     # keep the last velocity
            lim = (-q_lim, q_lim) if q_lim is not None else (None, None)
            d.ctrl[pan] = clamp_goal(target, d.qpos[0], *lim, delta)
        speed_before = abs(d.qvel[0])
        mujoco.mj_step(m, d)
        step += 1
        f = exp.post_force(m, d, post, f6)
        if f > 0 and out["impact_speed"] is None:
            out["impact_speed"] = speed_before
        forces.append(f)
        out["peak_force"] = max(out["peak_force"], f)
        out["peak_speed"] = max(out["peak_speed"], abs(d.qvel[0]))
        out["peak_tau"] = max(out["peak_tau"], abs(d.actuator_force[pan]))
        if silent:
            out["max_travel"] = max(out["max_travel"], abs(d.qpos[0] - q_last))
            if out["t_stop"] is None and d.time > t_last + 0.01 and abs(d.qvel[0]) < 0.01:
                out["t_stop"] = d.time
    out["hold_force"] = float(np.mean(forces[-int(0.3 / DT):]))
    out["grip_drop"] = gz0 - d.site("gripperframe").xpos[2]
    out["t_after"] = out["t_stop"] - t_last if out["t_stop"] is not None else float("nan")
    out["q_last"] = q_last
    return out, m, d


def diff(a, b, keys):
    return max(abs(a[k] - b[k]) for k in keys)


# ---------------------------------------------------------------- Step 1
print("\n== Step 1. The goal clamp: position limit 0.3 rad, relative-target limit 0.02 rad ==")
print("  target  q_now    goal")
for target, q_now in [(1.2, 0.0), (1.2, 0.29), (0.25, 0.24), (-1.0, -0.1)]:
    print(f"{target:8.2f}{q_now:7.2f}{clamp_goal(target, q_now, -0.3, 0.3, 0.02):8.3f}")
rng = np.random.default_rng(0)
tg, qn = rng.uniform(-3, 3, 100_000), rng.uniform(-0.3, 0.3, 100_000)
g = clamp_goal(tg, qn, -0.3, 0.3, 0.02)
lerobot_rule = qn + np.clip(np.clip(tg, -0.3, 0.3) - qn, -0.02, 0.02)     # present + clipped difference
print(f"100000 random: max |goal - q_now| = {np.abs(g - qn).max():.4f} rad, outside range: "
      f"{np.sum(np.abs(g) > 0.3 + 1e-12)}, vs present + clip(diff): {np.abs(g - lerobot_rule).max():.1e}")
print(f"goal speed bound delta / tick = {0.02 / (TICK * DT):.2f} rad/s")

# ---------------------------------------------------------------- Step 2
print("\n== Step 2. The capped servo vs MuJoCo's position actuator (SO-101, 2000 random states) ==")
m = exp.build()
d = mujoco.MjData(m)
kp, kv = m.actuator_gainprm[0, 0], -m.actuator_biasprm[0, 2]
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
print(f"kp = {kp:.2f}, kv = {kv:.3f}, cap = {m.actuator_forcerange[0, 1]:.2f} N m, "
      f"pan ctrlrange = [{lo[0]:.3f}, {hi[0]:.3f}] rad")
for cap in (2.94, 0.5):
    m.actuator_forcerange[:] = [-cap, cap]
    err = 0.0
    for _ in range(2000):
        d.qpos[:6] = rng.uniform(lo, hi)
        d.qvel[:6] = rng.uniform(-3, 3, 6)
        d.ctrl[:6] = rng.uniform(-3, 3, 6)                     # many targets beyond the joint range
        mujoco.mj_forward(m, d)
        ours = servo_torque(clamp_goal(d.ctrl[:6], d.qpos[:6], lo, hi), d.qpos[:6], d.qvel[:6], kp, kv, cap)
        err = max(err, np.abs(ours - d.actuator_force[:6]).max())
    print(f"cap {cap:4.2f} N m: max |ours - MuJoCo| = {err:.1e} N m; cap reached at an error of "
          f"{cap / kp * 1000:.2f} mrad")

# ---------------------------------------------------------------- Step 3
print("\n== Step 3. Part A with our layers in the loop (bad command: pan target 0 -> 1.2 rad at t = 0.1 s) ==")
step_cmd = lambda t: (1.2 if t >= 0.1 else 0.0, False)  # noqa: E731
keys = ["peak_force", "hold_force", "peak_speed", "peak_tau"]
print(f"{'safeguard':<12}{'peak N':>8}{'push N':>8}{'speed':>7}{'tau':>6}  max |ours - Exp.|")
cases = [("none", {}, {}), ("limit 0.3", dict(q_lim=0.3), dict(pos_limit=0.3)),
         ("delta 0.02", dict(delta=0.02), dict(rel_limit=0.02)),
         ("cap 0.5", dict(cap=0.5), dict(pan_cap=0.5)),
         ("all three", dict(q_lim=0.3, delta=0.02, cap=0.5), dict(pos_limit=0.3, rel_limit=0.02, pan_cap=0.5))]
for name, ours_kw, exp_kw in cases:
    r, _, _ = run_layers(step_cmd, **ours_kw)
    e = exp.run(step_cmd, **exp_kw)
    print(f"{name:<12}{r['peak_force']:8.2f}{r['hold_force']:8.2f}{r['peak_speed']:7.2f}{r['peak_tau']:6.2f}"
          f"  {diff(r, e, keys):.1e}")
print("steady push vs tau_max / r (r = lever of the post contacts about the pan axis):")
for cap in (2.94, 1.0, 0.5):
    r, m, d = run_layers(step_cmd, cap=cap)
    post, f6 = m.geom("post").id, np.zeros(6)
    a, ax = d.xanchor[m.joint("shoulder_pan").id], d.xaxis[m.joint("shoulder_pan").id]
    lever, fn = 0.0, 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        if post in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, k, f6)
            lever += f6[0] * abs(np.dot(np.cross(c.pos - a, c.frame[:3]), ax))
            fn += f6[0]
    lever /= fn
    print(f"  cap {cap:4.2f}: r = {lever * 100:.1f} cm, tau_max / r = {cap / lever:5.2f} N, push {r['hold_force']:5.2f} N")
print(f"stall test: kp * 0.005 = {kp * 0.005:.2f} N m >= 2.94 N m")

# ---------------------------------------------------------------- Step 4
print("\n== Step 4. Watchdog: jog at 1 rad/s, the last message at t = 0.18 s (part D) ==")
jog = lambda t: (1.0, True) if t < 0.2 - 1e-9 else (None, True)  # noqa: E731
keys = ["t_after", "max_travel", "peak_force"]
print(f"{'timeout s':>10}{'stopped after s':>17}{'travel deg':>12}{'peak N':>8}  max |ours - Exp.|")
for to in (None, 0.5, 0.2, 0.05):
    r, _, _ = run_layers(jog, T=2.5, timeout=to)
    e = exp.run(jog, T=2.5, watchdog=to)
    e["t_after"] = e["t_stop"] - e["t_last"]
    print(f"{'none' if to is None else f'{to:g}':>10}{r['t_after']:17.3f}{np.degrees(r['max_travel']):12.2f}"
          f"{r['peak_force']:8.2f}  {diff(r, e, keys):.1e}")

# ---------------------------------------------------------------- Step 5
print("\n== Step 5. E-stop latch, then part E: stop a 1 rad/s jog at t = 0.2 s ==")
es = EStop()
trace = [es.update(p) for p in (False, True, False, False)]
held = es.reset(pressed=True)
cleared = es.reset(pressed=False)
print(f"pressed for one tick: active {trace}; reset while pressed: {held}; after release: {cleared}")
keys = ["max_travel", "t_after", "grip_drop"]
print(f"{'stop':<12}{'travel deg':>11}{'stopped after s':>17}{'gripper drop cm':>17}  max |ours - Exp.|")
for kind in ("hold", "torque off"):
    r, _, _ = run_layers(lambda t: (1.0, True), T=1.7, stop=(kind, 0.2))
    e = exp.run(lambda t: (1.0, True), T=1.7, estop=(kind, 0.2))
    e["t_after"] = e["t_stop"] - e["t_last"]
    print(f"{kind:<12}{np.degrees(r['max_travel']):11.2f}{r['t_after']:17.3f}{r['grip_drop'] * 100:17.2f}"
          f"  {diff(r, e, keys):.1e}")

# ---------------------------------------------------------------- Use it
print("\n== Use it. The position limit and the cap as model attributes (MuJoCo clamps inside mj_step) ==")
build_plain = exp.build


def build_with(attr, lo_hi):
    """experiment.build, then set the pan actuator's ctrlrange or forcerange."""
    def build(cap=None):
        m = build_plain(cap)
        getattr(m, attr)[m.actuator("shoulder_pan").id] = lo_hi
        return m
    return build


keys = ["peak_force", "hold_force", "peak_speed", "peak_tau"]
for attr, lo_hi, ours_kw in [("actuator_ctrlrange", [-0.3, 0.3], dict(q_lim=0.3)),
                             ("actuator_forcerange", [-0.5, 0.5], dict(cap=0.5))]:
    exp.build = build_with(attr, lo_hi)
    lib = exp.run(step_cmd)                                   # no layer in the loop: MuJoCo enforces it
    exp.build = build_plain
    r, _, _ = run_layers(step_cmd, **ours_kw)
    print(f"{attr} = {lo_hi}: peak {lib['peak_force']:6.2f} N, push {lib['hold_force']:5.2f} N; "
          f"vs our layer {diff(lib, r, keys):.1e}")
