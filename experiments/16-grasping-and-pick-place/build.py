"""Lesson 16 · Build it: the grasp test and the scripted pick from scratch, checked at every step.

Step 1  The friction-cone ratio: recompute the Experiment's ratio columns from its own squeeze forces.
Step 2  Inside a held grasp: MuJoCo's contact forces balance the weight and stay inside their cones.
Step 3  The squeeze from the torque cap: 2 tau / r with the jaw lever measured from the contacts.
Step 4  The state machine and min-jerk blend: our pick reproduces the Experiment's pick step for step.
Step 5  Outcomes and Wilson intervals: three sweep rows rerun with our machine, Wilson vs SciPy.

The functions live in physai/grasp.py (the lesson shows them step by step).
The scene, IK and contact bookkeeping come from experiment.py, so both tell one story.
Run:  uv run python experiments/16-grasping-and-pick-place/build.py      (~5 s, deterministic)
"""
import importlib.util
import os
import re
import sys

import mujoco
import numpy as np
import scipy
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.grasp import (PickMachine, cone_ratio, cone_use, max_mass, minjerk, pick_outcome,  # noqa: E402
                          squeeze_from_cap, wilson)

spec = importlib.util.spec_from_file_location("grasp_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
HALF = exp.CUBE_HALF

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")


def scene(mu, mass, cap, xy=exp.CUBE_HOME):
    """The Experiment's scene with the jaw torque cap set, arm at home, jaw open."""
    m, d = exp.build(xy, mu=mu, mass=mass)
    m.actuator_forcerange[m.actuator("gripper").id] = [-cap, cap]
    exp.reset(m, d)
    return m, d


def our_pick(m, d, cube_pos, record=False):
    """The pick driven by our PickMachine (IK poses and contact sensing from the Experiment)."""
    I = exp.ids(m)
    poses, _ = exp.plan(m, d, I, cube_pos)
    poses["home"] = exp.HOME_Q
    sm = PickMachine(exp.DUR, exp.CLOSE_TIMEOUT, exp.SETTLED_SPEED, exp.SETTLED_TIME, t0=d.time)
    jaw_dof, squeeze, traj = m.joint("gripper").dofadr[0], (0.0, 0.0), []
    while sm.state != "done":
        d.ctrl[I["act"]], d.ctrl[I["grip"]] = sm.command(d.time, poses, exp.G_OPEN, exp.G_CLOSE)
        mujoco.mj_step(m, d)
        if record:
            traj.append(d.qpos.copy())
        if sm.state == "close":
            nf, nm = exp.jaw_contacts(m, d, I)
            if sm.update(d.time, nf > 0 and nm > 0, d.qvel[jaw_dof]) == "lift":
                squeeze = (nf, nm)
        else:
            sm.update(d.time)
    nf, nm = exp.jaw_contacts(m, d, I)
    lift = d.xpos[I["cube_body"]][2] - HALF
    return dict(outcome=pick_outcome(lift, nf > 0 and nm > 0, np.isfinite(sm.close_s)), lift=lift,
                close_s=sm.close_s, squeeze=squeeze, end=(nf, nm), traj=traj)


def cube_contacts(m, d):
    """Contact-frame force, world force on the cube and jaw (fixed/moving) for every jaw-cube contact."""
    I, f6, out = exp.ids(m), np.zeros(6), []
    for k in range(d.ncon):
        c = d.contact[k]
        if I["cube"] not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == I["cube"] else c.geom1
        if other not in I["fixed"] | I["moving"]:
            continue
        mujoco.mj_contactForce(m, d, k, f6)
        frame = c.frame.reshape(3, 3)                      # rows: normal (geom1 -> geom2), tangent1, tangent2
        on_cube = frame.T @ f6[:3] * (1 if c.geom2 == I["cube"] else -1)
        out.append((f6[:3].copy(), on_cube, "moving" if other in I["moving"] else "fixed", c.pos.copy(), frame[0]))
    return out


# ---------------------------------------------------------------- Step 1
print("\n== Step 1. The friction-cone ratio from the Experiment's own squeeze forces (output.txt) ==")
text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
rows = {}
for part, block in re.findall(r"^\[(A|B|C)\].*?\n(.*?)(?=^\[|\Z)", text, re.M | re.S):
    for line in block.splitlines()[1:]:
        f = line.split()
        if len(f) > 6 and f[5] != "-":                 # value, k/n, [lo, hi], close, squeeze, ratio, ...
            rows.setdefault(part, []).append((float(f[0]), float(f[5]), float(f[6])))
make = dict(A=lambda v: (v, 0.03), B=lambda v: (0.05, 0.03), C=lambda v: (0.3, v))
for part, rs in rows.items():
    err = max(abs(round(cone_ratio(make[part](v)[0], N, make[part](v)[1]), 2) - ratio) for v, N, ratio in rs)
    print(f"part {part}: {len(rs)} rows, max |ours - printed ratio| = {err:.2f}")
print(f"max_mass: mu 0.05, 12.74 N -> {max_mass(0.05, 12.74):.3f} kg; mu 0.3, 16.25 N -> {max_mass(0.3, 16.25):.3f} kg")

# ---------------------------------------------------------------- Step 2
print("\n== Step 2. Inside a held grasp (end of the hold, cube in the air) ==")
print(f"{'case':<16}{'contacts':>9}{'sum Fz N':>10}{'m g N':>8}{'max cone use':>14}")
for name, mu, mass, cap in [("mu 1.0, 30 g", 1.0, 0.03, 2.94), ("mu 0.3, 0.4 kg", 0.3, 0.4, 0.5)]:
    m, d = scene(mu, mass, cap)
    r = our_pick(m, d, np.array([*exp.CUBE_HOME, HALF]))
    cs = cube_contacts(m, d)
    fz = sum(on_cube[2] for _, on_cube, *_ in cs)
    use = max(cone_use(f, mu) for f, *_ in cs)
    print(f"{name:<16}{len(cs):>9}{fz:>10.3f}{mass * 9.81:>8.3f}{use:>14.3f}")

# ---------------------------------------------------------------- Step 3
print("\n== Step 3. Squeeze from the jaw torque cap (mu = 0.05, 30 g, nominal position) ==")
m0 = exp.build(exp.CUBE_HOME)[0]
loss = m0.dof_frictionloss[m0.joint("gripper").dofadr[0]]               # the jaw's static friction (N m)
print(f"jaw friction loss {loss:.3f} N m; band = 2 (tau +- loss) / r")
print(f"{'cap N m':>8}{'r_jaw cm':>10}{'2 tau / r N':>13}{'band N':>16}{'squeeze N':>11}{'N fixed / N moving':>20}")
for cap in (1.0, 0.5, 0.3):
    m, d = scene(0.05, 0.03, cap)
    r = our_pick(m, d, np.array([*exp.CUBE_HOME, HALF]))
    jid = m.joint("gripper").id
    a, ax = d.xanchor[jid], d.xaxis[jid]
    mov = [(f[0], abs(np.dot(np.cross(p - a, n), ax))) for f, _, side, p, n in cube_contacts(m, d) if side == "moving"]
    r_jaw = sum(N * lev for N, lev in mov) / sum(N for N, _ in mov)       # force-weighted lever of the moving jaw
    nf, nm = r["squeeze"]
    band = f"[{squeeze_from_cap(cap - loss, r_jaw):.2f}, {squeeze_from_cap(cap + loss, r_jaw):.2f}]"
    print(f"{cap:>8.2f}{r_jaw * 100:>10.2f}{squeeze_from_cap(cap, r_jaw):>13.2f}{band:>16}{nf + nm:>11.2f}{nf / nm:>20.3f}")

# ---------------------------------------------------------------- Step 4
print("\n== Step 4. Our state machine vs the Experiment's scripted_pick (nominal pick) ==")
u = np.linspace(0, 1, 100_001)
s = np.array([minjerk(x) for x in u])
v, acc = np.gradient(s, u), np.gradient(np.gradient(s, u), u)
print(f"minjerk: s(0.5) = {minjerk(0.5):.3f}, peak speed {v.max():.4f} (15/8 = 1.875), "
      f"|v|, |a| at the ends < {max(abs(v[0]), abs(v[-1]), abs(acc[1]), abs(acc[-2])):.0e} (finite differences)")
m, d = scene(1.0, 0.03, 2.94)
ours = our_pick(m, d, np.array([*exp.CUBE_HOME, HALF]), record=True)
m, d = scene(1.0, 0.03, 2.94)
ref = exp.scripted_pick(m, d, np.array([*exp.CUBE_HOME, HALF]), record=True)
dq = max(np.abs(a - b[2]).max() for a, b in zip(ours["traj"], ref["trajectory"]))
print(f"ours: {ours['outcome']}, lift {ours['lift'] * 100:.2f} cm, jaw forces {ours['end'][0]:.2f} / {ours['end'][1]:.2f} N, "
      f"close {ours['close_s']:.3f} s, {len(ours['traj'])} steps")
print(f"Exp.: success {ref['success']}, lift {ref['lift_cm']:.2f} cm, jaw forces {ref['jaw_fixed_N']:.2f} / "
      f"{ref['jaw_moving_N']:.2f} N, close {ref['close_s']:.3f} s, {len(ref['trajectory'])} steps; max |qpos ours - Exp.| = {dq:.1e}")

# ---------------------------------------------------------------- Step 5
print("\n== Step 5. Three sweep rows rerun with our machine (10 positions, seed 0) ==")
print(f"{'row':<12}{'held / not closed / dropped':>29}{'squeeze N':>11}{'ratio':>7}{'Wilson 95%':>16}{'vs SciPy':>10}")
for name, mu, mass in [("A mu 0.03", 0.03, 0.03), ("C 0.4 kg", 0.3, 0.4), ("C 0.45 kg", 0.3, 0.45)]:
    count, Ns = dict(held=0, **{"not closed": 0}, dropped=0), []
    for xy in exp.trial_positions(10):
        m, d = scene(mu, mass, 0.5, xy)
        r = our_pick(m, d, np.array([xy[0], xy[1], HALF]))
        count[r["outcome"]] += 1
        if np.isfinite(r["close_s"]):
            Ns.append(sum(r["squeeze"]))
    lo, hi = wilson(count["held"], 10)
    ci = binomtest(count["held"], 10).proportion_ci(method="wilson")
    N = float(np.mean(Ns))
    print(f"{name:<12}{count['held']:>13} / {count['not closed']} / {count['dropped']:<9}{N:>11.2f}"
          f"{cone_ratio(mu, N, mass):>7.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}{max(abs(lo - ci.low), abs(hi - ci.high)):>10.1e}")
