"""Lesson 39 · Domain randomization, measured in simulation.

Plant: one arm joint with the STS3215 joint parameters of the Menagerie SO-101 model
(armature 0.028, damping 0.60, frictionloss 0.052, torque limit 2.94 N m) carrying a 0.5 kg link
with its centre of mass 15 cm from the axis, under gravity. Torque control.
The controller sees the joint through a measurement delay (sensing + compute + bus).

Policy: tau = clip(Kp (q_ref - q~) + Kd (qd_ref - qd~) + kg cos(q_ref), +-2.94), three parameters,
tuned with the cross-entropy method (CEM) to minimise RMS tracking error of a 3 s reference.

  Training distributions: width w = 0 (nominal only) ... 1. With width w, each episode draws
     link mass, joint damping and friction loss scales log-uniformly in [2^-w, 2^w]
     and the delay uniformly in [5, 5 + 40 w] ms (nominal: all scales 1, delay 5 ms).
  Test: the nominal plant, and 32 held-out "real" plants (heavier, more friction, 15-30 ms delay),
     never seen in training.
  Break it: also train with w = 1.5 and 2.
  Exercise 3: w = 0.75 with only the delay randomized (scales fixed at 1).

Many copies of the joint live in one MuJoCo model, so one mj_step advances a whole batch.
Run:  python experiment.py      (deterministic; well under a minute of CPU)
"""
import math

import mujoco
import numpy as np

DT = 0.002
T_EP = 3.0
N_STEPS = int(round(T_EP / DT))
TAU_MAX = 2.94
LINK_MASS, LINK_COM = 0.5, 0.15
ARMATURE, DAMPING, FRICTIONLOSS = 0.028, 0.60, 0.052
NOM_DELAY = 0.005
MAX_DELAY_STEPS = 60                                   # 120 ms buffer

POP, ELITE, ITERS, K_SETS = 24, 6, 12, 10              # CEM: candidates, elites, iterations, plants per candidate
N_TEST = 32


def reference(t):
    q = 0.5 * np.sin(2 * np.pi * 0.4 * t) + 0.25 * np.sin(2 * np.pi * 1.1 * t + 1.0)
    qd = 0.5 * 2 * np.pi * 0.4 * np.cos(2 * np.pi * 0.4 * t) + 0.25 * 2 * np.pi * 1.1 * np.cos(2 * np.pi * 1.1 * t + 1.0)
    return q, qd


T_GRID = np.arange(N_STEPS) * DT
Q_REF, QD_REF = reference(T_GRID)


def build_batch(n):
    spec = mujoco.MjSpec()
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for i in range(n):
        b = spec.worldbody.add_body(name=f"link{i}", pos=[0, 0.5 * i, 0])
        b.add_joint(name=f"j{i}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                    armature=ARMATURE, damping=DAMPING, frictionloss=FRICTIONLOSS)
        g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 2 * LINK_COM, 0, 0], size=[0.01])
        g.contype, g.conaffinity, g.mass = 0, 0, 0
        b.mass, b.ipos, b.inertia = LINK_MASS, [LINK_COM, 0, 0], [1e-4, 1e-4, 1e-4]
        b.explicitinertial = True
        a = spec.add_actuator(name=f"m{i}", target=f"j{i}")
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm[0] = 1.0
        a.ctrllimited = True
        a.ctrlrange = [-TAU_MAX, TAU_MAX]
    return spec.compile()


class Batch:
    def __init__(self, n):
        self.n = n
        self.m = build_batch(n)
        self.d = mujoco.MjData(self.m)
        self.mass0 = self.m.body_mass[1:].copy()
        self.inertia0 = self.m.body_inertia[1:].copy()

    def rollout(self, gains, plants):
        """gains: (n, 3) Kp, Kd, kg. plants: dict of (n,) arrays mass, damping, friction scales and delay [s].
        Returns RMS tracking error per copy [rad] (diverged copies are capped at 1 rad)."""
        m, d, n = self.m, self.d, self.n
        m.body_mass[1:] = self.mass0 * plants["mass"]
        m.body_inertia[1:] = self.inertia0 * plants["mass"][:, None]
        m.dof_damping[:] = DAMPING * plants["damping"]
        m.dof_frictionloss[:] = FRICTIONLOSS * plants["friction"]
        nd = np.clip(np.round(plants["delay"] / DT).astype(int), 0, MAX_DELAY_STEPS - 1)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = Q_REF[0]
        d.qvel[:] = QD_REF[0]
        hq = np.tile(d.qpos.copy(), (MAX_DELAY_STEPS, 1))
        hv = np.tile(d.qvel.copy(), (MAX_DELAY_STEPS, 1))
        kp, kd, kg = gains[:, 0], gains[:, 1], gains[:, 2]
        idx = np.arange(n)
        sq = np.zeros(n)
        for k in range(N_STEPS):
            slot = k % MAX_DELAY_STEPS
            hq[slot], hv[slot] = d.qpos, d.qvel
            back = (slot - nd) % MAX_DELAY_STEPS
            q_seen, v_seen = hq[back, idx], hv[back, idx]
            d.ctrl[:] = kp * (Q_REF[k] - q_seen) + kd * (QD_REF[k] - v_seen) + kg * math.cos(Q_REF[k])
            mujoco.mj_step(m, d)
            e = d.qpos - Q_REF[k]
            sq += np.minimum(e * e, 1.0)
        return np.sqrt(sq / N_STEPS)


def sample_plants(rng, n, w, delay_only=False):
    if w == 0:
        return nominal_plants(n)
    if delay_only:                                         # exercise 3: scales fixed at 1
        return dict(mass=np.ones(n), damping=np.ones(n), friction=np.ones(n),
                    delay=rng.uniform(NOM_DELAY, NOM_DELAY + 0.040 * w, n))
    s = lambda: 2.0 ** rng.uniform(-w, w, n)  # noqa: E731
    return dict(mass=s(), damping=s(), friction=s(), delay=rng.uniform(NOM_DELAY, NOM_DELAY + 0.040 * w, n))


def nominal_plants(n):
    return dict(mass=np.ones(n), damping=np.ones(n), friction=np.ones(n), delay=np.full(n, NOM_DELAY))


def real_plants(n, seed=2026):
    """Held-out 'real robots': heavier link (payload), more damping and friction, more delay. Invented for the test."""
    rng = np.random.default_rng(seed)
    return dict(mass=rng.uniform(1.1, 1.4, n), damping=rng.uniform(0.8, 1.6, n),
                friction=rng.uniform(1.0, 2.5, n), delay=rng.uniform(0.015, 0.030, n))


def cem(batch, w, seed, delay_only=False):
    rng = np.random.default_rng(seed)
    mu = np.array([math.log(20.0), math.log(1.0), 0.0])        # log Kp, log Kd, kg
    sd = np.array([1.5, 1.5, 1.0])
    lo, hi = np.array([math.log(1.0), math.log(0.01), -3.0]), np.array([math.log(1000.0), math.log(30.0), 3.0])
    for it in range(ITERS):
        x = np.clip(mu + sd * rng.standard_normal((POP, 3)), lo, hi)
        plants1 = sample_plants(rng, K_SETS, w, delay_only)    # same K plants for every candidate
        plants = {k: np.tile(v, POP) for k, v in plants1.items()}
        gains = np.repeat(np.column_stack([np.exp(x[:, 0]), np.exp(x[:, 1]), x[:, 2]]), K_SETS, axis=0)
        cost = batch.rollout(gains, plants).reshape(POP, K_SETS).mean(axis=1)
        el = x[np.argsort(cost)[:ELITE]]
        mu, sd = el.mean(axis=0), el.std(axis=0) + 0.02
    return np.array([math.exp(mu[0]), math.exp(mu[1]), mu[2]])


def evaluate(batch_test, gains, plants):
    g = np.tile(gains, (N_TEST, 1))
    return batch_test.rollout(g, plants)


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    print(f"joint: link {LINK_MASS} kg at {LINK_COM * 100:.0f} cm (gravity torque max {LINK_MASS * 9.81 * LINK_COM:.3f} N m), "
          f"armature {ARMATURE}, damping {DAMPING}, frictionloss {FRICTIONLOSS}, torque limit {TAU_MAX} N m")
    print(f"reference: 0.5 sin(2 pi 0.4 t) + 0.25 sin(2 pi 1.1 t + 1) rad, {T_EP:.0f} s at {DT * 1e3:.0f} ms; "
          f"nominal delay {NOM_DELAY * 1e3:.0f} ms")
    print(f"CEM: {ITERS} iterations x {POP} candidates x {K_SETS} plants; test: nominal plant and {N_TEST} held-out real plants")
    rp = real_plants(N_TEST)
    print(f"real plants: mass x{rp['mass'].min():.2f}-{rp['mass'].max():.2f}, damping x{rp['damping'].min():.2f}-"
          f"{rp['damping'].max():.2f}, friction x{rp['friction'].min():.2f}-{rp['friction'].max():.2f}, "
          f"delay {rp['delay'].min() * 1e3:.1f}-{rp['delay'].max() * 1e3:.1f} ms")
    print("success = RMS tracking error below 2 deg")
    print()
    train = Batch(POP * K_SETS)
    test = Batch(N_TEST)
    thr = math.radians(2.0)
    print(f"    {'width w':>7} | {'delay range ms':>14} | {'scale range':>11} | {'Kp':>7} | {'Kd':>6} | {'kg':>6} | "
          f"{'nominal RMS deg':>15} | {'real mean deg':>13} | {'real worst deg':>14} | {'real success':>12}")
    for w, delay_only in [(0.0, False), (0.1, False), (0.25, False), (0.5, False), (0.75, False), (1.0, False),
                          (1.5, False), (2.0, False), (0.75, True)]:
        if w == 1.5:
            print("    --- break it: wider than needed ---")
        if delay_only:
            print("    --- exercise 3: randomize only the delay (scales fixed at 1) ---")
        gains = cem(train, w, seed=7, delay_only=delay_only)
        nom = evaluate(test, gains, nominal_plants(N_TEST))[0]
        real = evaluate(test, gains, rp)
        ok = int((real < thr).sum())
        lo_s, hi_s = (1.0, 1.0) if delay_only else (2 ** -w, 2 ** w)
        print(f"    {w:>7.2f} | {NOM_DELAY * 1e3:>5.0f}-{(NOM_DELAY + 0.04 * w) * 1e3:<8.0f} | "
              f"{lo_s:>4.2f}-{hi_s:<6.2f} | {gains[0]:>7.2f} | {gains[1]:>6.3f} | {gains[2]:>6.3f} | "
              f"{math.degrees(nom):>15.3f} | {math.degrees(real.mean()):>13.3f} | {math.degrees(real.max()):>14.3f} | "
              f"{ok:>9}/{N_TEST}")


if __name__ == "__main__":
    main()
