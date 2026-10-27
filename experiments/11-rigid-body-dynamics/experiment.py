"""Lesson 11 · Rigid-body dynamics: M(q) qdd + C(q, qd) qd + g(q) = tau.

  A. Check MuJoCo against a hand-derived double pendulum: mj_fullM vs M(q), qfrc_bias vs C qd + g,
     mj_inverse vs the analytic tau, and mj_forward vs M^-1 (tau - c), on 1000 random states.
  B. Sweep integrator x timestep: energy drift of the same double pendulum swinging passively for 10 s.
  C. The same quantities on real robot models (Panda, SO-101): gravity torques and the inertia matrix.

Run:  python experiment.py      (~10 s on Apple M5)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

G = 9.81
M1, M2 = 1.0, 0.5           # link masses [kg]
L1, L2 = 0.4, 0.3           # link lengths [m]
C1, C2 = 0.2, 0.15          # distance joint -> centre of mass [m]
I1, I2 = 0.02, 0.005        # inertia about the centre of mass, about the hinge axis [kg m^2]


def xml(integrator="Euler", dt=0.002):
    # Links lie along +x at q = 0; hinges about +y, so a positive angle swings the link down (towards -z).
    return f"""
<mujoco>
  <option integrator="{integrator}" timestep="{dt}" gravity="0 0 -{G}">
    <flag energy="enable"/>
  </option>
  <worldbody>
    <body name="link1">
      <joint name="q1" type="hinge" axis="0 1 0"/>
      <inertial pos="{C1} 0 0" mass="{M1}" diaginertia="{I1} {I1} {I1}"/>
      <body name="link2" pos="{L1} 0 0">
        <joint name="q2" type="hinge" axis="0 1 0"/>
        <inertial pos="{C2} 0 0" mass="{M2}" diaginertia="{I2} {I2} {I2}"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def analytic(q, qd):
    """Mass matrix, Coriolis/centrifugal vector and gravity vector of the planar double pendulum."""
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    a = M2 * L1 * C2
    M = np.array([[M1 * C1 ** 2 + I1 + M2 * (L1 ** 2 + C2 ** 2 + 2 * L1 * C2 * c2) + I2, M2 * (C2 ** 2 + L1 * C2 * c2) + I2],
                  [M2 * (C2 ** 2 + L1 * C2 * c2) + I2, M2 * C2 ** 2 + I2]])
    Cqd = np.array([-a * s2 * (2 * qd[0] * qd[1] + qd[1] ** 2), a * s2 * qd[0] ** 2])
    # height of each centre of mass: z = -(...) sin(.)  ->  V = sum m g z,  g(q) = dV/dq
    g = -G * np.array([(M1 * C1 + M2 * L1) * np.cos(q[0]) + M2 * C2 * np.cos(q[0] + q[1]),
                       M2 * C2 * np.cos(q[0] + q[1])])
    return M, Cqd, g


def part_a():
    m = mujoco.MjModel.from_xml_string(xml())
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    err = dict(M=0.0, bias=0.0, inverse=0.0, forward=0.0)
    tau_max = 0.0
    Mf = np.zeros((2, 2))
    for _ in range(1000):
        q, qd, qdd = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5, 5, 2), rng.uniform(-20, 20, 2)
        Ma, Ca, ga = analytic(q, qd)
        tau = Ma @ qdd + Ca + ga
        d.qpos[:], d.qvel[:], d.qacc[:] = q, qd, qdd
        mujoco.mj_inverse(m, d)                    # -> d.qfrc_inverse, also fills d.M (sparse inertia) and qfrc_bias
        mujoco.mj_fullM(m, d, Mf)
        err["M"] = max(err["M"], np.abs(Mf - Ma).max())
        err["bias"] = max(err["bias"], np.abs(d.qfrc_bias - (Ca + ga)).max())
        err["inverse"] = max(err["inverse"], np.abs(d.qfrc_inverse - tau).max())
        tau_max = max(tau_max, np.abs(tau).max())
        d.qfrc_applied[:] = tau                    # forward dynamics with the same torque must give back qdd
        mujoco.mj_forward(m, d)
        err["forward"] = max(err["forward"], np.abs(d.qacc - qdd).max())
        d.qfrc_applied[:] = 0
    print("\n[A] Double pendulum, 1000 random (q, qd, qdd), seed 0: max |MuJoCo - analytic|")
    print(f"  mj_fullM vs M(q)                      {err['M']:.2e} kg m^2")
    print(f"  qfrc_bias vs C(q,qd) qd + g(q)        {err['bias']:.2e} N m")
    print(f"  mj_inverse (qfrc_inverse) vs tau      {err['inverse']:.2e} N m   (largest |tau| = {tau_max:.1f} N m)")
    print(f"  mj_forward (qacc) vs qdd              {err['forward']:.2e} rad/s^2")


def drift(integrator, dt, T=10.0):
    m = mujoco.MjModel.from_xml_string(xml(integrator, dt))
    d = mujoco.MjData(m)
    d.qpos[:] = [0.0, 0.0]                         # both links horizontal, released from rest
    mujoco.mj_forward(m, d)
    e0 = d.energy.sum()
    worst = 0.0
    n = int(round(T / dt))
    for _ in range(n):
        mujoco.mj_step(m, d)
        e = d.energy.sum()
        if not np.isfinite(e):
            return float("inf"), float("inf")
        worst = max(worst, abs(e - e0))
    return worst, d.energy.sum() - e0


def part_b():
    scale = G * (M1 * C1 + M2 * (L1 + C2))        # potential energy released when both links hang down [J]
    print(f"\n[B] Passive double pendulum released horizontal, 10 s. Energy error in % of {scale:.3f} J "
          "(the potential energy it can release)")
    print("    cell = max |E(t) - E0| % / final E(10 s) - E0 %")
    dts = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
    print(f"{'integrator':<13}" + "".join(f"{f'dt {dt * 1000:g} ms':>22}" for dt in dts))
    for integ in ("Euler", "RK4", "implicit", "implicitfast"):
        cells = []
        for dt in dts:
            w, f = drift(integ, dt)
            cells.append("diverged" if not np.isfinite(w) else f"{100 * w / scale:.3f} / {100 * f / scale:+.3f}")
        print(f"{integ:<13}" + "".join(f"{c:>22}" for c in cells))


def part_c():
    print("\n[C] The same quantities on robot models (qvel = 0, so qfrc_bias = g(q))")
    m = mujoco.MjModel.from_xml_path(path("franka_emika_panda"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    Mf = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mf)
    print("  Panda 'home' keyframe, joints 1-7:")
    print("    g(q)  [N m]   " + " ".join(f"{x:7.2f}" for x in d.qfrc_bias[:7]))
    print("    diag M [kg m^2]" + " ".join(f"{x:7.3f}" for x in np.diag(Mf)[:7]))
    print("    of which armature" + " ".join(f"{x:7.3f}" for x in m.dof_armature[:7]))
    print(f"    cond(M[:7,:7]) = {np.linalg.cond(Mf[:7, :7]):.1f}")
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101"))
    d = mujoco.MjData(m)
    for name, q in (("stretched forward (all 0)", np.zeros(6)), ("folded (lift -1.57, elbow 1.57)", np.array([0, -1.57, 1.57, 0, 0, 0]))):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        print(f"  SO-101 {name:<32} g(q) [N m] " + " ".join(f"{x:6.3f}" for x in d.qfrc_bias[:5]))


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    part_a()
    part_b()
    part_c()


if __name__ == "__main__":
    main()
