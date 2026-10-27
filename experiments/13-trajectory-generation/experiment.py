"""Lesson 13 · Trajectory generation: how the reference is shaped decides the torque you need.

Panda (Menagerie) under joint-space torque control: tau = g(q) + Kp (q_ref - q) + Kd (qd_ref - qd),
torques clipped at the model's force limits (87 N m joints 1-4, 12 N m joints 5-7).
The model's built-in position actuators are switched off (gain and bias set to zero) and the torque
is written to qfrc_applied.

  A. One fixed joint move in T = 1 s, five reference shapes: step, linear, trapezoidal velocity, cubic, quintic.
     Reference peak acceleration / jerk, robot peak acceleration / jerk, tracking error, torque saturation.
  B. Sweep the duration T for cubic and quintic.
  C. Joint-space vs Cartesian straight line between two hand positions: how far the hand leaves the line.

Run:  python experiment.py      (~15 s on Apple M5)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

KP = np.array([600, 600, 600, 600, 250, 150, 50.0])
KD = np.array([50, 50, 50, 50, 20, 12, 4.0])
Q0 = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853])          # Menagerie "home" keyframe (arm joints)
DQ = np.array([0.6, 0.4, -0.4, 0.5, 0.5, -0.4, 0.6])             # the move [rad]
HOLD = 0.5                                                        # simulate this long after the move ends [s]


def load_panda():
    m = mujoco.MjModel.from_xml_path(path("franka_emika_panda"))
    m.actuator_gainprm[:7] = 0       # built-in position servos off: they would add their own PD
    m.actuator_biasprm[:7] = 0
    return m


# ---------------------------------------------------------------- reference shapes, s(u) on u in [0, 1]
def profile(kind, t, T):
    """Returns (s, ds/dt, d2s/dt2) of a 0 -> 1 normalised move of duration T at time t."""
    u = np.clip(t / T, 0.0, 1.0)
    inside = 0.0 < t / T < 1.0
    if kind == "step":
        return 1.0, 0.0, 0.0
    if kind == "linear":
        return u, (1 / T if inside else 0.0), 0.0
    if kind == "cubic":                     # s = 3u^2 - 2u^3: zero velocity at both ends
        return 3 * u**2 - 2 * u**3, (6 * u - 6 * u**2) / T, ((6 - 12 * u) / T**2 if inside else 0.0)
    if kind == "quintic":                   # s = 10u^3 - 15u^4 + 6u^5: zero velocity and acceleration at both ends
        return (10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4) / T,
                (60 * u - 180 * u**2 + 120 * u**3) / T**2)
    if kind == "trapezoid":                 # accelerate T/3, cruise T/3, decelerate T/3; v_max = 1.5 / T
        a = 4.5 / T**2
        ta = T / 3
        tt = np.clip(t, 0, T)
        if tt < ta:
            return 0.5 * a * tt**2, a * tt, (a if inside else 0.0)
        if tt < T - ta:
            return 0.5 * a * ta**2 + a * ta * (tt - ta), a * ta, 0.0
        r = T - tt
        return 1 - 0.5 * a * r**2, a * r, (-a if inside else 0.0)
    raise ValueError(kind)


def ref_peaks(kind, T, dt):
    """Peak |acceleration| and |jerk| of the joint reference, sampled at the control rate.
    'inf' marks a reference whose position (step), velocity or acceleration jumps."""
    jumps = {"step": (True, True), "linear": (True, True), "trapezoid": (False, True), "cubic": (False, True),
             "quintic": (False, False)}[kind]
    ts = np.arange(0, T + dt / 2, dt)
    acc = np.array([profile(kind, t, T)[2] for t in ts]) if kind != "step" else np.zeros(1)
    k = np.abs(DQ).max()
    a = float("inf") if jumps[0] else k * np.abs(acc).max()
    if jumps[1]:
        j = float("inf")
    else:
        j = k * np.abs(np.diff(acc) / dt).max()
    return a, j


def track(kind, T, accel_ff=False):
    m = load_panda()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    lim = m.actuator_forcerange[:7, 1]
    err_max, acc_max, jerk_max, sat, n, err_joint = 0.0, 0.0, 0.0, 0, 0, 0
    prev_acc = None
    tau_max = 0.0
    Mfull = np.zeros((m.nv, m.nv))
    while d.time < T + HOLD - 1e-9:
        s, sd, sdd = profile(kind, d.time, T)
        q_ref, qd_ref = Q0 + s * DQ, sd * DQ
        q, qd = d.qpos[:7], d.qvel[:7]
        tau = d.qfrc_bias[:7] + KP * (q_ref - q) + KD * (qd_ref - qd)
        if accel_ff:                          # model feedforward (lesson 11): M(q) qdd_ref, minus passive joint forces
            mujoco.mj_fullM(m, d, Mfull)
            tau = tau + Mfull[:7, :7] @ (sdd * DQ) - d.qfrc_passive[:7]
        sat += int(np.any(np.abs(tau) > lim))
        tau = np.clip(tau, -lim, lim)
        tau_max = max(tau_max, float(np.max(np.abs(tau) / lim)))
        d.qfrc_applied[:7] = tau
        mujoco.mj_step(m, d)
        n += 1
        acc = d.qacc[:7].copy()
        acc_max = max(acc_max, float(np.abs(acc).max()))
        if prev_acc is not None:
            jerk_max = max(jerk_max, float(np.abs(acc - prev_acc).max() / dt))
        prev_acc = acc
        if d.time <= T:                       # tracking error while the reference moves (step: whole window)
            s_now = profile(kind, d.time, T)[0]
            ej = np.abs(Q0 + s_now * DQ - d.qpos[:7])
            if ej.max() > err_max:
                err_max, err_joint = float(ej.max()), int(np.argmax(ej)) + 1
    final_err = float(np.abs(Q0 + DQ - d.qpos[:7]).max())
    track.err_joint = err_joint
    return err_max, acc_max, jerk_max, 100 * sat / n, final_err, dt


def part_a():
    T = 1.0
    print(f"\n[A] Panda joint move, |dq|max = {np.abs(DQ).max():.1f} rad, T = {T} s, PD + g(q), "
          f"Kp = {KP.astype(int).tolist()}, Kd = {KD.tolist()}")
    print("  ref = reference peak (largest joint); robot = measured in simulation; err = max |q_ref - q| during the move")
    print(f"{'shape':<31}{'ref acc':>10}{'ref jerk':>11}{'robot acc':>11}{'robot jerk':>12}{'err deg':>9}"
          f"{'saturated %':>12}{'err at T+0.5 s deg':>20}")
    for kind, ff in (("step", 0), ("linear", 0), ("trapezoid", 0), ("cubic", 0), ("quintic", 0), ("quintic", 1)):
        e, a, j, s, f, dt = track(kind, T, accel_ff=bool(ff))
        ra, rj = ref_peaks(kind, T, dt)
        label = kind + (" + M qdd_ref - passive" if ff else "")
        print(f"{label:<31}{ra:>10.2f}{rj:>11.2f}{a:>11.1f}{j:>12.0f}{np.degrees(e):>9.2f}{s:>12.1f}"
              f"{np.degrees(f):>20.3f}   (worst joint {track.err_joint})")
    print("  units: rad/s^2 (acc), rad/s^3 (jerk); control and physics at 2 ms")


def part_b():
    print("\n[B] Sweep the duration T (same move)")
    print(f"{'T s':>6}{'shape':>9}{'ref acc':>10}{'ref jerk':>11}{'robot jerk':>12}{'err deg':>9}{'saturated %':>12}")
    for T in (0.25, 0.5, 1.0, 2.0):
        for kind in ("cubic", "quintic"):
            e, a, j, s, f, dt = track(kind, T)
            ra, rj = ref_peaks(kind, T, dt)
            print(f"{T:>6.2f}{kind:>9}{ra:>10.2f}{rj:>11.1f}{j:>12.0f}{np.degrees(e):>9.2f}{s:>12.1f}")


# ---------------------------------------------------------------- C: joint space vs Cartesian line
def ik_position(m, d, site, target, q_init, iters=200):
    """Damped least squares on the 7 arm joints for the site position, keeping the hand orientation of q_init."""
    dd = mujoco.MjData(m)
    dd.qpos[:] = q_init
    mujoco.mj_kinematics(m, dd)
    R0 = dd.site_xmat[site].reshape(3, 3).copy()
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_comPos(m, dd)
        R = dd.site_xmat[site].reshape(3, 3)
        ep = target - dd.site_xpos[site]
        er = 0.5 * sum(np.cross(R[:, i], R0[:, i]) for i in range(3))
        if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
            break
        mujoco.mj_jacSite(m, dd, jp, jr, site)
        J = np.vstack([jp, jr])[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
        dd.qpos[:7] += dq
    return dd.qpos.copy()


def part_c():
    spec = mujoco.MjSpec.from_file(path("franka_emika_panda"))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_kinematics(m, d)
    site = m.site("tcp").id
    p_a = d.site_xpos[site].copy() + np.array([0.0, -0.25, -0.15])
    p_b = p_a + np.array([0.0, 0.5, 0.0])                       # 50 cm straight move along y
    qa = ik_position(m, d, site, p_a, d.qpos.copy())
    qb = ik_position(m, d, site, p_b, qa)
    res = []
    for q, p in ((qa, p_a), (qb, p_b)):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        res.append(np.linalg.norm(p - d.site_xpos[site]))
    print(f"\n[C] Hand (TCP) from A to B: 50 cm straight line along y, orientation held; "
          f"IK residual at A and B < {max(res) * 1000:.3f} mm")
    print(f"{'interpolation':<28}{'max distance from line cm':>27}{'hand path cm':>14}{'joint path rad':>16}")
    N = 200
    for name in ("joint space (quintic in q)", "Cartesian (IK per waypoint)"):
        pts, qs = [], []
        q_prev = qa.copy()
        for k in range(N + 1):
            s = profile("quintic", k / N, 1.0)[0]
            if name.startswith("joint"):
                q = qa + s * (qb - qa)
            else:
                q = ik_position(m, d, site, p_a + s * (p_b - p_a), q_prev, iters=50)
            q_prev = q
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            pts.append(d.site_xpos[site].copy())
            qs.append(q[:7].copy())
        pts = np.array(pts)
        u = (p_b - p_a) / np.linalg.norm(p_b - p_a)
        rel = pts - p_a
        dist = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        jpath = np.linalg.norm(np.diff(np.array(qs), axis=0), axis=1).sum()
        print(f"{name:<28}{dist.max() * 100:>27.2f}{length * 100:>14.2f}{jpath:>16.3f}")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    part_a()
    part_b()
    part_c()


if __name__ == "__main__":
    main()
