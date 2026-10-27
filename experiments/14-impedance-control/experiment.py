"""Lesson 14 · Operational-space and impedance control on a torque-controlled Panda.

Model changes (Menagerie franka_emika_panda/scene.xml, compiled with MjSpec):
  * the 7 built-in position actuators are switched off (gain and bias = 0); our torques go to qfrc_applied,
    clipped at the model's forcerange (87 N m joints 1-4, 12 N m joints 5-7);
  * a site "tcp" 10.34 cm in front of the hand flange and a 1 cm probe sphere just in front of it;
  * for part B a fixed rigid block whose top face is a horizontal wall under the hand.
    Only the probe collides with the block (contype/conaffinity = 2).

  A. Free-space 5 cm steps along x, y, z: impedance (tau = J^T F + bias) vs operational space (F = Lambda a).
     Hand points down (orientation of the Menagerie 'home' keyframe).
  A2. Null space: push the elbow for 0.3 s with and without a null-space posture term.
  B. Push down into the block: sweep the translational stiffness K -> contact force, penetration, saturation.

Run:  python experiment.py      (~30 s on Apple M5)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

PROBE_R = 0.01
TCP_Z = 0.1034
# desired tcp orientation = the 'home' keyframe orientation: approach axis (hand z) pointing down
R_DES = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
P_START = np.array([0.5, 0.0, 0.3])


def build(wall_z=None):
    spec = mujoco.MjSpec.from_file(path("franka_emika_panda"))
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_Z])
    hand.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PROBE_R, 0, 0],
                  pos=[0, 0, TCP_Z + PROBE_R], contype=2, conaffinity=2, mass=0.01)
    if wall_z is not None:                        # block whose top face is at z = wall_z
        spec.worldbody.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.05],
                                pos=[P_START[0], 0, wall_z - 0.05], contype=2, conaffinity=2)
    m = spec.compile()
    m.actuator_gainprm[:7] = 0
    m.actuator_biasprm[:7] = 0
    return m


def orient_err(R, Rd):
    return 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))


def ik(m, p_des, R_des, q_init, iters=500):
    """Damped least-squares IK for the tcp site (position + orientation), 7 arm joints."""
    d = mujoco.MjData(m)
    d.qpos[:] = q_init
    site = m.site("tcp").id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        e = np.concatenate([p_des - d.site_xpos[site], orient_err(d.site_xmat[site].reshape(3, 3), R_des)])
        if np.linalg.norm(e) < 1e-7:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp, jr])[:, :7]
        d.qpos[:7] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
        d.qpos[:7] = np.clip(d.qpos[:7], lo, hi)
    return d.qpos.copy()


class Controller:
    """Task-space controller for the tcp site.
    mode 'impedance': F = K e - D v (N, N m), tau = J^T F + qfrc_bias
    mode 'osc':       F = Lambda (kp e - kd v), Lambda = (J M^-1 J^T)^-1, tau = J^T F + qfrc_bias
    null: if True, add (I - J^T Jbar^T) (k0 (q0 - q) - d0 qd) with the dynamically consistent inverse Jbar."""

    def __init__(self, m, mode, K=600.0, D=50.0, Kr=30.0, Dr=3.0, kp=100.0, kd=14.0, null=False, q0=None):
        self.m, self.mode, self.K, self.D, self.Kr, self.Dr = m, mode, K, D, Kr, Dr
        self.kp, self.kd, self.null, self.q0 = kp, kd, null, q0
        self.site = m.site("tcp").id
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.M = np.zeros((m.nv, m.nv))
        self.lim = m.actuator_forcerange[:7, 1]

    def __call__(self, d, p_des):
        m = self.m
        mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
        J = np.vstack([self.jp, self.jr])[:, :7]
        qd = d.qvel[:7]
        v = J @ qd
        e = np.concatenate([p_des - d.site_xpos[self.site], orient_err(d.site_xmat[self.site].reshape(3, 3), R_DES)])
        mujoco.mj_fullM(m, d, self.M)
        Minv = np.linalg.inv(self.M[:7, :7])
        Lam = np.linalg.inv(J @ Minv @ J.T)
        if self.mode == "impedance":
            F = np.concatenate([self.K * e[:3] - self.D * v[:3], self.Kr * e[3:] - self.Dr * v[3:]])
        else:
            F = Lam @ (self.kp * e - self.kd * v)
        tau = J.T @ F + d.qfrc_bias[:7]
        if self.null:
            Jbar = Minv @ J.T @ Lam
            N = np.eye(7) - J.T @ Jbar.T
            tau += N @ (10.0 * (self.q0[:7] - d.qpos[:7]) - 2.0 * qd)
        sat = bool(np.any(np.abs(tau) > self.lim))
        return np.clip(tau, -self.lim, self.lim), sat, Lam


def start(m, q):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[7] = 255                               # fingers open (the finger actuator is left as it is)
    mujoco.mj_forward(m, d)
    return d


def q_start():
    m = build()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    q = ik(m, P_START, R_DES, d.qpos.copy())
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    print(f"start pose (IK from 'home'): q = {np.round(q[:7], 3).tolist()}, "
          f"tcp error {np.linalg.norm(d.site_xpos[m.site('tcp').id] - P_START) * 1000:.4f} mm")
    return q


def part_a(q0):
    m = build()
    print("\n[A] 5 cm step of the tcp target from (50, 0, 30) cm, hand pointing down; 2 s each")
    print("  impedance: K = 600 N/m, D = 50 N s/m   |   osc: kp = 100 1/s^2, kd = 14 1/s  (both + qfrc_bias)")
    print(f"{'controller':<11}{'axis':>5}{'Lambda_ii kg':>14}{'overshoot %':>13}{'t into 2 mm s':>15}{'off-axis peak mm':>18}")
    for mode in ("impedance", "osc"):
        for ax in range(3):
            d = start(m, q0)
            c = Controller(m, mode)
            target = P_START.copy()
            target[ax] += 0.05
            site = c.site
            hist, off, lam = [], 0.0, None
            while d.time < 2.0 - 1e-9:
                tau, _, L = c(d, target)
                lam = L[ax, ax] if lam is None else lam
                d.qfrc_applied[:7] = tau
                mujoco.mj_step(m, d)
                p = d.site_xpos[site]
                hist.append((d.time, p[ax] - P_START[ax]))
                others = [i for i in range(3) if i != ax]
                off = max(off, float(np.abs(p[others] - P_START[others]).max()))
            t, x = np.array(hist).T
            over = max(0.0, (x.max() - 0.05) / 0.05 * 100)
            outside = np.where(np.abs(x - 0.05) > 0.002)[0]
            ts = t[outside[-1]] if len(outside) and outside[-1] < len(t) - 1 else float("nan")
            print(f"{mode:<11}{'xyz'[ax]:>5}{lam:>14.2f}{over:>13.1f}{ts:>15.3f}{off * 1000:>18.2f}")


def part_a2(q0):
    m = build()
    print("\n[A2] Null space: hold the tcp still, push the elbow (link4 body) with 20 N along +y for 0.3 s, then release")
    print(f"{'controller':<28}{'tcp error mm (max)':>19}{'elbow moved cm (end)':>22}{'|q - q0| rad (end)':>20}")
    elbow = m.body("link4").id
    for name, null in (("osc, no null-space term", False), ("osc + null-space posture", True)):
        d = start(m, q0)
        c = Controller(m, "osc", null=null, q0=q0)
        e0 = d.xpos[elbow].copy()
        emax = 0.0
        while d.time < 3.0 - 1e-9:
            d.xfrc_applied[elbow, :3] = [0, 20.0, 0] if 0.5 <= d.time < 0.8 else 0.0
            tau, _, _ = c(d, P_START)
            d.qfrc_applied[:7] = tau
            mujoco.mj_step(m, d)
            emax = max(emax, float(np.linalg.norm(d.site_xpos[c.site] - P_START)))
        print(f"{name:<28}{emax * 1000:>19.2f}{np.linalg.norm(d.xpos[elbow] - e0) * 100:>22.2f}"
              f"{np.linalg.norm(d.qpos[:7] - q0[:7]):>20.3f}")


def part_b(q0):
    wall_z = P_START[2] - PROBE_R * 2 - 0.03           # wall surface 3 cm below the probe
    depth = 0.05                                       # target: 5 cm beyond the wall surface
    print(f"\n[B] Push into a wall: target ramps (quintic, 1 s) from the start to {depth * 100:.0f} cm beyond the wall "
          "surface, then holds 1.5 s")
    print("  impedance control, D = 2 sqrt(K Lambda_zz) (critically damped along z), rotation Kr = 30 N m/rad")
    print(f"{'K N/m':>8}{'K x 5cm N':>11}{'force N (hold)':>16}{'peak N':>9}{'penetration mm':>16}"
          f"{'saturated %':>13}{'contact lost':>14}{'max tilt deg':>14}")
    m = build(wall_z)
    wall = m.geom("wall").id
    f6 = np.zeros(6)
    for K in (100, 300, 1000, 3000, 10000, 30000, 100000):
        d = start(m, q0)
        c = Controller(m, "impedance", K=K)
        _, _, L = c(d, P_START)
        c.D = 2 * np.sqrt(K * L[2, 2])
        tcp_goal = wall_z - depth + 2 * PROBE_R      # tcp height that would put the probe bottom 5 cm in
        forces, pens, sat, n, lost, was, tilt = [], [], 0, 0, 0, False, 0.0
        while d.time < 2.5 - 1e-9:
            u = min(1.0, d.time / 1.0)
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            target = P_START.copy()
            target[2] = P_START[2] + s * (tcp_goal - P_START[2])
            tau, sflag, _ = c(d, target)
            d.qfrc_applied[:7] = tau
            mujoco.mj_step(m, d)
            sat += sflag
            n += 1
            fn = 0.0
            for k in range(d.ncon):
                con = d.contact[k]
                if wall in (con.geom1, con.geom2):
                    mujoco.mj_contactForce(m, d, k, f6)
                    fn += f6[0]
            touching = fn > 0
            if was and not touching and d.time > 1.0:
                lost += 1
            was = touching
            forces.append((d.time, fn))
            Rz = d.site_xmat[c.site].reshape(3, 3)[:, 2]
            tilt = max(tilt, float(np.degrees(np.arccos(np.clip(-Rz[2], -1, 1)))))
            bottom = d.geom_xpos[m.geom("probe").id][2] - PROBE_R
            pens.append(wall_z - bottom)
        t, f = np.array(forces).T
        hold = f[t > 2.2].mean()
        print(f"{K:>8}{K * depth:>11.1f}{hold:>16.2f}{f.max():>9.1f}{max(pens) * 1000:>16.3f}"
              f"{100 * sat / n:>13.1f}{lost:>14d}{tilt:>14.2f}")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    q0 = q_start()
    part_a(q0)
    part_a2(q0)
    part_b(q0)


if __name__ == "__main__":
    main()
