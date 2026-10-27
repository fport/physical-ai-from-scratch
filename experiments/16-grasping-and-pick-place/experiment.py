"""Lesson 16 · Grasping and scripted pick-and-place with the SO-101.

A 3 cm cube on the floor in front of the SO-101 (Menagerie scene.xml, timestep set to 2 ms).
State machine: home -> above the cube -> down -> close the jaw -> lift -> hold.
Joint targets come from damped-least-squares IK; every segment is a minimum-jerk (quintic) blend.
The model's own STS3215 position servos track the targets; the jaw squeezes until its servo hits
its torque limit. No weld, no attachment: success is measured physically at the end
(cube bottom >= 4 cm above the floor AND the cube touches both jaws).

  A. Friction sweep: cube friction mu -> success rate (10 random cube positions per value), compared with
     the friction-cone condition mu (N_fixed + N_moving) >= m g using the measured jaw squeeze force.
  B. Jaw torque cap sweep (grip force) at a low mu.
  C. Mass sweep.

Part 7 can import `scripted_pick(model, data, cube_pos)` from this file to record demonstrations.

Run:  python experiment.py      (~30 s on Apple M5)
"""
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

DT = 0.002
CUBE_HALF = 0.015                         # 3 cm cube
CUBE_HOME = np.array([0.22, 0.0])          # 22 cm in front of the base
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
G_OPEN, G_CLOSE = 1.2, -0.17               # jaw command [rad]: open / closed joint limit (the servo squeezes)
PRE_M, LIFT_M, SUCCESS_LIFT_M = 0.055, 0.07, 0.04
HOME_Q = np.array([0.0, -1.2, 0.5, 1.2, 1.68])
# state machine: (state, duration [s]). CLOSE ends on a sensed condition, not on a timer.
DUR = dict(settle=0.3, pre=1.2, down=0.8, close_ramp=0.3, lift=1.0, hold=0.8)
CLOSE_TIMEOUT = 4.0                        # give up if the jaw has not settled on the cube by then
SETTLED_SPEED, SETTLED_TIME = 0.05, 0.1    # jaw speed [rad/s] that counts as "stopped", for this long [s]


def build(cube_xy, mu=1.0, mass=0.03):
    """SO-101 scene + free cube. priority=2 makes the cube's friction the one used in every cube contact
    (the jaw pads have priority 1 and friction 1 in the Menagerie model)."""
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    spec.option.timestep = DT
    b = spec.worldbody.add_body(name="cube", pos=[cube_xy[0], cube_xy[1], CUBE_HALF])
    b.add_freejoint()
    b.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=mass,
               friction=[mu, 0.03, 0.003], condim=4, solref=[0.01, 1], priority=2)
    m = spec.compile()
    return m, mujoco.MjData(m)


def ids(m):
    jaw_fixed, jaw_moving = set(), set()
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        body = m.body(m.geom_bodyid[g]).name
        if body == "gripper":
            jaw_fixed.add(g)
        elif body == "moving_jaw_so101_v1":
            jaw_moving.add(g)
    return dict(cube=m.geom("cube_geom").id, cube_body=m.body("cube").id, site=m.site("gripperframe").id,
                qadr=[m.joint(j).qposadr[0] for j in ARM], dadr=[m.joint(j).dofadr[0] for j in ARM],
                act=[m.actuator(j).id for j in ARM], grip=m.actuator("gripper").id,
                grip_q=m.joint("gripper").qposadr[0], fixed=jaw_fixed, moving=jaw_moving)


def ik(m, I, q_seed, target, x_axis=(0, 0, -1), z_axis=(0, 1, 0), iters=1500):
    """DLS on the 5 arm joints: gripperframe site at `target`, its x axis (approach) pointing down,
    its z axis along world y (the jaws close along x, onto two opposite cube faces)."""
    dd = mujoco.MjData(m)
    dd.qpos[:] = q_seed
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    xa, za = np.asarray(x_axis, float), np.asarray(z_axis, float)
    for _ in range(iters):
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_comPos(m, dd)
        R = dd.site_xmat[I["site"]].reshape(3, 3)
        ep = np.asarray(target) - dd.site_xpos[I["site"]]
        er = 0.3 * (np.cross(R[:, 0], xa) + 0.5 * np.cross(R[:, 2], za))
        if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
            break
        mujoco.mj_jacSite(m, dd, jp, jr, I["site"])
        J = np.vstack([jp, 0.3 * jr])[:, I["dadr"]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
        for k, a in enumerate(I["qadr"]):
            lo, hi = m.jnt_range[m.dof_jntid[I["dadr"][k]]]
            dd.qpos[a] = np.clip(dd.qpos[a] + 0.7 * dq[k], lo, hi)
    mujoco.mj_kinematics(m, dd)
    q = np.array([dd.qpos[a] for a in I["qadr"]])
    return q, float(np.linalg.norm(np.asarray(target) - dd.site_xpos[I["site"]]))


def minjerk(u):
    u = min(1.0, max(0.0, u))
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def plan(m, d, I, cube_pos):
    """IK for the three arm poses of a top-down grasp: above, at and lifted over the cube."""
    x, y = cube_pos[0], cube_pos[1]
    tip = 0.004                                    # site 4 mm below the cube centre: pads cover the middle
    q_down, e2 = ik(m, I, d.qpos, [x, y, CUBE_HALF - tip])
    seed = d.qpos.copy()
    seed[I["qadr"]] = q_down                       # same elbow branch for all three poses
    q_pre, e1 = ik(m, I, seed, [x, y, CUBE_HALF + PRE_M])
    q_lift, e3 = ik(m, I, seed, [x, y, CUBE_HALF + LIFT_M])
    return dict(pre=q_pre, down=q_down, lift=q_lift), max(e1, e2, e3)


def jaw_contacts(m, d, I):
    """Normal force [N] on the cube from the fixed and the moving jaw."""
    f6, nf, nm = np.zeros(6), 0.0, 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        if I["cube"] not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == I["cube"] else c.geom1
        if other in I["fixed"] or other in I["moving"]:
            mujoco.mj_contactForce(m, d, k, f6)
            if other in I["fixed"]:
                nf += f6[0]
            else:
                nm += f6[0]
    return nf, nm


def scripted_pick(model, data, cube_pos, record=False):
    """Run the whole pick in (model, data) from the current state. The arm should start at rest in HOME_Q
    with the jaw open (see reset()). Returns a result dict; with record=True also the per-step
    (time, state, qpos, ctrl) trajectory, which is what a demonstration dataset stores.

    States: settle -> pre (above the cube) -> down -> close (ramp the jaw command, then wait until both jaws
    touch the cube and the jaw has stopped) -> lift -> hold. Every arm motion is a min-jerk blend."""
    I = ids(model)
    poses, ik_err = plan(model, data, I, cube_pos)
    moves = dict(settle=(HOME_Q, HOME_Q), pre=(HOME_Q, poses["pre"]), down=(poses["pre"], poses["down"]),
                 lift=(poses["down"], poses["lift"]), hold=(poses["lift"], poses["lift"]))
    state, t_state, still_since = "settle", data.time, None
    traj, forces, close_time, squeeze = [], [], float("nan"), (0.0, 0.0)
    gripper_dof = model.joint("gripper").dofadr[0]
    while True:
        t = data.time - t_state
        if state in moves:
            qa, qb = moves[state]
            q = qa + minjerk(t / DUR[state]) * (qb - qa)
            g = G_OPEN if state in ("settle", "pre", "down") else G_CLOSE
        else:                                              # close: arm still, jaw command ramps to G_CLOSE
            q = poses["down"]
            g = G_OPEN + minjerk(t / DUR["close_ramp"]) * (G_CLOSE - G_OPEN)
        data.ctrl[I["act"]] = q
        data.ctrl[I["grip"]] = g
        mujoco.mj_step(model, data)
        if record:
            traj.append((data.time, state, data.qpos.copy(), data.ctrl.copy()))
        if state == "hold" and t > 0.3:
            forces.append(jaw_contacts(model, data, I))
        # transitions
        t = data.time - t_state
        if state == "close":
            nf, nm = jaw_contacts(model, data, I)
            settled = nf > 0 and nm > 0 and abs(data.qvel[gripper_dof]) < SETTLED_SPEED and t > DUR["close_ramp"]
            still_since = (still_since or data.time) if settled else None
            if still_since is not None and data.time - still_since >= SETTLED_TIME:
                close_time, squeeze = t, (nf, nm)
                state, t_state = "lift", data.time
            elif t > CLOSE_TIMEOUT:
                squeeze = (nf, nm)
                state, t_state = "lift", data.time        # lift anyway; the success test will catch it
        elif t >= DUR[state] - 1e-9:
            if state == "hold":
                break
            state, t_state = {"settle": "pre", "pre": "down", "down": "close", "lift": "hold"}[state], data.time
    nf, nm = jaw_contacts(model, data, I)
    bottom = data.xpos[I["cube_body"]][2] - CUBE_HALF
    moved = float(np.linalg.norm(data.xpos[I["cube_body"]] - np.asarray(cube_pos, float)))
    res = dict(success=bool(bottom >= SUCCESS_LIFT_M and nf > 0 and nm > 0), lift_cm=bottom * 100, moved_cm=moved * 100,
               jaw_fixed_N=nf, jaw_moving_N=nm, ik_err_mm=ik_err * 1000, close_s=close_time, squeeze_N=squeeze,
               hold_N=np.mean(forces, axis=0) if forces else np.zeros(2), duration_s=data.time)
    if record:
        res["trajectory"] = traj
    return res


def reset(m, d):
    I = ids(m)
    d.qpos[I["qadr"]] = HOME_Q
    d.qpos[I["grip_q"]] = G_OPEN
    d.ctrl[I["act"]] = HOME_Q
    d.ctrl[I["grip"]] = G_OPEN
    mujoco.mj_forward(m, d)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def trial_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    return [CUBE_HOME + rng.uniform(-0.01, 0.01, 2) for _ in range(n)]


def sweep(label, values, make, n=10):
    """make(v) -> (mu, mass, jaw torque cap). Prints success rate, outcomes and the friction-cone check."""
    print(f"{label:>9}{'success':>9}{'Wilson 95%':>15}{'close s':>9}{'squeeze N1+N2 N':>17}{'mu (N1+N2) / mg':>17}"
          f"   outcomes (held / not closed / dropped; ejected > 30 cm)")
    for v in values:
        mu, mass, cap = make(v)
        out = dict(held=0, not_closed=0, dropped=0, ejected=0)
        Ns, Ts = [], []
        for xy in trial_positions(n):
            m, d = build(xy, mu=mu, mass=mass)
            m.actuator_forcerange[m.actuator("gripper").id] = [-cap, cap]
            reset(m, d)
            r = scripted_pick(m, d, np.array([xy[0], xy[1], CUBE_HALF]))
            if r["success"]:
                out["held"] += 1
            elif not np.isfinite(r["close_s"]):
                out["not_closed"] += 1
            else:
                out["dropped"] += 1
            out["ejected"] += r["moved_cm"] > 30
            if np.isfinite(r["close_s"]):
                Ns.append(sum(r["squeeze_N"]))
                Ts.append(r["close_s"])
        lo, hi = wilson(out["held"], n)
        N = float(np.mean(Ns)) if Ns else float("nan")
        ratio = mu * N / (mass * 9.81)
        Ntxt, rtxt, Ttxt = (f"{N:.2f}", f"{ratio:.2f}", f"{np.mean(Ts):.2f}") if Ns else ("-", "-", "-")
        print(f"{v:>9g}{f'{out['held']}/{n}':>9}{f'[{lo:.2f}, {hi:.2f}]':>15}{Ttxt:>9}{Ntxt:>17}{rtxt:>17}"
              f"   {out['held']} / {out['not_closed']} / {out['dropped']}; {out['ejected']}")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    m, d = build(CUBE_HOME)
    reset(m, d)
    r = scripted_pick(m, d, np.array([*CUBE_HOME, CUBE_HALF]), record=True)
    print(f"\nsingle pick at the nominal position (mu 1.0, 30 g): success {r['success']}, lift {r['lift_cm']:.2f} cm, "
          f"IK error {r['ik_err_mm']:.2f} mm, jaw normal force at the end {r['jaw_fixed_N']:.2f} / {r['jaw_moving_N']:.2f} N, "
          f"{len(r['trajectory'])} recorded steps")
    print("\nAll sweeps: 10 cube positions uniform in a 2 x 2 cm square around (22, 0) cm, seed 0.")
    print("close = time spent in the close state; squeeze = jaw normal force on the cube when it ends")
    print("(both: mean over the trials where the jaw closed; close times out after 4 s);")
    print("friction cone (lesson 04): the cube can hang in the jaws only if mu (N1 + N2) / (m g) >= 1.")
    print("\n[A] Friction sweep: 30 g cube, jaw torque cap 0.5 N m")
    sweep("mu", (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0), lambda v: (v, 0.03, 0.5))
    print("\n[B] Jaw torque cap sweep: 30 g cube, mu = 0.05 (Menagerie default cap 2.94 N m)")
    sweep("cap N m", (2.94, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1), lambda v: (0.05, 0.03, v))
    print("\n[C] Mass sweep: mu = 0.3, jaw torque cap 0.5 N m")
    sweep("mass kg", (0.03, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7), lambda v: (0.3, v, 0.5))

if __name__ == "__main__":
    main()
