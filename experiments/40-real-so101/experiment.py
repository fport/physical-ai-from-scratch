"""Lesson 40 · Preflight a commanded SO-101 trajectory in simulation before sending it to the real arm.

A pick-and-place goal sequence (what a policy or a replayed episode would send at 30 fps) is checked
against the Menagerie SO-101 model:

  1. range     every goal inside the model joint range with a margin (5 deg)
  2. step      largest goal change per 30 fps tick (our threshold 0.10 rad, cf. LeRobot max_relative_target)
  3. track     sim playback through the model's STS3215 servos: max tracking error (5 deg)
  4. sat       share of physics steps with any servo at its 2.94 N m limit (information only: with
               kp = 998 N m/rad the model servo saturates beyond 2.9 mrad of error, i.e. in almost any motion)
  5. gravity   static gravity torque along the path vs 80% of the 2.94 N m limit
  6. contact   arm-floor or arm-arm contacts during playback (none allowed)

  A. Sweep the playback speed (x0.5 ... x4).
  B. Sweep a payload in the gripper (0 ... 0.8 kg) at speed x1.
  Break it: a calibration offset on shoulder_lift (the real zero differs from the model zero).
  Exercise 3: lowest gripper-frame height during playback vs that offset (2 cm clearance).
None of this is measured on the real arm; it only says whether the command is sane for the model.

Run:  python experiment.py      (deterministic; a few seconds of CPU)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

FPS = 30
DT = 0.002
SEG_S = 1.2                                # seconds per waypoint segment at speed x1
WAYPOINTS = np.array([                     # model joint angles [rad]: pan, lift, elbow, wrist_flex, wrist_roll, gripper
    [0.0, -0.5, 0.5, 0.8, 0.0, 0.2],       # home (gripper ~13 cm above the floor)
    [0.4, -0.3, 0.6, 0.9, 0.0, 1.0],       # above object, open
    [0.4, -0.15, 0.5, 1.05, 0.0, 1.0],     # down (gripper frame ~4 cm up)
    [0.4, -0.15, 0.5, 1.05, 0.0, 0.0],     # close
    [0.4, -0.4, 0.6, 0.9, 0.0, 0.0],       # lift
    [-0.4, -0.4, 0.6, 0.9, 0.0, 0.0],      # carry
    [-0.4, -0.15, 0.5, 1.05, 0.0, 0.0],    # down
    [-0.4, -0.15, 0.5, 1.05, 0.0, 1.0],    # open
    [0.0, -0.5, 0.5, 0.8, 0.0, 0.2],       # home
])
LIMITS = dict(range_margin_deg=5.0, step_rad=0.10, track_deg=5.0, grav_frac=0.80)


def min_jerk(a, b, n):
    s = np.linspace(0, 1, n, endpoint=False)[:, None]
    return a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def trajectory(speed=1.0, offset=None):
    n = max(2, int(round(SEG_S / speed * FPS)))
    segs = [min_jerk(WAYPOINTS[i], WAYPOINTS[i + 1], n) for i in range(len(WAYPOINTS) - 1)]
    goals = np.vstack(segs + [WAYPOINTS[-1:]] * int(0.5 * FPS))       # hold home 0.5 s at the end
    if offset is not None:
        goals = goals + offset
    return goals


def build(payload=0.0):
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    spec.option.timestep = DT
    if payload > 0:
        grip = spec.body("gripper")
        site = [s for s in spec.sites if s.name == "gripperframe"][0]
        b = grip.add_body(name="payload", pos=site.pos)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015], contype=0, conaffinity=0, mass=0)
        b.mass, b.inertia, b.explicitinertial = payload, [1e-5, 1e-5, 1e-5], True
    return spec.compile()


def preflight(goals, payload=0.0):
    m = build(payload)
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
    margin = np.minimum(goals - lo, hi - goals).min()                  # rad, negative = outside
    step = np.abs(np.diff(goals, axis=0)).max()
    # static gravity torque at every goal (qvel = 0): qfrc_bias is gravity only
    grav = 0.0
    for g in goals:
        d.qpos[:6] = g
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        grav = max(grav, np.abs(d.qfrc_bias[:5]).max())
    # playback: goals held for one 30 fps tick each, servos as modelled
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = goals[0]
    d.ctrl[:6] = goals[0]
    floor = m.geom("floor").id
    per_tick = int(round(1 / FPS / DT))
    lim = m.actuator_forcerange[:6, 1] - 1e-6
    err, sat, n, floor_hits, self_hits = 0.0, 0, 0, 0, 0
    site = m.site("gripperframe").id
    z_min = np.inf
    for i, g in enumerate(goals):
        d.ctrl[:6] = g
        for _ in range(per_tick):
            mujoco.mj_step(m, d)
            n += 1
            sat += bool((np.abs(d.actuator_force[:6]) >= lim).any())
            z_min = min(z_min, d.site_xpos[site, 2])
            hit_f = hit_s = False
            for k in range(d.ncon):
                c = d.contact[k]
                if floor in (c.geom1, c.geom2):
                    hit_f = True
                else:
                    hit_s = True
            floor_hits += hit_f
            self_hits += hit_s
        err = max(err, np.abs(d.qpos[:5] - g[:5]).max())             # arm joints, end of the tick vs its goal
    return dict(margin=np.degrees(margin), step=step, track=np.degrees(err), sat=sat / n,
                grav=grav, floor=floor_hits, self=self_hits, tau_max=m.actuator_forcerange[0, 1], z_min=z_min)


def verdict(r):
    fails = []
    if r["margin"] < LIMITS["range_margin_deg"]:
        fails.append("range")
    if r["step"] > LIMITS["step_rad"]:
        fails.append("step")
    if r["track"] > LIMITS["track_deg"]:
        fails.append("track")
    if r["grav"] > LIMITS["grav_frac"] * r["tau_max"]:
        fails.append("gravity")
    if r["floor"] or r["self"]:
        fails.append("contact")
    return "PASS" if not fails else "FAIL: " + ",".join(fails)


def header(first):
    print(f"    {first:>10} | {'margin deg':>10} | {'step rad':>8} | {'track deg':>9} | {'sat %':>6} | "
          f"{'grav N m':>8} | {'floor/self steps':>16} | verdict")


def row(label, r):
    print(f"    {label:>10} | {r['margin']:>10.2f} | {r['step']:>8.3f} | {r['track']:>9.2f} | {r['sat'] * 100:>6.1f} | "
          f"{r['grav']:>8.3f} | {r['floor']:>7}/{r['self']:<8} | {verdict(r)}")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    g1 = trajectory(1.0)
    print(f"trajectory: {len(WAYPOINTS)} waypoints, min-jerk segments of {SEG_S} s at x1, "
          f"{len(g1)} goals at {FPS} fps ({len(g1) / FPS:.1f} s)")
    print(f"limits: margin >= {LIMITS['range_margin_deg']} deg, step <= {LIMITS['step_rad']} rad/tick, "
          f"track <= {LIMITS['track_deg']} deg, "
          f"gravity <= {LIMITS['grav_frac'] * 100:.0f}% of limit, no contacts (sat % is information only)")
    print()
    print("[A] Sweep playback speed (no payload)")
    header("speed")
    for s in [0.5, 1.0, 2.0, 3.0, 4.0]:
        row(f"x{s:g}", preflight(trajectory(s)))
    print()
    print("[B] Sweep payload at the gripper frame (speed x1)")
    header("payload kg")
    for p in [0.0, 0.2, 0.4, 0.6, 0.8]:
        row(f"{p:.1f}", preflight(g1, payload=p))
    print()
    print("[Break it] Calibration offset on shoulder_lift: the arm's zero is off by this much (speed x1)")
    header("offset deg")
    for off in [0, 5, 10, 20, 30]:
        o = np.zeros(6)
        o[1] = np.radians(off)
        row(f"{off}", preflight(trajectory(1.0, offset=o)))
    print()
    print("[Exercise 3] Lowest gripper-frame height above the floor during playback vs shoulder_lift offset (speed x1)")
    print(f"    {'offset deg':>10} | {'min height cm':>13} | 2 cm clearance")
    for off in range(0, 11):
        o = np.zeros(6)
        o[1] = np.radians(off)
        z = preflight(trajectory(1.0, offset=o))["z_min"]
        print(f"    {off:>10} | {z * 100:>13.2f} | {'ok' if z >= 0.02 else 'FAIL'}")


if __name__ == "__main__":
    main()
