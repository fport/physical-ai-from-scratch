"""Lesson 15 · Safety layers between a command and the motors, measured in simulation.

SO-101 (Menagerie scene.xml). Each joint is the model's STS3215 position servo
(force = kp (ctrl - q) - kv qdot, kp 998.22, kv 2.731, clipped at +-2.94 N m).
Timestep set to 2 ms; the command loop runs at 50 Hz (every 10 physics steps).
A fixed post (2 x 2 x 20 cm) stands in the swing path of the gripper.

  A. A bad command: shoulder_pan target jumps 0 -> 1.2 rad. Each safeguard alone vs none.
  B. Sweep the torque cap of shoulder_pan -> impact force and steady push force on the post.
  C. Sweep the per-tick relative-target limit (clamp goal to present +- delta) -> impact speed and force.
  D. Watchdog: the operator jogs shoulder_pan at 1 rad/s, then commands stop arriving.
     Sweep the timeout -> time to stop, travel after the last command, impact.
  E. Two kinds of stop from the same moving state: hold position vs torque off.

Run:  python experiment.py      (~20 s on Apple M5)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

DT = 0.002
TICK = 10                                       # physics steps per command (50 Hz)
POSE = np.array([0.0, -0.5, 0.5, 0.8, 0.0, 0.0])  # gripper ~29 cm out, 13 cm high
POST_ANGLE = 0.6                                # post centre at this pan angle [rad]


def build(pan_cap=None):
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    spec.option.timestep = DT
    r = 0.255                                    # gripper radius from the pan axis in POSE
    x0 = 0.0388                                  # pan axis x (shoulder body)
    spec.worldbody.add_geom(name="post", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.1],
                            pos=[x0 + r * np.cos(POST_ANGLE), -r * np.sin(POST_ANGLE), 0.1])
    m = spec.compile()
    if pan_cap is not None:
        m.actuator_forcerange[m.actuator("shoulder_pan").id] = [-pan_cap, pan_cap]
    return m


def post_force(m, d, post, f6):
    fn = 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        if post in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, k, f6)
            fn += f6[0]
    return fn


def run(command, T=2.0, pan_cap=None, pos_limit=None, rel_limit=None, watchdog=None, estop=None):
    """command(t) -> (pan target or None for 'no message this tick', is_velocity)."""
    m = build(pan_cap)
    d = mujoco.MjData(m)
    d.qpos[:6] = POSE
    d.ctrl[:6] = POSE
    mujoco.mj_forward(m, d)
    post, f6 = m.geom("post").id, np.zeros(6)
    pan = m.actuator("shoulder_pan").id
    gz0 = d.site("gripperframe").xpos[2]
    out = dict(peak_force=0.0, peak_speed=0.0, peak_tau=0.0, forces=[], t_last=None, t_stop=None, q_last=None,
               silent=False, max_travel=0.0)
    target, v_cmd, last_msg = 0.0, 0.0, 0.0
    step = 0
    out["impact_speed"] = None
    while d.time < T - 1e-9:
        if step % TICK == 0:
            t = d.time
            if estop is not None and t >= estop[1]:
                if not out["silent"]:
                    out["silent"], out["q_last"], out["t_last"] = True, d.qpos[0], t
                    if estop[0] == "hold":
                        target, v_cmd = d.qpos[0], 0.0           # hold where we are
                    else:                                        # torque off: every actuator stops producing force
                        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
            else:
                msg, is_vel = command(t)
                if msg is not None:
                    last_msg, out["q_last"], out["t_last"] = t, d.qpos[0], t
                    if is_vel:
                        v_cmd = msg
                    else:
                        target = msg
                else:
                    out["silent"] = True
                    if watchdog is not None and t - last_msg > watchdog and v_cmd != 0.0:
                        v_cmd, target = 0.0, d.qpos[0]           # timeout: stop and hold where we are
                if is_vel:
                    target += v_cmd * TICK * DT                  # no new message: keep the last velocity
            goal = target
            if pos_limit is not None:
                goal = float(np.clip(goal, -pos_limit, pos_limit))
            if rel_limit is not None:
                goal = float(np.clip(goal, d.qpos[0] - rel_limit, d.qpos[0] + rel_limit))
            d.ctrl[pan] = goal
        speed_before = abs(d.qvel[0])
        mujoco.mj_step(m, d)
        step += 1
        f = post_force(m, d, post, f6)
        if f > 0 and out["impact_speed"] is None:
            out["impact_speed"] = speed_before
        out["forces"].append(f)
        out["peak_force"] = max(out["peak_force"], f)
        out["peak_speed"] = max(out["peak_speed"], abs(d.qvel[0]))
        out["peak_tau"] = max(out["peak_tau"], abs(d.actuator_force[pan]))
        if out["silent"]:
            out["max_travel"] = max(out["max_travel"], abs(d.qpos[0] - out["q_last"]))
        if out["silent"] and out["t_stop"] is None and d.time > out["t_last"] + 0.01 and abs(d.qvel[0]) < 0.01:
            out["t_stop"] = d.time
    out["q_end"] = d.qpos[0]
    out["hold_force"] = float(np.mean(out["forces"][-int(0.3 / DT):]))
    out["grip_drop"] = gz0 - d.site("gripperframe").xpos[2]
    return out


def part_a():
    m = build()
    d = mujoco.MjData(m)
    d.qpos[:6] = POSE
    mujoco.mj_kinematics(m, d)
    p = d.site("gripperframe").xpos * 100
    print(f"\npose {POSE.tolist()}: gripper frame at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) cm")
    step_cmd = lambda t: (1.2 if t >= 0.1 else 0.0, False)
    print(f"\n[A] Bad command: pan target 0 -> 1.2 rad at t = 0.1 s; post at pan ~{POST_ANGLE} rad; 2 s")
    print(f"{'safeguard':<34}{'peak force N':>13}{'hold force N':>13}{'peak speed rad/s':>17}{'peak torque N m':>16}")
    cases = [("none", {}), ("joint limit |pan| <= 0.3 rad", dict(pos_limit=0.3)),
             ("relative target <= 0.02 rad/tick", dict(rel_limit=0.02)), ("torque cap 0.5 N m", dict(pan_cap=0.5)),
             ("all three", dict(pos_limit=0.3, rel_limit=0.02, pan_cap=0.5))]
    for name, kw in cases:
        r = run(step_cmd, **kw)
        print(f"{name:<34}{r['peak_force']:>13.2f}{r['hold_force']:>13.2f}{r['peak_speed']:>17.2f}{r['peak_tau']:>16.2f}")


def part_b():
    step_cmd = lambda t: (1.2 if t >= 0.1 else 0.0, False)
    print("\n[B] Torque cap sweep on shoulder_pan (same bad command, no other safeguard)")
    print(f"{'cap N m':>8}{'peak force N':>13}{'hold force N':>13}{'peak speed rad/s':>17}")
    for cap in (2.94, 2.0, 1.0, 0.5, 0.25, 0.1):
        r = run(step_cmd, pan_cap=cap)
        print(f"{cap:>8.2f}{r['peak_force']:>13.2f}{r['hold_force']:>13.2f}{r['peak_speed']:>17.2f}")


def part_c():
    step_cmd = lambda t: (1.2 if t >= 0.1 else 0.0, False)
    print("\n[C] Relative-target limit sweep (goal clamped to present +- delta every 20 ms tick)")
    print(f"{'delta rad':>10}{'speed at impact rad/s':>23}{'peak force N':>13}{'hold force N':>13}")
    for delta in (None, 0.2, 0.05, 0.02, 0.01, 0.005):
        r = run(step_cmd, rel_limit=delta)
        sp = r["impact_speed"] if r["impact_speed"] is not None else float("nan")
        name = "none" if delta is None else f"{delta:g}"
        print(f"{name:>10}{sp:>23.3f}{r['peak_force']:>13.2f}{r['hold_force']:>13.2f}")


def part_d():
    jog = lambda t: (1.0, True) if t < 0.2 - 1e-9 else (None, True)     # last message at t = 0.18 s
    print("\n[D] Watchdog: jog pan at 1 rad/s, the last message arrives at t = 0.18 s; post at pan ~0.6 rad")
    print(f"{'timeout s':>10}{'stopped after last msg s':>26}{'max travel after last msg deg':>31}{'peak force N':>13}")
    for to in (None, 1.0, 0.5, 0.2, 0.1, 0.05):
        r = run(jog, T=2.5, watchdog=to)
        if to is None:
            print(f"  (pan at the last message: {r['q_last']:.3f} rad)")
        stop = (r["t_stop"] - r["t_last"]) if r["t_stop"] is not None else float("nan")
        name = "none" if to is None else f"{to:g}"
        print(f"{name:>10}{stop:>26.3f}{np.degrees(r['max_travel']):>31.2f}{r['peak_force']:>13.2f}")


def part_e():
    jog = lambda t: (1.0, True)
    print("\n[E] Stop from the 1 rad/s jog at t = 0.2 s: hold position vs torque off (1.5 s)")
    print(f"{'stop':<14}{'max pan travel after stop deg':>31}{'pan stopped after s':>21}{'gripper drop cm':>17}")
    for kind in ("hold", "torque off"):
        r = run(jog, T=1.7, estop=(kind, 0.2))
        ts = (r["t_stop"] - r["t_last"]) if r["t_stop"] is not None else float("nan")
        print(f"{kind:<14}{np.degrees(r['max_travel']):>31.2f}{ts:>21.3f}{r['grip_drop'] * 100:>17.2f}")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()


if __name__ == "__main__":
    main()
