"""Lesson 26 · Teleoperation, replay and data collection.

  A. Open-loop replay vs closed-loop re-planning (port of notebooks/26-teleop-replay-data.ipynb).
     The SO-101 picks a 3 cm cube once along a fixed 4.5 s timeline; the 6 commands sent every 2 ms
     (5 arm joints + jaw) are recorded. Then the cube is shifted 0.5-5 cm in 8 directions (81 trials):
       replay   = the recorded command list, unchanged;
       feedback = the same timeline, poses re-solved with IK from the cube's true position in that trial.
     Success is physical (no weld): at t = 4.5 s cube bottom >= 4 cm above the floor and both jaws touch it.
  B. Demonstration generation for lessons 27-29: the scripted pick of lesson 16 (`scripted_pick`) on cube
     positions drawn uniformly in a 12 x 16 cm area, recorded at 25 Hz (one frame every 20 physics steps)
     and saved to demos.npz (observation, action, episode/frame index, timestamp, phase, success flag).
  C. Self-consistency check of the saved data: replay each episode's 25 Hz actions (zero-order hold)
     from the same start, same cube position, and count successes.

Run:  python experiment.py      (deterministic; ~0.5 min CPU on Apple M5, longer wall time on a loaded machine)
"""
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "16-grasping-and-pick-place"))
sys.path.insert(0, EXP)
import experiment as pick16  # noqa: E402  (lesson 16: scripted_pick, build, reset, ids, jaw_contacts)
from common.menagerie import path  # noqa: E402


def load_lesson(name, folder):
    """Import experiments/<folder>/experiment.py under a unique module name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, folder, "experiment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lesson25 = load_lesson("lesson25", "25-evaluating-policies")
wilson, Scene = lesson25.wilson, lesson25.Scene

# ------------------------------------------------------------------ A. replay vs feedback (notebook port)
DT = 0.002
CUBE_HALF = 0.015
CUBE_MASS = 0.03
CUBE_HOME = np.array([0.22, 0.0])            # where the recording is made (22 cm in front of the base)
OFFSETS_CM = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
DIRS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]   # 0 = forward (+x), 90 = +y (moving-jaw side)
ARM = pick16.ARM
G_OPEN, G_CLOSE = 1.2, -0.17
PRE_M, LIFT_M, SUCCESS_LIFT_M = 0.055, 0.07, 0.04
HOME_Q = np.array([0.0, -1.2, 0.5, 1.2, 1.68])
T = dict(settle=0.3, pre=1.5, down=2.3, close=2.9, lift=3.9, hold=4.5)   # fixed timeline [s]


def build_a(cube_xy):
    """so101.xml + a floor plane + a free 3 cm cube (the notebook's scene)."""
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101", "so101.xml"))
    spec.option.timestep = DT
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05])
    b = wb.add_body(name="cube", pos=[cube_xy[0], cube_xy[1], CUBE_HALF])
    b.add_freejoint()
    b.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=CUBE_MASS,
               friction=[1.0, 0.03, 0.003], condim=4, solref=[0.01, 1])
    m = spec.compile()
    return m, mujoco.MjData(m)


def ik(m, d, I, target, q0=None, iters=1500):
    return pick16.ik(m, I, d.qpos if q0 is None else q0, target, iters=iters)


def plan_a(m, d, I, cube_xy):
    """Fixed-timeline top-down pick; returns ctrl for every physics step (n x 6)."""
    tip = 0.004
    q_down, e2 = ik(m, d, I, [cube_xy[0], cube_xy[1], CUBE_HALF - tip])
    seed = d.qpos.copy()
    seed[I["qadr"]] = q_down
    q_pre, e1 = ik(m, d, I, [cube_xy[0], cube_xy[1], CUBE_HALF + PRE_M], q0=seed)
    q_lift, e3 = ik(m, d, I, [cube_xy[0], cube_xy[1], CUBE_HALF + LIFT_M], q0=seed)
    mj = pick16.minjerk
    n = int(round(T["hold"] / DT))
    ctrl = np.zeros((n, 6))
    for i in range(n):
        t = (i + 1) * DT
        if t < T["settle"]:
            q, g = HOME_Q, G_OPEN
        elif t < T["pre"]:
            q, g = HOME_Q + (q_pre - HOME_Q) * mj((t - T["settle"]) / (T["pre"] - T["settle"])), G_OPEN
        elif t < T["down"]:
            q, g = q_pre + (q_down - q_pre) * mj((t - T["pre"]) / (T["down"] - T["pre"])), G_OPEN
        elif t < T["close"]:
            q, g = q_down, G_OPEN + (G_CLOSE - G_OPEN) * mj((t - T["down"]) / (T["close"] - T["down"]))
        elif t < T["lift"]:
            q, g = q_down + (q_lift - q_down) * mj((t - T["close"]) / (T["lift"] - T["close"])), G_CLOSE
        else:
            q, g = q_lift, G_CLOSE
        ctrl[i, :5], ctrl[i, 5] = q, g
    return ctrl, max(e1, e2, e3)


def run_a(cube_xy, ctrl):
    """Play a ctrl sequence with the cube at cube_xy; physical success test at the end."""
    m, d = build_a(cube_xy)
    I = pick16.ids(m)
    d.qpos[I["qadr"]] = HOME_Q
    d.qpos[I["grip_q"]] = G_OPEN
    mujoco.mj_forward(m, d)
    start = d.xpos[I["cube_body"]].copy()
    touched = False
    for c in ctrl:
        d.ctrl[I["act"]] = c[:5]
        d.ctrl[I["grip"]] = c[5]
        mujoco.mj_step(m, d)
        if not touched:
            nf, nm = pick16.jaw_contacts(m, d, I)
            touched = nf > 0 or nm > 0
    nf, nm = pick16.jaw_contacts(m, d, I)
    end = d.xpos[I["cube_body"]]
    lifted = end[2] - CUBE_HALF >= SUCCESS_LIFT_M
    ok = bool(lifted and nf > 0 and nm > 0)
    moved = float(np.linalg.norm(end[:2] - start[:2]))
    if ok:
        kind = "held"
    elif not touched:
        kind = "missed"          # the jaws never touched the cube
    elif not lifted:
        kind = "pushed"          # touched it, but it stayed on the floor
    else:
        kind = "one jaw"
    return ok, kind, moved


def part_a():
    m, d = build_a(CUBE_HOME)
    I = pick16.ids(m)
    rec_ctrl, ik_err = plan_a(m, d, I, CUBE_HOME)
    print(f"recording: {rec_ctrl.shape[0]} steps x {rec_ctrl.shape[1]} commands (2 ms), IK error {ik_err * 1000:.2f} mm")
    rows = []
    for r in OFFSETS_CM:
        for a in (DIRS_DEG if r > 0 else [0]):
            xy = CUBE_HOME + r / 100 * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            ok_r, kind_r, _ = run_a(xy, rec_ctrl)
            m0, d0 = build_a(xy)
            ctrl, _ = plan_a(m0, d0, pick16.ids(m0), xy)
            ok_f, kind_f, _ = run_a(xy, ctrl)
            rows.append((r, a, ok_r, kind_r, ok_f, kind_f))
    print(f"\n{'shift cm':>9}{'replay':>9}{'Wilson 95%':>15}{'feedback':>10}   replay outcomes")
    for r in OFFSETS_CM:
        sel = [x for x in rows if x[0] == r]
        kr, kf, n = sum(x[2] for x in sel), sum(x[4] for x in sel), len(sel)
        lo, hi = wilson(kr, n)
        kinds = {}
        for x in sel:
            kinds[x[3]] = kinds.get(x[3], 0) + 1
        print(f"{r:>9g}{f'{kr}/{n}':>9}{f'[{lo:.2f}, {hi:.2f}]':>15}{f'{kf}/{n}':>10}   "
              + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    kr, kf = sum(x[2] for x in rows), sum(x[4] for x in rows)
    print(f"total: replay {kr}/{len(rows)}, feedback {kf}/{len(rows)}")
    print("\nfarthest successful replay shift per direction (0 = forward +x, 90 = +y moving-jaw side, 270 = -y fixed-jaw side):")
    for a in DIRS_DEG:
        ok = [x[0] for x in rows if x[1] == a and x[2]]
        pattern = "".join("o" if x[2] else "." for x in rows if x[1] == a and x[0] > 0)
        print(f"  {a:>3} deg: {max(ok) if ok else 0:>3g} cm   0.5..5 cm: {pattern}")


# ------------------------------------------------------------------ B. demonstrations for lessons 27-29
FPS = 25
K = int(round(1 / (FPS * DT)))               # physics steps per recorded frame (20)
X_RANGE, Y_RANGE = (0.16, 0.28), (-0.08, 0.08)
N_DEMOS = 120
PHASES = ["settle", "pre", "down", "close", "lift", "hold"]
OBS_NAMES = ARM + ["gripper", "cube_x", "cube_y", "cube_z"]
ACT_NAMES = ARM + ["gripper"]


def record_demo(scene, xy):
    """Scripted pick (lesson 16) recorded at 25 Hz. Frame k: observation at t = k/25 s, action = the joint
    targets the expert commands at the end of the next 40 ms (held for 20 physics steps on replay)."""
    scene.place(xy)
    o0 = scene.obs()
    r = pick16.scripted_pick(scene.m, scene.d, np.array([xy[0], xy[1], pick16.CUBE_HALF]), record=True)
    tr = r["trajectory"]
    obs, act, phase = [], [], []
    for k in range(len(tr) // K):
        obs.append(o0 if k == 0 else scene.obs_from_qpos(tr[k * K - 1][2]))
        act.append(tr[k * K + K - 1][3])
        phase.append(PHASES.index(tr[k * K + K - 1][1]))
    return np.array(obs, np.float32), np.array(act, np.float32), np.array(phase, np.int8), r["success"]


def sample_positions(n, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(*X_RANGE, n), rng.uniform(*Y_RANGE, n)])


def part_b(scene):
    xy_all = sample_positions(N_DEMOS, seed=26)
    O, A, P, EP, FR, OK = [], [], [], [], [], []
    for e, xy in enumerate(xy_all):
        o, a, p, ok = record_demo(scene, xy)
        O.append(o), A.append(a), P.append(p), OK.append(ok)
        EP.append(np.full(len(o), e, np.int16)), FR.append(np.arange(len(o), dtype=np.int16))
    lens = np.array([len(o) for o in O])
    out = os.path.join(HERE, "demos.npz")
    np.savez_compressed(out, obs=np.concatenate(O), action=np.concatenate(A), phase=np.concatenate(P),
                        episode_index=np.concatenate(EP), frame_index=np.concatenate(FR),
                        timestamp=(np.concatenate(FR) / FPS).astype(np.float32),
                        episode_success=np.array(OK), episode_cube_xy=xy_all.astype(np.float32),
                        fps=np.int16(FPS), obs_names=np.array(OBS_NAMES), action_names=np.array(ACT_NAMES),
                        phase_names=np.array(PHASES), task=np.array("pick up the cube"))
    print(f"{N_DEMOS} scripted demonstrations, cube x in [{X_RANGE[0] * 100:.0f}, {X_RANGE[1] * 100:.0f}] cm, "
          f"y in [{Y_RANGE[0] * 100:.0f}, {Y_RANGE[1] * 100:.0f}] cm (seed 26), recorded at {FPS} Hz")
    print(f"  successful (physical test): {sum(OK)}/{N_DEMOS}; failed episodes at cube xy [cm]: "
          + ", ".join(f"({xy[0] * 100:.1f}, {xy[1] * 100:.1f})" for xy, ok in zip(xy_all, OK) if not ok))
    print(f"  frames per episode: min {lens.min()}, max {lens.max()}, total {lens.sum()} "
          f"({lens.min() / FPS:.2f}-{lens.max() / FPS:.2f} s)")
    print(f"  observation {len(OBS_NAMES)} = {OBS_NAMES}")
    print(f"  action      {len(ACT_NAMES)} = {ACT_NAMES} (joint position targets [rad])")
    print(f"  saved demos.npz: {os.path.getsize(out) / 1e6:.2f} MB")
    return xy_all, A, OK


def part_c(scene, xy_all, A, OK, n=20):
    """Replay the saved 25 Hz actions of the first n episodes, same cube position, zero-order hold."""
    k_ok = 0
    for e in range(n):
        scene.place(xy_all[e])
        for a in A[e]:
            scene.apply(a)
            scene.step(K)
        k_ok += scene.success()
    lo, hi = wilson(k_ok, n)
    print(f"replay of the saved 25 Hz actions, same cube position: {k_ok}/{n} succeed "
          f"(Wilson 95% [{lo:.2f}, {hi:.2f}]); the original demos: {sum(OK[:n])}/{n}")


def main():
    t0 = time.process_time()
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    print("\n[A] Open-loop replay vs closed-loop re-planning, cube shifted 0.5-5 cm in 8 directions (81 trials)")
    part_a()
    print("\n[B] Scripted demonstrations (lesson 16 scripted_pick)")
    scene = Scene()
    xy_all, A, OK = part_b(scene)
    print("\n[C] Is the saved dataset self-consistent?")
    part_c(scene, xy_all, A, OK)
    print(f"CPU time {time.process_time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
