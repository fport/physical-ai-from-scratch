"""Lesson 26 · Build it: frames, replay and re-targeting from scratch, on the Experiment's SO-101 scenes.

Step 1  Cut a 500 Hz scripted demonstration into 25 Hz (observation, action) frames vs demos.npz.
Step 2  Zero-order hold: the fixed-timeline recording of part A at 25 Hz, replayed at the recorded spot.
Step 3  Open-loop replay with the cube shifted 1-3 cm in 8 directions.
Step 4  Re-target the recorded joints with the Jacobian: first-order error of the tool point.
Step 5  Replay of the re-targeted frames on the same 24 shifts.

The functions live in physai/data.py (the lesson shows them step by step); the scenes, the recording and the
physical success test come from experiment.py (lesson 26) and lesson 16.
Run:  uv run python experiments/26-teleop-replay-data/build.py      (~6 s, deterministic)
"""
import importlib.util
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.data import hold, retarget, to_frames  # noqa: E402
from physai.evaluation import wilson  # noqa: E402
from physai.kinematics import chain_from_mujoco, fk, jacobian  # noqa: E402

spec = importlib.util.spec_from_file_location("lesson26", os.path.join(HERE, "experiment.py"))
L26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L26)
pick16 = L26.pick16
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
K = 20                                         # physics steps per frame: 2 ms x 20 = 25 Hz

print("\n== Step 1. A scripted demonstration cut into 25 Hz frames vs demos.npz, episode 0 ==")
D = np.load(os.path.join(HERE, "demos.npz"))
ep0 = D["episode_index"] == 0
scene = L26.Scene()
xy0 = D["episode_cube_xy"][0].astype(float)
scene.place(xy0)
o0 = scene.obs()
traj = pick16.scripted_pick(scene.m, scene.d, np.array([xy0[0], xy0[1], pick16.CUBE_HALF]), record=True)["trajectory"]
obs, act = to_frames(o0, np.array([scene.obs_from_qpos(s[2]) for s in traj]), np.array([s[3] for s in traj]), K)
print(f"{len(traj)} physics steps -> {len(obs)} frames | demos.npz episode 0: {int(ep0.sum())} frames")
print(f"max |obs - saved| = {np.abs(obs.astype(np.float32) - D['obs'][ep0]).max():.1e} | "
      f"max |action - saved| = {np.abs(act.astype(np.float32) - D['action'][ep0]).max():.1e}")

print("\n== Step 2. Zero-order hold of the part-A recording (fixed 4.5 s timeline, cube at (22, 0) cm) ==")
m, d = L26.build_a(L26.CUBE_HOME)
I = pick16.ids(m)
rec, ik_err = L26.plan_a(m, d, I, L26.CUBE_HOME)
frames = rec[K - 1:len(rec) // K * K:K]          # the command at the end of every 40 ms
held = hold(frames, K)
dev = np.degrees(np.abs(held - rec[:len(held)]))
print(f"{len(rec)} commands -> {len(frames)} frames -> {len(held)} held commands")
print(f"largest gap between held and original command: arm {dev[:, :5].max():.2f} deg, jaw {dev[:, 5].max():.2f} deg")
print(f"replay of the held frames at the recorded spot: {L26.run_a(L26.CUBE_HOME, held)[1]}")

SHIFTS = [(r, a) for r in (1, 2, 3) for a in range(0, 360, 45)]


def shift_xy(r, a):
    return r / 100 * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])


def sweep(make_ctrl):
    rows = {}
    for r, a in SHIFTS:
        rows.setdefault(r, []).append(L26.run_a(L26.CUBE_HOME + shift_xy(r, a), make_ctrl(shift_xy(r, a)))[1])
    for r, kinds in rows.items():
        k, n = kinds.count("held"), len(kinds)
        lo, hi = wilson(k, n)
        fails = {x: kinds.count(x) for x in sorted(set(kinds)) if x != "held"}
        print((f"{r:>5} cm {f'{k}/{n}':>6}  [{lo:.2f}, {hi:.2f}]  " + ", ".join(f"{x} {v}" for x, v in fails.items())).rstrip())


print("\n== Step 3. Open-loop replay of the held frames, cube shifted in 8 directions ==")
print(f"{'shift':>8} {'held':>6}  Wilson 95%    failures")
sweep(lambda shift: held)

print("\n== Step 4. Re-target with the Jacobian: q' = q + pinv(J_p(q)) shift, error of the tool point ==")
chain = chain_from_mujoco(m, "gripperframe")
q_down = frames[60, :5]                         # frame 60 (t = 2.4 s): gripper down at the cube, jaw closing
for r in (1, 2, 3, 5):
    shift = np.r_[shift_xy(r, 90), 0.0]
    q_new = retarget(q_down[None], lambda q: jacobian(chain, q)[:3], shift)[0]
    err = np.linalg.norm(fk(chain, q_new)[2] - fk(chain, q_down)[2] - shift)
    print(f"shift {r} cm toward +y: tool point off by {1000 * err:.2f} mm")


def retargeted(shift):
    out = frames.copy()
    out[:, :5] = retarget(frames[:, :5], lambda q: jacobian(chain, q)[:3], np.r_[shift, 0.0])
    return hold(out, K)


print("\n== Step 5. Replay of the re-targeted frames (cube position read once), same 24 shifts ==")
print(f"{'shift':>8} {'held':>6}  Wilson 95%    failures")
sweep(retargeted)
