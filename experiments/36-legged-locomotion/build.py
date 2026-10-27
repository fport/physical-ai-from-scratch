"""Lesson 36 · Build it: a vectorized Go2 environment (one model, N MjData) with auto-reset and step accounting.

The Go2 is the Experiment's: Menagerie unitree_go2 with its motors turned into PD servos (Kp 20, Kd 0.5) around
the training config's default joint angles. No timing is printed: wall-clock rates are the Experiment's job;
everything here is deterministic.

Step 1  One model, N datas: the batch step equals N independent single-robot loops.
Step 2  What the policy sees: projected gravity and roll / pitch from the base quaternion, vs MuJoCo and SciPy.
Step 3  Auto-reset: push 16 robots with random joint targets; fallen robots restart in the same step.
Step 4  Accounting: policy steps, physics steps, samples per update, and the Go2 config's budget.
Use it  mujoco.rollout runs the same batch in C, open loop.

The functions live in physai/vecenv.py. Run:  uv run python experiments/36-legged-locomotion/build.py   (~3 s)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np
from mujoco import rollout
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.vecenv import VecEnv, projected_gravity, roll_pitch, samples_per_update  # noqa: E402

spec = importlib.util.spec_from_file_location("exp36", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)                       # load_go2() and the stance of the training config

m = exp.load_go2()
KEY = m.key("home").id
Q = exp.Q_STAND


def reset_fn(m, d, rng):
    """Keyframe base pose, the training stance, joint angles jittered by up to 0.05 rad."""
    mujoco.mj_resetDataKeyframe(m, d, KEY)
    d.qpos[7:] = Q + rng.uniform(-0.05, 0.05, len(Q))
    d.ctrl[:] = Q
    mujoco.mj_forward(m, d)


print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
print(f"Go2: nq {m.nq}, nv {m.nv}, nu {m.nu}, timestep {m.opt.timestep * 1000:.0f} ms, decimation 4 -> policy step "
      f"{4 * m.opt.timestep * 1000:.0f} ms")

print("\n== Step 1. One model, 8 MjData: the batch vs 8 independent loops (100 policy steps) ==")
env = VecEnv(m, 8, reset_fn, np.random.default_rng(0))
env.reset()
starts = [d.qpos.copy() for d in env.datas]
acts = Q + np.random.default_rng(1).uniform(-0.2, 0.2, (100, 8, m.nu))
for a in acts:
    env.step(a)
worst = 0.0
for i, q0 in enumerate(starts):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, KEY)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    for a in acts[:, i]:
        d.ctrl[:] = a
        mujoco.mj_step(m, d, nstep=4)
    worst = max(worst, np.abs(d.qpos - env.datas[i].qpos).max())
print(f"max |qpos batch - qpos alone| over 8 robots: {worst:.1e}")
print(f"episodes ended: {env.episodes} (base heights {', '.join(f'{d.qpos[2] * 100:.1f}' for d in env.datas)} cm)")

print("\n== Step 2. Projected gravity and roll / pitch from the base quaternion ==")
rng = np.random.default_rng(2)
d = mujoco.MjData(m)
g_err = rp_err = 0.0
for _ in range(200):
    d.qpos[3:7] = rng.normal(size=4)
    d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
    mujoco.mj_kinematics(m, d)
    R_mj = d.xmat[1].reshape(3, 3)                 # MuJoCo's own base orientation
    g_err = max(g_err, np.abs(projected_gravity(d.qpos[3:7]) - R_mj.T @ [0, 0, -1]).max())
    w, x, y, z = d.qpos[3:7]
    _, pitch_sp, roll_sp = Rotation.from_quat([x, y, z, w]).as_euler("ZYX")   # SciPy wants (x, y, z, w)
    rp_err = max(rp_err, np.abs(np.array(roll_pitch(d.qpos[3:7])) - [roll_sp, pitch_sp]).max())
print(f"projected gravity vs MuJoCo xmat^T g: max diff {g_err:.1e}")
print(f"roll, pitch vs SciPy as_euler('ZYX'): max diff {rp_err:.1e}")
reset_fn(m, d, np.random.default_rng(3))
print(f"standing start: projected gravity {np.round(projected_gravity(d.qpos[3:7]), 3)}, "
      f"base height {d.qpos[2] * 100:.1f} cm")

print("\n== Step 3. Auto-reset: 16 robots standing, a roll / pitch kick of up to 18 rad/s every 25 steps, "
      "episodes of at most 150 steps ==")
env = VecEnv(m, 16, reset_fn, np.random.default_rng(4), max_steps=150)
obs = env.reset()
push = np.random.default_rng(5)
falls, timeouts, first_fall, reset_heights = 0, 0, None, []
for k in range(300):
    if k % 25 == 0:                                # a push, as push_robots does in legged_gym (there: base velocity)
        for d in env.datas:
            d.qvel[3:5] += push.uniform(-18.0, 18.0, 2)
    obs, fell, timed_out = env.step(np.tile(Q, (16, 1)))
    falls += int(fell.sum())
    timeouts += int(timed_out.sum())
    if fell.any() and first_fall is None:
        first_fall = k + 1
    reset_heights += list(obs[fell | timed_out, 0])
print(f"300 policy steps ({300 * 4 * m.opt.timestep:.1f} s per robot): {falls} falls, {timeouts} time-outs, "
      f"episodes ended {env.episodes}")
print(f"first fall after {first_fall} policy steps; every ended row came back at base height "
      f"{min(reset_heights) * 100:.1f}-{max(reset_heights) * 100:.1f} cm (the keyframe height) in the same step")
print(f"steps since reset now, per robot: {env.t.tolist()}")

print("\n== Step 4. Accounting ==")
print(f"Step 3 counters: policy steps {env.policy_steps:,} = 16 x 300, physics steps {env.physics_steps:,} "
      f"= x4, simulated time {env.physics_steps * m.opt.timestep:.1f} robot-seconds")
n_env, T, iters, dec = 4096, 24, 1500, 4
print(f"Go2 training config: samples per update {samples_per_update(n_env, T):,}, "
      f"policy steps {samples_per_update(n_env, T) * iters:,}, physics steps {samples_per_update(n_env, T) * iters * dec:,}")
print(f"each robot advances {T} x 0.02 s = {T * 0.02:.2f} s per update")

print("\n== Use it. mujoco.rollout: the same batch in C, open loop ==")
env = VecEnv(m, 8, reset_fn, np.random.default_rng(0))
env.reset()
spec_ = mujoco.mjtState.mjSTATE_FULLPHYSICS
x0 = np.zeros((8, mujoco.mj_stateSize(m, spec_)))
for i, d in enumerate(env.datas):
    mujoco.mj_getState(m, d, x0[i], spec_)
U = np.repeat(acts.transpose(1, 0, 2), 4, axis=1)  # (8 robots, 400 physics steps, 12): each action held 4 steps
state, _ = rollout.rollout(m, [mujoco.MjData(m) for _ in range(4)], x0, U)
for a in acts:
    env.step(a)
q_ours = np.array([d.qpos for d in env.datas])
print(f"rollout (4 threads) vs VecEnv after 100 policy steps: max |qpos diff| = "
      f"{np.abs(state[:, -1, 1:1 + m.nq] - q_ours).max():.1e}")
