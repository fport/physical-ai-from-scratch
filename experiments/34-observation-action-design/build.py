"""Lesson 34 · Build it: the interface between policy and robot, checked at every step.

Step 1  Policy output -> actuator command in the three modes vs the lesson's ReachEnv.
Step 2  The servo's PD law by hand vs MuJoCo's actuator force; how small an error saturates it.
Step 3  Decimation: hold the command for k physics steps; torque rate and power vs ReachEnv's recorder.
Step 4  Policy rate: the discount that keeps a 0.4 s horizon, and the simulated time in a fixed budget.
Step 5  Action scale: what the scripted IK controller reaches through the clipped interface.

The functions live in physai/rl.py (the lesson shows them step by step).
Run:  uv run python experiments/34-observation-action-design/build.py      (~3 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "32-rl-environment"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.rl import action_to_ctrl, discount_for_rate, pd_torque, torque_rate_and_power  # noqa: E402
from reach_env import Q_HOME, TAU_MAX, ReachEnv  # noqa: E402

spec = importlib.util.spec_from_file_location("rl32_experiment", os.path.join(HERE, "..", "32-rl-environment", "experiment.py"))
exp32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp32)
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. Policy output -> actuator command, 3 episodes of random actions per mode ==")
rng = np.random.default_rng(0)
for mode in ("position", "delta", "torque"):
    env = ReachEnv(action_mode=mode)
    worst = 0.0
    for s in range(3):
        env.reset(seed=s)
        for _ in range(env.max_steps):
            a = rng.uniform(-1.2, 1.2, 4)                        # a little outside [-1, 1] to exercise the clip
            want = action_to_ctrl(a, mode, Q_HOME, env.scale, env.data.ctrl[env.act].copy(), env.lo, env.hi)
            env.step(a)
            worst = max(worst, np.abs(env.data.ctrl[env.act] - want).max())
    print(f"{mode:>8} (scale {env.scale}): max |ours - ReachEnv's command| = {worst:.1e}")

print("\n== Step 2. The servo's PD law vs MuJoCo, 1000 random states ==")
env = ReachEnv()
m, d = env.model, env.data
i = env.act[0]
kp, kv, tmax = m.actuator_gainprm[i, 0], -m.actuator_biasprm[i, 2], m.actuator_forcerange[i, 1]
print(f"STS3215 in the Menagerie model: kp = {kp:.2f} N m/rad, kv = {kv:.3f} N m s/rad, force range +-{tmax:.2f} N m")
worst, n_sat = 0.0, 0
for _ in range(1000):
    d.qpos[env.qadr] = Q_HOME + rng.uniform(-0.5, 0.5, 4)
    d.qvel[env.vadr] = rng.normal(0, 1.0, 4)
    d.ctrl[env.act] = d.qpos[env.qadr] + rng.normal(0, 0.003, 4)
    mujoco.mj_forward(m, d)
    tau = pd_torque(d.ctrl[env.act], d.qpos[env.qadr], d.qvel[env.vadr], kp, kv, tmax)
    worst = max(worst, np.abs(tau - d.actuator_force[env.act]).max())
    n_sat += int(np.sum(np.abs(tau) >= tmax))
print(f"max |ours - data.actuator_force| = {worst:.1e} N m; {n_sat} of 4000 torques at the limit")
print(f"a position error of {tmax / kp * 1000:.2f} mrad alone saturates the servo (2.94 / 998.22)")

print("\n== Step 3. Decimation and the physics-rate recorder: one episode per mode (position: scripted IK; torque: a = 0.1) ==")
for mode in ("position", "torque"):
    ref = ReachEnv(action_mode=mode, record=True)
    own = ReachEnv(action_mode=mode)
    ref.reset(seed=10_000)
    own.reset(seed=10_000)
    taus, qds = [own.data.actuator_force[own.act].copy()], []
    for _ in range(ref.max_steps):
        a = exp32.scripted_ik(ref, None) if mode == "position" else np.full(4, 0.1)
        ref.step(a)
        own.data.ctrl[own.act] = action_to_ctrl(a, mode, Q_HOME, own.scale, own.data.ctrl[own.act], own.lo, own.hi, TAU_MAX)
        for _ in range(own.frame_skip):                             # the command is held for k physics steps
            mujoco.mj_step(own.model, own.data)
            taus.append(own.data.actuator_force[own.act].copy())
            qds.append(own.data.qvel[own.vadr].copy())
    rate, power = torque_rate_and_power(np.array(taus), np.array(qds), own.model.opt.timestep)
    st = ref.stats
    print(f"{mode:>8}: {len(qds)} physics steps; torque rate {rate:8.1f} N m/s (ReachEnv {np.sqrt(st['tau_rate_sq'] / st['n']):8.1f}), "
          f"power {power:6.3f} W (ReachEnv {st['power'] / st['n']:6.3f})")

print("\n== Step 4. Policy rate: gamma = 1 - dt / 0.4 s, and 300,000 policy steps in simulated time ==")
print("frame_skip  rate Hz   gamma   simulated time")
for k in (1, 2, 4, 8, 16):
    dt = 0.005 * k
    print(f"{k:>10}  {1 / dt:7.1f}  {discount_for_rate(dt):.4f}  {300_000 * dt / 60:7.0f} min")

print("\n== Step 5. Action scale: scripted IK through the same clipped interface, 100 episodes (seeds 10000-10099) ==")
for scale in (1.0, 0.25):
    env = ReachEnv(action_scale=scale)
    k = 0
    for e in range(100):
        env.reset(seed=10_000 + e)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(exp32.scripted_ik(env, None))
            done = term or trunc
        k += int(info["is_success"])
    print(f"scale {scale:.2f} rad: reaches {k}/100")
inside = np.prod(np.minimum(0.25 / np.array([1.0, 0.6, 0.6, 0.6]), 1.0))
print(f"joint vectors drawn by the target sampler that lie within q0 +- 0.25 rad on all 4 joints: {inside * 100:.1f} %")
