"""Lesson 32 · Build it: the SO-101 reach environment from scratch, checked at every step.

Step 1  Timing: physics step, frame skip, policy period; k single steps vs one mj_step(nstep=k).
Step 2  reset(seed): targets and start poses vs the lesson's ReachEnv for 200 seeds.
Step 3  step(action): observations and rewards vs ReachEnv; the zero-action baseline vs the Experiment table.
Step 4  terminated vs truncated, and the value target that treats them differently.
Step 5  Gymnasium's own checker on our environment.

The functions live in physai/rl.py (the lesson shows them step by step).
Run:  uv run python experiments/32-rl-environment/build.py      (~4 s, deterministic)
"""
import importlib.util
import os
import re
import sys
import warnings

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.rl import ReachTask, value_target  # noqa: E402
from reach_env import ReachEnv  # noqa: E402

spec = importlib.util.spec_from_file_location("rl32_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Gymnasium {gym.__version__}")
SO101 = path("robotstudio_so101")

print("\n== Step 1. Timing ==")
env = ReachTask(SO101)
print(f"physics step h = {env.model.opt.timestep * 1000:.0f} ms, frame_skip {env.frame_skip} -> policy period "
      f"{env.dt * 1000:.0f} ms ({1 / env.dt:.0f} Hz), {env.max_steps} steps = {env.max_steps * env.dt:.1f} s")
env.reset(seed=0)
d1, d2 = mujoco.MjData(env.model), mujoco.MjData(env.model)
for d in (d1, d2):
    d.qpos[:], d.ctrl[:] = env.data.qpos, env.data.ctrl
    d.ctrl[env.act] += 0.3
for _ in range(4):
    mujoco.mj_step(env.model, d1)
mujoco.mj_step(env.model, d2, nstep=4)
print(f"4 x mj_step vs mj_step(nstep=4): max |qpos diff| = {np.abs(d1.qpos - d2.qpos).max():.1e}, "
      f"time {d1.time:.3f} s vs {d2.time:.3f} s")

print("\n== Step 2. reset(seed) vs the lesson's ReachEnv, seeds 0-199 ==")
ref = ReachEnv()
dt_max, do_max = 0.0, 0.0
for s in range(200):
    o1, i1 = env.reset(seed=s)
    o2, i2 = ref.reset(seed=s)
    dt_max = max(dt_max, np.abs(env.target - ref.target).max())
    do_max = max(do_max, np.abs(o1 - o2).max())
print(f"max |target ours - ReachEnv| = {dt_max:.1e} m, max |first observation diff| = {do_max:.1e}")
t7 = env.reset(seed=7)[1]["dist"], env.target.copy()
env.reset(seed=8)
env.reset(seed=7)
print(f"reset(seed=7), reset(seed=8), reset(seed=7): same target {bool(np.all(env.target == t7[1]))}, "
      f"target {np.round(t7[1], 3)} m, start distance {t7[0]:.3f} m")
env.reset()
print(f"reset() without a seed continues the generator: new target {np.round(env.target, 3)} m")

print("\n== Step 3. step(action) vs ReachEnv, and the zero-action baseline ==")
rng = np.random.default_rng(0)
worst = 0.0
for s in range(5):
    env.reset(seed=s)
    ref.reset(seed=s)
    for _ in range(80):
        a = rng.uniform(-1, 1, 4)
        o1, r1, *_ = env.step(a)
        o2, r2, *_ = ref.step(a)
        worst = max(worst, np.abs(o1 - o2).max(), abs(r1 - r2))
print(f"5 episodes of random actions: max |obs or reward diff| = {worst:.1e}")
rets, final, succ = [], [], 0
for s in range(200):
    env.reset(seed=s)
    R, done = 0.0, False
    while not done:
        _, r, term, trunc, info = env.step(np.zeros(4))
        R, done = R + r, term or trunc
    rets.append(R)
    final.append(info["dist"])
    succ += info["is_success"]
print(f"zero action, 200 episodes: return {np.mean(rets):.2f} +- {np.std(rets):.2f}, median final distance "
      f"{np.median(final) * 1000:.1f} mm, success {succ}/200")

print("\n== Step 4. terminated vs truncated ==")
ends = {"terminated": 0, "truncated": 0}
steps = []
for flag in (True, False):
    e = ReachTask(SO101, terminate_on_success=flag)
    for s in range(50):
        e.reset(seed=s)
        done = False
        while not done:
            _, r, term, trunc, info = e.step(exp.scripted_ik(e, None))
            done = term or trunc
        if flag:
            ends["terminated" if term else "truncated"] += 1
            steps.append(e.t)
print(f"scripted IK, terminate_on_success=True, 50 episodes: terminated {ends['terminated']}, "
      f"truncated {ends['truncated']}, mean length {np.mean(steps):.1f} steps")
print(f"last episode with terminate_on_success=False: terminated {term}, truncated {trunc}, t = {e.t}")
g, r, V = 0.95, -0.01, -0.01 / (1 - 0.95)                       # a state that pays -1 cm per step forever: V = -0.2
print(f"V(s) = r / (1 - gamma) = {V:.3f}; target at the time limit: bootstrap {value_target(r, V, 0.0, g):.3f}, "
      f"treated as terminal {value_target(r, V, 1.0, g):.3f}")

print("\n== Step 5. Gymnasium's env_checker ==")


class GymReach(ReachTask, gym.Env):
    """Our class plus the two spaces Gymnasium asks for; reset/step are physai's."""

    def __init__(self, **kw):
        super().__init__(SO101, **kw)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    check_env(GymReach(), skip_render_check=True)
print(f"check_env passed, {len(caught)} warnings")
for w in caught:
    print("  warning:", re.sub(r"\x1b\[\d+m", "", str(w.message)))
