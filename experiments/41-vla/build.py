"""Lesson 41 · Build it: the two VLA action heads in NumPy, checked on lesson 16's scripted SO-101 pick.

Step 1  Tokens: tokenize / de-tokenize per dimension; the error never exceeds half a bin.
Step 2  The RMS error against Delta / sqrt(12), bin count by bin count.
Step 3  What the error means at the gripper: forward kinematics of the de-tokenized actions (lesson 09's fk).
Step 4  The cost side: network passes per second for tokens and for chunks, from the configs' numbers.
Step 5  The flow head: Euler integration of the exact flow-matching velocity of the demonstration's chunks.

The functions live in physai/foundation.py. Run:  uv run python experiments/41-vla/build.py      (~2 s)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.foundation import detokenize, flow_velocity, sample_chunk, tokenize  # noqa: E402
from physai.kinematics import chain_from_mujoco, fk  # noqa: E402

spec = importlib.util.spec_from_file_location("exp41", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)                       # lesson 16's scripted pick, as the Experiment records it

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
_, A = exp.demo()                                  # 229 actions x 6 joint targets at 50 Hz
lo, hi = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0)
BINS = (4, 8, 16, 64, 256, 1024)

print("\n== Step 1. Tokens per dimension, bins between the 1st and 99th percentile ==")
print(f"demonstration: {A.shape[0]} actions x {A.shape[1]} dims; arm-joint ranges (deg): "
      + " ".join(f"{v:.1f}" for v in np.degrees(hi - lo)[:5]))
print(f"{'bins':>5s} {'same tokens as experiment.py':>29s} {'re-tokenized = same':>20s} {'max |err| / (Delta/2)':>22s}")
for n in BINS:
    k = tokenize(A, lo, hi, n)
    a_hat = detokenize(k, lo, hi, n)
    ratio = (np.abs(a_hat - A) / ((hi - lo) / n / 2)).max()
    print(f"{n:5d} {str(np.array_equal(k, exp.tokenize(A, lo, hi, n))):>29s} "
          f"{str(np.array_equal(tokenize(a_hat, lo, hi, n), k)):>20s} {ratio:22.3f}")

print("\n== Step 2. RMS error of the arm joints vs Delta / sqrt(12) ==")
print(f"{'bins':>5s} {'RMS err deg':>12s} {'Delta/sqrt(12) deg':>19s} {'max err deg':>12s} {'Delta/2 (widest) deg':>21s}")
for n in BINS:
    err = np.degrees(detokenize(tokenize(A, lo, hi, n), lo, hi, n) - A)[:, :5]
    delta = np.degrees(hi - lo)[:5] / n
    print(f"{n:5d} {np.sqrt(np.mean(err ** 2)):12.3f} {np.sqrt(np.mean(delta ** 2 / 12)):19.3f} "
          f"{np.abs(err).max():12.3f} {delta.max() / 2:21.3f}")

print("\n== Step 3. At the gripper: forward kinematics of the de-tokenized actions ==")
m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
chain = chain_from_mujoco(m, "gripperframe")
tips = np.array([fk(chain, a[:5])[2] for a in A])
print(f"{'bins':>5s} {'mean tip err mm':>16s} {'max tip err mm':>15s}")
for n in BINS:
    a_hat = detokenize(tokenize(A, lo, hi, n), lo, hi, n)
    e = np.linalg.norm(np.array([fk(chain, a[:5])[2] for a in a_hat]) - tips, axis=1) * 1000
    print(f"{n:5d} {e.mean():16.2f} {e.max():15.2f}")

print("\n== Step 4. Network passes per second at a 30 Hz control loop ==")
rate, dims = 30, 6
print(f"tokens, one action per call, {dims} dims: {rate} x {dims} = {rate * dims} decoder passes/s")
print(f"tokens for the 7-D OpenVLA action: {rate} x 7 = {rate * 7} decoder passes/s")
for chunk, n_exec, steps in ((50, 50, 10), (50, 10, 10)):
    calls = rate / n_exec
    print(f"chunk of {chunk}, execute {n_exec} before re-planning, {steps} flow steps: {calls:.1f} VLM passes/s, "
          f"{calls * steps:.0f} expert passes/s, {n_exec / rate * 1000:.0f} ms without reacting to new images")

print("\n== Step 5. Flow head: Euler steps through the exact velocity of the demonstration's chunks ==")
H = 50
data = np.stack([A[i:i + H].ravel() for i in range(len(A) - H + 1)])   # every 50-step window: 180 chunks of 300
rng = np.random.default_rng(0)
noise = rng.standard_normal((20, data.shape[1]))
print(f"training set: {data.shape[0]} chunks of {H} x 6 = {data.shape[1]} numbers; 20 samples from noise")
print(f"{'flow steps':>10s} {'RMS to nearest chunk rad':>25s} {'within 0.01 rad':>16s} {'RMS to the mean chunk rad':>26s}")
for n in (1, 2, 5, 10, 20):
    out = np.array([sample_chunk(lambda x, t: flow_velocity(x, t, data), z, n) for z in noise])
    near = np.sqrt(((out[:, None] - data[None]) ** 2).mean(axis=2)).min(axis=1)
    to_mean = np.sqrt(((out - data.mean(axis=0)) ** 2).mean(axis=1))
    print(f"{n:10d} {near.mean():25.4f} {int((near < 0.01).sum()):>13d}/20 {to_mean.mean():26.4f}")
