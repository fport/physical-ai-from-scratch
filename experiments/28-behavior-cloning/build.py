"""Lesson 28 · Build it: behavior cloning from scratch, checked at every step.

Step 1  MEAN_STD statistics of the lesson's features vs lesson 27's train_mlp.
Step 2  The MLP forward pass in NumPy vs torch.nn.Sequential.
Step 3  The training loop vs lesson 27's train_mlp (bit for bit), then one full run: open-loop RMSE.
Step 4  Closed loop: the cloned policy drives the SO-101 on the 20 test positions.

One training seed (0) and 20 demonstrations: a smaller run than the Experiment's 3 seeds x 5 dataset sizes.
The functions live in physai/imitation.py (the lesson shows them step by step).
Run:  uv run python experiments/28-behavior-cloning/build.py      (~6 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.imitation import bc_policy, fit_mean_std, mlp_forward, train_bc  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
spec = importlib.util.spec_from_file_location("bc28_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | PyTorch {torch.__version__}")

D = exp.load_demos()
O, A, EP, TS = D["obs"], D["action"], D["episode_index"], D["timestamp"]
tr, val = EP < 20, EP >= 100
X, Y = exp.features(O[tr], TS[tr]), A[tr]

print("\n== Step 1. Normalization statistics, 20 demonstrations ==")
xs, xk = fit_mean_std(X)
ref = exp.train_mlp(X, Y, steps=1, seed=0)                  # lesson 27's MLPPolicy keeps its statistics
print(f"{len(X)} frames, {X.shape[1]} inputs (5 joints, jaw, cube xyz, clock), {Y.shape[1]} actions")
print("mean  " + " ".join(f"{v:7.3f}" for v in xs))
print("std   " + " ".join(f"{v:7.3f}" for v in xk))
print(f"max |ours - lesson 27| = {max(np.abs(xs - ref.xs).max(), np.abs(xk - ref.xk).max()):.1e}")

print("\n== Step 2. Forward pass: NumPy vs torch.nn.Sequential (float64, random weights) ==")
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6)).double()
layers = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in net if isinstance(m, nn.Linear)]
x = np.random.default_rng(0).normal(size=(500, 10))
with torch.no_grad():
    y_torch = net(torch.tensor(x)).numpy()
print(f"500 inputs: max |mlp_forward - torch| = {np.abs(mlp_forward(layers, x) - y_torch).max():.1e}")

print("\n== Step 3. Training: ours vs lesson 27's train_mlp, then the full run ==")
noise = exp.noise_vector()                                   # 0.02 rad on joints and jaw, 1 mm on the cube, 0 on t
mine, _, _ = train_bc(X, Y, steps=200, seed=0, obs_noise=noise)
theirs = exp.train_mlp(X, Y, steps=200, seed=0, obs_noise=noise)
diff = max(np.abs(w1 - w2).max() for (w1, _), (w2, _) in zip(mine, theirs.layers))
print(f"200 steps, same seed: max |weights ours - lesson 27| = {diff:.1e}")
layers, x_stats, y_stats = train_bc(X, Y, steps=4000, seed=0, obs_noise=noise)
pi = bc_policy(layers, x_stats, y_stats)
pred = pi(exp.features(O[val], TS[val]))
rmse = np.degrees(np.sqrt(np.mean((pred - A[val]) ** 2)))
base = np.degrees(np.sqrt(np.mean((A[val] - A[tr].mean(0)) ** 2)))
print(f"4000 steps, seed 0: open-loop RMSE on episodes 100-119 = {rmse:.3f} deg "
      f"(predicting the mean action: {base:.3f} deg)")

print("\n== Step 4. Closed loop: 20 test positions (seed 28), 6 s each ==")
scene = exp.Scene()
outs = [exp.rollout(scene, lambda o, t: pi(exp.features(o, t))[0], xy) for xy in exp.TEST_XY]
k = sum(o == "held" for o in outs)
lo, hi = exp.wilson(k, len(outs))
print(f"success {k}/{len(outs)}, Wilson 95% [{lo:.2f}, {hi:.2f}]; outcomes: "
      + ", ".join(f"{o} {outs.count(o)}" for o in sorted(set(outs))))
