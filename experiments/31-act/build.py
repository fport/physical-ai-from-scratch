"""Lesson 31 · Build it: action chunks, the L1 optimum at a pause, and temporal ensembling, checked at every step.

Step 1  Chunk targets with end padding vs the Experiment's chunk_dataset.
Step 2  An idealized pause: per-entry median (L1) and mean (L2) of the demonstrated chunks.
Step 3  Temporal ensembling three ways: the list formula, a running average, the Experiment's accumulators.
Step 4  What ensembling does to a chunk that departs late (an idealized calculation, no simulation).

Steps 2 and 4 use hand-made chunks, not the trained policies: they isolate the arithmetic that the
Experiment's table depends on. The functions live in physai/imitation.py (the lesson shows them step by step).
Run:  uv run python experiments/31-act/build.py      (< 1 s, deterministic)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.imitation import TemporalEnsembler, chunk_targets, ensembled_action  # noqa: E402

print(f"NumPy {np.__version__}")
PAUSE, M = 25, 0.01                          # the Experiment's pause (control steps) and ensembling rate


def experiment_chunk_dataset(O, A, k):       # copied from experiments/31-act/experiment.py
    X, Y = [], []
    for o, a in zip(O, A):
        a_pad = np.vstack([a, np.repeat(a[-1:], k, 0)])
        for t in range(len(o)):
            X.append(o[t])
            Y.append(a_pad[t:t + k].reshape(-1))
    return np.array(X, np.float32), np.array(Y, np.float32)


print("\n== Step 1. Chunk targets ==")
rng = np.random.default_rng(0)
demos = [rng.normal(size=(150, 2)) for _ in range(5)]          # 5 episodes x 150 steps x 2 joints
for k in (1, 4, 32):
    mine = np.concatenate([chunk_targets(a, k) for a in demos])
    _, theirs = experiment_chunk_dataset(demos, demos, k)
    print(f"k = {k:>2}: targets {mine.shape}, max |ours - Experiment| = "
          f"{np.abs(mine.reshape(len(mine), -1).astype(np.float32) - theirs).max():.1e}")
c = chunk_targets(demos[0], 4)
print(f"last step's chunk repeats the last action: {bool(np.all(c[-1] == demos[0][-1]))}")

print("\n== Step 2. An idealized pause: 25 hold commands, then a move, all seen from the same resting observation ==")
# Joint target relative to the goal from the first pause step on: 25 holds, then 0.12 rad per step. The arm is still
# at rest when the first move is commanded, so 26 steps share the resting observation.
traj = np.r_[np.zeros(PAUSE), 0.12 * np.arange(1, 41)]
chunks = np.stack([chunk_targets(traj[:, None], 32)[j, :, 0] for j in range(PAUSE + 1)])   # one chunk per resting step
med, mean = np.median(chunks, 0), chunks.mean(0)
print("entry i          0      4      8     12     13     16     24     31")
print("moving %   " + " ".join(f"{100 * np.mean(chunks[:, i] > 0):6.0f}" for i in (0, 4, 8, 12, 13, 16, 24, 31)))
print("median     " + " ".join(f"{med[i]:6.3f}" for i in (0, 4, 8, 12, 13, 16, 24, 31)))
print("mean       " + " ".join(f"{mean[i]:6.3f}" for i in (0, 4, 8, 12, 13, 16, 24, 31)))
first = int(np.argmax(med > 0))
print(f"single-step policy (entry 0): median {med[0]:.3f} rad = hold forever, mean {mean[0]:.4f} rad = a creep")
print(f"first entry whose median departs: {first}; chunks of k <= {first} are all 'hold'")

print("\n== Step 3. Temporal ensembling, k = 32, m = 0.01, 60 steps of random chunks ==")
k, n = 32, 60
preds = rng.normal(size=(n, k, 2))                              # preds[s][j]: chunk queried at step s, entry j
ens = TemporalEnsembler(k, M)
num, den = np.zeros((n + k, 2)), np.zeros(n + k)
w_exp = np.exp(-M * (k - 1 - np.arange(k)))                     # the Experiment: offset j, older -> larger weight
err_run, err_exp = 0.0, 0.0
for t in range(n):
    a_list = ensembled_action(preds, t, k, M)
    err_run = max(err_run, np.abs(ens.update(preds[t]) - a_list).max())
    num[t:t + k] += w_exp[:, None] * preds[t]
    den[t:t + k] += w_exp
    err_exp = max(err_exp, np.abs(num[t] / den[t] - a_list).max())
print(f"max |running average - list formula| = {err_run:.1e}")
print(f"max |Experiment's accumulators - list formula| = {err_exp:.1e}")
w = np.exp(-M * np.arange(k))
print(f"full window: oldest weight {w[0] / w.sum():.4f}, newest {w[-1] / w.sum():.4f}, ratio {w[0] / w[-1]:.3f}")

print("\n== Step 4. A chunk that departs at entry i0, queried again and again at rest ==")
print("(i0 = the Experiment's median first departure index; same chunk at every query; 0.12 rad per step after i0)")
print("   k  i0  weight on departing entries  open-loop, entries i0 .. k-1  ensembled, at rest  [rad from the goal]")
for k, i0 in ((16, 9), (32, 8)):
    chunk = np.maximum(0, np.arange(k) - i0 + 1) * 0.12
    w = np.exp(-M * np.arange(k))[::-1]                          # entry j was predicted j steps ago: weight exp(-m (k-1-j))
    frac = w[i0:].sum() / w.sum()
    steady = ensembled_action(np.tile(chunk[:, None], (k, 1, 1)), k - 1, k, M)[0]
    print(f"{k:>4}  {i0:>2}  {frac:27.2f}  {chunk[i0]:12.2f} .. {chunk[-1]:.2f}  {steady:18.3f}")
