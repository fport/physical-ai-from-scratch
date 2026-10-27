"""Lesson 27 · Build it: an episode dataset from scratch on the lesson-26 demonstrations.

Step 1  Episode boundaries and the global row index.
Step 2  Normalization statistics, and merging per-episode statistics.
Step 3  Frame split vs episode split: what each puts on both sides.
Step 4  MEAN_STD and MIN_MAX with training statistics; the cube in m or in mm.
Step 5  A nearest-neighbour probe of the split leak, on the Experiment's splits.

The functions live in physai/data.py (the lesson shows them step by step).
Run:  uv run python experiments/27-robot-datasets/build.py      (~1 s, deterministic)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.data import (episode_starts, feature_stats, global_index, merge_stats, nearest_neighbour,  # noqa: E402
                         normalize, split_episodes, split_frames, unnormalize)

print(f"NumPy {np.__version__}")
D = np.load(os.path.join(HERE, "..", "26-teleop-replay-data", "demos.npz"))
O, A = D["obs"].astype(np.float64), D["action"].astype(np.float64)
EP, FR = D["episode_index"].astype(int), D["frame_index"].astype(int)

print("\n== Step 1. Episode boundaries and the global index ==")
starts = episode_starts(EP)
lengths = np.diff(np.r_[starts, len(EP)])
idx = global_index(EP, FR)
print(f"episodes {len(starts)} | starts {starts[:4].tolist()} ... {starts[-1]} | lengths {lengths.min()}-{lengths.max()}")
print(f"index == row number for all {len(idx)} frames: {np.array_equal(idx, np.arange(len(EP)))} | "
      f"episode 2, frame 5 -> index {idx[(EP == 2) & (FR == 5)][0]}")

print("\n== Step 2. Statistics (all frames) and merged per-episode statistics ==")
S = feature_stats(O)
names = list(D["obs_names"])
print(f"{'feature':>14}" + "".join(f"{k:>9}" for k in ("mean", "std", "min", "max", "q01", "q99")))
for j in (names.index("shoulder_lift"), names.index("gripper"), names.index("cube_x")):
    print(f"{names[j]:>14}" + "".join(f"{S[k][j]:>9.3f}" for k in ("mean", "std", "min", "max", "q01", "q99")))
merged = feature_stats(O[EP == 0])
for e in range(1, len(starts)):
    merged = merge_stats(merged, feature_stats(O[EP == e]))
print(f"120 per-episode stats merged vs computed on all frames: max |mean diff| {np.abs(merged['mean'] - S['mean']).max():.1e}, "
      f"max |std diff| {np.abs(merged['std'] - S['std']).max():.1e}, min/max equal: "
      f"{np.array_equal(merged['min'], S['min']) and np.array_equal(merged['max'], S['max'])}")

print("\n== Step 3. 10 episodes, 80/20: frame split vs episode split ==")
rng = np.random.default_rng(27)                 # the Experiment's generator and call order
eps = rng.permutation(100)[:10]
rows = np.flatnonzero(np.isin(EP, eps))
f_tr, f_va = split_frames(rows, 0.2, rng)
e_tr, e_va = split_episodes(EP, eps[:2])
e_tr &= np.isin(EP, eps)
both = np.intersect1d(EP[f_tr], EP[f_va])
gap = np.isin(f_va - 1, f_tr) | np.isin(f_va + 1, f_tr)
print(f"frame split:   {len(f_tr)} / {len(f_va)} frames, episodes on both sides {len(both)}/10, "
      f"validation frames with a training neighbour 40 ms away {gap.mean():.1%}")
print(f"episode split: {int(e_tr.sum())} / {int(e_va.sum())} frames, episodes on both sides "
      f"{len(np.intersect1d(EP[e_tr], EP[e_va]))}/10")

print("\n== Step 4. Normalize with training statistics (episodes 0-79) ==")
tr = EP < 80
mm = O.copy()
mm[:, 6:9] *= 1000.0                            # the same data with the cube logged in mm
for mode in ("mean_std", "min_max"):
    s_m, s_mm = feature_stats(O[tr]), feature_stats(mm[tr])
    z = normalize(O, s_m, mode)
    print(f"{mode:>8}: train range [{z[tr].min():.2f}, {z[tr].max():.2f}] | round trip error "
          f"{np.abs(unnormalize(z, s_m, mode) - O).max():.1e} | m vs mm after normalizing: "
          f"{np.abs(z - normalize(mm, s_mm, mode)).max():.1e}")

print("\n== Step 5. 1-nearest-neighbour probe, MEAN_STD, the Experiment's splits; test = episodes 100-119 ==")
te = np.isin(EP, np.arange(100, 120))


def rmse_deg(pred, y):
    return float(np.degrees(np.sqrt(np.mean((pred - y) ** 2))))


print(f"{'train eps':>10}{'split':>9}{'val RMSE deg':>14}{'test RMSE deg':>15}{'test / val':>12}")
rng = np.random.default_rng(27)
for n in (10, 30, 90):
    eps = rng.permutation(100)[:n]
    rows = np.flatnonzero(np.isin(EP, eps))
    f_tr, f_va = split_frames(rows, 0.2, rng)
    e_tr, e_va = split_episodes(EP, eps[:max(2, n // 5)])
    e_tr &= np.isin(EP, eps)
    for split, (a, b) in (("frame", (f_tr, f_va)), ("episode", (e_tr, e_va))):
        s = feature_stats(O[a])
        v = rmse_deg(nearest_neighbour(normalize(O[a], s), A[a], normalize(O[b], s)), A[b])
        t = rmse_deg(nearest_neighbour(normalize(O[a], s), A[a], normalize(O[te], s)), A[te])
        print(f"{n:>10}{split:>9}{v:>14.3f}{t:>15.3f}{t / v:>12.2f}")
