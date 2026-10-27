"""Lesson 29 · Distribution shift and DAgger.

Plain behavior cloning (lesson 28's MLP and features, but WITHOUT the input-noise augmentation) trained on 20 scripted
demonstrations, then DAgger (Ross, Gordon, Bagnell 2011): run the learner, label every state it visits with the
scripted expert's action, aggregate, retrain.

The labeler: lesson 16's scripted pick, re-planned for the cube's current position and recorded at 25 Hz
(lesson 26's record_demo); the label for learner state o at policy step k is the expert's action at step k.

  A. Distribution shift: distance of each visited arm state (5 joints + jaw) to the nearest frame of the 20
     demonstrations (max joint difference, deg), in five time windows, for plain BC, BC with input noise (lesson 28)
     and the final DAgger policy, plus the deviation from the expert's own trajectory for the same cube.
  B. DAgger iterations 0..4 (10 learner rollouts per iteration, beta_1 = 1 then 0) vs plain BC trained on the
     same number of expert-labelled episodes, 3 seeds, 20 test positions of lesson 28.
  C. Break it: DAgger without aggregation (train only on the newest learner data), seed 0.

Run:  python experiment.py      (deterministic; ~4.5 min CPU on Apple M5, longer wall time on a loaded machine)
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)


def load_lesson(name, folder):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, folder, "experiment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L28 = load_lesson("lesson28", "28-behavior-cloning")
L25, L26 = L28.L25, L28.L26
wilson, Scene, pick16 = L25.wilson, L25.Scene, L25.pick16
features, rollout, train_policy, as_policy, TEST_XY = L28.features, L28.rollout, L28.train_policy, L28.as_policy, L28.TEST_XY
N_DEMOS, ITERS, ROLLOUTS = 20, 4, 10


class ScriptedLabeler:
    """Expert action for any learner state: the lesson-16 scripted pick for the cube where it is now
    (re-planned when the learner has pushed the cube > 1 cm along the floor; plans are cached on a 2 mm grid),
    indexed by the policy step."""

    def __init__(self, scratch):
        self.scratch, self.cube, self.ref_obs, self.ref_act = scratch, None, None, None
        self.cache = {}                      # expert runs are deterministic: reuse them per cube position
        self.runs = 0

    def __call__(self, o, k):
        c = o[6:9]
        if k == 0:
            self.cube = None                 # new episode
        moved = self.cube is None or np.linalg.norm(self.cube - c[:2]) > 0.01
        if moved and c[2] < pick16.CUBE_HALF + 0.003:
            key = (round(float(c[0]) / 0.002) * 0.002, round(float(c[1]) / 0.002) * 0.002)
            if key not in self.cache:
                ro, ra, _, _ = L26.record_demo(self.scratch, np.array(key))
                self.cache[key] = (ro, ra)
                self.runs += 1
            self.ref_obs, self.ref_act = self.cache[key]
            self.cube = c[:2].copy()
        i = min(k, len(self.ref_act) - 1)
        return self.ref_act[i], self.ref_obs[i]


DEMO_Q = None      # (frames, 6) arm joints + jaw of the 20 demonstrations, set in main()


def run_episode(scene, labeler, pol, xy, label=False):
    """Learner in control. Returns (outcome, labelled frames X, labels Y, per-step [nn distance, expert deviation] deg)."""
    X, Y, drift = [], [], []

    def policy(o, t):
        k = int(round(t * L28.FPS))
        a_exp, o_exp = labeler(o, k)
        nn = np.abs(DEMO_Q - o[:6]).max(axis=1).min()
        drift.append((np.degrees(nn), np.degrees(np.abs(o[:6] - o_exp[:6]).max())))
        if label:
            X.append(features(o, t, pol.clock)[0])
            Y.append(a_exp)
        return pol(features(o, t, pol.clock))[0]

    out = rollout(scene, policy, xy)
    return out, X, Y, np.array(drift)


def evaluate(scene, labeler, pol):
    res = [run_episode(scene, labeler, pol, xy) for xy in TEST_XY]
    return sum(r[0] == "held" for r in res), np.array([r[3] for r in res])


def dagger(scene, labeler, O, A, TS, seed, aggregate=True):
    """Returns [(expert episodes, success k, drift)] for iterations 0..ITERS."""
    rng = np.random.default_rng(290 + seed)
    tr = L26_EP < N_DEMOS
    X, Y = features(O[tr], TS[tr]), A[tr]
    pol = L28.train_mlp(X, Y, seed=seed)
    pol.clock = True
    k, dr = evaluate(scene, labeler, pol)
    hist = [(N_DEMOS, k, dr)]
    for it in range(1, ITERS + 1):
        nx, ny = [], []
        for xy in L26.sample_positions(ROLLOUTS, seed=1000 * seed + it):
            _, x, y, _ = run_episode(scene, labeler, pol, xy, label=True)
            nx += x
            ny += y
        nx, ny = np.array(nx), np.array(ny)
        if aggregate:
            X, Y = np.concatenate([X, nx]), np.concatenate([Y, ny])
        else:
            X, Y = nx, ny
        pol = L28.train_mlp(X, Y, seed=seed)
        pol.clock = True
        k, dr = evaluate(scene, labeler, pol)
        hist.append((N_DEMOS + ROLLOUTS * it, k, dr))
    return hist, pol


def windows(drift):
    """Mean distance to the demonstrations [deg] over all test episodes in five time windows of the 6 s horizon."""
    edges = [(0, 25), (25, 50), (50, 75), (75, 113), (113, 150)]
    return [drift[:, a:b, 0].mean() for a, b in edges]


L26_EP = None


def main():
    global L26_EP, DEMO_Q
    t0 = time.process_time()
    import mujoco
    import torch
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · torch {torch.__version__}")
    D = L28.load_demos()
    O, A, EP, TS = D["obs"], D["action"], D["episode_index"], D["timestamp"]
    L26_EP = EP
    DEMO_Q = O[EP < N_DEMOS, :6].astype(np.float64)
    scene, labeler = Scene(), ScriptedLabeler(Scene())
    print(f"plain BC = lesson-28 MLP and features without input noise; {N_DEMOS} demonstrations; DAgger: "
          f"{ROLLOUTS} learner rollouts per iteration labelled by the scripted expert; {len(TEST_XY)} test positions")

    print("\n[B] DAgger iterations vs plain BC with the same number of expert-labelled episodes")
    print("(dist = mean distance of visited arm states to the nearest demonstration frame, deg)")
    print(f"{'iter':>5}{'episodes':>10}{'DAgger s0/s1/s2':>17}{'pooled':>8}{'Wilson 95%':>15}{'dist deg':>10}"
          f"{'BC s0/s1/s2':>14}{'pooled':>8}{'Wilson 95%':>15}{'dist deg':>10}")
    hists, finals = [], []
    for seed in (0, 1, 2):
        h, pol = dagger(scene, labeler, O, A, TS, seed)
        hists.append(h)
        finals.append(pol)
        print(f"  [seed {seed} done, {time.process_time() - t0:.0f} s CPU]", file=sys.stderr, flush=True)
    bc_res = {}
    for it in range(ITERS + 1):
        n = N_DEMOS + ROLLOUTS * it
        if it == 0:
            bc_res[it] = [(h[0][1], h[0][2]) for h in hists]
        elif it % 2 == 1:
            bc_res[it] = None                # BC baseline only at iterations 2 and 4 (CPU budget)
        else:
            row = []
            for seed in (0, 1, 2):
                tr = EP < n
                pol = L28.train_mlp(features(O[tr], TS[tr]), A[tr], seed=seed)
                pol.clock = True
                row.append(evaluate(scene, labeler, pol))
            bc_res[it] = row
        dk = [h[it][1] for h in hists]
        dd = np.mean([h[it][2][..., 0].mean() for h in hists])
        lo, hi = wilson(sum(dk), 60)
        line = f"{it:>5}{n:>10}{'/'.join(map(str, dk)):>17}{f'{sum(dk)}/60':>8}{f'[{lo:.2f}, {hi:.2f}]':>15}{dd:>10.2f}"
        if bc_res[it] is not None:
            bk = [r[0] for r in bc_res[it]]
            bd = np.mean([r[1][..., 0].mean() for r in bc_res[it]])
            lo2, hi2 = wilson(sum(bk), 60)
            line += f"{'/'.join(map(str, bk)):>14}{f'{sum(bk)}/60':>8}{f'[{lo2:.2f}, {hi2:.2f}]':>15}{bd:>10.2f}"
        print(line, flush=True)
        print(f"  [{time.process_time() - t0:.0f} s CPU]", file=sys.stderr, flush=True)

    print("\n[A] Distribution shift: mean distance of visited arm states to the nearest demonstration frame [deg],")
    print("    seed 0, 20 test positions; last column: mean max deviation from the expert's trajectory for that cube [deg]")
    print(f"{'policy':>40}{'0-1 s':>8}{'1-2 s':>8}{'2-3 s':>8}{'3-4.5 s':>9}{'4.5-6 s':>9}{'success':>9}{'vs expert':>11}")
    tr = EP < N_DEMOS
    plain = L28.train_mlp(features(O[tr], TS[tr]), A[tr], seed=0)
    plain.clock = True
    noisy = train_policy(O[tr], A[tr], TS[tr], seed=0)        # lesson 28: input noise 0.02 rad
    for name, pol in (("plain BC, 20 demos", plain), ("BC + input noise (lesson 28), 20 demos", noisy),
                      (f"DAgger after {ITERS} iterations", finals[0])):
        k, dr = evaluate(scene, labeler, pol)
        print(f"{name:>40}" + "".join(f"{w:>8.2f}" if j < 3 else f"{w:>9.2f}" for j, w in enumerate(windows(dr)))
              + f"{f'{k}/20':>9}{dr[..., 1].mean():>11.1f}")

    print("\n[C] Break it: DAgger without aggregation (each iteration trains only on its newest 10 learner rollouts), seed 0", flush=True)
    h, _ = dagger(scene, labeler, O, A, TS, 0, aggregate=False)
    print("  iteration:        " + "".join(f"{i:>8}" for i in range(ITERS + 1)))
    print("  success /20:      " + "".join(f"{x[1]:>8}" for x in h))
    print("  with aggregation: " + "".join(f"{x[1]:>8}" for x in hists[0]))
    print(f"\nscripted-expert runs used for labelling and drift reference: {labeler.runs}")
    print(f"CPU time {time.process_time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
