"""Lesson 28 · Behavior cloning.

An MLP policy maps the SO-101 state (5 arm joints, jaw, cube position) plus the episode clock to the next
6 joint targets, trained with MSE on the lesson-26 scripted demonstrations (25 Hz). It is evaluated closed loop
in MuJoCo: every 40 ms the policy reads the state and its output is held for 20 physics steps (2 ms each).
Success is physical (lesson 16/25): cube bottom >= 4 cm above the floor and both jaws touching it after 6 s.

  A. Sweep the number of demonstrations (5 / 10 / 20 / 50 / 100) x 3 training seeds, 20 fixed test positions:
     closed-loop success with Wilson 95% intervals, and the open-loop action error on unseen episodes.
  B. Break it: training-time input noise (0 ... 0.04 rad), and the same policy without the clock.

`features`, `rollout`, `train_policy` and `TEST_XY` are imported by lesson 29.

Run:  python experiment.py      (deterministic; ~1 min CPU on Apple M5, longer wall time on a loaded machine)
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


L25 = load_lesson("lesson25", "25-evaluating-policies")
L26 = load_lesson("lesson26", "26-teleop-replay-data")
L27 = load_lesson("lesson27", "27-robot-datasets")
pick16 = L25.pick16
wilson, Scene, train_mlp, load_demos = L25.wilson, L25.Scene, L27.train_mlp, L27.load_demos
FPS, K = L26.FPS, L26.K                     # 25 Hz, 20 physics steps per policy step
HORIZON_S = 6.0
T_MAX = 113 / FPS                            # last timestamp in the demonstrations (4.52 s)
TEST_XY = L26.sample_positions(20, seed=28)  # fixed test positions, never used for training
NOISE_RAD, NOISE_CUBE_M = 0.02, 0.001


def features(obs, t, clock=True):
    """Policy input: the 9-D observation, plus the episode clock (clipped to the demonstrated range)."""
    obs = np.atleast_2d(obs)
    if not clock:
        return obs
    t = np.minimum(np.reshape(t, (-1, 1)), T_MAX)
    return np.concatenate([obs, t], axis=1)


def noise_vector(sigma_rad=NOISE_RAD, clock=True):
    return np.array([sigma_rad] * 6 + [NOISE_CUBE_M] * 3 + ([0.0] if clock else []))


def train_policy(obs, act, t, seed=0, sigma_rad=NOISE_RAD, clock=True, steps=4000):
    """Behavior cloning: MSE regression from features to joint targets, MEAN_STD normalization,
    Gaussian noise on the inputs of every training batch (sigma_rad on joints, 1 mm on the cube)."""
    X = features(obs, t, clock)
    pol = train_mlp(X, act, steps=steps, seed=seed, obs_noise=noise_vector(sigma_rad, clock) if sigma_rad > 0 else None)
    pol.clock = clock
    return pol


def rollout(scene, policy, xy, perturb=None, log=None):
    """Closed-loop episode. policy(obs, t) -> 6 joint targets. perturb(k, action, scene) may modify the action
    or the scene at policy step k. log (a list) receives (obs, t) per step. Returns an outcome string."""
    scene.place(xy)
    min_jaw = 9.0
    for k in range(int(round(HORIZON_S * FPS))):
        o, t = scene.obs(), k / FPS
        if log is not None:
            log.append((o.copy(), t))
        a = np.array(policy(o, t), dtype=float).reshape(-1)
        if perturb is not None:
            a = perturb(k, a, scene)
        scene.apply(a)
        scene.step(K)
        min_jaw = min(min_jaw, scene.obs()[5])
    if scene.success():
        return "held"
    if min_jaw > 0.6:
        return "never closed"
    if scene.obs()[8] < pick16.CUBE_HALF + 0.01:
        return "closed, cube on floor"
    return "lifted, not held"


def as_policy(pol):
    return lambda o, t: pol(features(o, t, pol.clock))[0]


def evaluate(scene, pol, positions=TEST_XY):
    outs = [rollout(scene, as_policy(pol), xy) for xy in positions]
    return sum(o == "held" for o in outs), outs


def main():
    t0 = time.process_time()
    import mujoco
    import torch
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · torch {torch.__version__}")
    D = load_demos()
    O, A, EP, TS = D["obs"], D["action"], D["episode_index"], D["timestamp"]
    scene = Scene()
    val = (EP >= 100)
    print(f"MLP 2 x 256 ReLU, 4000 Adam steps (batch 256, lr 1e-3), MEAN_STD normalization, input noise "
          f"{NOISE_RAD} rad on joints / {NOISE_CUBE_M * 1000:.0f} mm on cube; {len(TEST_XY)} test positions (seed 28); "
          f"horizon {HORIZON_S:.0f} s")

    print("\n[A] Number of demonstrations x 3 training seeds")
    print(f"{'demos':>6}{'frames':>8}{'seed 0':>8}{'seed 1':>8}{'seed 2':>8}{'pooled':>9}{'Wilson 95%':>16}"
          f"{'open-loop RMSE deg':>20}   failures (pooled)")
    pooled = {}
    for n in (5, 10, 20, 50, 100):
        tr = EP < n
        ks, fails, errs, where = [], {}, [], {}
        for seed in (0, 1, 2):
            pol = train_policy(O[tr], A[tr], TS[tr], seed=seed)
            k, outs = evaluate(scene, pol)
            ks.append(k)
            for i, o in enumerate(outs):
                if o != "held":
                    fails[o] = fails.get(o, 0) + 1
                    where[i] = where.get(i, 0) + 1
            pred = pol(features(O[val], TS[val]))
            errs.append(np.degrees(np.sqrt(np.mean((pred - A[val]) ** 2))))
        K_, N_ = sum(ks), 3 * len(TEST_XY)
        pooled[n] = (K_, N_)
        lo, hi = wilson(K_, N_)
        print(f"{n:>6}{int(tr.sum()):>8}" + "".join(f"{f'{k}/20':>8}" for k in ks) + f"{f'{K_}/{N_}':>9}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>16}{np.mean(errs):>20.3f}   "
              + (", ".join(f"{k} {v}" for k, v in sorted(fails.items())) or "-")
              + ("" if not where else "; at test positions " + ", ".join(
                  f"#{i} ({TEST_XY[i][0] * 100:.1f}, {TEST_XY[i][1] * 100:.1f}) cm x{c}" for i, c in sorted(where.items()))))

    (k50, n50), (k100, n100) = pooled[50], pooled[100]
    print(f"50 vs 100 demos, pooled {k50}/{n50} vs {k100}/{n100}: two-sided Fisher exact p = "
          f"{L25.fisher_p(k50, n50, k100, n100):.3f} (treats the 60 rollouts as independent; "
          f"the 3 seeds share the same 20 test positions)")

    print("\n[B] Break it: 50 demonstrations, seed 0")
    tr = EP < 50
    print(f"{'input noise rad':>16}{'clock':>7}{'success':>9}{'Wilson 95%':>16}   failures")
    for sigma, clock in ((0.0, True), (0.005, True), (0.01, True), (0.02, True), (0.04, True), (0.02, False)):
        pol = train_policy(O[tr], A[tr], TS[tr], seed=0, sigma_rad=sigma, clock=clock)
        k, outs = evaluate(scene, pol)
        lo, hi = wilson(k, len(TEST_XY))
        fails = {}
        for o in outs:
            if o != "held":
                fails[o] = fails.get(o, 0) + 1
        print(f"{sigma:>16g}{'yes' if clock else 'no':>7}{f'{k}/20':>9}{f'[{lo:.2f}, {hi:.2f}]':>16}   "
              + (", ".join(f"{a} {b}" for a, b in sorted(fails.items())) or "-"))
    print(f"CPU time {time.process_time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
