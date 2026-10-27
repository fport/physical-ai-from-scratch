# 25 · Evaluating robot policies

<!-- meta -->
| | |
|---|---|
| **Part** | [7 · Learning from demonstration](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md) |
| **Learn first** | [Statistics for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/15-statistics-for-ml) · [Model Evaluation](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/09-model-evaluation) |
| **Time** | ~40 min |
| **Code** | [`experiments/25-evaluating-policies/experiment.py`](../experiments/25-evaluating-policies/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A success rate without its number of trials is not a measurement. Report k/n, a Wilson interval, and the physical test that decided each trial.

## The problem

Part 7 trains policies and asks one question of each: does it pick up the cube? Answers like "90% success" usually rest on 10 or 20 rollouts. With 10 trials, a policy whose long-run rate is 0.74 scores 9/10 or better about one time in five (measured below), so a single "9/10" cannot tell a 0.74 policy from a 0.88 one. Before training anything, this lesson builds the evaluation protocol the rest of the book uses: what counts as success, how many trials, what interval to report, and when two policies are actually different.

## Core idea

**Success is a Bernoulli outcome.** Trial $i$ ends in $X_i \in \{0, 1\}$, decided by a physical test. Ours, from [lesson 16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md): at the end the cube bottom is ≥ 4 cm above the floor and both jaws touch it, with no weld or attachment. If trials are independent with the same rate $p$, the count $k = \sum X_i$ is binomial and $\hat p = k/n$ (see AIEFS: [Statistics for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/15-statistics-for-ml)).

**Wilson interval.** The textbook (Wald) interval $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ collapses to zero width at $k = 0$ or $k = n$ and under-covers at small $n$. Wilson (1927) inverts the score test instead:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}},\qquad z = 1.96 \text{ for 95\%}.$$

Brown, Cai and DasGupta (2001) show the Wald interval's coverage is erratic and recommend Wilson (or Jeffreys) for small $n$.

**Trials needed.** The half-width shrinks like $1/\sqrt n$: halving it costs four times the trials.

**Variance has more than one source.** The binomial interval covers evaluation noise only. A learned policy also varies across training seeds and data subsets; a claim about a *method* needs several training runs, each evaluated on the same set of test positions. Lesson 28 reports per-seed and pooled results.

**Comparing two policies.** Overlapping intervals do not prove equality, and disjoint 95% intervals are a stricter test than a direct comparison. For $k_A/n_A$ vs $k_B/n_B$ use a two-sample test, e.g. Fisher's exact test on the 2 × 2 table ([SciPy `fisher_exact`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)).

## Experiment

A "policy" with an unknown rate: the scripted pick of lesson 16 aiming at a noisy cube estimate, true position + $\mathcal N(0, \sigma^2)$ per axis, as if from an imperfect perception module. Cube positions are uniform in x ∈ [16, 28] cm, y ∈ [−8, 8] cm. Policy A has σ = 9 mm, policy B σ = 6 mm. Each gets a pool of 300 simulated trials, whose rate stands in for the long-run rate $p$. Evaluations of $n$ trials are then drawn from a pool with replacement. Full script: [`experiments/25-evaluating-policies/experiment.py`](../experiments/25-evaluating-policies/experiment.py), output: [`output.txt`](../experiments/25-evaluating-policies/output.txt). MuJoCo 3.17.0, NumPy 2.5.4, SciPy 1.18.1, Apple Silicon Mac (Apple M5), deterministic, about 1 min of CPU (approximate; wall time was several times longer on a heavily loaded machine).

```python
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)
```

**A. The pools.**

| Policy | Aim noise | Success | Wilson 95% | "Jaw closed on cube" |
|---|---|---|---|---|
| A | 9 mm | 221/300 (0.737) | [0.684, 0.783] | 259/300 (0.863) |
| B | 6 mm | 263/300 (0.877) | [0.835, 0.909] | 288/300 (0.960) |

The last column is a looser criterion counted on the same trials: the jaw settled with both jaws on the cube. It adds 38 "successes" to A and 25 to B, cubes that were touched or squeezed but not lifted and held. The success test has to be written down before the evaluation, and it has to be physical.

**B. Number of trials.** *Predict:* at $n = 10$, how wide is a 95% interval around $p \approx 0.74$? 2000 evaluations per row, from pool A:

| n | mean Wilson width | Wilson covers p | Wald covers p | exact Wilson | exact Wald | 5–95% of k/n |
|---|---|---|---|---|---|---|
| 5 | 0.577 | 0.987 | 0.764 | 0.981 | 0.764 | 0.40–1.00 |
| 10 | 0.463 | 0.971 | 0.928 | 0.975 | 0.927 | 0.50–0.90 |
| 20 | 0.354 | 0.967 | 0.919 | 0.962 | 0.908 | 0.55–0.90 |
| 50 | 0.235 | 0.951 | 0.916 | 0.947 | 0.914 | 0.62–0.84 |
| 100 | 0.169 | 0.942 | 0.942 | 0.946 | 0.945 | 0.66–0.81 |

"Exact" is the coverage computed from the binomial distribution at $p$ = 0.737; the resampled columns agree with it. At 10 trials the interval is 0.46 wide and the observed rate lands anywhere from 0.50 to 0.90. Wilson stays within about one point of 95% (0.942–0.987; it is slightly conservative at small $n$); Wald falls to 0.76 at $n = 5$.

**C. What "9/10" means.** Wilson 95% for 9/10 is [0.596, 0.982]; for 10/10 it is [0.722, 1.000]. The probability of seeing ≥ 9/10:

| Policy | long-run p | P(≥ 9/10), binomial | drawn from the pool |
|---|---|---|---|
| A | 0.737 | 0.215 | 0.208 |
| B | 0.877 | 0.645 | 0.634 |

A 9/10 is consistent with both policies, and B fails to produce one in about a third of evaluations.

**D. Trials for a target half-width** (Wilson 95%):

| observed rate | ±0.10 | ±0.05 | ±0.02 |
|---|---|---|---|
| 0.50 | 93 | 381 | 2398 |
| 0.70 | 78 | 320 | 2014 |
| 0.90 | 34 | 141 | 864 |
| 0.95 | 24 | 82 | 465 |

**E. Is B better than A?** The true gap is 0.14. 1000 evaluation pairs of $n$ trials each:

| n per policy | Fisher p < 0.05 | Wilson intervals disjoint | B scores ≤ A |
|---|---|---|---|
| 10 | 0.037 | 0.006 | 0.286 |
| 20 | 0.113 | 0.025 | 0.170 |
| 50 | 0.335 | 0.152 | 0.041 |
| 100 | 0.673 | 0.423 | 0.005 |

With 20 trials each, the better policy scores the same or worse 17% of the time and the difference is significant only 11% of the time. Even 100 trials each detect a 14-point gap only two times in three.

## Break it

Push $n$ down and $p$ toward 1 and the Wald interval breaks. At $n = 5$ it covers the true rate in 76% of evaluations instead of 95%. At 10/10 it reports [1.000, 1.000], a claim of certainty from ten trials, while Wilson gives [0.722, 1.000]. The mechanism: Wald plugs $\hat p$ into the variance, so the estimated variance is zero exactly when $\hat p$ hits 0 or 1, and the normal approximation is poor when $np$ or $n(1-p)$ is small. Wilson centers the interval toward 0.5 by $z^2/2n$ and keeps a $z^2/4n^2$ term that never vanishes.

## On the real robot

- **Protocol** (not yet run on our SO-101): fix 20 cube positions on a grid marked on the table, and use the same positions for every policy compared. Write the success test before running: cube lifted ≥ 4 cm and held for 1 s, judged from video. Reset the cube by hand to the mark. Report k/n, Wilson 95%, and failure types (missed, pushed, dropped).
- **Independence is the assumption that fails first.** Servo heating, a cube that wears, a shifted camera and the operator's resets all make trials correlated. Interleave the policies (A, B, A, B, …) rather than running all of A first.
- LeRobot's `lerobot-rollout` can record evaluation episodes to a dataset (e.g. `--strategy.type=sentry`; `base` runs without recording) ([il_robots](https://huggingface.co/docs/lerobot/il_robots)). The success decision and the interval are still yours to define and report.

## Check yourself

1. **Easy.** A policy scores 18/20. Give its Wilson 95% interval using the code above.
2. **Medium.** From table D, about how many trials does a ±0.05 claim need at an observed rate of 0.70? Why is it fewer at 0.95?
3. **Hard.** Run 3 training seeds of the same method and evaluate each on the same 20 positions. Propose how to report the result so both seed variance and trial variance are visible.

### Answers

1. [0.699, 0.972] (`wilson(18, 20)`; also printed in part C of the output).
2. 320. The binomial variance $p(1-p)$ is 0.21 at 0.70 and 0.0475 at 0.95, so the same precision needs fewer trials near the edges.
3. One acceptable answer: report each seed's k/20 with its Wilson interval, plus the pooled k/60 with its interval and the min–max across seeds. The pooled interval alone hides seed variance.

## Takeaways and sources

- Report k/n, a Wilson 95% interval and the physical success test. In simulation, a looser "jaw closed" test inflated policy A from 0.737 to 0.863.
- Ten trials give an interval about 0.46 wide; a 0.74 policy produces ≥ 9/10 in about 21% of evaluations.
- Telling a 0.74 from a 0.88 policy reliably takes on the order of 100+ trials each; at 20 each, Fisher's test detected the gap 11% of the time.

Next: [lesson 26 · Teleoperation, replay and data collection](26-teleop-replay-data.md).

Sources:
- E. B. Wilson, ["Probable inference, the law of succession, and statistical inference"](https://doi.org/10.1080/01621459.1927.10502953), *J. Amer. Statist. Assoc.* 22 (158), 209–212, 1927.
- L. D. Brown, T. T. Cai, A. DasGupta, ["Interval Estimation for a Binomial Proportion"](https://doi.org/10.1214/ss/1009213286), *Statistical Science* 16 (2), 101–133, 2001.
- SciPy, [`scipy.stats.fisher_exact`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
- LeRobot, [Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots).
- All measured numbers: [`experiments/25-evaluating-policies/output.txt`](../experiments/25-evaluating-policies/output.txt).
