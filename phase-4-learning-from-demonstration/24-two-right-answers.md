# 24 · The average of two right answers is wrong

> The robot learned to pass the obstacle on the left and on the right. It hit it dead center.

> **Status: draft.** The experiment has been run, the notebook is ready and the numbers are verified. The text is not yet finished in book format.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Type** | Experiment |
| **Prerequisites** | [21 · Behavior cloning](21-behavior-cloning.md) |
| **Time** | ~40 min |
| **Tools** | NumPy · Colab CPU or Mac (no GPU needed, ~1 min) |
| **Code** | [notebook](../notebooks/24-two-right-answers.ipynb) |
| **Status** | 📝 draft · notebook ready |

## Why it matters

In 2023, researchers from Columbia, Toyota Research Institute and MIT published *Diffusion Policy* ([Chi et al., arXiv 2303.04137](https://arxiv.org/abs/2303.04137)). One of the paper's starting points is **multimodal action distributions**: there can be more than one correct motion in the same state, and methods that learn a single average break down in that case. This lesson shows that idea with 200 demonstrations passing on both sides of an obstacle, in 2 dimensions and in one minute.

## Concept

The prediction that minimizes squared error (MSE) is the **conditional mean** of the data (e.g. Bishop, *Pattern Recognition and Machine Learning* §1.5.5). The average of demonstrations that start from the same place and pass half on the left and half on the right is a path straight through the middle of the obstacle.

## Experiment (in the notebook)

In a bird's-eye 2D plane a point moves from bottom to top, with a round obstacle in between. Of the 200 demonstrations, 100 pass on the left and 100 on the right; the side is chosen independently of the starting point.

- **MSE model:** a small MLP (64-64, tanh) that looks at the start and predicts the whole path (40 waypoints).
- **Mixture model (MDN, 2 components):** the same network, whose output is two possible paths and their weights. In a trial, one component is chosen first, then that path is followed.

## Results in numbers

NumPy 2.5.4, seed 0, 1500 epochs, 200 trials.

| | Collisions |
|---|---|
| MSE, demonstrations half left / half right | **200/200** |
| MSE, all demonstrations on the left | 0/200 |
| Mixture model (MDN) | **0/200** (96 left, 104 right) |

At the obstacle's level (y = 0) the demonstrations pass at x = −0.55 and +0.55, the MSE path at x = +0.03 (obstacle radius 0.30).

**More demonstrations do not fix it**, quite the opposite: the MSE collision rate is 8% at 10 demonstrations, 46% at 20, 85% at 50, 91.5% at 100, and 100% at 200 and 500. With few demonstrations the network memorizes small differences in the start by tying them to the demonstration's side, and most of the time it veers to one side; as data grows it approaches the true average.

Across three different seeds MSE collided 97% / 93.5% / 100%, the mixture model 0 in all three.

## Limits

- The "robot" is a 2D point.
- The result holds in the setup where the observation cannot distinguish the two paths (here: predicting the whole path from the start). A step-by-step, closed-loop MSE policy ((x, y) → next step) collided **0/200** on the same data: the average flow in the middle is unstable, and a small difference pushes the robot to one side.
- In the mixture model σ is fixed (0.2) and the components are initialized with two distant demonstrations; with a learned σ the components collapsed in the first attempt.
- The toy model is not the same as Diffusion Policy. The shared idea: representing several possible motions instead of a single average. ACT ([arXiv 2304.13705](https://arxiv.org/abs/2304.13705)) is trained as a CVAE but sets the latent variable to zero at inference; saying it "samples the possibilities" would be wrong.

## Exercises

1. **Easy.** If 80% of the demonstrations passed on the left, would the MSE model collide? Try it in the notebook with `LEFT_FRACS`. (The answer is in the notebook.)
2. **Medium.** Shrink the obstacle (`OBS_R = 0.1`). Does the average path now pass beside the obstacle?
3. **Hard.** Write a step-by-step policy and show why it usually escapes by plotting a flow field. Is this escape reliable?

## Sources

- [Chi et al., *Diffusion Policy*, arXiv 2303.04137](https://arxiv.org/abs/2303.04137).
- [Zhao et al., *ACT*, arXiv 2304.13705](https://arxiv.org/abs/2304.13705).
- Bishop, *Pattern Recognition and Machine Learning* (2006), §1.5.5.
- Numbers from the results file of the Koddan Kasa 24 experiment and the internal verification note (27.10.2026).
