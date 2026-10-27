# 26 · End-to-end imitation on a real arm

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Prerequisites** | 19–25 |
| **Tools** | LeRobot + real SO-101 + rented GPU |
| **Status** | 🚧 coming soon |

## Learning goal

End-to-end imitation on a real arm: recording → training → evaluation over 20+ trials; the effect of data distribution.

## Lesson question

Are 25 demonstrations enough, or 50?

## Why it matters

LeRobot recommends at least 50 episodes and 10 per position; the SmolVLA docs say 25 episodes were not enough. Per the hardware guide, ACT training on Apple Silicon (MPS) takes ~6–14 hours. Sources: [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots), [LeRobot · SmolVLA](https://huggingface.co/docs/lerobot/smolvla), [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide).

> Note: The 25 vs 50 episode comparison for ACT will not be stated until we run it ourselves.

## Planned experiment

ACT with 50 episodes (5 positions × 10), plus 25 episodes for comparison. Training on Colab GPU / HF Jobs. Evaluation: `lerobot-rollout`, 20 trials + confidence interval.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/26/result.json`.

## Sources

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide)
- [LeRobot · SmolVLA](https://huggingface.co/docs/lerobot/smolvla)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
