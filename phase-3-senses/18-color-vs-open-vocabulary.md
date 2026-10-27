# 18 · I turned on the lamp, the robot couldn't see red

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [3 · Senses](README.md) |
| **Prerequisites** | [16](../phase-3-senses/16-camera-calibration.md) |
| **Tools** | OpenCV + Hugging Face (Mac CPU/MPS) |
| **Status** | 🚧 coming soon |

## Learning goal

See that color thresholding is fragile under lighting changes, and that open-vocabulary detection is more robust but slower.

## Lesson question

Under three lighting conditions, how often do HSV thresholding and OWL-ViT find the object, and how many ms per frame?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: OWL-ViT's speed on a Mac has not been measured yet.

## Planned experiment

3 lighting conditions × (HSV, OWL-ViT) → detection rate and ms per frame.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/18/result.json`.

## Sources

- [LeRobot docs](https://huggingface.co/docs/lerobot/index) (lighting during recording)
- [OWL-ViT, arXiv 2205.06230](https://arxiv.org/abs/2205.06230)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
