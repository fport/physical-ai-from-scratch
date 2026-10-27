# 19 · Replay is not learning

> I played back the exact same motion. When the cube moved 3 cm, it missed.

> **Status: draft.** The experiment has been run in simulation, the notebook is ready and the numbers are verified. The `lerobot-replay` sweep on a real SO-101 has not been done yet.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Type** | Experiment |
| **Prerequisites** | [13 · Trajectory](../phase-2-muscle/13-trajectory.md), [14 · Safety](../phase-2-muscle/14-safety.md) (not required for the sim experiment) |
| **Time** | ~45 min |
| **Tools** | MuJoCo 3.17 · Menagerie `robotstudio_so101` · Colab or Mac |
| **Code** | [notebook](../notebooks/19-replay.ipynb) |
| **Status** | 📝 draft · notebook ready |

## Why it matters

The LeRobot workflow has an official command for playing a recording back: `lerobot-replay` sends the recorded actions to the robot again, exactly as they are ([LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)). When the arm repeats the recorded motion flawlessly, it is easy to think "the robot learned". This lesson measures that misconception: a small difference between the recording and the world is enough.

## Concept

**Open-loop** control applies commands without measuring the output; **closed-loop** (feedback) control changes the command based on what it measures (Lynch & Park, *Modern Robotics* §11.1; Åström & Murray, *Feedback Systems* §1.1). Replay is pure open loop: it sends the list in order and never looks at the cube.

## Experiment (in the notebook)

- The arm picks up and lifts a cube **once** (a motion written with IK, 4.5 s). The 6 commands sent every 2 ms (5 joints + gripper) are recorded: **2250 steps × 6**.
- **Replay:** the same list, completely unchanged. The cube is shifted 0.5–5 cm (10 distances in 0.5 cm steps) × 8 directions, plus the recorded position: **81 trials**.
- **Feedback rule:** the same timeline, but the target poses are recomputed with IK from the cube's position in that trial. No learned parameters.
- **Success is physical:** the cube is held by friction alone (no gluing/weld). Criterion: at t = 4.5 s the bottom of the cube is ≥ 4 cm above the floor and the cube touches both gripper jaws.

## Results in numbers

MuJoCo 3.17.0, deterministic (two runs, same result).

| Cube shift | Replay success (over 8 directions) |
|---|---|
| 0 cm (recorded position) | held |
| 0.5 cm | 8/8 |
| 1 cm | 7/8 |
| 1.5 cm | 4/8 |
| 2 cm | 3/8 |
| 2.5 cm | 1/8 |
| 3 cm | 1/8 |
| 4 cm | 1/8 |
| 4.5 and 5 cm | 0/8 |

- **Replay total: 27/81.** Feedback rule: **81/81**.
- When the cube moves 3 cm forward, the gripper closes where the cube used to be and never touches it.
- Direction matters: when the cube shifted toward the moving jaw (+y) it held up to 4 cm; when it shifted toward the fixed jaw (−y) it missed at 1 cm. Forward/backward tolerance is 1–1.5 cm.
- The confidence intervals of 8-trial rates are wide: e.g. for 1/8 the Wilson 95% interval is 0.02–0.47. This topic is in [lesson 22](22-evaluation.md).

The full table (including 3.5 cm) is in the notebook's results file.

## Limits

- **The feedback rule is not learning either.** The cube position is read **perfectly** from the simulation (no camera, no noise) and is read once before the motion starts. The more precise term is "replanning from perception".
- The recording in simulation is every 2 ms. On a real SO-101 a LeRobot recording is ~30 fps ([il_robots](https://huggingface.co/docs/lerobot/il_robots)).
- The Menagerie model's gripper collisions are primitive shapes tuned for manipulation; the actuator gains do not exactly match the real servo's. The tolerance numbers from the sim do not carry over to the real arm as they are.
- Each (distance, direction) is a single run.
- With the first home pose, one of the feedback runs failed due to self-collision (80/81); the home pose was changed. Across three different home poses, replay always came out 27/81 and feedback always 81/81.

## Exercises

1. **Easy.** In section 6 of the notebook, open the gripper less (`G_OPEN`). Does the replay tolerance shrink?
2. **Medium.** Make the cube larger (`CUBE_HALF`). In which directions does the tolerance grow?
3. **Hard.** Add ±5 mm noise to the cube position reading in the feedback rule. Does it stay at 81/81?

## Sources

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots): recording, replay, fps.
- [MuJoCo Menagerie · `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (Apache-2.0).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Chapter 11.
- Åström & Murray, [*Feedback Systems*](https://fbswiki.org), Chapter 1.
- Numbers from the results file of the Koddan Kasa 19 experiment and the internal verification note (27.10.2026).
