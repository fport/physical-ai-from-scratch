# 27 · Robot datasets

<!-- meta -->
| | |
|---|---|
| **Part** | [7 · Learning from demonstration](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [26](26-teleop-replay-data.md) |
| **Learn first** | none |
| **Time** | ~40 min |
| **Code** | [`experiments/27-robot-datasets/experiment.py`](../experiments/27-robot-datasets/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A robot dataset is a time series cut into episodes. Normalize with its statistics, and split it by episode, never by frame.

## The problem

Lesson 26 produced 13680 frames of scripted SO-101 demonstrations. Before training on them, three decisions decide whether the numbers you see during training mean anything: what each frame contains and where episodes start, how the features are scaled, and how the data is split into training and validation. The last one is the easiest to get wrong. Consecutive frames 40 ms apart are almost identical, so a random frame-level split puts near-duplicates on both sides, and the validation error turns into a measure of interpolation between neighbouring frames. This lesson measures how large that illusion is on our data.

## Core idea

**What a frame holds.** A robot dataset is a set of episodes $e$, each a sequence of frames $t = 0 \dots T_e - 1$ at a fixed rate:

| Column | Meaning | Our data |
|---|---|---|
| `observation.state` | proprioception (joint positions), here plus the cube pose | 9 floats |
| `observation.images.*` | camera frames | none (state-based) |
| `action` | command sent at this frame | 6 joint targets |
| `timestamp` | seconds since episode start | $t/25$ |
| `frame_index`, `episode_index`, `index` | position in episode, episode id, global row | int |
| `task_index` | which natural-language task | 0: "pick up the cube" |

**The LeRobot format (v3.0).** Verified in the docs and code ([LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3); `CODEBASE_VERSION = "v3.0"` in [`dataset_metadata.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/dataset_metadata.py)):

- `meta/info.json`: schema (feature names, dtypes, shapes), fps, codebase version, path templates.
- `meta/stats.json`: per-feature statistics for normalization; the code computes min, max, mean, std, count and the quantiles 0.01, 0.10, 0.50, 0.90, 0.99 ([`compute_stats.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py)).
- `meta/episodes/`: per-episode records (length, tasks, offsets into the shared files) as chunked Parquet.
- `meta/tasks.parquet`: task strings mapped to integer ids (the docs page still says `tasks.jsonl`, which the code keeps as the legacy path, [`utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py)).
- `data/chunk-000/file-000.parquet`: frame rows, many episodes per file; `videos/<camera>/chunk-000/file-000.mp4` likewise.

`timestamp`, `frame_index`, `episode_index`, `index` and `task_index` are filled in by the recorder, not by the user ([`feature_utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/feature_utils.py)). In v3 episode boundaries live in the metadata, not in file names.

**Normalization.** Inputs and targets are rescaled with dataset statistics before training:

$$\text{MEAN\_STD: } \tilde x = \frac{x - \mu}{\sigma}, \qquad \text{MIN\_MAX: } \tilde x = 2\,\frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1 .$$

LeRobot defines MIN_MAX, MEAN_STD, IDENTITY, QUANTILES and QUANTILE10 ([`types.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/types.py)); the ACT config defaults to MEAN_STD for state and action, the Diffusion Policy config to MIN_MAX ([ACT](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py), [Diffusion](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py)). The statistics must come from the training data and travel with the policy: a policy evaluated with different statistics sees different inputs.

**Splitting.** Frames within an episode are strongly correlated; episodes (here: cube positions) are the independent units. Validation must hold out whole episodes, as the evaluation in [lesson 25](25-evaluating-policies.md) holds out test positions.

## Experiment

Input: [`demos.npz`](../experiments/26-teleop-replay-data/demos.npz) from lesson 26. Model: an MLP (2 × 256 ReLU, Adam, 4000 steps of batch 256) predicting the 6 action values from the 9 state values, the same network lesson 28 trains, there with two extra ingredients (see AIEFS: [Model Evaluation](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/09-model-evaluation)). Error is the action RMSE in degrees, open loop (no simulation). Full script: [`experiments/27-robot-datasets/experiment.py`](../experiments/27-robot-datasets/experiment.py), output: [`output.txt`](../experiments/27-robot-datasets/output.txt). NumPy 2.5.4, PyTorch 2.14.1 (CPU), Apple Silicon Mac (Apple M5), deterministic.

**A. Anatomy.** 120 episodes, 13680 frames, 25 fps, all 120 successful; episode boundaries at rows 0, 114, 228, …, 13566. Statistics as `stats.json` would hold them (excerpt):

| Feature | mean | std | min | max | q01 | q99 |
|---|---|---|---|---|---|---|
| shoulder_lift [rad] | −0.240 | 0.515 | −1.200 | 0.702 | −1.200 | 0.620 |
| gripper [rad] | 0.731 | 0.505 | 0.141 | 1.200 | 0.162 | 1.200 |
| cube_x [m] | 0.222 | 0.037 | 0.160 | 0.332 | 0.161 | 0.321 |
| cube_z [m] | 0.036 | 0.032 | 0.015 | 0.114 | 0.015 | 0.101 |
| action gripper [rad] | 0.561 | 0.674 | −0.170 | 1.200 | −0.170 | 1.200 |

Two details show up only in the statistics. The commanded gripper reaches −0.170 rad (the closed limit) while the measured jaw never goes below 0.141 rad: the cube stops it, and the servo squeezes. And cube_x reaches 0.332 m although cubes were placed at most 0.28 m out: in the farthest episodes the lifted cube moves outward with the gripper during lift and hold (we checked the per-episode trajectories).

**B. Normalization** (train on episodes 0–79, validate on 80–99):

| Inputs | normalization | val RMSE [deg] | worst joint (RMSE) |
|---|---|---|---|
| cube in m | none | 0.306 | gripper (0.480) |
| cube in mm | none | 4.649 | gripper (8.164) |
| cube in m | MEAN_STD | 0.264 | gripper (0.507) |
| cube in mm | MEAN_STD | 0.265 | gripper (0.514) |
| cube in m | MIN_MAX | 0.245 | gripper (0.448) |

In metres our raw features are already of order 0.01–1, so normalization gains little (0.306 → 0.264). Log the cube in millimetres and the unnormalized model is 15 times worse, while the normalized one is unchanged. Normalization makes the model independent of the units someone chose when logging.

**C. Frame split vs episode split.** *Predict:* with 10 episodes, which split reports the lower validation error, and which one is right? Both splits are 80/20: the frame split shuffles all frames of the $n$ episodes, the episode split trains on 80% of the episodes (8, 24, 72) and validates on the rest (2, 6, 18). The test set is 20 episodes (100–119) that no model ever sees:

| Episodes used | split | val RMSE [deg] | test RMSE [deg] | test / val |
|---|---|---|---|---|
| 10 | frame | 0.252 | 1.595 | 6.33 |
| 10 | episode | 0.640 | 1.804 | 2.82 |
| 30 | frame | 0.280 | 0.446 | 1.59 |
| 30 | episode | 0.544 | 0.479 | 0.88 |
| 90 | frame | 0.270 | 0.279 | 1.03 |
| 90 | episode | 0.343 | 0.323 | 0.94 |

The frame-level validation error stays at 0.25–0.28° whatever the dataset size. It measures how well the model fills the 40 ms gaps between frames it has already seen, and with 10 episodes it understates the real error on new cube positions by a factor of 6.33. The episode-level number moves with the test error. With only 2 validation episodes (the 10-episode row) it is still optimistic, because 2 episodes are a small sample.

## Break it

The breaking parameter is the dataset size under a frame split. With 90 episodes the leak barely matters (test/val 1.03), because the training episodes already cover the cube area densely. With 10 it hides most of the error. Small datasets are where the frame split does the most damage, and robot datasets are small: LeRobot's own guide starts from 50 episodes ([il_robots](https://huggingface.co/docs/lerobot/il_robots)). The same leak appears in any per-frame shuffle that ignores `episode_index`: an early-stopping rule or a hyperparameter search driven by a leaky validation set picks overfitted models.

## On the real robot

- A real SO-101 recording made with `lerobot-record` already contains `episode_index`, `frame_index`, `timestamp` and `task_index`, and its `meta/stats.json` is written by the recorder ([LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)). Split with `episode_index`; with several recording sessions, hold out a whole session, because lighting, calibration and the operator drift between sessions.
- Real observations contain images, and their statistics are per-channel. Real actions come from the leader arm, so they are noisier than our scripted ones, and two operators produce different action distributions for the same task.
- Not yet measured on the real arm: the frame-vs-episode gap of part C on a teleoperated SO-101 dataset. Procedure: record 50 episodes, train the same MLP on state → action, and compare both validation numbers with 10 held-out episodes.

## Check yourself

1. **Easy.** Our episodes all have 114 frames at 25 fps. How long is each episode, and what is `index` of frame 5 of episode 2?
2. **Medium.** Why does the measured jaw angle never go below 0.141 rad while the action goes to −0.170 rad? What would a policy trained on `observation.state` alone have to learn about this gap?
3. **Hard.** Split the 100 training episodes by cube position instead of at random: train on cubes with y < 4 cm, validate on y ≥ 4 cm. Predict the validation error relative to the random episode split, then run it.

### Answers

1. 114 / 25 = 4.56 s. `index` = 2 · 114 + 5 = 233.
2. The cube blocks the jaw; the command is the closed limit so the servo pushes until its torque cap (lesson 16). The policy must output −0.17 rad when it sees the jaw stopped at the cube width, i.e. learn a command that is never observed as a state.
3. Not computed here; run it. An extrapolation split tests a different question from a random episode split.

## Takeaways and sources

- A robot dataset is (observation, action, timestamp) frames grouped by `episode_index`, with task strings and statistics in metadata; LeRobot v3.0 stores many episodes per Parquet/MP4 file and resolves episodes through `meta/episodes/`.
- Normalize with training-set statistics: logging the cube in mm raised the unnormalized error from 0.306° to 4.649°, the normalized one stayed at 0.265°.
- Split by episode: with 10 episodes the frame split reported 0.252° against a true 1.595° on unseen episodes.

Next: [lesson 28 · Behavior cloning](28-behavior-cloning.md).

Sources:
- LeRobot, [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) (read 2026-10-27).
- LeRobot source (main, read 2026-10-27): [`datasets/utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py), [`datasets/compute_stats.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py), [`datasets/feature_utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/feature_utils.py), [`datasets/dataset_metadata.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/dataset_metadata.py), [`configs/types.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/types.py), [ACT config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py), [Diffusion config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py).
- LeRobot, [Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots).
- All measured numbers: [`experiments/27-robot-datasets/output.txt`](../experiments/27-robot-datasets/output.txt).
