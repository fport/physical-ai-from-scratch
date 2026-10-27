# Table of contents

* [Introduction](README.md)

## Getting started

* [Setup: one Mac, zero cost](getting-started/setup.md)
* [How to read a lesson](getting-started/how-to-read-a-lesson.md)
* [Numbers and sources](getting-started/numbers-and-sources.md)
* [Reading order](getting-started/reading-order.md)

## Phase 0 · World, frame by frame

* [Introduction](phase-0-world-frame-by-frame/README.md)
  * [01 · The world is a text file](phase-0-world-frame-by-frame/01-world-as-text.md)
  * [02 · Timestep: why did the cube go through the table?](phase-0-world-frame-by-frame/02-timestep.md)
  * [03 · In simulation the table is a sponge: soft contact](phase-0-world-frame-by-frame/03-soft-contact.md)
  * [04 · Friction: does a heavier cube slip later?](phase-0-world-frame-by-frame/04-friction.md)

## Phase 1 · Space and body

* [Introduction](phase-1-space-and-body/README.md)
  * [05 · There is no "left" for a robot: coordinate frames](phase-1-space-and-body/05-coordinate-frames.md)
  * [06 · 6 motors, but not 6 degrees of freedom](phase-1-space-and-body/06-degrees-of-freedom.md)
  * [07 · Two angles, one point: forward kinematics](phase-1-space-and-body/07-forward-kinematics.md)
  * [08 · Can this arm touch its own base? Workspace](phase-1-space-and-body/08-workspace.md)
  * [09 · Same point, two different elbows: inverse kinematics](phase-1-space-and-body/09-inverse-kinematics.md)

## Phase 2 · Muscle

* [Introduction](phase-2-muscle/README.md)
  * [10 · You give the motor an angle, not a force: position servos](phase-2-muscle/10-position-servo.md)
  * [11 · 12 joints, two numbers: 20 and 0.5](phase-2-muscle/11-pd-control.md)
  * [12 · The arm isn't weak, physics makes it sag: gravity](phase-2-muscle/12-gravity.md)
  * [13 · Joints moved in a straight line, the tip drew a curve: trajectories](phase-2-muscle/13-trajectory.md)
  * [14 · A robot's first rule: being able to stop itself](phase-2-muscle/14-safety.md)

## Phase 3 · Senses

* [Introduction](phase-3-senses/README.md)
  * [15 · I cleaned the noise, the robot got late: filtering and latency](phase-3-senses/15-noise-and-latency.md)
  * [16 · Your camera sees straight lines as curves: calibration](phase-3-senses/16-camera-calibration.md)
  * [17 · I clicked on the screen, the arm touched that spot](phase-3-senses/17-pixel-to-table.md)
  * [18 · I turned on the lamp, the robot couldn't see red](phase-3-senses/18-color-vs-open-vocabulary.md)

## Phase 4 · Learning from demonstration

* [Introduction](phase-4-learning-from-demonstration/README.md)
  * [19 · Replay is not learning](phase-4-learning-from-demonstration/19-replay.md)
  * [20 · What does a robot dataset look like?](phase-4-learning-from-demonstration/20-datasets.md)
  * [21 · Nobody wrote code: behavior cloning](phase-4-learning-from-demonstration/21-behavior-cloning.md)
  * [22 · I said 90% success. How sure am I? Evaluation](phase-4-learning-from-demonstration/22-evaluation.md)
  * [23 · Does a perfect imitator sink at the first mistake? Distribution shift](phase-4-learning-from-demonstration/23-distribution-shift.md)
  * [24 · The average of two right answers is wrong](phase-4-learning-from-demonstration/24-two-right-answers.md)
  * [25 · The robot changed its mind every frame: action chunking](phase-4-learning-from-demonstration/25-action-chunking.md)
  * [26 · End-to-end imitation on a real arm](phase-4-learning-from-demonstration/26-end-to-end-imitation.md)

## Phase 5 · Agents

* [Introduction](phase-5-agents/README.md)
  * [27 · The agent loop in 25 lines](phase-5-agents/27-agent-loop.md)
  * [28 · The big model doesn't move the arm, it gives orders](phase-5-agents/28-two-brains.md)
  * [29 · I told the agent "reach under the table"](phase-5-agents/29-agent-safety.md)
  * [30 · Three agents, one arm: MCP](phase-5-agents/30-mcp.md)
  * [31 · Getting a robot to work by talking: Strands](phase-5-agents/31-strands-agent.md)

## Phase 6 · Learning by trial

* [Introduction](phase-6-learning-by-trial/README.md)
  * [32 · No directions, just rewards: Q-learning](phase-6-learning-by-trial/32-q-learning.md)
  * [33 · Thousands of robots falling at once: PPO intuition](phase-6-learning-by-trial/33-parallel-rl.md)
  * [34 · When the agent games the reward: reward hacking](phase-6-learning-by-trial/34-reward-hacking.md)

## Phase 7 · Sim to real

* [Introduction](phase-7-sim-to-real/README.md)
  * [35 · How many milliseconds behind is the robot's brain?](phase-7-sim-to-real/35-latency-and-control-rate.md)
  * [36 · My simulated motor wasn't like the real one: system identification](phase-7-sim-to-real/36-system-identification.md)
  * [37 · Breaking the sim on purpose: domain randomization](phase-7-sim-to-real/37-domain-randomization.md)

## Phase 8 · Big models

* [Introduction](phase-8-big-models/README.md)
  * [38 · Inside a VLA: image + sentence → action](phase-8-big-models/38-vla.md)
  * [39 · Paths from noise: diffusion and flow](phase-8-big-models/39-diffusion-and-flow.md)

## Phase 9 · Capstone

* [Introduction](phase-9-capstone/README.md)
  * [40 · "Put the blue cube in the box"](phase-9-capstone/40-capstone.md)

## Appendix

* [Appendix lessons](appendix/README.md)
  * [Appendix A · Singularities: why does the arm go crazy when fully stretched?](appendix/a-singularities.md)
  * [Appendix B · Gimbal lock and quaternions](appendix/b-gimbal-lock.md)
  * [Appendix C · Kalman intuition: fusing two noisy measurements](appendix/c-kalman.md)
  * [Appendix D · Grasp points with SAM](appendix/d-sam-grasp-points.md)
  * [Appendix E · Fine-tuning SmolVLA on your own data](appendix/e-smolvla-finetuning.md)
  * [Appendix F · World models: a small model that learns "what happens next"](appendix/f-world-models.md)
  * [Appendix G · π0, GR00T, Gemini Robotics: three scales of the same idea](appendix/g-big-robot-models.md)
  * [Appendix H · How many Hz on a Mac? Inference speed on CPU and MPS](appendix/h-inference-speed-on-mac.md)

## Reference

* [Glossary](glossary.md)
* [References](references.md)
* [Notebooks](notebooks/README.md)
* [Contributing](contributing.md)
