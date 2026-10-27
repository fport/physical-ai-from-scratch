# Table of contents

* [Introduction](README.md)

## Start here

* [The physical AI stack](start-here/physical-ai-stack.md)
* [Prerequisites](start-here/prerequisites.md)
* [Setup](start-here/setup.md)
* [How to read a lesson](start-here/how-to-read-a-lesson.md)
* [Numbers and sources](start-here/numbers-and-sources.md)
* [Reading paths](start-here/reading-paths.md)
* [Videos and lessons](start-here/video-map.md)

## Part 1 · Simulation

* [Overview](part-1-simulation/README.md)
  * [01 · The world as text: MJCF and URDF](part-1-simulation/01-world-as-text.md)
  * [02 · Time integration and the timestep](part-1-simulation/02-timestep.md)
  * [03 · Contact: soft constraints, solref and solimp](part-1-simulation/03-contact.md)
  * [04 · Friction and friction cones](part-1-simulation/04-friction.md)
  * [05 · Joints and actuators](part-1-simulation/05-joints-and-actuators.md)

## Part 2 · Spatial math

* [Overview](part-2-spatial-math/README.md)
  * [06 · Rotations: matrices, Euler angles, axis-angle, quaternions](part-2-spatial-math/06-rotations.md)
  * [07 · Rigid transforms and frame trees](part-2-spatial-math/07-rigid-transforms.md)

## Part 3 · Kinematics

* [Overview](part-3-kinematics/README.md)
  * [08 · Forward kinematics, degrees of freedom and workspace](part-3-kinematics/08-forward-kinematics.md)
  * [09 · Jacobians and differential kinematics](part-3-kinematics/09-jacobians.md)
  * [10 · Inverse kinematics and singularities](part-3-kinematics/10-inverse-kinematics.md)

## Part 4 · Dynamics and control

* [Overview](part-4-dynamics-and-control/README.md)
  * [11 · Rigid-body dynamics](part-4-dynamics-and-control/11-rigid-body-dynamics.md)
  * [12 · Joint control: PD, PID, gravity feedforward and position servos](part-4-dynamics-and-control/12-joint-control.md)
  * [13 · Trajectory generation](part-4-dynamics-and-control/13-trajectory-generation.md)
  * [14 · Operational-space and impedance control](part-4-dynamics-and-control/14-impedance-control.md)
  * [15 · Safety: limits, torque caps, e-stop and watchdogs](part-4-dynamics-and-control/15-safety.md)
  * [16 · Grasping and scripted pick-and-place](part-4-dynamics-and-control/16-grasping-and-pick-place.md)

## Part 5 · Perception and estimation

* [Overview](part-5-perception-and-estimation/README.md)
  * [17 · Sensors, noise and filtering](part-5-perception-and-estimation/17-sensors-noise-filtering.md)
  * [18 · The Kalman filter](part-5-perception-and-estimation/18-kalman-filter.md)
  * [19 · Camera model, calibration and hand-eye calibration](part-5-perception-and-estimation/19-camera-calibration.md)
  * [20 · From pixels to the table: homography and depth back-projection](part-5-perception-and-estimation/20-pixels-to-table.md)
  * [21 · Object perception for manipulation](part-5-perception-and-estimation/21-object-perception.md)

## Part 6 · Planning and optimal control

* [Overview](part-6-planning-and-optimal-control/README.md)
  * [22 · Configuration space and sampling-based planning](part-6-planning-and-optimal-control/22-motion-planning.md)
  * [23 · Trajectory optimization](part-6-planning-and-optimal-control/23-trajectory-optimization.md)
  * [24 · LQR and model predictive control](part-6-planning-and-optimal-control/24-lqr-and-mpc.md)

## Part 7 · Learning from demonstration

* [Overview](part-7-learning-from-demonstration/README.md)
  * [25 · Evaluating robot policies](part-7-learning-from-demonstration/25-evaluating-policies.md)
  * [26 · Teleoperation, replay and data collection](part-7-learning-from-demonstration/26-teleop-replay-data.md)
  * [27 · Robot datasets](part-7-learning-from-demonstration/27-robot-datasets.md)
  * [28 · Behavior cloning](part-7-learning-from-demonstration/28-behavior-cloning.md)
  * [29 · Distribution shift and DAgger](part-7-learning-from-demonstration/29-dagger.md)
  * [30 · Multimodal actions: diffusion and flow policies](part-7-learning-from-demonstration/30-diffusion-and-flow-policies.md)
  * [31 · Action chunking and ACT](part-7-learning-from-demonstration/31-act.md)

## Part 8 · Reinforcement learning for robots

* [Overview](part-8-rl-for-robots/README.md)
  * [32 · A MuJoCo RL environment](part-8-rl-for-robots/32-rl-environment.md)
  * [33 · PPO for continuous control](part-8-rl-for-robots/33-ppo.md)
  * [34 · Observation and action design](part-8-rl-for-robots/34-observation-action-design.md)
  * [35 · Reward design and reward hacking](part-8-rl-for-robots/35-reward-design.md)
  * [36 · Legged locomotion and massively parallel training](part-8-rl-for-robots/36-legged-locomotion.md)

## Part 9 · Sim to real

* [Overview](part-9-sim-to-real/README.md)
  * [37 · Control rate and end-to-end latency](part-9-sim-to-real/37-latency.md)
  * [38 · System identification](part-9-sim-to-real/38-system-identification.md)
  * [39 · Domain randomization](part-9-sim-to-real/39-domain-randomization.md)
  * [40 · Deploying on the real SO-101](part-9-sim-to-real/40-real-so101.md)

## Part 10 · Foundation models and agents

* [Overview](part-10-foundation-models-and-agents/README.md)
  * [41 · Vision-language-action models](part-10-foundation-models-and-agents/41-vla.md)
  * [42 · LLM agents with robot tools](part-10-foundation-models-and-agents/42-llm-agents.md)
  * [43 · World models: learned dynamics](part-10-foundation-models-and-agents/43-world-models.md)

## Capstone

* [Overview](capstone/README.md)
  * [44 · Say, see, plan, grasp](capstone/44-capstone.md)

## Appendix

* [Appendix](appendix/README.md)
  * [A · Large robot models: π0, GR00T, Gemini Robotics](appendix/a-large-robot-models.md)
  * [B · Fine-tuning SmolVLA](appendix/b-smolvla-finetuning.md)
  * [C · Inference speed on a Mac](appendix/c-inference-speed-on-mac.md)
  * [D · Segmentation for grasp points](appendix/d-segmentation-for-grasping.md)

## Reference

* [Glossary](glossary.md)
* [References](references.md)
* [Notebooks](notebooks/README.md)
* [Contributing](contributing.md)
