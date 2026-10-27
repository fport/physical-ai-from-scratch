# Table of contents

* [Giriş](README.md)

## Başlarken

* [Physical AI stack'i](start-here/physical-ai-stack.md)
* [Önkoşullar](start-here/prerequisites.md)
* [Kurulum](start-here/setup.md)
* [Bir ders nasıl okunur](start-here/how-to-read-a-lesson.md)
* [Sayı ve kaynak kuralları](start-here/numbers-and-sources.md)
* [Okuma yolları](start-here/reading-paths.md)
* [Videolar ve dersler](start-here/video-map.md)

## Bölüm 1 · Simulation

* [Genel bakış](part-1-simulation/README.md)
  * [01 · Dünya bir metin dosyası: MJCF ve URDF](part-1-simulation/01-world-as-text.md)
  * [02 · Time integration ve timestep](part-1-simulation/02-timestep.md)
  * [03 · Contact: soft constraint, solref ve solimp](part-1-simulation/03-contact.md)
  * [04 · Friction ve friction cone](part-1-simulation/04-friction.md)
  * [05 · Joint'ler ve actuator'ler](part-1-simulation/05-joints-and-actuators.md)

## Bölüm 2 · Spatial math

* [Genel bakış](part-2-spatial-math/README.md)
  * [06 · Rotation: matrix, Euler açıları, axis-angle, quaternion](part-2-spatial-math/06-rotations.md)
  * [07 · Rigid transform'lar ve frame tree'leri](part-2-spatial-math/07-rigid-transforms.md)

## Bölüm 3 · Kinematics

* [Genel bakış](part-3-kinematics/README.md)
  * [08 · Forward kinematics, degrees of freedom ve workspace](part-3-kinematics/08-forward-kinematics.md)
  * [09 · Jacobian ve differential kinematics](part-3-kinematics/09-jacobians.md)
  * [10 · Inverse kinematics ve singularity](part-3-kinematics/10-inverse-kinematics.md)

## Bölüm 4 · Dynamics ve control

* [Genel bakış](part-4-dynamics-and-control/README.md)
  * [11 · Rigid-body dynamics](part-4-dynamics-and-control/11-rigid-body-dynamics.md)
  * [12 · Joint control: PD, PID, gravity feedforward ve position servo](part-4-dynamics-and-control/12-joint-control.md)
  * [13 · Trajectory generation](part-4-dynamics-and-control/13-trajectory-generation.md)
  * [14 · Operational-space ve impedance control](part-4-dynamics-and-control/14-impedance-control.md)
  * [15 · Safety: limit'ler, torque cap, e-stop ve watchdog](part-4-dynamics-and-control/15-safety.md)
  * [16 · Grasping ve scripted pick-and-place](part-4-dynamics-and-control/16-grasping-and-pick-place.md)

## Bölüm 5 · Perception ve estimation

* [Genel bakış](part-5-perception-and-estimation/README.md)
  * [17 · Sensor'ler, noise ve filtering](part-5-perception-and-estimation/17-sensors-noise-filtering.md)
  * [18 · Kalman filter](part-5-perception-and-estimation/18-kalman-filter.md)
  * [19 · Camera model, calibration ve hand-eye calibration](part-5-perception-and-estimation/19-camera-calibration.md)
  * [20 · Pixel'den masaya: homography ve depth back-projection](part-5-perception-and-estimation/20-pixels-to-table.md)
  * [21 · Manipulation için object perception](part-5-perception-and-estimation/21-object-perception.md)

## Bölüm 6 · Planning ve optimal control

* [Genel bakış](part-6-planning-and-optimal-control/README.md)
  * [22 · Configuration space ve sampling-based planning](part-6-planning-and-optimal-control/22-motion-planning.md)
  * [23 · Trajectory optimization](part-6-planning-and-optimal-control/23-trajectory-optimization.md)
  * [24 · LQR ve model predictive control](part-6-planning-and-optimal-control/24-lqr-and-mpc.md)

## Bölüm 7 · Learning from demonstration

* [Genel bakış](part-7-learning-from-demonstration/README.md)
  * [25 · Robot policy'lerini evaluate etmek](part-7-learning-from-demonstration/25-evaluating-policies.md)
  * [26 · Teleoperation, replay ve data collection](part-7-learning-from-demonstration/26-teleop-replay-data.md)
  * [27 · Robot dataset'leri](part-7-learning-from-demonstration/27-robot-datasets.md)
  * [28 · Behavior cloning](part-7-learning-from-demonstration/28-behavior-cloning.md)
  * [29 · Distribution shift ve DAgger](part-7-learning-from-demonstration/29-dagger.md)
  * [30 · Multimodal action'lar: diffusion ve flow policy'ler](part-7-learning-from-demonstration/30-diffusion-and-flow-policies.md)
  * [31 · Action chunking ve ACT](part-7-learning-from-demonstration/31-act.md)

## Bölüm 8 · Robotlar için reinforcement learning

* [Genel bakış](part-8-rl-for-robots/README.md)
  * [32 · Bir MuJoCo RL environment'ı](part-8-rl-for-robots/32-rl-environment.md)
  * [33 · Continuous control için PPO](part-8-rl-for-robots/33-ppo.md)
  * [34 · Observation ve action tasarımı](part-8-rl-for-robots/34-observation-action-design.md)
  * [35 · Reward tasarımı ve reward hacking](part-8-rl-for-robots/35-reward-design.md)
  * [36 · Legged locomotion ve massively parallel training](part-8-rl-for-robots/36-legged-locomotion.md)

## Bölüm 9 · Sim-to-real

* [Genel bakış](part-9-sim-to-real/README.md)
  * [37 · Control rate ve end-to-end latency](part-9-sim-to-real/37-latency.md)
  * [38 · System identification](part-9-sim-to-real/38-system-identification.md)
  * [39 · Domain randomization](part-9-sim-to-real/39-domain-randomization.md)
  * [40 · Gerçek SO-101'e deploy etmek](part-9-sim-to-real/40-real-so101.md)

## Bölüm 10 · Foundation model'ler ve agent'lar

* [Genel bakış](part-10-foundation-models-and-agents/README.md)
  * [41 · Vision-language-action model'leri](part-10-foundation-models-and-agents/41-vla.md)
  * [42 · Robot tool'ları olan LLM agent'ları](part-10-foundation-models-and-agents/42-llm-agents.md)
  * [43 · World model'ler: öğrenilmiş dynamics](part-10-foundation-models-and-agents/43-world-models.md)

## Bitirme projesi

* [Genel bakış](capstone/README.md)
  * [44 · Söyle, gör, planla, tut](capstone/44-capstone.md)

## Ekler

* [Ekler](appendix/README.md)
  * [A · Büyük robot model'leri: π0, GR00T, Gemini Robotics](appendix/a-large-robot-models.md)
  * [B · SmolVLA fine-tuning](appendix/b-smolvla-finetuning.md)
  * [C · Mac'te inference hızı](appendix/c-inference-speed-on-mac.md)
  * [D · Grasp point için segmentation](appendix/d-segmentation-for-grasping.md)

## Başvuru

* [Sözlük](glossary.md)
* [Kaynakça](references.md)
* [Notebook'lar](notebooks/README.md)
* [Katkı](contributing.md)
