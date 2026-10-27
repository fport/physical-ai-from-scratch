# Kaynakça

Müfredatın dayandığı kurslar, kitaplar, belgeler ve makaleler. Her dersin kendi kaynak listesi sayfasının sonunda.

## Müfredat iskeleti

Bu kitabın bölümleri ve ders sırası, aşağıdaki müfredatların ortak zinciri incelenerek kuruldu (space & transforms → kinematics → dynamics & control → perception → estimation → learning → sim-to-real).

| Kaynak | Bu kitaba kattığı |
|---|---|
| Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) | Frames, FK/IK, dynamics, trajectory, control, grasping (Bölüm 2–4) |
| Tedrake, [*Robotic Manipulation*](https://manipulation.mit.edu/) | Spatial algebra, pose estimation, perception (Bölüm 2, 5) |
| Tedrake, [*Underactuated Robotics*](https://underactuated.mit.edu/) | System identification, state estimation (Bölüm 5, 6, 9) |
| [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) | Classical robotics → RL → imitation akışı |
| Capuano vd., [*Robot Learning: A Tutorial*, arXiv 2510.12403](https://arxiv.org/abs/2510.12403) | RL → behavior cloning → generalist policy'ler |
| [LeRobot belgeleri](https://huggingface.co/docs/lerobot/index) | Calibration → teleop → recording → training → evaluation workflow'u (Bölüm 7, 9) |
| [NVIDIA · SO-101 Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) | Sim-to-real stratejileri (Bölüm 9) |
| [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | Behavior cloning ve RL sırası (Bölüm 7, 8) |
| [MuJoCo Playground](https://playground.mujoco.org/), [arXiv 2502.08844](https://arxiv.org/abs/2502.08844) | GPU'da parallel locomotion training (Bölüm 8) |
| Rohit Ghumare vd., [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (MIT) | Ders formatı ve "sıfırdan" yaklaşımı (yapı ilhamı; içerik kopyalanmadı) |

## Araçlar ve modeller

- [MuJoCo belgeleri](https://mujoco.readthedocs.io/en/stable/) · [XML referansı](https://mujoco.readthedocs.io/en/stable/XMLreference.html) · [Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `robotstudio_so101` (Apache-2.0), `unitree_go2` (BSD-3-Clause), `boston_dynamics_spot`
- [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) (BSD-3-Clause) · [`legged_gym`](https://github.com/leggedrobotics/legged_gym)
- [`strands-robots`](https://pypi.org/project/strands-robots/) (Apache-2.0)
- [openpi](https://github.com/Physical-Intelligence/openpi)

## Makaleler

| Kısaltma | Makale |
|---|---|
| ACT | Zhao vd., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, [arXiv 2304.13705](https://arxiv.org/abs/2304.13705) |
| DAgger | Ross vd., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, [arXiv 1011.0686](https://arxiv.org/abs/1011.0686) |
| Diffusion Policy | Chi vd., [arXiv 2303.04137](https://arxiv.org/abs/2303.04137) |
| Domain randomization | Tobin vd., [arXiv 1703.06907](https://arxiv.org/abs/1703.06907) |
| Sim-to-real locomotion | Tan vd., [arXiv 1804.10332](https://arxiv.org/abs/1804.10332) |
| π0 | Black vd., [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) |
| SmolVLA | Shukor vd., [arXiv 2506.01844](https://arxiv.org/abs/2506.01844) |
| PPO | Schulman vd., [arXiv 1707.06347](https://arxiv.org/abs/1707.06347) |
| MuJoCo Playground | Zakka vd., [arXiv 2502.08844](https://arxiv.org/abs/2502.08844) |

## Fizik

- OpenStax, [*University Physics* Cilt 1](https://openstax.org/books/university-physics-volume-1/pages/1-introduction) (sürtünme, eğik düzlem)
- Åström & Murray, [*Feedback Systems*](https://fbswiki.org) (open-loop ve closed-loop)
- Bishop, *Pattern Recognition and Machine Learning* (2006)
