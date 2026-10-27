# References

The courses, books, docs and papers the curriculum rests on. Each lesson has its own source list at the end of its page.

## Curriculum backbone

The phases and lesson order were built by studying the common chain shared by these curricula (space and transforms → kinematics → dynamics and control → perception → estimation → learning → sim-to-real).

| Source | What it contributed |
|---|---|
| Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) | Frames, FK/IK, dynamics, trajectories, control, grasping (Phases 1–2) |
| Tedrake, [*Robotic Manipulation*](https://manipulation.mit.edu/) | Spatial algebra, pose estimation, perception (Phases 1, 3) |
| Tedrake, [*Underactuated Robotics*](https://underactuated.mit.edu/) | System identification, state estimation (Phases 3, 7) |
| [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) | Classical robotics → RL → imitation flow |
| Capuano et al., [*Robot Learning: A Tutorial*, arXiv 2510.12403](https://arxiv.org/abs/2510.12403) | RL → behavior cloning → generalist policies |
| [LeRobot docs](https://huggingface.co/docs/lerobot/index) | Calibration → teleop → recording → training → evaluation workflow (Phase 4) |
| [NVIDIA · SO-101 Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) | Sim-to-real strategies (Phase 7) |
| [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | Order of behavior cloning and RL (Phases 4, 6) |
| [MuJoCo Playground](https://playground.mujoco.org/), [arXiv 2502.08844](https://arxiv.org/abs/2502.08844) | Parallel locomotion training on GPU (Phase 6) |
| Rohit Ghumare et al., [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (MIT) | Lesson format and the "from scratch" approach (structural inspiration; no content copied) |

## Tools and models

- [MuJoCo docs](https://mujoco.readthedocs.io/en/stable/) · [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) · [Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `robotstudio_so101` (Apache-2.0), `unitree_go2` (BSD-3-Clause), `boston_dynamics_spot`
- [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) (BSD-3-Clause) · [`legged_gym`](https://github.com/leggedrobotics/legged_gym)
- [`strands-robots`](https://pypi.org/project/strands-robots/) (Apache-2.0)
- [openpi](https://github.com/Physical-Intelligence/openpi)

## Papers

| Short name | Paper |
|---|---|
| ACT | Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, [arXiv 2304.13705](https://arxiv.org/abs/2304.13705) |
| DAgger | Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, [arXiv 1011.0686](https://arxiv.org/abs/1011.0686) |
| Diffusion Policy | Chi et al., [arXiv 2303.04137](https://arxiv.org/abs/2303.04137) |
| Domain randomization | Tobin et al., [arXiv 1703.06907](https://arxiv.org/abs/1703.06907) |
| Sim-to-real locomotion | Tan et al., [arXiv 1804.10332](https://arxiv.org/abs/1804.10332) |
| π0 | Black et al., [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) |
| SmolVLA | Shukor et al., [arXiv 2506.01844](https://arxiv.org/abs/2506.01844) |
| PPO | Schulman et al., [arXiv 1707.06347](https://arxiv.org/abs/1707.06347) |
| MuJoCo Playground | Zakka et al., [arXiv 2502.08844](https://arxiv.org/abs/2502.08844) |

## Physics

- OpenStax, [*University Physics* Volume 1](https://openstax.org/books/university-physics-volume-1/pages/1-introduction) (friction, inclined plane)
- Åström & Murray, [*Feedback Systems*](https://fbswiki.org) (open and closed loop)
- Bishop, *Pattern Recognition and Machine Learning* (2006)
