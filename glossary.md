# Glossary

Each term links to the lesson where it first appears. The glossary grows as lessons are written.

| Term | In short | First appears |
|---|---|---|
| ACT (Action Chunking with Transformers) | An imitation policy that predicts a chunk of actions instead of a single next action | [24](phase-4-learning-from-demonstration/24-two-right-answers.md), [25](phase-4-learning-from-demonstration/25-action-chunking.md) |
| Behavior cloning (BC) | Learning a state → action mapping from demonstrations with supervised learning | [21](phase-4-learning-from-demonstration/21-behavior-cloning.md) |
| `body` / `geom` | In MJCF: a moving frame / the collision and mass shape attached to it | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| Contact normal | The direction the engine picks to push two bodies apart | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| Creep | Slow, steady sliding below the slip threshold caused by soft contact | [04](phase-0-world-frame-by-frame/04-friction.md) |
| Decimation | In RL training, how many physics steps pass between policy decisions | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| Domain randomization | Randomizing sim parameters during training to make the policy robust | [04](phase-0-world-frame-by-frame/04-friction.md), [37](phase-7-sim-to-real/37-domain-randomization.md) |
| FK (forward kinematics) | Computing the tip position from joint angles | [07](phase-1-space-and-body/07-forward-kinematics.md) |
| `freejoint` | A joint that frees a body with 6 degrees of freedom | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| Friction cone | The set of contact-force directions that friction can sustain | [04](phase-0-world-frame-by-frame/04-friction.md) |
| IK (inverse kinematics) | Finding the joint angles that reach a desired tip position | [09](phase-1-space-and-body/09-inverse-kinematics.md) |
| Kp / Kd | Gains that turn position error and velocity into torque in PD control | [11](phase-2-muscle/11-pd-control.md) |
| MDN (mixture density network) | A network whose output is several possible answers with weights | [24](phase-4-learning-from-demonstration/24-two-right-answers.md) |
| MJCF | MuJoCo's XML format for describing a world | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| μ (friction coefficient) | Ratio of the friction limit to the normal force; the tangent of the slip angle | [04](phase-0-world-frame-by-frame/04-friction.md) |
| Open-loop / closed-loop | Applying commands without measuring the output / adjusting commands based on what is measured | [19](phase-4-learning-from-demonstration/19-replay.md) |
| Overshoot | How far past the target (or the settled value) the response goes | [11](phase-2-muscle/11-pd-control.md) |
| Replay | Sending recorded actions again without changing them | [19](phase-4-learning-from-demonstration/19-replay.md) |
| Semi-implicit Euler | An integrator that updates velocity first, then position with the new velocity | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| `timestep` (dt) | How far the physics engine advances time in one step | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| Tool calling | A language model calling a function by name with arguments | [31](phase-5-agents/31-strands-agent.md) |
| Tunneling | A fast body crossing a thin obstacle in a single step | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| URDF | The ROS world's robot description format | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| VLA (vision-language-action) | A model that produces actions from images and a sentence | [38](phase-8-big-models/38-vla.md) |
