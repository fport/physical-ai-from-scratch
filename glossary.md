# Glossary

Each term links to the lesson that teaches it.

| Term | In short | Lesson |
|---|---|---|
| ACT (Action Chunking with Transformers) | Imitation policy that predicts a chunk of future actions with a CVAE-trained transformer | [31](part-7-learning-from-demonstration/31-act.md) |
| Action chunking | Predicting several future actions at once and executing them before re-querying | [31](part-7-learning-from-demonstration/31-act.md) |
| Antipodal grasp | Two contacts whose normals point at each other, inside both friction cones | [16](part-4-dynamics-and-control/16-grasping-and-pick-place.md) |
| Armature | Reflected rotor inertia added to a joint; stabilizes stiff position control | [05](part-1-simulation/05-joints-and-actuators.md) |
| Behavior cloning (BC) | Supervised learning of observation → action from demonstrations | [28](part-7-learning-from-demonstration/28-behavior-cloning.md) |
| `body` / `geom` | In MJCF: a moving frame / a collision and mass shape attached to it | [01](part-1-simulation/01-world-as-text.md) |
| Collocation | Trajectory optimization with states and controls as decision variables tied by dynamics constraints | [23](part-6-planning-and-optimal-control/23-trajectory-optimization.md) |
| Configuration space (C-space) | The space of all joint configurations; obstacles become forbidden regions in it | [22](part-6-planning-and-optimal-control/22-motion-planning.md) |
| Contact normal | The direction the engine uses to push two bodies apart | [02](part-1-simulation/02-timestep.md) |
| Creep | Slow steady sliding below the slip threshold caused by soft contact | [04](part-1-simulation/04-friction.md) |
| DAgger | Dataset aggregation: label the states the learner visits with an expert, retrain | [29](part-7-learning-from-demonstration/29-dagger.md) |
| Damped least squares (DLS) | IK update $(J^\top J + \lambda^2 I)^{-1} J^\top e$ that stays bounded near singularities | [10](part-3-kinematics/10-inverse-kinematics.md) |
| Decimation | Number of physics steps between two policy decisions | [34](part-8-rl-for-robots/34-observation-action-design.md) |
| Domain randomization | Training across randomized physics so the policy covers the real system | [39](part-9-sim-to-real/39-domain-randomization.md) |
| Flow matching | Generative model that learns a velocity field carrying noise to data; used for action generation | [30](part-7-learning-from-demonstration/30-diffusion-and-flow-policies.md) |
| Forward kinematics (FK) | Joint angles → pose of a frame on the robot | [08](part-3-kinematics/08-forward-kinematics.md) |
| Friction cone | Set of contact forces friction can sustain without slipping | [04](part-1-simulation/04-friction.md) |
| Gimbal lock | Loss of one rotational degree of freedom in an Euler-angle parameterization | [06](part-2-spatial-math/06-rotations.md) |
| Gravity feedforward | Adding the model's gravity torque to the controller output to remove sag | [12](part-4-dynamics-and-control/12-joint-control.md) |
| Homogeneous transform | 4×4 matrix combining rotation and translation; composes frame changes | [07](part-2-spatial-math/07-rigid-transforms.md) |
| Homography | 3×3 projective map between an image and a plane | [20](part-5-perception-and-estimation/20-pixels-to-table.md) |
| Impedance control | Making the end effector behave like a spring–damper around a target | [14](part-4-dynamics-and-control/14-impedance-control.md) |
| Intrinsics | Camera matrix $K$: focal lengths and principal point | [19](part-5-perception-and-estimation/19-camera-calibration.md) |
| Inverse kinematics (IK) | Finding joint angles that reach a desired pose | [10](part-3-kinematics/10-inverse-kinematics.md) |
| Jacobian | Matrix mapping joint velocities to end-effector velocity | [09](part-3-kinematics/09-jacobians.md) |
| Kalman filter | Recursive optimal state estimator for linear-Gaussian systems | [18](part-5-perception-and-estimation/18-kalman-filter.md) |
| Kp / Kd | Proportional and derivative gains of a PD controller | [12](part-4-dynamics-and-control/12-joint-control.md) |
| LQR | Optimal linear state feedback for linear dynamics and quadratic cost | [24](part-6-planning-and-optimal-control/24-lqr-and-mpc.md) |
| Manipulability | Measure of how well an arm can move in all directions at a configuration | [10](part-3-kinematics/10-inverse-kinematics.md) |
| MJCF | MuJoCo's XML format for describing a world | [01](part-1-simulation/01-world-as-text.md) |
| MPC | Re-solving a finite-horizon optimal control problem at every step | [24](part-6-planning-and-optimal-control/24-lqr-and-mpc.md) |
| Operational-space control | Task-space control using the end-effector inertia $\Lambda = (J M^{-1} J^\top)^{-1}$ | [14](part-4-dynamics-and-control/14-impedance-control.md) |
| Overshoot | How far the response goes past its final value | [12](part-4-dynamics-and-control/12-joint-control.md) |
| PPO | Policy-gradient algorithm with a clipped surrogate objective | [33](part-8-rl-for-robots/33-ppo.md) |
| Quaternion | Unit 4-vector representing a rotation; MuJoCo stores it w-first | [06](part-2-spatial-math/06-rotations.md) |
| Replay | Sending recorded actions again without looking at the world (open loop) | [26](part-7-learning-from-demonstration/26-teleop-replay-data.md) |
| Reward shaping | Adding terms to the reward to guide learning; potential-based shaping keeps the optimum | [35](part-8-rl-for-robots/35-reward-design.md) |
| RRT / RRT-Connect | Sampling-based motion planners that grow random trees in C-space | [22](part-6-planning-and-optimal-control/22-motion-planning.md) |
| Semi-implicit Euler | Integrator that updates velocity first, then position with the new velocity | [02](part-1-simulation/02-timestep.md) |
| Singularity | Configuration where the Jacobian loses rank | [10](part-3-kinematics/10-inverse-kinematics.md) |
| `solref` / `solimp` | MuJoCo contact stiffness/damping and impedance parameters | [03](part-1-simulation/03-contact.md) |
| System identification | Fitting model parameters to measured data | [38](part-9-sim-to-real/38-system-identification.md) |
| Temporal ensembling | Averaging overlapping action chunks with exponential weights | [31](part-7-learning-from-demonstration/31-act.md) |
| `timestep` (dt) | How far the physics engine advances time in one step | [02](part-1-simulation/02-timestep.md) |
| Tunneling | A fast body crossing a thin obstacle between two steps | [02](part-1-simulation/02-timestep.md) |
| URDF | ROS robot description format | [01](part-1-simulation/01-world-as-text.md) |
| VLA | Vision-language-action model: images + language + state → actions | [41](part-10-foundation-models-and-agents/41-vla.md) |
| Watchdog | Timer that stops the robot when commands stop arriving | [15](part-4-dynamics-and-control/15-safety.md) |
| Wilson interval | Confidence interval for a success rate from k successes in n trials | [25](part-7-learning-from-demonstration/25-evaluating-policies.md) |
| World model | Learned model that predicts the next state from state and action | [43](part-10-foundation-models-and-agents/43-world-models.md) |
