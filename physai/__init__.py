"""physai: the small robotics library you write while working through the course.

Every function here is built step by step in a lesson's "Build it" section and checked against
MuJoCo, SciPy or OpenCV there (and in tests/). Later lessons import it instead of re-deriving the math.

    sim          inertia, integrators, soft contact, friction cones, actuators   (lessons 01-05)
    spatial      quaternions, Euler angles, slerp, rigid transforms, frame trees (06-07)
    kinematics   forward kinematics, workspace, Jacobians, damped IK              (08-10)
    dynamics     spatial inertia, recursive Newton-Euler, mass matrix            (11)
    control      PD/PID with gravity feedforward, task-space impedance           (12, 14)
    trajectory   cubic, quintic and trapezoid time scalings                      (13)
    safety       goal clamps, servo torque limits, watchdog, e-stop              (15)
    grasp        friction-cone margin, squeeze from the torque cap, pick machine (16)
    estimation   encoder model, filters, Kalman filter                           (17-18)
    vision       pinhole model, calibration, homography, back-projection, color  (19-21)
    planning     collision-checked edges, RRT-Connect, shortcutting              (22)
    optimal      collocation, shooting, LQR, MPPI                                (23-24)
    evaluation   Wilson/Wald intervals, coverage, Fisher test, power             (25)
    data         episodes, splits, normalization statistics, replay/retarget     (26-27)
    imitation    behavior cloning, DAgger, flow matching, action chunking        (28-31)
    rl           a MuJoCo reach task, GAE, PPO loss, action and reward design    (32-35)
    vecenv       batched environments with auto-reset                            (36)
    sim2real     delay lines and margins, system identification, randomization   (37-40)
    foundation   action tokenization, tool-call validation, world models         (41-43)
    capstone     the say -> see -> plan -> grasp pipeline from the modules above  (44)

Use from the repo root:  uv run python -c "import physai"
Scripts in experiments/ add the repo root to sys.path first.
"""
