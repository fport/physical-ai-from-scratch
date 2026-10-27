"""Lesson 07 · Build it: 4x4 transforms and a frame-tree walk from scratch, checked against MuJoCo.

The model is the Experiment's: the SO-101 with its base moved to (0.10, -0.05, 0) and yawed 30 deg, plus a free cube.

Step 1  transform, transform_inv, composition vs np.linalg.inv, mju_negPose and mju_mulPose.
Step 2  Our frame-tree walk (static model data + qpos) vs xpos / xmat for every body.
Step 3  Sites, cameras and inertial frames hang off bodies: vs site_xmat, cam_xmat, ximat.
Step 4  The cube in every frame vs the framepos sensors (the Experiment's Result 1).
Step 5  Pixels, and the Break it mistakes, from our own transforms.

The functions live in physai/spatial.py (the lesson shows them step by step).
Run:  uv run python experiments/07-rigid-transforms/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.spatial import frame_tree, pose_to_transform, transform_inv, transform_point  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
CUBE_W = np.array([0.2905, 0.0600, 0.015])
Q_ARM = np.array([0.13, -1.65, 1.15, 1.33, -2.13])
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def build():
    """The Experiment's model: base moved and yawed, a free cube, four framepos sensors."""
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101", "so101.xml"))
    base = spec.body("base")
    base.pos = [0.10, -0.05, 0.0]
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0, 0, np.radians(30)]), "xyz")
    base.quat = q
    cube = spec.worldbody.add_body(name="cube")
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015] * 3)
    for name, rt, rn in [("in_base", mujoco.mjtObj.mjOBJ_XBODY, "base"),
                         ("in_base_inertial", mujoco.mjtObj.mjOBJ_BODY, "base"),
                         ("in_gripper", mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
                         ("in_camera", mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")]:
        spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_XBODY, objname="cube", reftype=rt, refname=rn)
    return spec.compile()


m = build()
d = mujoco.MjData(m)
CUBE, SITE, CAM, BASE = m.body("cube").id, m.site("gripperframe").id, m.camera("wrist_cam").id, m.body("base").id


def set_pose(q_arm):
    d.qpos[:] = m.qpos0
    for j, v in zip(ARM, q_arm):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    a = m.jnt_qposadr[m.body_jntadr[CUBE]]
    d.qpos[a:a + 7] = np.r_[CUBE_W, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)


def mj_pose(fn, *args):
    pos, quat = np.zeros(3), np.zeros(4)
    fn(pos, quat, *args)
    return pose_to_transform(pos, quat)


print("\n== Step 1. Transforms: inverse and composition (1000 random poses) ==")
rng = np.random.default_rng(0)
poses = []
for _ in range(1000):
    q = rng.normal(size=4)
    poses.append((rng.uniform(-1, 1, 3), q / np.linalg.norm(q)))
e_inv = max(np.abs(transform_inv(pose_to_transform(p, q)) - np.linalg.inv(pose_to_transform(p, q))).max()
            for p, q in poses)
e_neg = max(np.abs(transform_inv(pose_to_transform(p, q)) - mj_pose(mujoco.mju_negPose, p, q)).max() for p, q in poses)
e_mul = max(np.abs(pose_to_transform(*a) @ pose_to_transform(*b) - mj_pose(mujoco.mju_mulPose, *a, *b)).max()
            for a, b in zip(poses, poses[1:] + poses[:1]))
print(f"transform_inv vs np.linalg.inv   {e_inv:.1e}")
print(f"transform_inv vs mju_negPose     {e_neg:.1e}")
print(f"T_a @ T_b vs mju_mulPose         {e_mul:.1e}")

print("\n== Step 2. Frame-tree walk vs MuJoCo's xpos / xmat (every body, 50 random arm poses) ==")
lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
e_pos = e_rot = 0.0
for q in rng.uniform(lo, hi, size=(50, 5)):
    set_pose(q)
    T = frame_tree(m, d.qpos)
    e_pos = max(e_pos, max(np.abs(T[b][:3, 3] - d.xpos[b]).max() for b in range(m.nbody)))
    e_rot = max(e_rot, max(np.abs(T[b][:3, :3] - d.xmat[b].reshape(3, 3)).max() for b in range(m.nbody)))
print(f"{m.nbody} bodies (world, 8 SO-101 links, the free cube) · max |pos error| {e_pos:.1e} m · "
      f"max |rotation error| {e_rot:.1e}")

print("\n== Step 3. Frames hanging off bodies: site, camera, inertial frame ==")
set_pose(Q_ARM)
T = frame_tree(m, d.qpos)
T_WG = T[m.site_bodyid[SITE]] @ pose_to_transform(m.site_pos[SITE], m.site_quat[SITE])
T_WC = T[m.cam_bodyid[CAM]] @ pose_to_transform(m.cam_pos[CAM], m.cam_quat[CAM])
T_WI = T[BASE] @ pose_to_transform(m.body_ipos[BASE], m.body_iquat[BASE])     # the base's inertial frame
for name, Tw, pos, mat in (("site gripperframe", T_WG, d.site_xpos[SITE], d.site_xmat[SITE]),
                           ("camera wrist_cam", T_WC, d.cam_xpos[CAM], d.cam_xmat[CAM]),
                           ("base inertial frame", T_WI, d.xipos[BASE], d.ximat[BASE])):
    err = max(np.abs(Tw[:3, 3] - pos).max(), np.abs(Tw[:3, :3] - mat.reshape(3, 3)).max())
    print(f"{name:20s} max error {err:.1e}")

print("\n== Step 4. The cube in each frame: our transforms vs the framepos sensors ==")
p_w = T[CUBE][:3, 3]
frames = {"base B": (T[BASE], "in_base"), "gripper G": (T_WG, "in_gripper"), "camera C": (T_WC, "in_camera"),
          "base INERTIAL frame": (T_WI, "in_base_inertial")}
cube_in = {}
for name, (Tw, sensor) in frames.items():
    cube_in[name] = transform_point(transform_inv(Tw), p_w)
    print(f"{name:20s} {' '.join(f'{v:7.4f}' for v in cube_in[name])} m · "
          f"max |ours - sensor| {np.abs(cube_in[name] - d.sensor(sensor).data).max():.1e}")
print(f"inertial frame vs regular frame of the base: {1000 * np.linalg.norm(cube_in['base INERTIAL frame'] - cube_in['base B']):.1f} mm apart")

print("\n== Step 5. Use the transforms: pixels, and five ways to get the cube into the gripper frame ==")
x, y, z = cube_in["camera C"]
f_px = 240 / np.tan(np.radians(m.cam_fovy[CAM]) / 2)       # 640 x 480 image, fovy is the vertical field of view
print(f"f = {f_px:.1f} px · cube centre at u = {320 + f_px * x / -z:.1f}, v = {240 - f_px * y / -z:.1f} px "
      "(camera looks along -z, image v grows downward)")
R, t = T_WG[:3, :3], T_WG[:3, 3]
T_BG = transform_inv(T[BASE]) @ T_WG
truth = d.sensor("in_gripper").data
for label, guess in (("T_WG^-1 p (correct)", transform_point(transform_inv(T_WG), p_w)),
                     ("R (p - t): R instead of R^T", R @ (p_w - t)),
                     ("(T_BG T_WB)^-1 p: wrong order", transform_point(transform_inv(T_BG @ T[BASE]), p_w)),
                     ("R^T p: forgot the translation", R.T @ p_w),
                     ("T_WG p: transform, not inverse", transform_point(T_WG, p_w))):
    print(f"{label:32s} error {1000 * np.linalg.norm(guess - truth):6.1f} mm")
print(f"chain: max |T_WB @ T_BG - T_WG| = {np.abs(T[BASE] @ T_BG - T_WG).max():.1e}")
