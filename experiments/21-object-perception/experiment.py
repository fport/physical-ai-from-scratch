"""Lesson 21 · Object perception for manipulation.

A red 3 cm cube lies at a random spot in front of the SO-101 (MuJoCo). A fixed
overhead camera (0.6 m above the table, intrinsics from fovy as in lesson 19)
renders RGB + depth. Perception = HSV threshold -> mask -> back-project the mask
pixels with depth (lesson 20) -> cube (x, y). The arm then picks at that (x, y)
with a small self-contained IK + scripted grasp.

Sweeps:
  1. Pick tolerance: aim error injected around the TRUE position -> pick success.
  2. Light intensity (headlight + directional light scaled) -> detection rate,
     position error, pick success with the PERCEIVED position.
  3. Light direction (headlight off, sun elevation) -> same.

Run:  python experiment.py
"""
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

HALF = 0.015                      # cube half-size (m)
W, H, FOVY = 640, 480, 50.0
CAM_POS = np.array([0.20, 0.0, 0.60])
TCP = np.array([0.012, 0.0, -0.085])   # grasp point in the gripper body frame (between the jaws)
PAN_AXIS_XY = np.array([0.0388353, 0.0])
Q_PARK = np.array([0.0, -1.74, 0.0, 0.0, 0.0, 1.0])   # arm upright, out of the camera's view
OPEN, CLOSE = 1.0, -0.17
N_POS = 30
SEED = 0
FLIP = np.diag([1.0, -1.0, -1.0])

# HSV thresholds (OpenCV: H in 0..179). Tuned once at nominal light, then frozen.
RED_LO = [(0, 120, 70), (170, 120, 70)]
RED_HI = [(10, 255, 255), (179, 255, 255)]


# ---------------- scene ----------------
def build():
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    b = spec.worldbody.add_body(name="cube", pos=[0.2, 0, HALF])
    b.add_freejoint()
    b.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, rgba=[0.85, 0.1, 0.1, 1],
               mass=0.05, friction=[1, 0.005, 0.0001])
    spec.worldbody.add_camera(name="top", pos=CAM_POS.tolist(), fovy=FOVY)
    return spec.compile()


def set_cube(m, d, xy):
    mujoco.mj_resetData(m, d)
    yaw = np.arctan2(xy[1] - PAN_AXIS_XY[1], xy[0] - PAN_AXIS_XY[0])   # cube faces the robot (simplification)
    d.qpos[6:9] = [xy[0], xy[1], HALF]
    d.qpos[9:13] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qpos[:6] = Q_PARK
    d.ctrl[:] = Q_PARK
    mujoco.mj_forward(m, d)


def set_light(m, scale=1.0, elev_deg=None):
    """scale: multiplies headlight and directional light. elev_deg: headlight off, sun at this elevation."""
    m.vis.headlight.diffuse[:] = 0.6 * scale
    m.vis.headlight.ambient[:] = 0.3 * scale
    m.light_diffuse[0] = 0.7 * scale
    m.light_dir[0] = [0, 0, -1]
    if elev_deg is not None:
        m.light_diffuse[0] = 0.7
        m.vis.headlight.diffuse[:] = 0.0
        m.vis.headlight.ambient[:] = 0.1
        e = np.radians(elev_deg)
        m.light_dir[0] = [-np.cos(e), 0.0, -np.sin(e)]     # sun from +x, shining toward the robot


# ---------------- perception ----------------
def intrinsics():
    f = (H / 2) / np.tan(np.radians(FOVY) / 2)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])


def detect(rgb, v_min=70):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    los = [(h, s, v_min) for h, s, _ in RED_LO]
    mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in zip(los, RED_HI)).clip(0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 30:
        return None
    return lab == k


def locate(mask, depth, K, R_wc, c):
    v, u = np.nonzero(mask)
    z = depth[v, u]
    rays = np.c_[u, v, np.ones(len(u))] @ np.linalg.inv(K).T @ R_wc.T
    P = c + z[:, None] * rays
    top = P[P[:, 2] > 0.02]                 # keep the top face (cube height is 0.03 m)
    return (top if len(top) else P)[:, :2].mean(axis=0)


class Camera:
    def __init__(self, m, d):
        self.rgb = mujoco.Renderer(m, H, W)
        self.dep = mujoco.Renderer(m, H, W)
        self.dep.enable_depth_rendering()
        self.K = intrinsics()
        cid = m.camera("top").id
        self.R_wc = d.cam_xmat[cid].reshape(3, 3) @ FLIP
        self.c = d.cam_xpos[cid].copy()

    def perceive(self, d, v_min=70):
        self.rgb.update_scene(d, "top")
        self.dep.update_scene(d, "top")
        img, depth = self.rgb.render(), self.dep.render()
        mask = detect(img, v_min)
        return None if mask is None else locate(mask, depth, self.K, self.R_wc, self.c)


# ---------------- IK + scripted pick ----------------
def ik(m, target, yaw, q0, iters=200):
    """Damped least squares: TCP position + gripper pointing down + jaw axis along yaw."""
    dd = mujoco.MjData(m)
    gid = m.body("gripper").id
    q = q0.copy()
    zd, xd = np.array([0, 0, 1.0]), np.array([np.cos(yaw), np.sin(yaw), 0])
    for _ in range(iters):
        dd.qpos[:6] = q
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_comPos(m, dd)
        R = dd.xmat[gid].reshape(3, 3)
        p = dd.xpos[gid] + R @ TCP
        ep = target - p
        er = np.cross(R[:, 2], zd) + np.cross(R[:, 0], xd)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, dd, jp, jr, p, gid)
        J = np.vstack([jp[:, :5], 0.3 * jr[:, :5]])
        e = np.r_[ep, 0.3 * er]
        q[:5] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q[:5] = np.clip(q[:5], m.jnt_range[:5, 0], m.jnt_range[:5, 1])
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
    return q


def move(m, d, q_from, q_to, T, grip):
    n = int(T / m.opt.timestep)
    for i in range(n):
        a = min(1.0, (i + 1) / (0.8 * n))
        s = a * a * (3 - 2 * a)
        d.ctrl[:5] = (1 - s) * q_from[:5] + s * q_to[:5]
        d.ctrl[5] = grip
        mujoco.mj_step(m, d)


def pick(m, d, aim_xy):
    """Scripted pick at aim_xy. Success = cube lifted more than 5 cm."""
    yaw = np.arctan2(aim_xy[1] - PAN_AXIS_XY[1], aim_xy[0] - PAN_AXIS_XY[0])
    seed = np.array([yaw, 0.0, 0.5, 1.2, 0.0, OPEN])
    q_pre = ik(m, np.r_[aim_xy, HALF + 0.07], yaw, seed)
    q_grasp = ik(m, np.r_[aim_xy, HALF + 0.012], yaw, q_pre)
    q_lift = ik(m, np.r_[aim_xy, HALF + 0.10], yaw, q_grasp)
    move(m, d, Q_PARK, q_pre, 1.0, OPEN)
    move(m, d, q_pre, q_grasp, 0.8, OPEN)
    n = int(0.6 / m.opt.timestep)
    for i in range(n):
        d.ctrl[:5] = q_grasp[:5]
        d.ctrl[5] = OPEN + (CLOSE - OPEN) * min(1.0, (i + 1) / (0.5 * n))
        mujoco.mj_step(m, d)
    move(m, d, q_grasp, q_lift, 1.0, CLOSE)
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return bool(d.qpos[8] > HALF + 0.05)


def top_hsv(m, d, cam, xy):
    """HSV of the pixel that sees the centre of the cube's top face (diagnostic)."""
    set_cube(m, d, xy)
    cam.rgb.update_scene(d, "top")
    hsv = cv2.cvtColor(cam.rgb.render(), cv2.COLOR_RGB2HSV)
    pc = (np.r_[xy, 2 * HALF] - cam.c) @ cam.R_wc
    u, v = np.round((cam.K @ (pc / pc[2]))[:2]).astype(int)
    return "(" + ",".join(f"{x:d}" for x in hsv[v, u]) + ")"


def run_condition(m, d, cam, positions, v_min=70):
    det, errs, ok = 0, [], 0
    for xy in positions:
        set_cube(m, d, xy)
        est = cam.perceive(d, v_min)
        if est is None:
            continue
        det += 1
        errs.append(1000 * np.linalg.norm(est - xy))
        ok += pick(m, d, est)
    med = f"{np.median(errs):.1f}" if errs else "-"
    mx = f"{np.max(errs):.1f}" if errs else "-"
    return det, med, mx, ok


def main():
    print(f"mujoco {mujoco.__version__} · opencv {cv2.__version__} · numpy {np.__version__} · seed {SEED}")
    m = build()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(SEED)
    positions = [np.array([rng.uniform(0.15, 0.24), rng.uniform(-0.07, 0.07)]) for _ in range(N_POS)]
    print(f"{N_POS} cube positions, x in [0.15, 0.24] m, y in [-0.07, 0.07] m; timestep {m.opt.timestep*1000:.0f} ms")

    print("\n[1] pick tolerance: aim = true position + error of fixed size, random direction")
    print(f"{'aim error mm':>13} {'picks ok':>9}")
    for err in [0, 4, 8, 12, 16, 20]:
        ok = 0
        for xy in positions:
            a = rng.uniform(0, 2 * np.pi)
            set_cube(m, d, xy)
            ok += pick(m, d, xy + err / 1000 * np.array([np.cos(a), np.sin(a)]))
        print(f"{err:>13} {ok:>6}/{N_POS}")

    set_cube(m, d, positions[0])
    cam = Camera(m, d)
    print("\n[2] light intensity (headlight and directional light scaled together)")
    print(f"    thresholds: H in [0,10] or [170,179], S >= 120, V >= 70; 'top HSV' = top-face centre, position 1")
    print(f"{'scale':>6} {'top HSV':>14} {'detected':>9} {'median err mm':>14} {'max err mm':>11} {'picks ok':>9}")
    for s in [0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0, 3.0]:
        set_light(m, scale=s)
        hsv = top_hsv(m, d, cam, positions[0])
        det, med, mx, ok = run_condition(m, d, cam, positions)
        print(f"{s:>6} {hsv:>14} {det:>6}/{N_POS} {med:>14} {mx:>11} {ok:>6}/{N_POS}")

    print("\n[3] light direction (headlight off, ambient 0.1, sun from +x at elevation e)")
    print(f"{'elev deg':>9} {'top HSV':>14} {'detected':>9} {'median err mm':>14} {'max err mm':>11} {'picks ok':>9}")
    for e in [90, 45, 20, 10, 5]:
        set_light(m, elev_deg=e)
        hsv = top_hsv(m, d, cam, positions[0])
        det, med, mx, ok = run_condition(m, d, cam, positions)
        print(f"{e:>9} {hsv:>14} {det:>6}/{N_POS} {med:>14} {mx:>11} {ok:>6}/{N_POS}")

    print("\n[4] same light directions, value threshold lowered from 70 to 25")
    print(f"{'elev deg':>9} {'detected':>9} {'median err mm':>14} {'max err mm':>11} {'picks ok':>9}")
    for e in [20, 10, 5]:
        set_light(m, elev_deg=e)
        det, med, mx, ok = run_condition(m, d, cam, positions, v_min=25)
        print(f"{e:>9} {det:>6}/{N_POS} {med:>14} {mx:>11} {ok:>6}/{N_POS}")
    set_light(m, scale=1.0)
    det, med, mx, ok = run_condition(m, d, cam, positions, v_min=25)
    print(f"{'nominal':>9} {det:>6}/{N_POS} {med:>14} {mx:>11} {ok:>6}/{N_POS}   (scale 1.0, headlight on)")


if __name__ == "__main__":
    main()
