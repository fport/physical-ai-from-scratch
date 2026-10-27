"""Lesson 20 · From pixels to the table: homography and depth back-projection.

A MuJoCo camera looks at a table (tilted 30 deg from vertical). For a pixel (u, v)
we want the 3D point it sees. Two methods:
  1. Homography H (table plane <-> image), fitted with cv2.findHomography from
     known table points. Only valid for points ON the plane.
  2. Back-projection with the rendered depth image: X = Z K^-1 [u, v, 1]^T, then
     camera -> world with the camera pose.
Ground truth for a pixel = the intersection of its ray with the surface it sees.

Sweeps: pixel noise (homography, 4 vs 12 points); camera height; object height
above the table (break it: the homography's plane assumption).

Run:  python experiment.py
"""
import cv2
import mujoco
import numpy as np

W, H, FOVY = 640, 480, 50.0
TILT = np.radians(30)            # camera optical axis vs vertical
TARGET = np.array([0.25, 0.0, 0.0])
FLIP = np.diag([1.0, -1.0, -1.0])
SEED = 0
BOX_H = [0.01, 0.02, 0.05, 0.10]           # box heights for the break-it (m)
BOX_XY = [(0.17, -0.07), (0.21, 0.07), (0.30, -0.07), (0.34, 0.07)]


def scene_xml(h):
    cx = TARGET[0] - h * np.tan(TILT)
    f = TARGET - np.array([cx, 0, h])
    f /= np.linalg.norm(f)
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(-f, x)
    boxes = "".join(
        f'<geom type="box" size="0.02 0.02 {bh/2}" pos="{bx} {by} {bh/2}" rgba="0.8 0.2 0.2 1"/>'
        for bh, (bx, by) in zip(BOX_H, BOX_XY))
    return f"""
<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="table" type="plane" size="1 1 0.01" rgba="0.7 0.7 0.7 1"/>
    {boxes}
    <camera name="cam" pos="{cx} 0 {h}" xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" fovy="{FOVY}"/>
  </worldbody>
</mujoco>"""


def setup(h):
    m = mujoco.MjModel.from_xml_string(scene_xml(h))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W)
    r.enable_depth_rendering()
    r.update_scene(d, camera="cam")
    depth = r.render().copy()
    r.close()
    f = (H / 2) / np.tan(np.radians(FOVY) / 2)
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    R_wc = d.cam_xmat[0].reshape(3, 3) @ FLIP          # OpenCV camera axes in world
    return m, depth, K, R_wc, d.cam_xpos[0].copy()


def rays(uv, K, R_wc):
    """World-frame ray directions with unit z-depth in the camera frame."""
    p = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T     # [x/z, y/z, 1]
    return p @ R_wc.T


def hit_plane(uv, K, R_wc, c, z0=0.0):
    r = rays(uv, K, R_wc)
    s = (z0 - c[2]) / r[:, 2]
    return c + s[:, None] * r, s                 # s = z-depth (along optical axis)


def project(P, K, R_wc, c):
    pc = (P - c) @ R_wc
    return (pc[:, :2] / pc[:, 2:3]) @ K[:2, :2].T + K[:2, 2]


def bilinear(img, uv):
    return cv2.remap(img, uv[:, 0].astype(np.float32).reshape(-1, 1),
                     uv[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()


def backproject(uv, depth, K, R_wc, c):
    z = bilinear(depth, uv)
    return c + z[:, None] * rays(uv, K, R_wc)


def table_pixels(depth, K, R_wc, c, rng, n):
    """Random integer pixels that see the bare table (not a box)."""
    out = []
    while len(out) < n:
        uv = np.array([[rng.integers(20, W - 20), rng.integers(20, H - 20)]], float)
        P, s = hit_plane(uv, K, R_wc, c)
        if abs(depth[int(uv[0, 1]), int(uv[0, 0])] - s[0]) < 1e-3 and np.all(np.abs(P[0, :2] - [0.25, 0]) < 0.4):
            out.append(uv[0])
    return np.array(out)


def fit_h(table_xy, K, R_wc, c, rng, sigma):
    P = np.c_[table_xy, np.zeros(len(table_xy))]
    uv = project(P, K, R_wc, c) + rng.normal(0, sigma, (len(P), 2))
    Hm, _ = cv2.findHomography(uv.astype(np.float32), table_xy.astype(np.float32), 0)
    return Hm


def apply_h(Hm, uv):
    p = np.c_[uv, np.ones(len(uv))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


MARK4 = np.array([[0.15, -0.10], [0.35, -0.10], [0.35, 0.10], [0.15, 0.10]])
MARK12 = np.array([[x, y] for x in [0.15, 0.20, 0.25, 0.30, 0.35] for y in [-0.10, 0.10]] + [[0.25, -0.05], [0.25, 0.05]])


def main():
    print(f"mujoco {mujoco.__version__} · opencv {cv2.__version__} · numpy {np.__version__} · seed {SEED}")
    rng = np.random.default_rng(SEED)

    # 1. what the depth image contains
    h = 0.6
    m, depth, K, R_wc, c = setup(h)
    ext = m.stat.extent
    print(f"\n[1] camera height {h} m, tilt 30 deg; model extent {ext:.3f} m -> near {m.vis.map.znear*ext*1000:.1f} mm, "
          f"far {m.vis.map.zfar*ext:.1f} m")
    uv = table_pixels(depth, K, R_wc, c, rng, 300)
    P_true, s = hit_plane(uv, K, R_wc, c)
    zimg = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    ray_len = s * np.linalg.norm(rays(uv, K, R_wc), axis=1)
    print(f"    rendered depth vs z-depth along optical axis: median |diff| {1000*np.median(np.abs(zimg-s)):.3f} mm, "
          f"max {1000*np.max(np.abs(zimg-s)):.3f} mm")
    print(f"    rendered depth vs Euclidean ray length:      median |diff| {1000*np.median(np.abs(zimg-ray_len)):.1f} mm, "
          f"max {1000*np.max(np.abs(zimg-ray_len)):.1f} mm")
    P_bp = backproject(uv, depth, K, R_wc, c)
    e_bp = 1000 * np.linalg.norm(P_bp - P_true, axis=1)
    dirs = rays(uv, K, R_wc)
    P_wrong = c + (zimg / np.linalg.norm(dirs, axis=1))[:, None] * dirs   # depth misread as ray length
    e_wrong = 1000 * np.linalg.norm(P_wrong - P_true, axis=1)
    print(f"    back-projection error (depth used as z):          median {np.median(e_bp):.3f} mm, max {e_bp.max():.3f} mm")
    print(f"    back-projection error (depth misused as ray len): median {np.median(e_wrong):.1f} mm, max {e_wrong.max():.1f} mm")

    # 2. homography: pixel-noise sweep
    print(f"\n[2] homography, camera height {h} m: table-point error vs pixel noise (200 test pixels, 50 trials, median mm)")
    print(f"{'noise px':>9} {'H from 4 pts':>13} {'H from 12 pts':>14} {'depth back-proj':>16}")
    for sigma in [0, 0.5, 1, 2, 4]:
        e4, e12, ed = [], [], []
        for _ in range(50):
            q = uv[rng.choice(len(uv), 200, replace=False)]
            Pq, _ = hit_plane(q, K, R_wc, c)
            qn = q + rng.normal(0, sigma, q.shape)
            for marks, acc in [(MARK4, e4), (MARK12, e12)]:
                xy = apply_h(fit_h(marks, K, R_wc, c, rng, sigma), qn)
                acc.append(np.median(1000 * np.linalg.norm(xy - Pq[:, :2], axis=1)))
            ed.append(np.median(1000 * np.linalg.norm(backproject(qn, depth, K, R_wc, c) - Pq, axis=1)))
        print(f"{sigma:>9} {np.median(e4):>13.2f} {np.median(e12):>14.2f} {np.median(ed):>16.2f}")

    # 3. camera height sweep at 1 px noise
    print("\n[3] camera height sweep, 1 px pixel noise (12-point H; 50 trials, median mm)")
    print(f"{'height m':>9} {'mm per px at target':>20} {'homography':>11} {'depth back-proj':>16}")
    for hh in [0.3, 0.5, 0.8, 1.2, 2.0]:
        _, dep, Kh, Rh, ch = setup(hh)
        uvh = table_pixels(dep, Kh, Rh, ch, rng, 300)
        t_uv = project(np.array([TARGET]), Kh, Rh, ch)
        foot = 1000 * np.linalg.norm(hit_plane(t_uv + [1, 0], Kh, Rh, ch)[0] - hit_plane(t_uv, Kh, Rh, ch)[0])
        eh, ed = [], []
        for _ in range(50):
            q = uvh[rng.choice(len(uvh), 200, replace=False)]
            Pq, _ = hit_plane(q, Kh, Rh, ch)
            qn = q + rng.normal(0, 1.0, q.shape)
            eh.append(np.median(1000 * np.linalg.norm(apply_h(fit_h(MARK12, Kh, Rh, ch, rng, 1.0), qn) - Pq[:, :2], axis=1)))
            ed.append(np.median(1000 * np.linalg.norm(backproject(qn, dep, Kh, Rh, ch) - Pq, axis=1)))
        print(f"{hh:>9} {foot:>20.2f} {np.median(eh):>11.2f} {np.median(ed):>16.2f}")

    # 4. break it: points above the plane
    print(f"\n[4] break it: pixel on the top face of a box, camera height {h} m, no pixel noise")
    print(f"{'box height mm':>14} {'homography xy err mm':>21} {'depth back-proj err mm':>23}")
    Hm = fit_h(MARK12, K, R_wc, c, rng, 0.0)
    for bh, (bx, by) in zip(BOX_H, BOX_XY):
        top = np.array([[bx, by, bh]])
        q = np.round(project(top, K, R_wc, c))
        Pq, _ = hit_plane(q, K, R_wc, c, z0=bh)            # what this pixel really sees
        xy = apply_h(Hm, q)
        eb = 1000 * np.linalg.norm(backproject(q, depth, K, R_wc, c) - Pq, axis=1)[0]
        print(f"{1000*bh:>14.0f} {1000*np.linalg.norm(xy[0]-Pq[0, :2]):>21.1f} {eb:>23.2f}")


if __name__ == "__main__":
    main()
