"""Lesson 10 · Inverse kinematics and singularities (SO-101, MuJoCo Menagerie).

Reusable:  from experiment import ik_solve   (see lessons that import experiments/10-inverse-kinematics)

Part A  Sweep the DLS damping lambda (plus pseudo-inverse and Jacobian transpose)
        over 500 random reachable position targets -> success rate, iterations, final error.
Part B  Multiple solutions: position + "point the fingers down", 100 random seeds -> distinct solutions.
Part C  Manipulability (Yoshikawa) and singular values while the arm stretches out.
Break   Near the stretched singularity: joint speeds of the pseudo-inverse vs DLS for a 5 cm/s request.

Run:  python experiment.py      (~1 min on Apple M5)
"""
import os
import platform
import sys
from dataclasses import dataclass

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass
class IKResult:
    q: np.ndarray          # joint values for `joints`, in order
    success: bool
    iters: int
    err: float             # final task error (m for position; position + axis error otherwise)
    max_step: float        # largest |dq| taken in one iteration (rad)


def ik_solve(model, data, target_pos, site="gripperframe", joints=ARM, q0=None,
             method="dls", damping=0.03, target_axis=None, axis_index=0, axis_weight=0.1,
             max_iters=200, tol=5e-4, respect_limits=True):
    """Iterative IK for one site.

    method: "dls"       dq = J^T (J J^T + damping^2 I)^-1 e    (damped least squares)
            "pinv"      dq = J^+ e                              (Newton-Raphson / Gauss-Newton)
            "transpose" dq = alpha J^T e, alpha = <e, J J^T e> / |J J^T e|^2   (Buss 2009)
    target_axis: optional unit vector; the site's local axis `axis_index` should point along it
                 (e.g. fingers down). Its residual is weighted by axis_weight (m per unit of axis error).
    Joint limits are enforced by clamping after each step. data.qpos is modified.
    Returns IKResult; success means |e| < tol.
    """
    sid = model.site(site).id
    jid = [model.joint(j).id for j in joints]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    dadr = np.array([model.jnt_dofadr[j] for j in jid])
    lo, hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
    limited = model.jnt_limited[jid].astype(bool) if respect_limits else np.zeros(len(jid), bool)
    if q0 is not None:
        data.qpos[qadr] = q0
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    max_step = 0.0

    def residual_and_jac():
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        e = target_pos - data.site_xpos[sid]
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        J = jp[:, dadr]
        if target_axis is not None:
            a = data.site_xmat[sid].reshape(3, 3)[:, axis_index]
            ea = np.cross(a, target_axis)            # rotation vector that turns a toward the target
            e = np.concatenate([e, axis_weight * ea])
            J = np.vstack([J, axis_weight * jr[:, dadr]])
        return e, J

    for it in range(max_iters + 1):
        e, J = residual_and_jac()
        err = float(np.linalg.norm(e))
        if err < tol:
            return IKResult(data.qpos[qadr].copy(), True, it, err, max_step)
        if it == max_iters:
            break
        if method == "pinv":
            dq = np.linalg.pinv(J) @ e
        elif method == "transpose":
            JJe = J @ (J.T @ e)
            dq = (e @ JJe) / max(JJe @ JJe, 1e-300) * (J.T @ e)
        else:
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
        max_step = max(max_step, float(np.linalg.norm(dq)))
        q = data.qpos[qadr] + dq
        q[limited] = np.clip(q[limited], lo[limited], hi[limited])
        data.qpos[qadr] = q
    return IKResult(data.qpos[qadr].copy(), False, max_iters, err, max_step)


def site_jacobian(model, data, site="gripperframe", joints=ARM):
    """Translational Jacobian of the site w.r.t. `joints` at data.qpos."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, None, model.site(site).id)
    return jp[:, [model.jnt_dofadr[model.joint(j).id] for j in joints]]


if __name__ == "__main__":
    print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Python {platform.python_version()} "
          f"| {platform.machine()}")
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    d = mujoco.MjData(m)
    SITE = m.site("gripperframe").id
    NA = len(ARM)
    LO, HI = m.jnt_range[:NA, 0], m.jnt_range[:NA, 1]

    def fk(q):
        d.qpos[:NA] = q
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[SITE].copy()

    # ---------- Part A ----------
    N = 500
    rng = np.random.default_rng(0)
    targets = np.array([fk(q) for q in rng.uniform(LO, HI, size=(N, NA))])   # reachable by construction
    print(f"\n== A. Sweep damping: {N} random reachable position targets, seed q0 = 0, "
          f"tol 0.5 mm, max 200 iterations ==")
    print(f"{'method':>16s} {'success':>8s} {'median it':>10s} {'p95 it':>7s} {'median err (mm)':>16s} "
          f"{'max err (mm)':>13s} {'median max|dq| (rad)':>21s} {'fails ending at a limit':>24s}")
    rows = [("transpose", None)] + [("pinv", 0.0)] + [("dls", lam) for lam in
                                                      (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)]
    for method, lam in rows:
        res = [ik_solve(m, d, t, q0=np.zeros(NA), method=method, damping=lam or 0.0) for t in targets]
        ok = np.array([r.success for r in res])
        its = np.array([r.iters for r in res if r.success])
        errs = np.array([r.err for r in res]) * 1000
        steps = np.array([r.max_step for r in res])
        at_lim = sum(1 for r in res if not r.success and
                     np.any((np.abs(r.q - LO) < 1e-9) | (np.abs(r.q - HI) < 1e-9)))
        label = method if lam is None or method == "pinv" else f"dls l={lam:g}"
        print(f"{label:>16s} {ok.mean() * 100:7.1f}% {np.median(its) if len(its) else float('nan'):10.0f} "
              f"{np.percentile(its, 95) if len(its) else float('nan'):7.0f} {np.median(errs):16.3f} "
              f"{errs.max():13.1f} {np.median(steps):21.3f} {at_lim:>13d} of {int((~ok).sum()):3d}")

    # restarts: if the solve from q0 = 0 fails, retry from random seeds
    rs = np.random.default_rng(5)
    ok, its_total = 0, []
    for t in targets:
        n_it = 0
        for k in range(5):
            q0 = np.zeros(NA) if k == 0 else rs.uniform(LO, HI)
            r = ik_solve(m, d, t, q0=q0, method="dls", damping=0.1)
            n_it += r.iters
            if r.success:
                ok += 1
                break
        its_total.append(n_it)
    print(f"{'dls l=0.1 +4 restarts':>16s} {ok / N * 100:7.1f}%  (median total iterations "
          f"{np.median(its_total):.0f}, max {max(its_total)})")

    # ---------- Part B ----------
    print("\n== B. Multiple solutions: reach (0.20, 0.05, 0.05) m, fingers pointing down, wrist_roll held at 0, "
          "100 random seeds ==")
    tgt = np.array([0.20, 0.05, 0.05])
    for limits in (False, True):
        sols = []
        rs = np.random.default_rng(1)
        for _ in range(100):
            s = rs.uniform(LO[:4], HI[:4]) if limits else rs.uniform(-np.pi, np.pi, 4)
            d.qpos[:] = 0
            r = ik_solve(m, d, tgt, joints=ARM[:4], q0=s, method="dls", damping=0.01,
                         target_axis=np.array([0, 0, -1.0]), max_iters=500, tol=1e-5, respect_limits=limits)
            if r.success:
                sols.append((r.q + np.pi) % (2 * np.pi) - np.pi)      # wrap to [-180, 180) deg
        clusters = []
        for q in sols:
            for c in clusters:
                if np.abs(q - c[0]).max() < 1e-2:
                    c[1] += 1
                    break
            else:
                clusters.append([q, 1])
        clusters.sort(key=lambda c: -c[1])
        print(f"joint limits {'ON ' if limits else 'OFF'}: converged {len(sols)} / 100 seeds; "
              f"distinct solutions (tol 0.01 rad): {len(clusters)}")
        print(f"  {'count':>5s}  {'pan':>7s} {'lift':>7s} {'elbow':>7s} {'wrist':>7s}  (deg)  inside limits?")
        for q, c in clusters:
            inside = bool(np.all((q >= LO[:4]) & (q <= HI[:4])))
            print(f"  {c:5d}  " + " ".join(f"{v:7.1f}" for v in np.rad2deg(q)) + f"         {inside}")

    # ---------- Part C ----------
    # find the stretched pose: minimize sigma_min of the position Jacobian over elbow and wrist
    def sig(q):
        d.qpos[:NA] = q
        return np.linalg.svd(site_jacobian(m, d), compute_uv=False)

    best = None
    for e in np.linspace(-1.45, -1.15, 301):
        for w in np.linspace(-0.15, 0.25, 81):
            s = sig([0, 1.3, e, w, 0])[-1]
            if best is None or s < best[0]:
                best = (s, e, w)
    q_bent = np.array([0, 1.3, 0.0, 0.0, 0])
    q_str = np.array([0, 1.3, best[1], best[2], 0])
    print(f"\n== C. Stretching the arm: q(s) = q_bent + s (q_straight - q_bent); "
          f"straight elbow {np.rad2deg(best[1]):.2f} deg, wrist {np.rad2deg(best[2]):.2f} deg ==")
    print(f"q_bent = {q_bent} rad, q_straight = {np.round(q_str, 4)} rad")
    print(f"{'s':>5s} {'sigma1':>8s} {'sigma2':>8s} {'sigma3':>9s} {'w=sqrt(det JJ^T)':>17s} {'cond':>9s} "
          f"{'reach (m)':>10s}")
    sh = None
    for s_ in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]:
        q = q_bent + s_ * (q_str - q_bent)
        sv = sig(q)
        if sh is None:
            sh = d.xpos[m.body("upper_arm").id].copy()
        reach = np.linalg.norm(d.site_xpos[SITE] - d.xpos[m.body("upper_arm").id])
        print(f"{s_:5.2f} {sv[0]:8.4f} {sv[1]:8.4f} {sv[2]:9.2e} {np.prod(sv):17.3e} {sv[0] / sv[2]:9.1e} "
              f"{reach:10.4f}")

    # ---------- Break it ----------
    print("\n== Break it. Request v = 5 cm/s along the weakest direction (u3) near the stretched pose ==")
    print(f"{'s':>5s} {'|qdot| pinv (rad/s)':>20s} {'|qdot| dls l=0.03':>18s} {'dls |J qdot|/|v|':>17s} "
          f"{'|qdot| dls l=0.1':>17s} {'dls l=0.1 |J qdot|/|v|':>23s}")
    for s_ in [0.5, 0.9, 0.99, 0.999, 1.0]:
        q = q_bent + s_ * (q_str - q_bent)
        d.qpos[:NA] = q
        J = site_jacobian(m, d)
        U, S, Vt = np.linalg.svd(J)
        v = 0.05 * U[:, 2]
        qd_p = np.linalg.pinv(J) @ v
        out = [np.linalg.norm(qd_p)]
        for lam in (0.03, 0.1):
            qd = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), v)
            out += [np.linalg.norm(qd), np.linalg.norm(J @ qd) / 0.05]
        print(f"{s_:5.3f} {out[0]:20.2f} {out[1]:18.3f} {out[2]:17.3f} {out[3]:17.3f} {out[4]:23.3f}")
