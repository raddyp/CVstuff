#!/usr/bin/env python3
"""Fast, simplified incremental SfM (speed-focused variant of sfm_pipeline.py).

Same core (SIFT -> windowed matching + Fundamental RANSAC -> tracks -> two-view
init -> incremental PnP -> triangulation -> scipy BA -> .ply), but trimmed for
speed:
  * lighter defaults (smaller images, fewer features, narrower match window)
  * BA runs infrequently during growth and is BA-only (no mid-loop prune)
  * ONE final refinement pass (BA -> prune -> re-triangulate -> BA),
    instead of the 4-iteration loop in the reference

Tradeoff: fewer BA/prune passes = faster but a bit more drift / higher RMS.
The single final prune+re-triangulate is kept as the drift safety-net.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# --------------------------------------------------------------------------- #
# IO / features
# --------------------------------------------------------------------------- #
def load_images(img_dir, stride, max_dim):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                   glob.glob(os.path.join(img_dir, "*.png")))
    paths = paths[::stride]
    images, grays = [], []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        s = max_dim / max(h, w)
        if s < 1.0:
            im = cv2.resize(im, (int(round(w * s)), int(round(h * s))),
                            interpolation=cv2.INTER_AREA)
        images.append(im)
        grays.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    print(f"[load] {len(images)} images @ {images[0].shape[1]}x{images[0].shape[0]} "
          f"(stride={stride})")
    return images, grays


def detect(grays, nfeatures):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kpts, descs = [], []
    for g in grays:
        kp, ds = sift.detectAndCompute(g, None)
        kpts.append(np.array([k.pt for k in kp], dtype=np.float64))
        descs.append(ds)
    print(f"[sift] mean {np.mean([len(k) for k in kpts]):.0f} kpts/img")
    return kpts, descs


# --------------------------------------------------------------------------- #
# matching (sequential window with wraparound = good for turntable video)
# --------------------------------------------------------------------------- #
def match_pairs(kpts, descs, window, ratio):
    n = len(descs)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pair_matches = {}
    pairs = set()
    for i in range(n):
        for d in range(1, window + 1):
            pairs.add((i, (i + d) % n))
    for (i, j) in sorted(pairs):
        if i == j:
            continue
        raw = bf.knnMatch(descs[i], descs[j], k=2)
        good = [(m.queryIdx, m.trainIdx)
                for m, n2 in raw if m.distance < ratio * n2.distance]
        if len(good) < 30:
            continue
        good = np.array(good)
        pi = kpts[i][good[:, 0]]
        pj = kpts[j][good[:, 1]]
        F, mask = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, 2.0, 0.999)
        if F is None or mask is None:
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < 25:
            continue
        pair_matches[(i, j)] = good[mask]
    print(f"[match] {len(pair_matches)} verified pairs")
    return pair_matches


# --------------------------------------------------------------------------- #
# tracks via union-find
# --------------------------------------------------------------------------- #
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_tracks(kpts, pair_matches):
    counts = [len(k) for k in kpts]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    uf = UF(int(offsets[-1]))
    for (i, j), m in pair_matches.items():
        for ki, kj in m:
            uf.union(offsets[i] + ki, offsets[j] + kj)

    groups = {}
    for i in range(len(kpts)):
        for k in range(counts[i]):
            r = uf.find(offsets[i] + k)
            groups.setdefault(r, []).append((i, k))

    tracks, obs_to_track = [], {}
    for obs in groups.values():
        imgs = [o[0] for o in obs]
        if len(imgs) != len(set(imgs)) or len(obs) < 2:
            continue
        tid = len(tracks)
        tracks.append(obs)
        for o in obs:
            obs_to_track[o] = tid
    print(f"[tracks] {len(tracks)} tracks, mean len "
          f"{np.mean([len(t) for t in tracks]):.2f}")
    return tracks, obs_to_track


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def proj_matrix(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def triangulate_multiview(Pmats, pts2d):
    A = []
    for P, (x, y) in zip(Pmats, pts2d):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reproj_err(K, R, t, X, uv):
    x = R @ X + t
    if x[2] <= 1e-6:
        return 1e9, x[2]
    u = K[0, 0] * x[0] / x[2] + K[0, 2]
    v = K[1, 1] * x[1] / x[2] + K[1, 2]
    return np.hypot(u - uv[0], v - uv[1]), x[2]


def cam_center(R, t):
    return -R.T @ t


def max_tri_angle(cameras, views, X):
    dirs = []
    for img, _ in views:
        C = cam_center(*cameras[img])
        d = X - C
        n = np.linalg.norm(d)
        if n > 1e-9:
            dirs.append(d / n)
    best = 0.0
    for a in range(len(dirs)):
        for b in range(a + 1, len(dirs)):
            cang = np.clip(float(dirs[a] @ dirs[b]), -1.0, 1.0)
            best = max(best, np.degrees(np.arccos(cang)))
    return best


# --------------------------------------------------------------------------- #
# bundle adjustment (scipy)
# --------------------------------------------------------------------------- #
def rotate(pts, rvecs):
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.nan_to_num(rvecs / theta)
    dot = np.sum(pts * v, axis=1)[:, None]
    c = np.cos(theta)
    s = np.sin(theta)
    return c * pts + s * np.cross(v, pts) + dot * (1 - c) * v


def ba_residuals(params, n_cam, cam_idx, pt_idx, obs2d, K):
    cams = params[:n_cam * 6].reshape(n_cam, 6)
    pts = params[n_cam * 6:].reshape(-1, 3)
    P = rotate(pts[pt_idx], cams[cam_idx, :3]) + cams[cam_idx, 3:6]
    u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
    v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    return np.concatenate([u - obs2d[:, 0], v - obs2d[:, 1]])


def ba_sparsity(n_cam, n_pts, cam_idx, pt_idx):
    m = len(cam_idx) * 2
    n = n_cam * 6 + n_pts * 3
    S = lil_matrix((m, n), dtype=int)
    ar = np.arange(len(cam_idx))
    for s in range(6):
        S[2 * ar, cam_idx * 6 + s] = 1
        S[2 * ar + 1, cam_idx * 6 + s] = 1
    for s in range(3):
        S[2 * ar, n_cam * 6 + pt_idx * 3 + s] = 1
        S[2 * ar + 1, n_cam * 6 + pt_idx * 3 + s] = 1
    return S


def run_ba(cameras, reg_order, points3d, tracks, kpts, K, ftol=1e-4):
    cam_ids = list(reg_order)
    cam_pos = {c: i for i, c in enumerate(cam_ids)}
    pt_ids = [tid for tid, X in points3d.items() if X is not None]
    pt_pos = {tid: i for i, tid in enumerate(pt_ids)}

    cam_params = np.zeros((len(cam_ids), 6))
    for c in cam_ids:
        R, t = cameras[c]
        rvec, _ = cv2.Rodrigues(R)
        cam_params[cam_pos[c], :3] = rvec.ravel()
        cam_params[cam_pos[c], 3:6] = t.ravel()
    pt_params = np.array([points3d[tid] for tid in pt_ids])

    cam_idx, pt_idx, obs2d = [], [], []
    for tid in pt_ids:
        for (img, k) in tracks[tid]:
            if img in cam_pos:
                cam_idx.append(cam_pos[img])
                pt_idx.append(pt_pos[tid])
                obs2d.append(kpts[img][k])
    cam_idx = np.array(cam_idx)
    pt_idx = np.array(pt_idx)
    obs2d = np.array(obs2d)

    x0 = np.concatenate([cam_params.ravel(), pt_params.ravel()])
    S = ba_sparsity(len(cam_ids), len(pt_ids), cam_idx, pt_idx)
    res = least_squares(
        ba_residuals, x0, jac_sparsity=S, verbose=0, x_scale="jac",
        ftol=ftol, xtol=1e-8, method="trf", loss="huber", f_scale=2.0,
        args=(len(cam_ids), cam_idx, pt_idx, obs2d, K))

    p = res.x
    cams = p[:len(cam_ids) * 6].reshape(-1, 6)
    for c in cam_ids:
        R, _ = cv2.Rodrigues(cams[cam_pos[c], :3])
        cameras[c] = (R, cams[cam_pos[c], 3:6].copy())
    pts = p[len(cam_ids) * 6:].reshape(-1, 3)
    for tid in pt_ids:
        points3d[tid] = pts[pt_pos[tid]]
    return np.sqrt(np.mean(res.fun ** 2))


# --------------------------------------------------------------------------- #
# incremental SfM
# --------------------------------------------------------------------------- #
def two_view_init(pair_matches, kpts, K, obs_to_track, tracks):
    best = None
    for (i, j), m in pair_matches.items():
        pi = kpts[i][m[:, 0]]
        pj = kpts[j][m[:, 1]]
        E, mask = cv2.findEssentialMat(pi, pj, K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            continue
        _, R, t, mask_pose = cv2.recoverPose(E, pi, pj, K, mask=mask)
        inl = int(mask_pose.sum())
        if best is None or inl > best[0]:
            best = (inl, i, j, R, t.ravel(), mask_pose.ravel().astype(bool), m)
    inl, i, j, R, t, mp, m = best
    print(f"[init] pair ({i},{j}) inliers={inl}")

    cameras = {i: (np.eye(3), np.zeros(3)), j: (R, t)}
    P0 = proj_matrix(K, *cameras[i])
    P1 = proj_matrix(K, *cameras[j])
    points3d = {}
    for (ki, kj), ok in zip(m, mp):
        if not ok:
            continue
        tid = obs_to_track.get((i, ki))
        if tid is None:
            continue
        X = triangulate_multiview([P0, P1], [kpts[i][ki], kpts[j][kj]])
        e0, z0 = reproj_err(K, *cameras[i], X, kpts[i][ki])
        e1, z1 = reproj_err(K, *cameras[j], X, kpts[j][kj])
        ang = max_tri_angle(cameras, [(i, ki), (j, kj)], X)
        if z0 > 0 and z1 > 0 and e0 < 4 and e1 < 4 and ang > 1.5:
            points3d[tid] = X
    print(f"[init] triangulated {len(points3d)} points")
    return cameras, points3d, {i, j}


def triangulate_new(cameras, reg, points3d, tracks, kpts, K,
                    max_err=4.0, min_angle=1.5):
    added = 0
    for tid, obs in enumerate(tracks):
        if points3d.get(tid) is not None:
            continue
        views = [(img, k) for (img, k) in obs if img in reg]
        if len(views) < 2:
            continue
        Pmats = [proj_matrix(K, *cameras[img]) for img, _ in views]
        pts = [kpts[img][k] for img, k in views]
        X = triangulate_multiview(Pmats, pts)
        if max_tri_angle(cameras, views, X) < min_angle:
            continue
        ok = True
        for (img, k) in views:
            e, z = reproj_err(K, *cameras[img], X, kpts[img][k])
            if z <= 0 or e > max_err:
                ok = False
                break
        if ok:
            points3d[tid] = X
            added += 1
    return added


def prune_points(cameras, points3d, tracks, kpts, K, max_err, min_angle):
    removed = 0
    for tid, X in points3d.items():
        if X is None:
            continue
        views = [(img, k) for (img, k) in tracks[tid] if img in cameras]
        bad = False
        for (img, k) in views:
            e, z = reproj_err(K, *cameras[img], X, kpts[img][k])
            if z <= 0 or e > max_err:
                bad = True
                break
        if not bad and max_tri_angle(cameras, views, X) < min_angle:
            bad = True
        if bad:
            points3d[tid] = None
            removed += 1
    return removed


def register_next(cameras, reg, points3d, tracks, obs_to_track, kpts, K):
    n_img = len(kpts)
    best = None
    for img in range(n_img):
        if img in reg:
            continue
        pts3d, pts2d = [], []
        for k in range(len(kpts[img])):
            tid = obs_to_track.get((img, k))
            if tid is not None and points3d.get(tid) is not None:
                pts3d.append(points3d[tid])
                pts2d.append(kpts[img][k])
        if len(pts3d) >= 12 and (best is None or len(pts3d) > best[1]):
            best = (img, len(pts3d), np.array(pts3d), np.array(pts2d))
    if best is None:
        return None
    img, cnt, obj, imgpts = best
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj.astype(np.float64), imgpts.astype(np.float64), K, None,
        reprojectionError=4.0, confidence=0.999, iterationsCount=200,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        obj[inliers.ravel()].astype(np.float64),
        imgpts[inliers.ravel()].astype(np.float64), K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    cameras[img] = (R, tvec.ravel())
    reg.add(img)
    return img, cnt, len(inliers)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def filter_points(points3d, tracks, cameras, kpts, K, max_err, min_views):
    keep = {}
    for tid, X in points3d.items():
        if X is None:
            continue
        errs, nv = [], 0
        for (img, k) in tracks[tid]:
            if img in cameras:
                e, z = reproj_err(K, *cameras[img], X, kpts[img][k])
                if z > 0:
                    errs.append(e)
                    nv += 1
        if nv >= min_views and np.mean(errs) < max_err:
            keep[tid] = X
    return keep


def save_ply(path, points3d, tracks, images, kpts):
    xyz, rgb = [], []
    for tid, X in points3d.items():
        img, k = tracks[tid][0]
        px = kpts[img][k]
        h, w = images[img].shape[:2]
        u = int(np.clip(px[0], 0, w - 1))
        v = int(np.clip(px[1], 0, h - 1))
        b, g, r = images[img][v, u]
        xyz.append(X)
        rgb.append((r, g, b))
    xyz = np.array(xyz)
    c = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - c, axis=1)
    thr = np.median(d) + 3 * (np.percentile(d, 75) - np.percentile(d, 25) + 1e-9)
    m = d < thr
    xyz, rgb = xyz[m], np.array(rgb)[m]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"[ply] wrote {len(xyz)} points -> {path}")
    return len(xyz)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="reconstruction_fast.ply")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-dim", type=int, default=800)      # was 1024
    ap.add_argument("--nfeatures", type=int, default=3000)   # was 8000
    ap.add_argument("--window", type=int, default=3)         # was 4
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ba-every", type=int, default=10)      # was 5
    args = ap.parse_args()

    t0 = time.time()
    images, grays = load_images(args.images, args.stride, args.max_dim)
    h, w = images[0].shape[:2]
    f = 1.2 * max(w, h)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    print(f"[intrinsics] f={f:.1f} (assumed), principal=({w/2:.0f},{h/2:.0f})")

    kpts, descs = detect(grays, args.nfeatures)
    pair_matches = match_pairs(kpts, descs, args.window, args.ratio)
    tracks, obs_to_track = build_tracks(kpts, pair_matches)

    cameras, points3d, reg = two_view_init(pair_matches, kpts, K, obs_to_track, tracks)
    reg_order = list(reg)
    run_ba(cameras, reg_order, points3d, tracks, kpts, K)

    # Incremental growth: BA-only, infrequent (no mid-loop prune/re-triangulate).
    since_ba = 0
    while True:
        r = register_next(cameras, reg, points3d, tracks, obs_to_track, kpts, K)
        if r is None:
            break
        img, cnt, inl = r
        reg_order.append(img)
        added = triangulate_new(cameras, reg, points3d, tracks, kpts, K)
        since_ba += 1
        print(f"[reg] img {img:3d} | 2D-3D {cnt:4d} inl {inl:4d} | "
              f"+{added} pts | total {sum(1 for v in points3d.values() if v is not None)}")
        if since_ba >= args.ba_every:
            rms = run_ba(cameras, reg_order, points3d, tracks, kpts, K)
            print(f"      [BA] rms={rms:.3f}px  cams={len(reg_order)}")
            since_ba = 0

    # Single final refinement pass (drift safety-net).
    rms = run_ba(cameras, reg_order, points3d, tracks, kpts, K, ftol=1e-6)
    rm = prune_points(cameras, points3d, tracks, kpts, K, max_err=4.0, min_angle=1.5)
    add = triangulate_new(cameras, reg, points3d, tracks, kpts, K,
                          max_err=4.0, min_angle=1.5)
    rms = run_ba(cameras, reg_order, points3d, tracks, kpts, K, ftol=1e-6)
    print(f"[final] rms={rms:.3f}px  pruned={rm}  +{add}")

    pts = filter_points(points3d, tracks, cameras, kpts, K, max_err=2.0, min_views=2)
    print(f"[filter] {len(pts)} points kept (of "
          f"{sum(1 for v in points3d.values() if v is not None)})")
    save_ply(args.out, pts, tracks, images, kpts)
    print(f"[done] {len(reg)} / {len(images)} images registered in "
          f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
