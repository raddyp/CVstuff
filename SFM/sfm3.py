#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:14:20 2026

@author: mohamedathiq
"""

import numpy as np
import cv2
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def plot_image(img, title = "", grey= False):

    fig, ax = plt.subplots()
    plt.title(title)
    vmin = 0
    vmax = 1.5
    if(grey):
        im = ax.imshow(img, cmap='grey',vmin=vmin, vmax=vmax)
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = ax.imshow(img_rgb,vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img, cmap='turbo',vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

    return

def load_images(folder, max_dim=1600):
    
    imgs, names = [], []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        img = cv2.imread(p)
        if img is None:
            continue
        # h, w = img.shape[:2]
        # s = max_dim / max(h, w)
        # if s < 1.0:
        #     img = cv2.resize(img, (int(w * s), int(h * s)))
        imgs.append(img)
        names.append(os.path.basename(p))
    return imgs, names


def guess_intrinsics(img):
    h, w = img.shape[:2]
    f = 1.2 * max(h, w)   # rough focal guess in pixels
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]])

def extract_features(image_dir):
    sift = cv2.SIFT_create(nfeatures=10000)
    kpts, descs = [],[]
    for path in sorted(Path(image_dir).glob("*.jpg")):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # scale = min(1.0, max_dim / max(img.shape))
        # if scale < 1.0:
        #     img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        descs.append(descriptors)    
        kpts.append(np.array([k.pt for k in keypoints], dtype=np.float64))

    return kpts, descs

def pair_matches(keypoints, descriptors):
        
    ratio=0.8
    min_matches=30
    n = len(descriptors)
    window = 4

    bf = cv2.BFMatcher(cv2.NORM_L2)
    pair_matches = {}
    pairs = set()
    for i in range(n):
        for d in range(1, window + 1):
            pairs.add((i, (i + d) % n))
    for (i, j) in sorted(pairs):
        if i == j:
            continue
        raw = bf.knnMatch(descriptors[i], descriptors[j], k=2)
        good = [(m.queryIdx, m.trainIdx) for m, n2 in raw if m.distance < ratio * n2.distance]
        if len(good) < min_matches:
            continue
        good = np.array(good)
        pi = keypoints[i][good[:, 0]]
        pj = keypoints[j][good[:, 1]]
        F, mask = cv2.findEssentialMat(pi, pj,K, cv2.RANSAC, 0.999,1.0)
        if F is None or mask is None:
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < 25:
            continue
        pair_matches[(i, j)] = good[mask]
    return pair_matches

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

def build_tracks(keypoints, pair_matches):
        
    counts = [len(k) for k in keypoints]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    uf = UF(int(offsets[-1]))
    
    for (i, j), m in pair_matches.items():
        for ki, kj in m:
            uf.union(offsets[i] + ki, offsets[j] + kj)
    
    groups = {}
    for i in range(len(keypoints)):
        for k in range(counts[i]):
            r = uf.find(offsets[i] + k)
            groups.setdefault(r, []).append((i, k))
    
    tracks = []            # list of list[(img, kp_idx)]
    obs_to_track = {}      # (img, kp_idx) -> track_id
    for obs in groups.values():
        imgs = [o[0] for o in obs]
        if len(imgs) != len(set(imgs)):   # drop tracks with 2 feats in one image
            continue
        if len(obs) < 2:
            continue
        tid = len(tracks)
        tracks.append(obs)
        for o in obs:
            obs_to_track[o] = tid

    return tracks, obs_to_track


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


def max_tri_angle(cameras, views, X):
    dirs = []
    for img, _ in views:
        R,t = cameras[img]
        C = -R.T @ t
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

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t.reshape(3, 1)])
    
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

# BA
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
        rvec = cams[cam_pos[c], :3]
        R, _ = cv2.Rodrigues(rvec)
        cameras[c] = (R, cams[cam_pos[c], 3:6].copy())
    pts = p[len(cam_ids) * 6:].reshape(-1, 3)
    for tid in pt_ids:
        points3d[tid] = pts[pt_pos[tid]]
    rms = np.sqrt(np.mean(res.fun ** 2))
    return rms


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


def proj_matrix(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


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
    """COLMAP-style: drop points that reproject badly, go behind a camera,
    or have too small a triangulation angle. Pruned points can be
    re-triangulated later once more cameras are added."""
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


def save_ply(path, points3d, tracks, images, kpts, cameras, K):
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
    # drop statistical outliers (far from centroid)
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

#%% init

images = '/Volumes/WH/Data/phoenix_evt1/sfm/data/teddybear/34_1479_4753/images'

input_dir = Path(images)

images, names = load_images(images)

h, w = images[0].shape[:2]
f = 1.2 * max(w, h)
K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])

# extract kp & descriptors
keypoints, descriptors = extract_features(input_dir)

# pair matches
pair_matches = pair_matches(keypoints,descriptors)

# build tracks between same keypoints
tracks, obs_to_track = build_tracks(keypoints, pair_matches)

#%%
# incremental build of sfm for all valid tracked keypoints
cameras, points3d, reg = two_view_init(pair_matches, keypoints, K, obs_to_track, tracks)


# bundle adjustment
reg_order = list(reg)
run_ba(cameras, reg_order, points3d, tracks, keypoints, K)

# triangualtion
since_ba = 0
while True:
    r = register_next(cameras, reg, points3d, tracks, obs_to_track, keypoints, K)
    if r is None:
        break
    img, cnt, inl = r
    reg_order.append(img)
    added = triangulate_new(cameras, reg, points3d, tracks, keypoints, K)
    since_ba += 1
    print(f"[reg] img {img:3d} | 2D-3D {cnt:4d} inl {inl:4d} | "
          f"+{added} pts | total {sum(1 for v in points3d.values() if v is not None)}")
    ba_args = 5
    if since_ba >= ba_args:
        rms = run_ba(cameras, reg_order, points3d, tracks, keypoints, K)
        rm = prune_points(cameras, points3d, tracks, keypoints, K,
                          max_err=6.0, min_angle=1.5)
        triangulate_new(cameras, reg, points3d, tracks, keypoints, K)
        print(f"      [BA] rms={rms:.3f}px  cams={len(reg_order)}  pruned={rm}")
        since_ba = 0


# pts = filter_points(points3d, tracks, cameras, keypoints, K, max_err=2.0, min_views=2)
#%%


# Final refinement: iterate BA -> prune outliers -> re-triangulate.
for it in range(10):
    rms = run_ba(cameras, reg_order, points3d, tracks, keypoints, K, ftol=1e-6)
    rm = prune_points(cameras, points3d, tracks, keypoints, K,
                      max_err=4.0, min_angle=1.5)
    add = triangulate_new(cameras, reg, points3d, tracks, keypoints, K,
                          max_err=4.0, min_angle=1.5)
    npts = sum(1 for v in points3d.values() if v is not None)
    print(f"[final {it}] rms={rms:.3f}px  pruned={rm}  +{add}  pts={npts}")

pts = filter_points(points3d, tracks, cameras, keypoints, K, max_err=2.0, min_views=2)
print(f"[filter] {len(pts)} points kept (of "
      f"{sum(1 for v in points3d.values() if v is not None)})")


out_path = '/Volumes/WH/Data/phoenix_evt1/sfm/sample2.ply'
save_ply(out_path, pts, tracks, images, keypoints, cameras, K)






