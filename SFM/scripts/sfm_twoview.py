#!/usr/bin/env python3
"""Minimal two-view Structure-from-Motion — a teaching demo.

Given TWO overlapping images, recover their relative pose and triangulate a
sparse 3D point cloud. This is the irreducible core of SfM: no feature tracks,
no incremental registration, no bundle adjustment — just the essential-matrix
geometry that everything else is built on.

Steps:
  1. Load two images + assume intrinsics K
  2. Detect & describe SIFT features
  3. Match with Lowe's ratio test
  4. Estimate the Essential matrix (RANSAC)
  5. Recover relative pose R, t (recoverPose)
  6. Build the two 3x4 projection matrices
  7. Triangulate matched points (cv2.triangulatePoints)
  8. Keep points in front of both cameras (cheirality)
  9. Save a colored .ply

Run:
    
  python sfm_twoview.py --images <dir> --i 0 --j 8 --out twoview.ply
  
  i & j are just image file indices ina folder full of n images.
"""

import argparse

import cv2
import numpy as np
from pathlib import Path


def load_gray_color(path, max_dim):
    """Load an image, optionally downscale, return (color_bgr, gray)."""
    color = cv2.imread(str(path))
    h, w = color.shape[:2]
    s = max_dim / max(h, w)
    if s < 1.0:
        color = cv2.resize(color, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of .jpg frames")
    ap.add_argument("--i", type=int, default=0, help="index of first image")
    ap.add_argument("--j", type=int, default=8, help="index of second image")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--out", default="twoview.ply")
    args = ap.parse_args()

    paths = sorted(Path(args.images).glob("*.jpg"))
    p1, p2 = paths[args.i], paths[args.j]
    print(f"[pair] {p1.name}  <->  {p2.name}")

    # ---- 1. Load images + assume intrinsics --------------------------------
    color1, gray1 = load_gray_color(p1, args.max_dim)
    color2, gray2 = load_gray_color(p2, args.max_dim)
    h, w = gray1.shape[:2]
    f = 1.2 * max(w, h)                      # rough focal guess (~45 deg FOV)
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]])
    print(f"[K] f={f:.1f}, principal=({w/2:.0f},{h/2:.0f})")

    # ---- 2. Detect & describe SIFT features --------------------------------
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"[sift] {len(kp1)} and {len(kp2)} keypoints")

    # ---- 3. Match with Lowe's ratio test -----------------------------------
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < args.ratio * n.distance]
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
    print(f"[match] {len(good)} good matches")

    # ---- 4. Essential matrix (RANSAC rejects outlier matches) --------------
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    mask = mask.ravel().astype(bool)
    pts1, pts2 = pts1[mask], pts2[mask]
    print(f"[E] {mask.sum()} inliers after RANSAC")

    # ---- 5. Recover relative pose R, t -------------------------------------
    # Camera 1 sits at the origin; camera 2 is at [R|t] relative to it.
    # (t is a unit direction — two-view SfM recovers scale only up to a factor.)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    mask_pose = mask_pose.ravel().astype(bool)
    pts1, pts2 = pts1[mask_pose], pts2[mask_pose]
    print(f"[pose] {mask_pose.sum()} points pass cheirality in recoverPose")

    # ---- 6. Projection matrices  P = K [R | t] -----------------------------
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])   # reference camera
    P2 = K @ np.hstack([R, t])                          # second camera

    # ---- 7. Triangulate (OpenCV's built-in linear DLT) ---------------------
    # triangulatePoints wants 2xN inputs and returns 4xN homogeneous coords.
    X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X_h[:3] / X_h[3]).T                             # -> Nx3 Euclidean

    # ---- 8. Keep points in front of BOTH cameras ---------------------------
    z1 = X[:, 2]                                         # depth in camera 1
    X_cam2 = (R @ X.T + t).T
    z2 = X_cam2[:, 2]                                    # depth in camera 2
    front = (z1 > 0) & (z2 > 0)
    X, pts1 = X[front], pts1[front]
    print(f"[triangulate] {len(X)} 3D points in front of both cameras")

    # ---- 9. Color each point from image 1 and save .ply --------------------
    colors = []
    for (u, v) in pts1:
        ui = int(np.clip(u, 0, w - 1))
        vi = int(np.clip(v, 0, h - 1))
        b, g, r = color1[vi, ui]
        colors.append((r, g, b))

    with open(args.out, "w") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(X)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(X, colors):
            fp.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"[ply] wrote {len(X)} points -> {args.out}")


if __name__ == "__main__":
    main()
