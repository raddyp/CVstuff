#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 10:08:12 2026

"""

"""Minimal two-view Structure-from-Motion — a teaching demo.

Given two overlapping images, recover their relative pose and triangulate a
sparse 3D point cloud.

Use grayscale images for faster compute, color for added detail.

Steps:
  1. Load two images + assume intrinsics K
  2. Detect SIFT features
  3. Match using Lowe's ratio test
  4. Estimate the Essential matrix using RANSAC
  5. Recover relative pose R, t 
  6. Build the two 3x4 projection matrices
  7. Triangulate matched points (cv2.triangulatePoints)
  8. Keep points in front of both cameras (cheirality)
  9. Save a colored .ply

Run:
  python sfm_twoview.py -i /data/image1.jpg -j /data/image2.jpg -o /data/twoview.ply
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True, help="first image path")
    ap.add_argument("-j", required=True, help="second image path")
    ap.add_argument("-o", required=True, help="output dir")
    args = ap.parse_args()

    p1, p2 = Path(args.i), Path(args.j)
    
    # 1. Load images + assume intrinsics
    color1 = cv2.imread(p1, 1)
    gray1  = cv2.imread(p1, 0)
    gray2  = cv2.imread(p2, 0)

    h, w = gray1.shape[:2]
    # assume rough focal length for most consumer cameras (~45 deg FOV) 
    # f = d/2×tan(FOV/2) # d is the max dimension of image
    f = 1.2 * max(w, h)  
    # camera matrix                    
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]])
    print(f"[K] f={f:.1f}, principal=({w/2:.0f},{h/2:.0f})")

    #  2. Detect SIFT features 
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"[sift] {len(kp1)} and {len(kp2)} keypoints")

    #  3. Match with Lowe's ratio test, only keep good points
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    ratio = 0.7 # default
    good = [m for m, n in knn if m.distance <ratio * n.distance]
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
    print(f"[match] {len(good)} good matches")

    #  4. Essential matrix (RANSAC rejects outlier matches) 
    # technically can use findFundamental matrix if K is unknown
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    mask = mask.ravel().astype(bool)
    pts1, pts2 = pts1[mask], pts2[mask]
    print(f"[E] {mask.sum()} inliers after RANSAC")

    #  5. Recover relative pose R, t between thw two images
    # Camera 1 is the reference; camera 2 is at [R|t] relative to it.
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    mask_pose = mask_pose.ravel().astype(bool)
    pts1, pts2 = pts1[mask_pose], pts2[mask_pose]
    print(f"[pose] {mask_pose.sum()} points pass cheirality in recoverPose")

    #  6. Projection matrices  P = K [R | t] 
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])   # reference camera
    P2 = K @ np.hstack([R, t])                          # second camera

    #  7. Triangulate using OpenCV's DLT 
    # triangulatePoints wants 2xN inputs and returns 4xN homogeneous coords.
    X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X_h[:3] / X_h[3]).T                             # -> Nx3 Euclidean

    #  8. Keep points in front of both cameras  (cheirality)
    z1 = X[:, 2]                                         # depth in camera 1
    X_cam2 = (R @ X.T + t).T
    z2 = X_cam2[:, 2]                                    # depth in camera 2
    front = (z1 > 0) & (z2 > 0)
    X, pts1 = X[front], pts1[front]

    # ---- 9. Color each point from image 1 and save .ply --------------------
    colors = []
    for (u, v) in pts1:
        # clipping projected pixel loacations within orig image size
        ui = int(np.clip(u, 0, w - 1))
        vi = int(np.clip(v, 0, h - 1))
        b, g, r = color1[vi, ui]
        colors.append((r, g, b))
    
    # write to ply format
    with open(args.o, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(X)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(X, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"[ply] wrote {len(X)} points -> {args.o}")


if __name__ == "__main__":
    main()
