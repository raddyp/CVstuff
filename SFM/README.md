# Structure-from-Motion (SfM)

Two self-contained Python scripts for reconstructing sparse 3D point clouds from
overlapping images. They share the same core geometry (SIFT features →
matching → pose estimation → triangulation) but differ in scope:

| Script | Scope | Use it for |
| --- | --- | --- |
| `sfm_twoview.py` | Two images, no bundle adjustment | Learning the irreducible core of SfM |
| `sfm_fast.py` | Full incremental pipeline over many images | Reconstructing a scene from an image sequence |

Both write a colored [`.ply`](https://en.wikipedia.org/wiki/PLY_(file_format))
point cloud you can open in MeshLab, CloudCompare, or Blender.

## Overview

Both scripts start from the same feature-matching core, then diverge: the
two-view script recovers a single relative pose, while the fast script grows an
incremental reconstruction with bundle adjustment.

![SfM conceptual overview](flow_concept.png)

## Requirements

```bash
pip install opencv-python numpy scipy
```

- `opencv-python` — SIFT, matching, RANSAC pose estimation, triangulation
- `numpy` — linear algebra
- `scipy` — sparse bundle adjustment (`sfm_fast.py` only)

Input images should be `.jpg` or `.png` frames of a static scene with good
overlap between consecutive views.

---

## `sfm_twoview.py` — minimal two-view SfM

A teaching demo. Given **two** overlapping images, it recovers their relative
pose and triangulates a sparse point cloud. No feature tracks, no incremental
registration, no bundle adjustment — just the essential-matrix geometry that
everything else is built on.

**Pipeline**

1. Load two images and assume camera intrinsics `K`
2. Detect & describe SIFT features
3. Match with Lowe's ratio test
4. Estimate the Essential matrix (RANSAC)
5. Recover relative pose `R, t` (`cv2.recoverPose`)
6. Build the two 3×4 projection matrices
7. Triangulate matched points (`cv2.triangulatePoints`)
8. Keep points in front of both cameras (cheirality check)
9. Save a colored `.ply`

![sfm_twoview.py pipeline](flow_twoview.png)

**Usage**

```bash
python sfm_twoview.py --images <dir> --i 0 --j 8 --out twoview.ply
```

**Arguments**

| Flag | Default | Description |
| --- | --- | --- |
| `--images` | *(required)* | Folder of `.jpg` frames |
| `--i` | `0` | Index of the first image |
| `--j` | `8` | Index of the second image |
| `--max-dim` | `1024` | Longest image side in pixels (downscales larger images) |
| `--ratio` | `0.75` | Lowe's ratio-test threshold |
| `--out` | `twoview.ply` | Output point cloud path |

> Note: two-view SfM recovers translation only up to scale, so the resulting
> cloud has no absolute metric size.

---

## `sfm_fast.py` — fast incremental SfM

A full incremental pipeline over an image sequence, tuned for speed. It keeps
the same core as the two-view demo but adds feature tracks, incremental camera
registration, and bundle adjustment.

**Pipeline**

1. Load & downscale images, assume intrinsics `K`
2. SIFT detection on every frame
3. Windowed pairwise matching + Fundamental-matrix RANSAC verification
4. Build feature tracks across frames via union-find
5. Two-view initialization from the best-conditioned pair
6. Incremental growth: register the next best image with PnP RANSAC, then
   triangulate new points
7. Sparse bundle adjustment (`scipy.optimize.least_squares`) every `--ba-every`
   registrations
8. One final refinement pass: BA → prune outliers → re-triangulate → BA
9. Save both a filtered `.ply` and a full unfiltered `_full.ply`

**Speed tradeoff:** lighter defaults (smaller images, fewer features, narrower
match window) and infrequent bundle adjustment make it fast but allow a bit more
drift / higher RMS. Increase `--nfeatures`, `--window`, or lower `--ba-every`
for a more accurate result.

**Usage**

```bash
python sfm_fast.py --images <dir> --out reconstruction_fast.ply
```

**Arguments**

| Flag | Default | Description |
| --- | --- | --- |
| `--images` | *(required)* | Folder of `.jpg`/`.png` frames |
| `--out` | `reconstruction_fast.ply` | Filtered output point cloud |
| `--out-full` | `<out>_full.ply` | Full unfiltered point cloud |
| `--stride` | `5` | Use every Nth frame |
| `--max-dim` | `800` | Longest image side in pixels |
| `--nfeatures` | `3000` | SIFT features per image |
| `--window` | `3` | Match each frame against the next N frames |
| `--ratio` | `0.75` | Lowe's ratio-test threshold |
| `--ba-every` | `10` | Run bundle adjustment every N registrations |

**Outputs**

- `<out>` — statistical-outlier-filtered cloud (cleaner, fewer points)
- `<out>_full` — every triangulated point, unfiltered (denser, noisier)

The two output clouds share the entire pipeline and diverge only at the final
write step:

| Filtered output | Full unfiltered output |
| --- | --- |
| ![sfm_fast.py filtered pipeline](flow_fast.png) | ![sfm_fast.py full pipeline](flow_fast_full.png) |

---

## Results

![Two-view vs. incremental reconstruction](ply_compare.png)

Comparison of the point clouds produced by the two scripts on the same scene:
the sparse two-view result (`sfm_twoview.py`) versus the denser incremental
reconstruction (`sfm_fast.py`).

---

## Notes

- Camera intrinsics are **assumed**, not calibrated: focal length is guessed as
  `f = 1.2 · max(width, height)` with the principal point at the image center.
  For metric accuracy, substitute your camera's real intrinsics.
- Both scripts expect a static scene; moving objects will produce noise or be
  rejected as outliers.
