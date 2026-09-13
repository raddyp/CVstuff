# KITTI Depth Completion (U-Net)

Three PyTorch scripts that train a small U-Net to turn **sparse LiDAR depth** into
**dense depth** on the KITTI dataset. They differ only in their input and data
layout; the model, loss, metrics, and training loop are shared.

| Script | Tier | Input | Data layout |
| --- | --- | --- | --- |
| `kitti_depth.py` | **1** | sparse depth (1-ch) | nested depth tree (`train/`, `val/`) |
| `kitti_depth_rgb_subset.py` | **2** | RGB + sparse (4-ch) | flat 1000-frame subset (self-split) |
| `kitti_depth_rgb_full.py` | **3** | RGB + sparse (4-ch) | depth tree + raw-KITTI RGB (joined by drive) |

## Overview

All three tiers feed the same U-Net encoder–decoder and the same masked-loss
training loop; only the input channels and dataset differ.

![KITTI depth completion — conceptual overview](flow_concept.png)

## Requirements

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

- `torch` — model, training, MPS/CPU device
- `opencv-python` — 16-bit PNG depth I/O, RGB decode, colorized visualizations
- `numpy` — array ops, crop/flip augmentation
- `matplotlib` — training-curve plots

Runs on Apple Silicon (`mps`) if available, otherwise `cpu`. Each script is run
directly (no CLI args) — configuration lives in the `# 1. Config` block at the
top of each file (`DATA_ROOT`, `CROP`, `BATCH`, `LR`, `EPOCHS`, `OUT_DIR`).

```bash
python kitti_depth.py            # Tier 1
python kitti_depth_rgb_subset.py # Tier 2
python kitti_depth_rgb_full.py   # Tier 3
```

## Shared components

Every script defines the same core:

- **`read_depth`** — reads a KITTI 16-bit depth PNG → meters (`uint16 / 256`) plus
  a validity mask (`png > 0`); `0` means "no measurement".
- **`UNet`** (with `DoubleConv`) — 4-level encoder / bottleneck / 4-level decoder
  with skip connections, `base=32`, and a `softplus` head so depth is `>= 0`.
  Input side must be divisible by 16, hence the crop sizes.
- **`masked_loss`** — MSE (or L1) computed over valid pixels only.
- **KITTI metrics** (`compute_metrics` → `finalize`): `RMSE_mm`, `MAE_mm`,
  `iRMSE`, `iMAE`, and `delta1` (fraction with `max(p/g, g/p) < 1.25`).
- **`train_one_epoch` / `evaluate`** — training pass and no-grad metric pass.
- **`save_metrics` / `plot_history` / `run_test`** — write JSON+TXT metrics,
  plot loss/error/accuracy curves, and save colorized comparison strips.

The best checkpoint (`best.pt`) is kept whenever validation `RMSE_mm` improves.

---

## Tier 1 — `kitti_depth.py` (depth-only)

Sparse depth in, dense depth out (single channel). The simplest baseline: no
RGB, so the network has only the sparse LiDAR points to work from.

- **Dataset** `KittiDepthOnly`: pairs `proj_depth/velodyne_raw/image_02/*.png`
  (input) with the matching `groundtruth` map (target).
- **Input** 1-channel sparse depth → `UNet(in_ch=1)`.
- **Config** crop `256×512`, batch 4, LR 1e-3, 5 epochs.

![kitti_depth.py pipeline](flow_tier1_depth.png)

---

## Tier 2 — `kitti_depth_rgb_subset.py` (RGB + depth, subset)

Adds RGB guidance on a self-contained flat subset of 1000 frames. RGB gives the
network appearance cues (edges, surfaces) that sharpen the completed depth.

- **Split** `make_splits` deterministically partitions the frames 900 / 100
  (fixed seed) — no pre-defined train/val split needed.
- **Dataset** `KittiDepthRGB`: concatenates ImageNet-normalized RGB with sparse
  depth → 4-channel input; also returns the per-frame intrinsics `K`
  (adjusted for the crop offset).
- **Input** 4-channel RGB+sparse → `UNet(in_ch=4)`.
- **Config** crop `352×1216` (full frame), batch 4, LR 1e-3, 25 epochs.

![kitti_depth_rgb_subset.py pipeline](flow_tier2_rgb_subset.png)

---

## Tier 3 — `kitti_depth_rgb_full.py` (RGB + depth, full)

Same 4-channel RGB+depth model as Tier 2, but on the full dataset. RGB is not
bundled with the depth tree, so it is pulled from a separate raw-KITTI download
and joined per frame.

- **Dataset** `KittiDepthRGBFull`: reads sparse+GT from the depth tree and finds
  the matching RGB in `RAW_ROOT/<date>/<drive>/image_02/data/<frame>.png`
  (date = `drive[:10]`); crops RGB and depth to a common size.
- **Input** 4-channel RGB+sparse → `UNet(in_ch=4)`.
- **Config** crop `256×512`, batch 4, LR 1e-3, 8 epochs. Requires `RAW_ROOT` to
  point at the raw-KITTI RGB drives.

![kitti_depth_rgb_full.py pipeline](flow_tier3_rgb_full.png)

---

## Outputs

Each run writes to its `OUT_DIR`:

- `best.pt` — best-by-RMSE model checkpoint
- `val_metrics.json` / `.txt` — final validation metrics
- `history.json` + `curves.png` — per-epoch loss / error / accuracy
- `test_vis/sample_*.png` — colorized comparison strips
  (Tier 1: `[sparse | pred | GT]`; Tiers 2–3: `[RGB | sparse | pred | GT]`)

## Notes

- Depth PNGs are 16-bit and must **not** be rescaled on load — only divided by
  256 to convert to meters.
- Crops are applied identically to every map (input / GT / mask / RGB); alignment
  is critical, and horizontal-flip augmentation is train-only.
- KITTI metrics are computed over valid ground-truth pixels only.
