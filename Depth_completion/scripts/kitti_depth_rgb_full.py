#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:13:41 2026

"""


import numpy as np
import cv2, json, os
import matplotlib.pyplot as plt

import time
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch, torch.nn as nn, torch.nn.functional as F


# 1. Config 
DATA_ROOT = "/Volumes/WH/Data/ML/data/kitti_depth_only_full"   # depth tree (sparse+GT)
RAW_ROOT  = "/Volumes/WH/Data/ML/data/kitti_raw"               # raw KITTI RGB
# CROP = (352, 1216)          # full frame, no cropping
CROP = (256, 512)          #  cropping, to speed up training, augmentation, increase epochs to ~8
BATCH, LR, EPOCHS, NUM_WORKERS = 4, 1e-3, 8, 6
OUT_DIR   = "/Volumes/WH/Data/ML/res/depth_rgb_full"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# 2. data IO
#data prep is very important here


def read_depth(path):
    """KITTI depth PNG -> (meters float32 HxW, valid mask float32 HxW). 0 == invalid."""
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)      # uint16; do NOT rescale
    assert png is not None, f"unreadable: {path}"
    return png.astype(np.float32) / 256.0, (png > 0).astype(np.float32)

# read intrinsics
def read_K(path):
    return np.loadtxt(path).astype(np.float32).reshape(3,3)   # fx=[0,0] cx=[0,2] fy=[1,1] cy=[1,2]

# 3. Dataloader 
class KittiDepthRGBFull(Dataset):
    """Tree depth (sparse+GT) + RGB joined from raw KITTI by drive/date. 4-channel."""
    def __init__(self, root, raw_root, split="train", crop=CROP, train=True):
        self.raw_root, self.crop, self.train = Path(raw_root), crop, train
        # image_02 only (that's what we downloaded)
        self.samples = sorted((Path(root)/split).glob("*/proj_depth/velodyne_raw/image_02/*.png"))
        assert self.samples, f"no samples under {Path(root)/split}"

    def __len__(self):
        return len(self.samples)

    def _rgb_path(self, sp):
        # sp = .../<drive>_sync/proj_depth/velodyne_raw/image_02/000005.png
        p = sp.parts
        drive, cam, fname = p[-5], p[-2], p[-1]      # drive has _sync suffix
        return self.raw_root / drive[:10] / drive / cam / "data" / fname

    def __getitem__(self, i):
        sp = self.samples[i]
        sparse, _ = read_depth(sp)
        gt, mask  = read_depth(Path(str(sp).replace("velodyne_raw", "groundtruth")))

        bgr = cv2.imread(str(self._rgb_path(sp)))
        if bgr is None:
            raise FileNotFoundError(f"RGB missing: {self._rgb_path(sp)}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # raw RGB can be a few px larger than the depth map -> crop to common size
        h = min(rgb.shape[0], sparse.shape[0]); w = min(rgb.shape[1], sparse.shape[1])
        rgb, sparse, gt, mask = rgb[:h,:w], sparse[:h,:w], gt[:h,:w], mask[:h,:w]

        # one crop box for all maps
        ch, cw = min(self.crop[0], h), min(self.crop[1], w)
        if self.train:
            top, left = np.random.randint(0, h-ch+1), np.random.randint(0, w-cw+1)
        else:
            top, left = h-ch, (w-cw)//2
        box = (slice(top, top+ch), slice(left, left+cw))
        rgb, sparse, gt, mask = rgb[box], sparse[box], gt[box], mask[box]

        if self.train and np.random.rand() < 0.5:
            rgb, sparse, gt, mask = rgb[:,::-1], sparse[:,::-1], gt[:,::-1], mask[:,::-1]

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return {"input": torch.cat([t(rgb).permute(2,0,1), t(sparse).unsqueeze(0)], 0),  # (4,H,W)
                "gt": t(gt).unsqueeze(0), "mask": t(mask).unsqueeze(0)}


# 4. Model (small U-Net) 
class DoubleConv(nn.Module):
    """(conv-BN-ReLU)x2. (B,in,H,W)->(B,out,H,W)."""
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """(B,1,H,W) sparse -> (B,1,H,W) dense depth (>=0). H,W divisible by 16."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.d1, self.d2 = DoubleConv(in_ch, base), DoubleConv(base, base*2)
        self.d3, self.d4 = DoubleConv(base*2, base*4), DoubleConv(base*4, base*8)
        self.bott = DoubleConv(base*8, base*16); self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.u4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.u3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.u2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2);   self.u1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1)); c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3)); b = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return F.softplus(self.out(x))                          # depth >= 0


# 5. METRICS 
def masked_loss(pred, gt, mask, kind="l2"):
    """Loss over valid pixels only. pred/gt/mask (B,1,H,W) -> scalar."""
    diff = (pred - gt) * mask
    n = mask.sum().clamp(min=1.0)
    return diff.abs().sum()/n if kind == "l1" else (diff**2).sum()/n


@torch.no_grad()
def compute_metrics(pred, gt, mask):
    """Accumulate summed errors over valid px for one batch -> dict of sums+n."""
    m = mask > 0.5
    p, g = pred[m].clamp(min=1e-3), gt[m].clamp(min=1e-3)
    r = torch.max(p/g, g/p)
    return {"se": ((p-g)**2).sum().item(), "ae": (p-g).abs().sum().item(),
            "ise": ((1/p-1/g)**2).sum().item(), "iae": (1/p-1/g).abs().sum().item(),
            "d1": (r < 1.25).float().sum().item(), "n": p.numel()}


def finalize(acc):
    """Summed metrics -> KITTI units (RMSE/MAE mm, iRMSE/iMAE 1/km, delta1)."""
    n = max(acc["n"], 1)
    return {"RMSE_mm": 1000*(acc["se"]/n)**0.5, "MAE_mm": 1000*acc["ae"]/n,
            "iRMSE": 1000*(acc["ise"]/n)**0.5, "iMAE": 1000*acc["iae"]/n,
            "delta1": acc["d1"]/n}


# 6. Epochs
def train_one_epoch(model, loader, opt, ep):
    """One pass over train set. Returns avg loss (float). Updates weights."""
    model.train(); run = seen = 0; t0 = time.time()
    for it, b in enumerate(loader):
        inp, gt, mk = b["input"].to(DEVICE), b["gt"].to(DEVICE), b["mask"].to(DEVICE)
        loss = masked_loss(model(inp), gt, mk)
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item()*inp.size(0); seen += inp.size(0)
        if it % 50 == 0:
            print(f"  ep{ep} it{it:4d}/{len(loader)} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
    return run/max(seen, 1)


@torch.no_grad()
def evaluate(model, loader):
    """Compute KITTI metrics over a loader (no grad). Returns dict of floats."""
    model.eval(); acc = {k: 0 for k in ["se", "ae", "ise", "iae", "d1", "n"]}
    for b in loader:
        inp, gt, mk = b["input"].to(DEVICE), b["gt"].to(DEVICE), b["mask"].to(DEVICE)
        for k, v in compute_metrics(model(inp), gt, mk).items(): acc[k] += v
    return finalize(acc)



# ---------- A. save metrics ----------
def save_metrics(metrics: dict, out_dir="res", name="metrics"):
    """Write a metrics dict to both human-readable .txt and machine .json."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/f"{name}.json").write_text(json.dumps(metrics, indent=2))
    with open(out/f"{name}.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n" if isinstance(v, float) else f"{k:12s}: {v}\n")
    print(f"wrote {out/name}.txt and .json")

# ---------- B. plot training history ----------
def plot_history(history: list[dict], out_dir="res", name="curves"):
    """history = list of per-epoch dicts, e.g.
       [{"epoch":0,"train_loss":..,"RMSE_mm":..,"MAE_mm":..,"delta1":..}, ...]"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", color="#1f5c8a")
    ax[0].set(title="Train loss (masked MSE)", xlabel="epoch", ylabel="loss"); ax[0].grid(alpha=.3)

    ax[1].plot(ep, [h["RMSE_mm"] for h in history], "o-", label="RMSE", color="#b5651d")
    ax[1].plot(ep, [h["MAE_mm"]  for h in history], "s-", label="MAE",  color="#2e7d32")
    ax[1].set(title="Val error (mm)", xlabel="epoch", ylabel="mm"); ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(ep, [h["delta1"] for h in history], "o-", color="#6a3d9a")
    ax[2].set(title="Val δ<1.25 (accuracy)", xlabel="epoch", ylabel="fraction", ylim=(0, 1)); ax[2].grid(alpha=.3)

    fig.tight_layout(); fig.savefig(out/f"{name}.png", dpi=130); plt.close(fig)
    print(f"wrote {out/name}.png")
 
    
def loader(split, train):
    ds = KittiDepthRGBFull(DATA_ROOT, RAW_ROOT, split, crop=CROP, train=train)
    return DataLoader(ds, batch_size=BATCH, shuffle=train, num_workers=NUM_WORKERS,
                      pin_memory=False, drop_last=train)


@torch.no_grad()
def run_test(model, ds, num, out_dir):
    """Save [RGB | sparse | pred | gt] strips for `num` samples."""
    model.eval();
    os.makedirs(out_dir, exist_ok=True)
    color = lambda d: cv2.applyColorMap((np.clip(d/80, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for i in range(min(num, len(ds))):
        s = ds[i]
        pred = model(s["input"].unsqueeze(0).to(DEVICE))[0, 0].cpu().numpy()
        # un-normalize RGB (first 3 channels) back to viewable BGR
        rgb = s["input"][:3].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
        rgb = cv2.cvtColor((np.clip(rgb, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        strip = np.concatenate([rgb, color(s["input"][3].numpy()), color(pred), color(s["gt"][0].numpy())], 1)
        cv2.imwrite(os.path.join(out_dir, f"sample_{i:02d}.png"), strip)
    print(f"wrote {min(num, len(ds))} visualizations to {out_dir}")


#%%


    
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tr, va = loader("train", True), loader("val", False)
    print(f"[TIER 3] device={DEVICE}  train batches={len(tr)}  val batches={len(va)}")

    model = UNet(in_ch=4).to(DEVICE)                           # 4ch: RGB + sparse
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    chkpt = os.path.join(OUT_DIR, "best.pt")
    best = float("inf")

    history = []
    for ep in range(EPOCHS):
        loss = train_one_epoch(model, tr, opt, ep)            # ONE train pass
        m = evaluate(model, va)
        history.append({"epoch": ep, "train_loss": loss, **m})
        print(f"[ep{ep}] loss={loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        if m["RMSE_mm"] < best:
            best = m["RMSE_mm"]; torch.save(model.state_dict(), chkpt)
            print(f"  saved {chkpt} (RMSE {best:.1f}mm)")

    # logs + curves
    save_metrics(history[-1], OUT_DIR, "val_metrics")
    plot_history(history, OUT_DIR, "curves")
    json.dump(history, open(os.path.join(OUT_DIR, "history.json"), "w"), indent=2)

    # best-model eval + visualizations
    model.load_state_dict(torch.load(chkpt, map_location=DEVICE))
    print("best val:", " ".join(f"{k}={v:.3f}" for k, v in evaluate(model, va).items()))
    run_test(model, KittiDepthRGBFull(DATA_ROOT, RAW_ROOT, "val", crop=CROP, train=False), 8,
             os.path.join(OUT_DIR, "test_vis"))


if __name__ == "__main__":
    main()
    
    
    