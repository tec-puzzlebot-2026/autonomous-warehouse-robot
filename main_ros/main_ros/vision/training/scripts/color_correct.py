"""
color_correct.py - apply the camera's colour calibration (colorCalibration.npz)
to every frame, so the training images look like what the calibrated robot camera sees.

Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    source ~/venvs/sem9-vision/bin/activate
    python vision/color_correct.py --inspect          # 1. look inside the .npz first
    python vision/color_correct.py                    # 2. correct all frames

Input : dataset/frames/<folder>/*.jpg                 (originals, never modified)
Output: dataset/frames_color/<folder>/*.jpg           (same names, colour-corrected)
        vision/check/color_correct_preview.jpg         (left original | right corrected)

Only pixel COLOURS change - nothing moves - so the SAM 2 labels stay valid.

How the correction works:
  The calibration stores a GAIN per colour channel (B, G, R). The summary
  printed during calibration showed min/mean/max per channel, which means the
  gain is stored per PIXEL (a gain map): it corrects the colour cast (pink tint)
  AND the darker image corners (vignetting) at the same time.
      corrected_pixel = original_pixel * gain_at_that_pixel
  The map was computed at 820x616; our frames are 640x480 (same 4:3 view,
  just smaller), so the map is resized to 640x480.
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

TEAM = os.path.expanduser("~/data/cosmostests/sem9_team")


def load_gain(npz_path, key=None, order="bgr"):
    """Return a gain array: (H, W, 3) map or (3,) per-channel gains, in BGR order."""
    data = np.load(npz_path)
    if key is None:
        # pick the most likely array: a 3-channel map, else a 3-vector
        cands = [k for k in data.files if data[k].ndim == 3 and 3 in data[k].shape]
        cands += [k for k in data.files if data[k].shape == (3,)]
        prefer = [k for k in cands if "gain" in k.lower()]
        if not (prefer or cands):
            raise SystemExit(f"Could not find a gain array in {npz_path}. Keys: {data.files}. "
                             "Run with --inspect and pass --key NAME.")
        key = (prefer or cands)[0]
    g = data[key].astype(np.float32)
    if g.ndim == 3 and g.shape[0] == 3 and g.shape[-1] != 3:      # (3, H, W) -> (H, W, 3)
        g = np.transpose(g, (1, 2, 0))
    if order == "rgb":
        g = g[..., ::-1]
    return key, g


def apply_gain(img, gain):
    h, w = img.shape[:2]
    if gain.ndim == 3 and gain.shape[:2] != (h, w):
        gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=f"{TEAM}/calibration/colorCalibration.npz")
    ap.add_argument("--src", default=f"{TEAM}/dataset/frames")
    ap.add_argument("--dst", default=f"{TEAM}/dataset/frames_color")
    ap.add_argument("--key", default=None, help="array name inside the .npz (auto if omitted)")
    ap.add_argument("--order", choices=["bgr", "rgb"], default="bgr",
                    help="channel order of the stored gains (calibration printed B, G, R)")
    ap.add_argument("--inspect", action="store_true", help="only print what is inside the .npz")
    args = ap.parse_args()

    data = np.load(args.calib)
    print(f"Calibration file: {args.calib}")
    for k in data.files:
        a = data[k]
        info = f"shape {a.shape}, dtype {a.dtype}"
        if a.dtype.kind == "f" and a.size:
            if a.ndim == 3:
                ch = (a.reshape(-1, 3) if a.shape[-1] == 3 else a.reshape(3, -1).T).astype(np.float64)
                info += " | per channel mean " + ", ".join(f"{m:.3f}" for m in ch.mean(0))
            else:
                info += f" | min {a.min():.3f} mean {a.mean():.3f} max {a.max():.3f}"
        print(f"  {k:20s} {info}")
    if args.inspect:
        return

    key, gain = load_gain(args.calib, args.key, args.order)
    desc = (f"gain map {gain.shape[1]}x{gain.shape[0]}" if gain.ndim == 3
            else f"gains B,G,R = {gain.round(3).tolist()}")
    print(f"\nUsing '{key}': {desc}")

    folders = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    preview = []
    for fo in folders:
        os.makedirs(os.path.join(args.dst, fo), exist_ok=True)
        files = sorted(f for f in os.listdir(os.path.join(args.src, fo))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for i, f in enumerate(tqdm(files, desc=f"  {fo:14s}", unit="img", dynamic_ncols=True)):
            img = cv2.imread(os.path.join(args.src, fo, f))
            out = apply_gain(img, gain)
            cv2.imwrite(os.path.join(args.dst, fo, f), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if i == len(files) // 2:                       # one example per folder
                pair = np.hstack([cv2.resize(img, (320, 240)), cv2.resize(out, (320, 240))])
                cv2.putText(pair, fo, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                preview.append(pair)

    os.makedirs(f"{TEAM}/vision/check", exist_ok=True)
    cv2.imwrite(f"{TEAM}/vision/check/color_correct_preview.jpg", np.vstack(preview))
    print(f"\nDone -> {args.dst}")
    print(f"Preview: {TEAM}/vision/check/color_correct_preview.jpg  (left original | right corrected)")


if __name__ == "__main__":
    main()
