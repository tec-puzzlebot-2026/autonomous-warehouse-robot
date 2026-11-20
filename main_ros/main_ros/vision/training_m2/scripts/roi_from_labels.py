"""
roi_from_labels.py - pick the ROI (how many rows to cut off the top of the frame) from YOUR
SAM 2 labels instead of guessing. Run on the workstation, before build_dataset_seg.py:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/roi_from_labels.py
    ~/venvs/sem9-vision/bin/python vision_m2/roi_from_labels.py --min-size 120   # other size filter

For every labeled frame it finds the TARGET the robot would pick (the largest box with
size >= MIN_SIZE_PX, size = sqrt(w*h) in frame px, the same rule as near_box.py). Then,
for each candidate ROI top row y0, it measures:
    kept%      targets still >= min size AFTER the crop (the robot measures the cut box)
    visible    mean fraction of the target's box area that stays inside the ROI
    far cut%   small boxes (< min size) removed completely by the crop
    far/frame  small boxes still inside the ROI per frame (the size filter drops those)
    net        network input, and how many fewer pixels than the old 640x640 export
The suggested y0 is the largest one that keeps >= --keep % of the targets.

It also prints a size histogram (is there a gap between far and near boxes? that is where
MIN_SIZE_PX belongs) and writes check/roi_heatmap.jpg: where the targets (red) and the far
boxes (blue) are in the frame, with the candidate ROI lines.
Labels in labels_auto/ are full-frame, so nothing here changes any file.
"""
import argparse
import math
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C
import near_box as NB

CANDIDATES = [0, 32, 96, 160, 224, 288]       # ROI heights 480, 448, 384, 320, 256, 192
HIST = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 1e9]


def net_for(y0):
    """Network input for ROI (0, y0, FRAME_W, FRAME_H) at the configured scale (NET_W / ROI width)."""
    s = C.NET_W / float(C.ROI[2] - C.ROI[0])
    h = (C.FRAME_H - y0) * s
    return C.NET_W, int(math.ceil(h / 32.0) * 32)


def clip_stats(B, y0, min_size):
    """B: (n,4) boxes. -> clipped size, visible fraction, inside(any pixel)."""
    if not len(B):
        return np.zeros(0), np.zeros(0), np.zeros(0, bool)
    c = B.copy()
    c[:, [1, 3]] = c[:, [1, 3]].clip(y0, C.FRAME_H)
    area = lambda b: (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    a0, a1 = area(B), area(c)
    return np.sqrt(a1), a1 / np.maximum(a0, 1), a1 > 0


def heat(B, H, W):
    """How many boxes cover each pixel (difference array, fast for 100k boxes)."""
    d = np.zeros((H + 1, W + 1), np.float32)
    if len(B):
        b = np.round(B).astype(int)
        x1, y1 = b[:, 0].clip(0, W), b[:, 1].clip(0, H)
        x2, y2 = b[:, 2].clip(0, W), b[:, 3].clip(0, H)
        np.add.at(d, (y1, x1), 1)
        np.add.at(d, (y1, x2), -1)
        np.add.at(d, (y2, x1), -1)
        np.add.at(d, (y2, x2), 1)
    return d.cumsum(0).cumsum(1)[:H, :W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=C.LABELS)
    ap.add_argument("--frames", default=C.FRAMES)
    ap.add_argument("--min-size", type=float, default=C.MIN_SIZE_PX, help="near-box size filter (px)")
    ap.add_argument("--keep", type=float, default=98.0, help="%% of targets the ROI must keep")
    args = ap.parse_args()

    targets, far, sizes, n_frames, bg, bg_n = [], [], [], 0, None, 0
    rng = random.Random(0)
    for clip in sorted(os.listdir(args.labels)):
        lab_dir, img_dir = os.path.join(args.labels, clip), os.path.join(args.frames, clip)
        if not os.path.isdir(lab_dir):
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        H, W = cv2.imread(os.path.join(img_dir, imgs[0])).shape[:2]
        sx, sy = C.FRAME_W / float(W), C.FRAME_H / float(H)     # everything in config frame px
        for f in sorted(f for f in os.listdir(lab_dir) if f.endswith(".txt")):
            dets = []
            for line in open(os.path.join(lab_dir, f)):
                v = line.split()
                if len(v) < 7:
                    continue
                p = np.array(v[1:], np.float32).reshape(-1, 2) * [W * sx, H * sy]
                dets.append({"box": [*p.min(0), *p.max(0)]})
            n_frames += 1
            sizes += [NB.box_size(d["box"]) for d in dets]
            t = NB.select_target(dets, args.min_size)
            if t is not None:
                targets.append(t["box"])
            far += [d["box"] for d in dets if NB.box_size(d["box"]) < args.min_size]
        for f in rng.sample(imgs, min(5, len(imgs))):          # average frame = background
            im = cv2.resize(cv2.imread(os.path.join(img_dir, f)), (C.FRAME_W, C.FRAME_H)).astype(np.float32)
            bg = im if bg is None else bg + im
            bg_n += 1
    if not n_frames:
        raise SystemExit(f"No labels in {args.labels}")
    T, F = np.array(targets, np.float32).reshape(-1, 4), np.array(far, np.float32).reshape(-1, 4)

    # ---- size histogram ----
    print(f"{n_frames} labeled frames, {len(sizes)} masks, {len(T)} frames with a near target "
          f"(size >= {args.min_size:g} px)\n")
    print("Box sizes, sqrt(w*h) in frame px:")
    cnt, _ = np.histogram(sizes, HIST)
    for lo, hi, c in zip(HIST[:-1], HIST[1:], cnt):
        mark = "  <- min size" if lo <= args.min_size < hi else ""
        label = f"{lo:4.0f}-{hi:<4.0f}" if hi < 1e9 else f"{lo:4.0f}+    "
        print(f"  {label} {c:7d} {'#' * int(50 * c / max(cnt.max(), 1))}{mark}")
    print("  A good min size sits in an empty valley between the far and the near peak.\n")

    # ---- ROI candidates ----
    print(f"{'y0':>4s} {'ROI':>9s} {'net':>8s} {'px vs 640x640':>14s} {'kept%':>6s} {'visible':>8s} "
          f"{'far cut%':>9s} {'far/frame':>10s}")
    rows, best = [], CANDIDATES[0]
    for y0 in CANDIDATES:
        ts, tv, _ = clip_stats(T, y0, args.min_size)
        _, _, fin = clip_stats(F, y0, args.min_size)
        kept = 100.0 * (ts >= args.min_size).mean() if len(T) else 100.0
        nw, nh = net_for(y0)
        rows.append((y0, kept))
        if kept >= args.keep:
            best = y0
        print(f"{y0:4d} {C.FRAME_W:4d}x{C.FRAME_H - y0:<4d} {nw:3d}x{nh:<4d} {640 * 640 / (nw * nh):13.1f}x "
              f"{kept:6.1f} {tv.mean() if len(T) else 1:8.2f} "
              f"{100.0 * (~fin).mean() if len(F) else 0:9.1f} {fin.sum() / n_frames:10.2f}")
    nw, nh = net_for(best)
    print(f"\nSuggested: ROI = (0, {best}, {C.FRAME_W}, {C.FRAME_H})  NET_W, NET_H = {nw}, {nh}"
          f"   (largest y0 keeping >= {args.keep:g} % of targets)")
    print(f"config_m2 now: ROI = {C.ROI}  NET = {C.NET_W}x{C.NET_H}")
    if len(T):
        q = np.percentile(T, [1, 50, 99], axis=0)
        print(f"Target box rows: top y1 p1/p50/p99 = {q[0, 1]:.0f}/{q[1, 1]:.0f}/{q[2, 1]:.0f}, "
              f"bottom y2 p50 = {q[1, 3]:.0f};  columns x1 p1 = {q[0, 0]:.0f}, x2 p99 = {q[2, 2]:.0f}")
    if len(T) < 300 or len(T) < 0.2 * n_frames:
        print(f"\n[!] Only {len(T)} of {n_frames} frames contain a near box. The robot's real view (a box "
              "10-30 cm away) must be well represented: record clips driving up to boxes, at different "
              "angles and offsets, before training the ROI model.")

    # ---- heat map ----
    img = (bg / bg_n * 0.45).astype(np.uint8) if bg is not None else np.zeros((C.FRAME_H, C.FRAME_W, 3), np.uint8)
    for B, ch in ((F, 0), (T, 2)):                          # blue = far, red = targets
        h = heat(B, C.FRAME_H, C.FRAME_W)
        if h.max() > 0:
            h = np.log1p(h) / np.log1p(h.max())
            img[..., ch] = np.maximum(img[..., ch], (255 * h).astype(np.uint8))
    for y0, kept in rows:
        col, th = ((255, 255, 255), 2) if y0 == best else ((150, 150, 150), 1)
        cv2.line(img, (0, y0), (C.FRAME_W - 1, y0), col, th)
        cv2.putText(img, f"y0={y0} kept {kept:.0f}%", (C.FRAME_W - 175, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(img, "red = near targets   blue = far boxes", (8, C.FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    os.makedirs(C.CHECK, exist_ok=True)
    out = os.path.join(C.CHECK, "roi_heatmap.jpg")
    cv2.imwrite(out, img)
    print(f"\nHeat map: {out}")


if __name__ == "__main__":
    main()