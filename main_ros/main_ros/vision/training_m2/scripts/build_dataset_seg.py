"""
build_dataset_seg.py - SAM 2 segmentation labels + frames -> YOLO-seg dataset,
with photometric augmentation of the TRAIN set. Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/build_dataset_seg.py          # ROI-cropped (Jetson)
    ~/venvs/sem9-vision/bin/python vision_m2/build_dataset_seg.py --full   # old full-frame dataset

ROI mode (default): every frame is cut to config_m2.ROI, the SAME crop the Jetson makes,
and every SAM 2 polygon is clipped to it. A box partly inside keeps its visible part
(the robot also sees cut boxes); pieces smaller than --min-area px are dropped.
The SAM 2 labels in labels_auto/ stay full-frame, so you can change the ROI and rebuild
without re-running SAM 2.

Result:
    vision_m2/yolo_dataset_roi/   (or yolo_dataset/ with --full)
        images/train/  labels/train/   clips WITHOUT 'test'/'val' in the name (+ augmented copies)
        images/val/    labels/val/     clips WITH 'test'/'val' in the name - never augmented
        data.yaml
    vision_m2/check/augment_preview.jpg   original vs augmented, outlines drawn

Image names get the clip name as prefix (bag_boxes_train_1__frame_00012.jpg) so
frames of different bags never overwrite each other.
Photometric augmentation never moves pixels, so the polygons stay correct.
Geometric augmentation (zoom, shift, mosaic) is done by YOLO while training.
"""
import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


# ----------------------------------------------------------------------------
# Photometric augmentations (each takes and returns a BGR uint8 image)
# ----------------------------------------------------------------------------
def brightness_contrast(img, rng):
    alpha = rng.uniform(0.6, 1.4)            # contrast
    beta = rng.uniform(-45, 45)              # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gamma(img, rng):
    g = rng.uniform(0.5, 1.8)                # <1 brighter, >1 darker (exposure)
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def color_temperature(img, rng):
    shift = rng.uniform(-30, 30)             # + warmer (more red), - cooler (more blue)
    out = img.astype(np.float32)
    out[..., 2] += shift
    out[..., 0] -= shift
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(img, rng):
    k = rng.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, rng):
    k = rng.choice([5, 7, 9, 11])            # robot moving / camera shake
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def noise(img, rng):
    sigma = rng.uniform(4, 18)               # camera sensor noise
    n = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def shadow(img, rng):
    h, w = img.shape[:2]                     # dark polygon, like a shadow on the scene
    pts = np.array([[rng.randint(0, w), rng.randint(0, h)] for _ in range(rng.randint(3, 6))],
                   np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)
    mask = cv2.GaussianBlur(mask, (51, 51), 0).astype(np.float32) / 255.0
    factor = 1.0 - rng.uniform(0.3, 0.6) * mask[..., None]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def jpeg(img, rng):
    q = rng.randint(25, 70)                  # compression artifacts
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


LIGHT = [brightness_contrast, gamma, color_temperature]
DEGRADE = [gaussian_blur, motion_blur, noise, shadow, jpeg]


def augment(img, rng):
    """1 lighting change + 1-2 degradations, in random order."""
    ops = [rng.choice(LIGHT)] + rng.sample(DEGRADE, rng.randint(1, 2))
    rng.shuffle(ops)
    for op in ops:
        img = op(img, rng)
    return img, [op.__name__ for op in ops]


# ----------------------------------------------------------------------------
# ROI cropping of polygons
# ----------------------------------------------------------------------------
def clip_polygon(pts, x0, y0, x1, y1):
    """Sutherland-Hodgman: clip a polygon (Nx2, any shape) to the rectangle. Returns Mx2 or []."""
    def clip(poly, inside, cross):
        out = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def at_x(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def at_y(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    poly = [tuple(map(float, p)) for p in pts]
    for inside, cross in ((lambda p: p[0] >= x0, at_x(x0)), (lambda p: p[0] <= x1, at_x(x1)),
                          (lambda p: p[1] >= y0, at_y(y0)), (lambda p: p[1] <= y1, at_y(y1))):
        if not poly:
            break
        poly = clip(poly, inside, cross)
    return np.array(poly, np.float32).reshape(-1, 2)


def crop_label_lines(lines, W, H, roi, min_area):
    """Full-frame YOLO-seg lines -> lines normalised to the ROI crop. Returns (lines, n_dropped)."""
    x0, y0, x1, y1 = roi
    cw, ch = x1 - x0, y1 - y0
    out, dropped = [], 0
    for line in lines:
        v = line.split()
        pts = np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]
        p = clip_polygon(pts, x0, y0, x1, y1)
        if len(p) < 3 or cv2.contourArea(p) < min_area:
            dropped += 1
            continue
        p = (p - [x0, y0]) / [cw, ch]
        out.append(v[0] + " " + " ".join(f"{x:.6f} {y:.6f}" for x, y in p.clip(0, 1)))
    return out, dropped


def scaled_roi(W, H):
    """config ROI is for FRAME_W x FRAME_H; scale it if the frames have another size."""
    x0, y0, x1, y1 = C.ROI
    sx, sy = W / C.FRAME_W, H / C.FRAME_H
    return int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))


# ----------------------------------------------------------------------------
def draw_labels(img, label_lines):
    """Draw YOLO-seg polygons:  cls x1 y1 x2 y2 ..."""
    h, w = img.shape[:2]
    out, overlay = img.copy(), img.copy()
    polys = []
    for line in label_lines:
        v = line.split()
        c = int(v[0])
        pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
        polys.append((c, pts))
        cv2.fillPoly(overlay, [pts], C.COLORS[c])
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for c, pts in polys:
        cv2.polylines(out, [pts], True, C.COLORS[c], 2)
        x, y = pts.min(axis=0)
        cv2.putText(out, C.CLASSES[c], (int(x), max(14, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C.COLORS[c], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=C.FRAMES)
    ap.add_argument("--labels", default=C.LABELS)
    ap.add_argument("--out", help="default: yolo_dataset_roi (or yolo_dataset with --full)")
    ap.add_argument("--full", action="store_true", help="no ROI crop (old behaviour)")
    ap.add_argument("--min-area", type=float, default=150,
                    help="ROI mode: drop clipped polygons smaller than this (px), same as SAM 2 --min-area")
    ap.add_argument("--aug-per-image", type=int, default=1,
                    help="augmented copies per TRAIN image (1 = train set doubles)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out = args.out or (C.DATASET if args.full else C.DATASET_ROI)
    rng = random.Random(args.seed)
    nc = len(C.CLASSES)
    clipped_away = 0
    if not args.full:
        print(f"ROI {C.ROI} of {C.FRAME_W}x{C.FRAME_H}  (network input {C.NET_W}x{C.NET_H})")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        os.makedirs(f"{args.out}/images/{split}")
        os.makedirs(f"{args.out}/labels/{split}")

    clips = sorted(d for d in os.listdir(args.labels) if os.path.isdir(os.path.join(args.labels, d)))
    stats = {s: [0, 0, [0] * nc, 0] for s in ("train", "val")}   # orig, aug, masks/class, empty
    preview = []

    for clip in clips:
        split = "val" if C.is_val(clip) else "train"
        lab_dir, img_dir = os.path.join(args.labels, clip), os.path.join(args.frames, clip)
        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lab_dir) if f.endswith(".txt"))
        print(f"{clip:24s} -> {split:5s}  {len(stems)} images"
              + (f" + {len(stems) * args.aug_per_image} augmented" if split == "train" else ""))

        for stem in stems:
            img_path = os.path.join(img_dir, imgs[stem])
            with open(os.path.join(lab_dir, stem + ".txt")) as f:
                lines = [l.strip() for l in f if l.strip()]
            name = f"{clip}__{stem}"
            img = cv2.imread(img_path)
            if not args.full:
                H, W = img.shape[:2]
                x0, y0, x1, y1 = scaled_roi(W, H)
                lines, nd = crop_label_lines(lines, W, H, (x0, y0, x1, y1), args.min_area)
                clipped_away += nd
                img = np.ascontiguousarray(img[y0:y1, x0:x1])
            label_text = "".join(l + "\n" for l in lines)
            # always save as .jpg so image and label names match
            cv2.imwrite(f"{args.out}/images/{split}/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(f"{args.out}/labels/{split}/{name}.txt", "w") as fh:
                fh.write(label_text)
            stats[split][0] += 1
            stats[split][3] += not lines
            for l in lines:
                stats[split][2][int(l.split()[0])] += 1

            if split == "train":
                for k in range(args.aug_per_image):
                    aug, ops = augment(img, rng)
                    cv2.imwrite(f"{args.out}/images/train/{name}_aug{k}.jpg", aug,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(f"{args.out}/labels/train/{name}_aug{k}.txt", "w") as fh:
                        fh.write(label_text)
                    stats["train"][1] += 1
                    for l in lines:
                        stats["train"][2][int(l.split()[0])] += 1
                    if len(preview) < 8 and lines and rng.random() < 0.02:
                        pw, ph = 320, int(round(320 * img.shape[0] / img.shape[1]))
                        o = cv2.resize(draw_labels(img, lines), (pw, ph))
                        a = cv2.resize(draw_labels(aug, lines), (pw, ph))
                        cv2.putText(a, "+".join(ops)[:40], (5, ph - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        preview.append(np.hstack([o, a]))

    with open(f"{args.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n"
                "train: images/train\n"
                "val: images/val\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(C.CLASSES)))

    if preview:
        while len(preview) % 2:
            preview.append(np.zeros_like(preview[0]))
        os.makedirs(C.CHECK, exist_ok=True)
        cv2.imwrite(os.path.join(C.CHECK, "augment_preview.jpg"),
                    np.vstack([np.hstack(preview[i:i + 2]) for i in range(0, len(preview), 2)]))

    print("\nDataset summary")
    for split in ("train", "val"):
        o, a, m, e = stats[split]
        print(f"  {split:5s}: {o + a:5d} images ({o} original + {a} augmented, {e} originals with no box)"
              " | masks: " + ", ".join(f"{C.CLASSES[i]} {m[i]}" for i in range(nc)))
    if not args.full:
        print(f"  ROI crop: {clipped_away} masks outside the ROI or smaller than {args.min_area:g} px dropped")
        print(f"  Train with:  train_yolo_seg.py --imgsz {C.NET_W}   (= the {C.NET_W}x{C.NET_H} network input)")
    if stats["val"][0] == 0:
        print("[!] No validation clips found - clip names need 'test' or 'val' in them.")
    print(f"\nWrote {args.out}/data.yaml")
    print(f"Preview: {C.CHECK}/augment_preview.jpg  (left = original, right = augmented)")


if __name__ == "__main__":
    main()