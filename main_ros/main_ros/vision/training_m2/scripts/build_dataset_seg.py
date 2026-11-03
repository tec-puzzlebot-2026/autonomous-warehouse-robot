"""
build_dataset_seg.py - SAM 2 segmentation labels + frames -> YOLO-seg dataset,
with photometric augmentation of the TRAIN set. Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/build_dataset_seg.py

Result:
    vision_m2/yolo_dataset/
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
    ap.add_argument("--out", default=C.DATASET)
    ap.add_argument("--aug-per-image", type=int, default=1,
                    help="augmented copies per TRAIN image (1 = train set doubles)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nc = len(C.CLASSES)

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
            # always save as .jpg so image and label names match
            cv2.imwrite(f"{args.out}/images/{split}/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(os.path.join(lab_dir, stem + ".txt"), f"{args.out}/labels/{split}/{name}.txt")
            stats[split][0] += 1
            stats[split][3] += not lines
            for l in lines:
                stats[split][2][int(l.split()[0])] += 1

            if split == "train":
                for k in range(args.aug_per_image):
                    aug, ops = augment(img, rng)
                    cv2.imwrite(f"{args.out}/images/train/{name}_aug{k}.jpg", aug,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    shutil.copy(os.path.join(lab_dir, stem + ".txt"),
                                f"{args.out}/labels/train/{name}_aug{k}.txt")
                    stats["train"][1] += 1
                    for l in lines:
                        stats["train"][2][int(l.split()[0])] += 1
                    if len(preview) < 8 and lines and rng.random() < 0.02:
                        o = cv2.resize(draw_labels(img, lines), (320, 240))
                        a = cv2.resize(draw_labels(aug, lines), (320, 240))
                        cv2.putText(a, "+".join(ops)[:40], (5, 232),
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
    if stats["val"][0] == 0:
        print("[!] No validation clips found - clip names need 'test' or 'val' in them.")
    print(f"\nWrote {args.out}/data.yaml")
    print(f"Preview: {C.CHECK}/augment_preview.jpg  (left = original, right = augmented)")


if __name__ == "__main__":
    main()