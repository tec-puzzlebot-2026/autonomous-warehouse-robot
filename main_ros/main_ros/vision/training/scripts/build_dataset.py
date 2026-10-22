"""
build_dataset.py - SAM 2 labels + frames -> YOLO dataset, with augmentation of the TRAIN set.
Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision/build_dataset.py

Result:
    vision/yolo_dataset/
        images/train/   originals + augmented copies (logo_*_train folders)
        labels/train/   one .txt per image (augmented copies reuse the original's label)
        images/val/     originals only (logo_*_val folders) - NEVER augmented
        labels/val/
        data.yaml       tells YOLO where everything is + the class names
    vision/check/augment_preview.jpg   originals vs augmented, with boxes drawn

Only PHOTOMETRIC augmentation is done here (light, colour, blur, noise, shadow,
JPEG). These change pixel values but never move the logo, so the label boxes
stay correct. GEOMETRIC changes (zoom, shift, perspective, mosaic) are done by
YOLO itself during training, where the boxes are moved together with the image.
"""
import argparse
import os
import random
import shutil

import cv2
import numpy as np

TEAM = os.path.expanduser("~/data/cosmostests/sem9_team")
CLASSES = ["yinyang", "emezon", "walnut"]
COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0)]


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
    h, w = img.shape[:2]
    out = img.copy()
    for line in label_lines:
        c, cx, cy, bw, bh = line.split()
        c = int(c)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1 = (int(cx - bw / 2), int(cy - bh / 2))
        p2 = (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(out, p1, p2, COLORS[c], 2)
        cv2.putText(out, CLASSES[c], (p1[0], max(14, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=f"{TEAM}/dataset/frames")
    ap.add_argument("--labels", default=f"{TEAM}/vision/labels_auto")
    ap.add_argument("--out", default=f"{TEAM}/vision/yolo_dataset")
    ap.add_argument("--aug-per-image", type=int, default=1,
                    help="augmented copies per TRAIN image (1 = dataset doubles)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        os.makedirs(f"{args.out}/images/{split}")
        os.makedirs(f"{args.out}/labels/{split}")

    folders = sorted(d for d in os.listdir(args.labels)
                     if os.path.isdir(os.path.join(args.labels, d)))
    stats = {"train": [0, 0, [0, 0, 0]], "val": [0, 0, [0, 0, 0]]}   # orig, aug, boxes/class
    preview = []

    for fo in folders:
        split = "val" if fo.endswith("_val") else "train"
        lab_dir = os.path.join(args.labels, fo)
        img_dir = os.path.join(args.frames, fo)
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lab_dir) if f.endswith(".txt"))
        print(f"{fo:14s} -> {split:5s}  {len(stems)} images"
              + (f" + {len(stems) * args.aug_per_image} augmented" if split == "train" else ""))

        for stem in stems:
            img_path = os.path.join(img_dir, stem + ".jpg")
            with open(os.path.join(lab_dir, stem + ".txt")) as f:
                lines = [l.strip() for l in f if l.strip()]
            for l in lines:
                stats[split][2][int(l.split()[0])] += 1

            # original (copied, unchanged)
            shutil.copy(img_path, f"{args.out}/images/{split}/{stem}.jpg")
            shutil.copy(os.path.join(lab_dir, stem + ".txt"),
                        f"{args.out}/labels/{split}/{stem}.txt")
            stats[split][0] += 1

            # augmented copies (train only), same label file
            if split == "train":
                img = cv2.imread(img_path)
                for k in range(args.aug_per_image):
                    aug, ops = augment(img, rng)
                    name = f"{stem}_aug{k}"
                    cv2.imwrite(f"{args.out}/images/train/{name}.jpg", aug,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    shutil.copy(os.path.join(lab_dir, stem + ".txt"),
                                f"{args.out}/labels/train/{name}.txt")
                    stats["train"][1] += 1
                    for l in lines:
                        stats["train"][2][int(l.split()[0])] += 1
                    if len(preview) < 8 and rng.random() < 0.02:
                        o = cv2.resize(draw_labels(img, lines), (320, 240))
                        a = cv2.resize(draw_labels(aug, lines), (320, 240))
                        cv2.putText(a, "+".join(ops)[:40], (5, 232),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        preview.append(np.hstack([o, a]))

    with open(f"{args.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n"
                "train: images/train\n"
                "val: images/val\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))

    if preview:
        while len(preview) % 2:
            preview.append(np.zeros_like(preview[0]))
        rows = [np.hstack(preview[i:i + 2]) for i in range(0, len(preview), 2)]
        os.makedirs(os.path.join(os.path.dirname(args.out), "check"), exist_ok=True)
        cv2.imwrite(os.path.join(os.path.dirname(args.out), "check", "augment_preview.jpg"),
                    np.vstack(rows))

    print("\nDataset summary")
    for split in ("train", "val"):
        o, a, b = stats[split]
        print(f"  {split:5s}: {o + a:5d} images ({o} original + {a} augmented) | boxes: "
              + ", ".join(f"{CLASSES[i]} {b[i]}" for i in range(3)))
    print(f"\nWrote {args.out}/data.yaml")
    print("Preview: vision/check/augment_preview.jpg  (left = original, right = augmented)")


if __name__ == "__main__":
    main()
