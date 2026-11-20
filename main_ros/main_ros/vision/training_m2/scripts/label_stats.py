"""
label_stats.py - sanity check of the SAM 2 labels, per clip.
Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/label_stats.py

Per clip it prints:
  masks/frame   average number of labeled boxes per frame
  tiny%         masks smaller than 20 px (sqrt of area)
  fill          polygon area / its bounding-box area. A box seen from the front
                fills its rectangle (~0.8-1.0). Low values = odd shapes: a mask
                leaking onto the rack, two boxes merged, thin slivers.
  bad%          masks with fill < 0.5
  overlap%      masks whose bounding box overlaps another mask's by IoU > 0.5
                (duplicates or merged boxes)
It also saves vision_m2/check/bad_masks/<clip>_*.jpg : frames with the most bad masks
drawn in red, so you can see what they are.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    out_dir = os.path.join(C.CHECK, "bad_masks")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'clip':22s} {'frames':>6s} {'masks':>6s} {'masks/frame':>11s} {'tiny%':>6s} "
          f"{'fill':>5s} {'bad%':>5s} {'overlap%':>8s}")
    for clip in sorted(os.listdir(C.LABELS)):
        lab_dir, img_dir = os.path.join(C.LABELS, clip), os.path.join(C.FRAMES, clip)
        files = sorted(f for f in os.listdir(lab_dir) if f.endswith(".txt"))
        first = next(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        H, W = cv2.imread(os.path.join(img_dir, first)).shape[:2]
        n = tiny = bad = ovl = 0
        fills, worst = [], []
        for f in files:
            polys = []
            for line in open(os.path.join(lab_dir, f)):
                v = line.split()
                if len(v) < 7:
                    continue
                p = (np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
                x, y, w, h = cv2.boundingRect(p)
                fill = cv2.contourArea(p) / max(w * h, 1)
                polys.append((p, [x, y, x + w, y + h], fill))
            nb = 0
            for i, (p, b, fill) in enumerate(polys):
                n += 1
                fills.append(fill)
                tiny += np.sqrt((b[2] - b[0]) * (b[3] - b[1])) < 20
                bad += fill < 0.5
                nb += fill < 0.5
                ovl += any(box_iou(b, polys[j][1]) > 0.5 for j in range(len(polys)) if j != i)
            worst.append((nb, f, polys))
        worst.sort(key=lambda t: -t[0])
        for nb, f, polys in worst[:5]:
            if nb == 0:
                break
            img = cv2.imread(os.path.join(img_dir, f.replace(".txt", ".jpg")))
            for p, b, fill in polys:
                cv2.polylines(img, [p], True, (0, 0, 255) if fill < 0.5 else (0, 200, 0), 2 if fill < 0.5 else 1)
            cv2.imwrite(os.path.join(out_dir, f"{clip}_{f.replace('.txt', '.jpg')}"), img)
        print(f"{clip:22s} {len(files):6d} {n:6d} {n / max(len(files), 1):11.1f} {100 * tiny / max(n, 1):6.1f} "
              f"{np.median(fills) if fills else 0:5.2f} {100 * bad / max(n, 1):5.1f} {100 * ovl / max(n, 1):8.1f}")
    print(f"\nWorst frames (red = fill < 0.5): {out_dir}")


if __name__ == "__main__":
    main()
    