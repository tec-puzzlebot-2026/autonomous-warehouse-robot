"""
recall_by_size.py - which boxes does the model miss? Recall split by box size.
Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/recall_by_size.py
    ~/venvs/sem9-vision/bin/python vision_m2/recall_by_size.py --model vision_m2/runs/boxes_v2/weights/best.pt --imgsz 960

For every validation label (SAM 2 mask -> its bounding box) it checks whether the
model predicted a box of the SAME class with IoU >= 0.5. Labels are grouped by
size = sqrt(box area) in pixels of the 640x480 frame:
    tiny < 20 px   (far racks)    small 20-40   medium 40-80   large >= 80 (close boxes)
Also writes vision_m2/check/missed/ : val frames with misses drawn in red
(green = found), so you can see WHICH boxes are missed.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C

BINS = [("tiny <20", 0, 20), ("small 20-40", 20, 40), ("medium 40-80", 40, 80), ("large >=80", 80, 1e9)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.RUNS, "boxes_v1", "weights", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", type=int, default=20, help="frames with misses to save")
    args = ap.parse_args()

    model = YOLO(args.model)
    imgs = sorted(glob.glob(os.path.join(C.DATASET, "images", "val", "*.jpg")))
    out_dir = os.path.join(C.CHECK, "missed")
    os.makedirs(out_dir, exist_ok=True)
    found = {b[0]: [0, 0] for b in BINS}          # bin -> [found, total]
    n_pred, n_tp, saved = 0, 0, 0

    for k in range(0, len(imgs), 32):
        batch = imgs[k:k + 32]
        res = model.predict(batch, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False)
        for path, r in zip(batch, res):
            img = cv2.imread(path)
            H, W = img.shape[:2]
            lab = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            gts = []
            for line in open(lab):
                v = line.split()
                if not v:
                    continue
                p = np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]
                gts.append((int(v[0]), [*p.min(0), *p.max(0)]))
            preds = [(int(c), b.tolist()) for c, b in zip(r.boxes.cls.cpu(), r.boxes.xyxy.cpu())]
            n_pred += len(preds)
            used = set()
            draw = False
            for c, g in gts:
                best, bj = 0, -1
                for j, (pc, pb) in enumerate(preds):
                    if pc == c and j not in used:
                        o = iou(g, pb)
                        if o > best:
                            best, bj = o, j
                ok = best >= 0.5
                if ok:
                    used.add(bj)
                    n_tp += 1
                size = np.sqrt((g[2] - g[0]) * (g[3] - g[1]))
                for name, lo, hi in BINS:
                    if lo <= size < hi:
                        found[name][0] += ok
                        found[name][1] += 1
                col = (0, 200, 0) if ok else (0, 0, 255)
                draw |= not ok
                cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), col, 1 if ok else 2)
            if draw and saved < args.save and (k // 32) % 2 == 0:
                cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), img)
                saved += 1

    print(f"\nModel {args.model}  imgsz {args.imgsz}  conf {args.conf}")
    print(f"{'size (px)':14s} {'labels':>7s} {'found':>7s} {'recall':>7s}")
    for name, _, _ in BINS:
        f, t = found[name]
        print(f"{name:14s} {t:7d} {f:7d} {f / max(t, 1):7.2f}")
    tot = sum(t for _, t in found.values())
    print(f"{'ALL':14s} {tot:7d} {n_tp:7d} {n_tp / max(tot, 1):7.2f}   precision {n_tp / max(n_pred, 1):.2f}")
    print(f"\nFrames with misses (red = missed, green = found): {out_dir}")


if __name__ == "__main__":
    main()