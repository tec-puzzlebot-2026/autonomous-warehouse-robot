"""
dedup_labels.py - remove duplicate masks from the SAM 2 labels (no need to re-run SAM 2).
Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/dedup_labels.py              # all clips
    ~/venvs/sem9-vision/bin/python vision_m2/dedup_labels.py --dry-run    # only report

Two masks are the same box when most of the smaller one lies inside the larger one
(intersection / smaller area > --overlap). Then the SMALLER one is removed (the larger
one is the whole box; the smaller is a duplicate or only a part of it, e.g. the picture).
Both masks also have to be the same class, unless --any-class is given.

Before changing anything the original labels are copied to labels_auto_backup/
(only the first time). Example frames are saved to check/dedup/<clip>_*.jpg :
green = kept, red = removed.
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--any-class", action="store_true", help="also merge masks of different classes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    backup = C.LABELS + "_backup"
    if not args.dry_run and not os.path.exists(backup):
        shutil.copytree(C.LABELS, backup)
        print(f"Backup of the original labels: {backup}")
    out_dir = os.path.join(C.CHECK, "dedup")
    os.makedirs(out_dir, exist_ok=True)

    for clip in sorted(os.listdir(C.LABELS)):
        lab_dir, img_dir = os.path.join(C.LABELS, clip), os.path.join(C.FRAMES, clip)
        files = sorted(f for f in os.listdir(lab_dir) if f.endswith(".txt"))
        first = next(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        H, W = cv2.imread(os.path.join(img_dir, first)).shape[:2]
        before = after = saved = 0
        for fi, f in enumerate(files):
            path = os.path.join(lab_dir, f)
            items = []
            for line in open(path):
                v = line.split()
                if len(v) < 7:
                    continue
                p = (np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [p], 1)
                items.append({"line": line.strip(), "cls": int(v[0]), "poly": p,
                              "mask": m.astype(bool), "area": int(m.sum())})
            before += len(items)
            items.sort(key=lambda d: -d["area"])          # larger masks first = kept first
            kept, removed = [], []
            for d in items:
                dup = any((args.any_class or k["cls"] == d["cls"]) and
                          (d["mask"] & k["mask"]).sum() / max(d["area"], 1) > args.overlap
                          for k in kept)
                (removed if dup else kept).append(d)
            after += len(kept)
            if not args.dry_run:
                with open(path, "w") as fh:
                    fh.write("".join(k["line"] + "\n" for k in kept))
            if removed and saved < 6 and fi % 20 == 0:
                img = cv2.imread(os.path.join(img_dir, f.replace(".txt", ".jpg")))
                for k in kept:
                    cv2.polylines(img, [k["poly"]], True, (0, 200, 0), 1)
                for r in removed:
                    cv2.polylines(img, [r["poly"]], True, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(out_dir, f"{clip}_{f.replace('.txt', '.jpg')}"), img)
                saved += 1
        print(f"{clip:22s} masks {before:6d} -> {after:6d}  ({before - after} removed) | "
              f"per frame {before / len(files):5.1f} -> {after / len(files):5.1f}")
    print(f"\nExamples (green kept, red removed): {out_dir}")
    if args.dry_run:
        print("Dry run: nothing was changed.")


if __name__ == "__main__":
    main()