"""
recall_by_size.py - does the model find THE near box the robot needs? Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/recall_by_size.py                     # ROI model, ROI val set
    ~/venvs/sem9-vision/bin/python vision_m2/recall_by_size.py --model vision_m2/runs/boxes_v1/weights/best.pt \
        --data vision_m2/yolo_dataset --imgsz 480,640                            # old full-frame model

Uses the same input size, NMS, size filter and "largest box" rule as near_box.py on the
robot (read from near_box.yaml next to the model, otherwise from config_m2).

1. Recall per box size (size = sqrt(box area) in frame px; the ROI crop keeps frame scale):
       tiny <20   small 20-40   medium 40-<min>   near >= min size (the boxes that matter)
   A label is found if a prediction of the SAME class has IoU >= 0.5.
2. Target check per frame - the number that matters for the robot:
       hit             picked the right near box (IoU >= 0.5); also class and mask IoU
       correctly empty no near box in the labels, none picked
       wrong box       picked a different box
       missed          a near box was there, nothing picked
       false target    nothing near in the labels, but something picked
Writes vision_m2/check/missed/: frames with target errors. Thin green/red = every label
found/missed, thick green = labeled target, thick yellow = what the robot would pick.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C
import near_box as NB


def mask_iou(p1, p2, H, W):
    a, b = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    cv2.fillPoly(a, [np.round(p1).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(p2).astype(np.int32)], 1)
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / float(u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.RUNS, "boxes_roi_v1", "weights", "best.pt"))
    ap.add_argument("--data", default=C.DATASET_ROI, help="dataset folder with images/val, labels/val")
    ap.add_argument("--imgsz", help="network input 'H,W' (default: near_box.yaml next to the model)")
    ap.add_argument("--conf", type=float, help="default: conf from near_box.yaml / config_m2")
    ap.add_argument("--min-size", type=float, help="default: min_size_px from near_box.yaml / config_m2")
    ap.add_argument("--save", type=int, default=20, help="frames with target errors to save")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    s = C.near_box_settings()
    yml = os.path.join(os.path.dirname(os.path.abspath(args.model)), "near_box.yaml")
    if os.path.exists(yml):
        s.update(yaml.safe_load(open(yml)) or {})
    nw, nh = s["net"]
    if args.imgsz:
        nh, nw = [int(v) for v in args.imgsz.split(",")] if "," in args.imgsz else [int(args.imgsz)] * 2
    conf = args.conf if args.conf is not None else s["conf"]
    min_size = args.min_size if args.min_size is not None else s["min_size_px"]
    bins = [("tiny <20", 0, 20), ("small 20-40", 20, 40), (f"medium 40-{min_size:g}", 40, min_size),
            (f"near >={min_size:g}", min_size, 1e9)]

    model = YOLO(args.model)
    imgs = sorted(glob.glob(os.path.join(args.data, "images", "val", "*.jpg")))
    if not imgs:
        raise SystemExit(f"No validation images in {args.data}/images/val")
    out_dir = os.path.join(C.CHECK, "missed")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(f)
    found = {b[0]: [0, 0] for b in bins}          # bin -> [found, total]
    n_pred = n_tp = 0
    outcome = {"hit": 0, "correctly empty": 0, "wrong box": 0, "missed": 0, "false target": 0}
    hit_iou, hit_cls, hit_miou, errors = [], [], [], []

    for k in range(0, len(imgs), 32):
        batch = imgs[k:k + 32]
        res = model.predict(batch, imgsz=(nh, nw), conf=conf, iou=s["nms_iou"], agnostic_nms=True,
                            max_det=int(s["max_det"]), device=args.device, verbose=False)
        for path, r in zip(batch, res):
            H, W = r.orig_shape
            lab = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            gts = []
            for line in open(lab):
                v = line.split()
                if len(v) < 7:
                    continue
                p = np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]
                gts.append({"cls": int(v[0]), "box": [*p.min(0), *p.max(0)], "poly": p})
            polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
            preds = [{"cls": int(c), "conf": float(cf), "box": b.tolist(), "poly": p}
                     for c, cf, b, p in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                            r.boxes.xyxy.cpu().numpy(), polys)]
            n_pred += len(preds)

            # 1. per-label recall
            used = set()
            for g in gts:
                best, bj = 0, -1
                for j, p in enumerate(preds):
                    if p["cls"] == g["cls"] and j not in used:
                        o = NB.box_iou(g["box"], p["box"])
                        if o > best:
                            best, bj = o, j
                g["found"] = best >= 0.5
                if g["found"]:
                    used.add(bj)
                    n_tp += 1
                size = NB.box_size(g["box"])
                for name, lo, hi in bins:
                    if lo <= size < hi:
                        found[name][0] += g["found"]
                        found[name][1] += 1

            # 2. the target the robot would pick
            gt_t, pr_t = NB.select_target(gts, min_size), NB.select_target(preds, min_size)
            if gt_t is None:
                o = "correctly empty" if pr_t is None else "false target"
            elif pr_t is None:
                o = "missed"
            else:
                iou_t = NB.box_iou(gt_t["box"], pr_t["box"])
                o = "hit" if iou_t >= 0.5 else "wrong box"
                if o == "hit":
                    hit_iou.append(iou_t)
                    hit_cls.append(gt_t["cls"] == pr_t["cls"])
                    if pr_t["poly"] is not None and len(pr_t["poly"]) >= 3:
                        hit_miou.append(mask_iou(gt_t["poly"], pr_t["poly"], H, W))
            outcome[o] += 1
            if o not in ("hit", "correctly empty"):
                errors.append((path, gts, gt_t, pr_t, o))

    print(f"\nModel {args.model}\n  input {nw}x{nh} | conf {conf} | min size {min_size:g} px | "
          f"{len(imgs)} val frames from {args.data}")
    print(f"\n{'size (px)':16s} {'labels':>7s} {'found':>7s} {'recall':>7s}")
    for name, _, _ in bins:
        f, t = found[name]
        print(f"{name:16s} {t:7d} {f:7d} {f / max(t, 1):7.2f}")
    tot = sum(t for _, t in found.values())
    print(f"{'ALL':16s} {tot:7d} {n_tp:7d} {n_tp / max(tot, 1):7.2f}   precision {n_tp / max(n_pred, 1):.2f}")

    n = float(len(imgs))
    print(f"\nTarget the robot would pick (largest box >= {min_size:g} px):")
    print(f"  correct           {100 * (outcome['hit'] + outcome['correctly empty']) / n:5.1f} %")
    for key, val in outcome.items():
        extra = ""
        if key == "hit" and hit_iou:
            extra = (f"   box IoU {np.mean(hit_iou):.2f} | class right {100 * np.mean(hit_cls):.1f} %"
                     + (f" | mask IoU {np.mean(hit_miou):.2f}" if hit_miou else ""))
        print(f"  {key:16s} {val:6d}{extra}")

    pick = np.unique(np.linspace(0, len(errors) - 1, min(args.save, len(errors))).astype(int)) if errors else []
    for i in pick:
        path, gts, gt_t, pr_t, o = errors[i]
        img = cv2.imread(path)
        for g in gts:
            b = [int(v) for v in g["box"]]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 0) if g["found"] else (0, 0, 255), 1)
        for t, col in ((gt_t, (0, 200, 0)), (pr_t, (0, 255, 255))):
            if t is not None:
                b = [int(v) for v in t["box"]]
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 3)
        cv2.putText(img, o, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, o, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), img)
    print(f"\n{len(pick)} frames with target errors: {out_dir}  (thick green = label, thick yellow = picked)")


if __name__ == "__main__":
    main()