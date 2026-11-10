"""
auto_prompt.py - let a trained model write the SAM 2 prompts for new clips.
Run on the WORKSTATION:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/auto_prompt.py --clips bag_boxes_train_2 bag_boxes_test_2
    ~/venvs/sem9-vision/bin/python vision_m2/auto_prompt.py --clips warehouse_1 \
        --model vision_m2/runs/boxes_v2/weights/best.pt --imgsz 960

How it works:
  1. The model runs over the clip WITH A TRACKER (ByteTrack), so the same physical
     box keeps the same id from frame to frame (= one SAM 2 object).
  2. Tracks that are too short or never confident are dropped (false detections).
  3. Each track becomes an object: class = the class the model voted most for it,
     prompt box = the track's first confident detection, plus a re-prompt every
     --every frames (best confident detection in that window) to stop SAM 2 drifting.
  4. A new track that sits on top of an existing one (IoU > 0.6) is a duplicate -> dropped.
  5. The prompts are ADDED to vision_m2/prompts.json in the same format as the
     prompt tool, so you can open the clip in prompt_tool_seg.py and fix/add boxes
     before running SAM 2. Existing clips in prompts.json are only replaced if you
     pass them in --clips.

  6. --roi-only: a track that never has >= 20 % of its box inside config_m2.ROI is
     dropped. Those boxes are cut away by build_dataset_seg.py anyway, so SAM 2 does not
     have to track them (fewer objects = faster SAM 2, fewer junk masks to review).
     Small boxes INSIDE the ROI are deliberately kept: a far box that is in the crop but
     unlabeled would teach the model "this box is background", which fights the near
     boxes (scale augmentation shrinks those to the same size). Far boxes are removed
     at runtime by the size filter in near_box.py, not by leaving them unlabeled.

Use a FULL-FRAME model here (boxes_v1/v2): it runs on whole frames. The ROI model is
for the robot.

Then:  sam_label_seg.py --clips <the same clips>   ->  review videos as before.
"""
import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def tracker_cfg(buffer):
    """Ultralytics' ByteTrack config with a longer memory (boxes hidden for a while keep their id)."""
    import ultralytics
    src = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
    cfg = yaml.safe_load(open(src))
    cfg["track_buffer"] = buffer
    path = os.path.join(tempfile.gettempdir(), "bytetrack_m2.yaml")
    yaml.safe_dump(cfg, open(path, "w"))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--model", default=os.path.join(C.RUNS, "boxes_v1", "weights", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640, help="same size the model was trained with")
    ap.add_argument("--conf", type=float, default=0.25, help="detections used for tracking")
    ap.add_argument("--prompt-conf", type=float, default=0.5, help="detections good enough to be a prompt")
    ap.add_argument("--min-len", type=int, default=5, help="drop tracks shorter than this (frames)")
    ap.add_argument("--every", type=int, default=25, help="re-prompt each object every N frames (0 = never)")
    ap.add_argument("--buffer", type=int, default=60, help="frames a lost track is remembered")
    ap.add_argument("--roi-only", action="store_true", help="drop tracks that never enter config_m2.ROI")
    ap.add_argument("--device", default="0")
    ap.add_argument("--frames", default=C.FRAMES)
    ap.add_argument("--prompts", default=C.PROMPTS)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts)) if os.path.exists(args.prompts) else {}
    tcfg = tracker_cfg(args.buffer)

    for clip in args.clips:
        model = YOLO(args.model)                       # fresh model = fresh tracker per clip
        src = os.path.join(args.frames, clip)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        tracks = {}                                     # track id -> {frame idx: (cls, conf, box)}
        for fi, f in enumerate(tqdm(files, desc=f"{clip}: tracking", unit="frame", dynamic_ncols=True)):
            r = model.track(cv2.imread(os.path.join(src, f)), persist=True, tracker=tcfg,
                            conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            if r.boxes.id is None:
                continue
            for tid, c, cf, b in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(),
                                     r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                tracks.setdefault(tid, {})[fi] = (c, cf, [int(round(v)) for v in b])

        # ---- tracks -> objects ----
        H, W = cv2.imread(os.path.join(src, files[0])).shape[:2] if files else (C.FRAME_H, C.FRAME_W)
        rx0, ry0 = C.ROI[0] * W / C.FRAME_W, C.ROI[1] * H / C.FRAME_H
        rx1, ry1 = C.ROI[2] * W / C.FRAME_W, C.ROI[3] * H / C.FRAME_H

        def in_roi(b):
            """fraction of the box area inside the ROI"""
            iw = max(0, min(b[2], rx1) - max(b[0], rx0))
            ih = max(0, min(b[3], ry1) - max(b[1], ry0))
            return iw * ih / max((b[2] - b[0]) * (b[3] - b[1]), 1)

        kept, kept_cls, entries = [], [], []
        dropped = {"short / weak": 0, "duplicate": 0, "outside ROI": 0}
        for tid in sorted(tracks, key=lambda t: min(tracks[t])):
            det = tracks[tid]
            good = {fi: d for fi, d in det.items() if d[1] >= args.prompt_conf}
            if len(det) < args.min_len or not good:
                dropped["short / weak"] += 1
                continue
            if args.roi_only and max(in_roi(d[2]) for d in det.values()) < 0.2:
                dropped["outside ROI"] += 1
                continue
            votes = np.zeros(len(C.CLASSES))
            for c, cf, _ in det.values():
                votes[c] += cf
            cls = int(votes.argmax())
            first = min(good)
            if any(first in det2 and iou(good[first][2], det2[first][2]) > 0.6 for det2 in kept):
                dropped["duplicate"] += 1
                continue
            kept.append(det)
            kept_cls.append(cls)
            obj = len(kept)
            chosen = [first]
            if args.every:
                for start in range(first + args.every, max(good) + 1, args.every):
                    window = [fi for fi in good if start <= fi < start + args.every]
                    if window:
                        chosen.append(max(window, key=lambda fi: good[fi][1]))
            for fi in chosen:
                entries.append({"frame": files[fi], "obj": obj, "cls": cls, "box": good[fi][2]})

        prompts[clip] = sorted(entries, key=lambda e: (e["frame"], e["obj"]))
        per_cls = [kept_cls.count(k) for k in range(len(C.CLASSES))]   # same class as in the prompts
        print(f"  {clip}: {len(tracks)} tracks -> {len(kept)} objects "
              f"({', '.join(f'{C.CLASSES[k]} {per_cls[k]}' for k in range(len(C.CLASSES)))}), "
              f"{len(entries)} prompts | dropped: {dropped}")

    json.dump(prompts, open(args.prompts, "w"), indent=1)
    print(f"\nPrompts added to {args.prompts}")
    print("Next (optional): copy prompts.json to the laptop and check/fix the clip in prompt_tool_seg.py")
    print("Then: sam_label_seg.py --clips " + " ".join(args.clips))


if __name__ == "__main__":
    main()