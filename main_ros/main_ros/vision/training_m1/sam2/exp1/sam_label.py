"""
sam_label.py - SAM 2 auto-labeling: your prompt boxes -> YOLO labels for every frame.
Run on the WORKSTATION with our environment:

    cd ~/data/cosmostests/sem9_team
    .venv-vision/bin/python vision/sam_label.py
    .venv-vision/bin/python vision/sam_label.py --folders logo_2_train   # redo one folder

What it does, per folder (logo_1_train, logo_1_val, ...):
  1. Makes numbered links 00000.jpg, 00001.jpg ... to your frames
     (SAM 2's video loader needs pure numbers; your originals are not touched).
  2. Loads SAM 2 and gives it your boxes from prompts.json.
  3. Tracks every boxed logo forward through all frames -> one mask per frame.
  4. Mask -> bounding box -> YOLO label line "class cx cy w h" (normalised 0-1).
     A logo only gets boxes from the frame where you first marked it onwards.
     Frames with no logo get an EMPTY label file (= background image).
  5. Review material: a video and contact sheets with the boxes drawn, plus a
     list of frames where a box suddenly jumps or changes size (likely errors).

Outputs (in vision/):
    labels_auto/<folder>/<frame>.txt      YOLO labels
    check/<folder>_review.mp4             boxes drawn, frame numbers shown
    check/<folder>/sheet_XX.jpg           24 frames per image
    check/<folder>_suspicious.txt         frames worth a closer look
    check/summary.txt                     boxes per class per folder
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

CLASSES = ["yinyang", "emezon", "walnut"]
COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0)]   # BGR per class
TEAM = os.path.expanduser("~/data/cosmostests/sem9_team")


# ----------------------------------------------------------------------------
# mask -> box
# ----------------------------------------------------------------------------
def mask_to_box(mask, min_area):
    """Largest connected blob of the mask -> (x1, y1, x2, y2), or None."""
    m = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))      # skip background
    x, y, w, h, area = stats[i]
    if area < min_area:
        return None
    return int(x), int(y), int(x + w), int(y + h)


def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    return (f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")


# ----------------------------------------------------------------------------
# review material
# ----------------------------------------------------------------------------
def draw(img, boxes, idx, prompted):
    out = img.copy()
    for cls, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[cls], 2)
        cv2.putText(out, CLASSES[cls], (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls], 2)
    tag = f"#{idx}" + ("  PROMPT" if prompted else "")
    cv2.putText(out, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def find_suspicious(per_frame_boxes):
    """Frames where a class box jumps far or changes size a lot vs previous frame."""
    sus, last = [], {}
    for idx, boxes in enumerate(per_frame_boxes):
        for cls, (x1, y1, x2, y2) in boxes:
            cx, cy, a = (x1 + x2) / 2, (y1 + y2) / 2, max(1, (x2 - x1) * (y2 - y1))
            if cls in last:
                pcx, pcy, pa, pw = last[cls]
                if abs(cx - pcx) > 0.5 * pw + 40 or abs(cy - pcy) > 0.5 * pw + 40:
                    sus.append((idx, CLASSES[cls], "box jumped"))
                elif a / pa > 3 or pa / a > 3:
                    sus.append((idx, CLASSES[cls], f"size changed x{max(a / pa, pa / a):.1f}"))
            last[cls] = (cx, cy, a, x2 - x1)
        present = {c for c, _ in boxes}
        for cls in list(last):
            if cls not in present:
                del last[cls]           # logo left the view; restart comparison
    return sus


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=f"{TEAM}/dataset/frames")
    ap.add_argument("--prompts", default=f"{TEAM}/vision/prompts.json")
    ap.add_argument("--out", default=f"{TEAM}/vision")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/data/weights/sam2_hiera_large.pt"))
    ap.add_argument("--cfg", default="sam2_hiera_l.yaml")
    ap.add_argument("--folders", nargs="*", help="only these folders (default: all in prompts.json)")
    ap.add_argument("--min-area", type=int, default=60, help="ignore masks smaller than this (pixels)")
    ap.add_argument("--review-fps", type=int, default=8)
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    folders = args.folders or sorted(prompts)
    for fo in folders:
        if fo not in prompts or not prompts[fo]:
            raise SystemExit(f"No prompts for '{fo}' in {args.prompts}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - run with .venv-vision/bin/python on the workstation")
    from sam2.build_sam import build_sam2_video_predictor
    print(f"Loading SAM 2 ({args.cfg}) on {torch.cuda.get_device_name(0)} ...")
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")

    summary = []
    for fo in folders:
        src = os.path.join(args.frames, fo)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg")))
        H, W = cv2.imread(os.path.join(src, files[0])).shape[:2]
        print(f"\n=== {fo}: {len(files)} frames ({W}x{H}), {len(prompts[fo])} prompt boxes")

        # 1. numbered links for SAM 2
        link_dir = os.path.join(args.out, "sam_jpg", fo)
        shutil.rmtree(link_dir, ignore_errors=True)
        os.makedirs(link_dir)
        for i, f in enumerate(files):
            os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(link_dir, f"{i:05d}.jpg"))

        # 2-3. prompts + tracking
        first_seen = {}                       # obj_id -> first prompted frame
        prompted_frames = set()
        per_frame = [dict() for _ in files]   # idx -> {cls: box}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=link_dir, offload_video_to_cpu=True)
            for p in prompts[fo]:
                if p["frame"] not in files:
                    raise SystemExit(f"Prompt frame {p['frame']} not found in {src}")
                fi, obj = files.index(p["frame"]), p["cls"] + 1
                predictor.add_new_points_or_box(inference_state=state, frame_idx=fi,
                                                obj_id=obj, box=np.array(p["box"], dtype=np.float32))
                first_seen[obj] = min(fi, first_seen.get(obj, fi))
                prompted_frames.add(fi)

            for fi, obj_ids, logits in tqdm(predictor.propagate_in_video(state),
                                            total=len(files) - min(first_seen.values()),
                                            desc="  tracking", unit="frame", dynamic_ncols=True):
                for k, obj in enumerate(obj_ids):
                    if fi < first_seen[obj]:
                        continue          # before you marked it: not visible yet
                    mask = (logits[k, 0] > 0).cpu().numpy()
                    box = mask_to_box(mask, args.min_area)
                    if box:
                        per_frame[fi][obj - 1] = box
            predictor.reset_state(state)
        del state
        torch.cuda.empty_cache()

        # 4. YOLO labels (one file per frame; empty = no logo)
        lab_dir = os.path.join(args.out, "labels_auto", fo)
        shutil.rmtree(lab_dir, ignore_errors=True)
        os.makedirs(lab_dir)
        for f, boxes in zip(files, per_frame):
            lines = [yolo_line(c, b, W, H) for c, b in sorted(boxes.items())]
            with open(os.path.join(lab_dir, os.path.splitext(f)[0] + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))

        # 5. review material
        chk = os.path.join(args.out, "check")
        sheet_dir = os.path.join(chk, fo)
        shutil.rmtree(sheet_dir, ignore_errors=True)
        os.makedirs(sheet_dir)
        vw = cv2.VideoWriter(os.path.join(chk, f"{fo}_review.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.review_fps, (W, H))
        thumbs = []
        tw, th = W // 3, H // 3
        for idx, (f, boxes) in enumerate(zip(files, per_frame)):
            img = draw(cv2.imread(os.path.join(src, f)), list(boxes.items()), idx,
                       idx in prompted_frames)
            vw.write(img)
            thumbs.append(cv2.resize(img, (tw, th)))
            if len(thumbs) == 24 or idx == len(files) - 1:
                while len(thumbs) < 24:
                    thumbs.append(np.zeros((th, tw, 3), np.uint8))
                rows = [np.hstack(thumbs[r * 6:(r + 1) * 6]) for r in range(4)]
                cv2.imwrite(os.path.join(sheet_dir, f"sheet_{idx // 24:02d}.jpg"), np.vstack(rows))
                thumbs = []
        vw.release()

        sus = find_suspicious([list(d.items()) for d in per_frame])
        with open(os.path.join(chk, f"{fo}_suspicious.txt"), "w") as fh:
            for i, c, why in sus:
                fh.write(f"#{i}  {files[i]}  {c}: {why}\n")

        counts = [sum(c in d for d in per_frame) for c in range(3)]
        empty = sum(not d for d in per_frame)
        line = (f"{fo:14s} frames {len(files):4d} | " +
                " ".join(f"{CLASSES[c]} {counts[c]:4d}" for c in range(3)) +
                f" | no logo {empty:4d} | suspicious {len(sus)}")
        print("  " + line)
        summary.append(line)

    with open(os.path.join(args.out, "check", "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\nDone. Summary:\n  " + "\n  ".join(summary))
    print(f"\nReview: {args.out}/check/  (videos, contact sheets, *_suspicious.txt)")


if __name__ == "__main__":
    main()
