"""
sam_label_seg.py - SAM 2: prompt boxes -> a MASK per box per frame -> YOLO segmentation labels.
Run on the WORKSTATION:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/sam_label_seg.py
    ~/venvs/sem9-vision/bin/python vision_m2/sam_label_seg.py --clips bag_boxes_train_1   # redo one clip

Per clip:
  1. numbered links 00000.jpg ... to the frames (SAM 2 needs pure numbers)
  2. SAM 2 receives every prompt box (object number = SAM 2 object id)
  3. tracks every object forward -> one mask per object per frame
  4. mask -> outer outline (largest piece) -> simplified polygon -> YOLO-seg line:
         <class> x1 y1 x2 y2 x3 y3 ...    (normalised 0-1)
     An object only gets labels from the frame where you first prompted it.
  5. review: video + contact sheets with the outlines drawn, and a list of
     frames where an object's area suddenly jumps (possible tracking errors)

Outputs in vision_m2/:  labels_auto/<clip>/*.txt, check/<clip>_review.mp4,
                        check/<clip>/sheet_XX.jpg, check/<clip>_suspicious.txt, check/summary.txt
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


# ----------------------------------------------------------------------------
def mask_to_polygon(mask, min_area, eps_frac=0.003):
    """Largest outer contour of the mask, simplified. Returns Nx2 int array or None."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    c = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
    return c.reshape(-1, 2) if len(c) >= 3 else None


def yolo_seg_line(cls, poly, W, H):
    pts = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly)
    return f"{cls} {pts}"


def draw(img, objs, idx, prompted):
    """objs: list of (obj_id, cls, polygon)."""
    out = img.copy()
    overlay = img.copy()
    for obj, cls, poly in objs:
        cv2.fillPoly(overlay, [poly], C.COLORS[cls])
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for obj, cls, poly in objs:
        cv2.polylines(out, [poly], True, C.COLORS[cls], 2)
        x, y = poly.min(axis=0)
        cv2.putText(out, f"#{obj} {C.CLASSES[cls]}", (int(x), max(14, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C.COLORS[cls], 2)
    tag = f"#{idx}" + ("  PROMPT" if prompted else "")
    cv2.putText(out, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def find_suspicious(per_frame):
    """per_frame: list of {obj: (cls, area, cx, cy, w)}; flag big jumps/size changes."""
    sus, last = [], {}
    for idx, d in enumerate(per_frame):
        for obj, (cls, a, cx, cy, w) in d.items():
            if obj in last:
                pa, pcx, pcy, pw = last[obj]
                if abs(cx - pcx) > 0.5 * pw + 40 or abs(cy - pcy) > 0.5 * pw + 40:
                    sus.append((idx, obj, "jumped"))
                elif a / max(pa, 1) > 2.5 or pa / max(a, 1) > 2.5:
                    sus.append((idx, obj, f"area x{max(a / max(pa, 1), pa / max(a, 1)):.1f}"))
            last[obj] = (a, cx, cy, w)
        for obj in list(last):
            if obj not in d:
                del last[obj]
    return sus


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=C.FRAMES)
    ap.add_argument("--prompts", default=C.PROMPTS)
    ap.add_argument("--out", default=C.WORK)
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--review-fps", type=int, default=8)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    clips = args.clips or sorted(prompts)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - run with ~/venvs/sem9-vision/bin/python")
    from sam2.build_sam import build_sam2_video_predictor
    print(f"Loading SAM 2 on {torch.cuda.get_device_name(0)} ...")
    predictor = build_sam2_video_predictor(C.SAM2_CFG, C.SAM2_CKPT, device="cuda")

    summary = []
    for clip in clips:
        src = os.path.join(args.frames, clip)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        H, W = cv2.imread(os.path.join(src, files[0])).shape[:2]
        obj_cls = {p["obj"]: p["cls"] for p in prompts[clip]}
        print(f"\n=== {clip}: {len(files)} frames ({W}x{H}), {len(prompts[clip])} prompts, "
              f"{len(obj_cls)} objects")

        link_dir = os.path.join(args.out, "sam_jpg", clip)
        shutil.rmtree(link_dir, ignore_errors=True)
        os.makedirs(link_dir)
        for i, f in enumerate(files):
            os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(link_dir, f"{i:05d}.jpg"))

        first_seen, prompted = {}, set()
        polys = [dict() for _ in files]           # idx -> {obj: (cls, poly)}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=link_dir, offload_video_to_cpu=True)
            for p in prompts[clip]:
                if p["frame"] not in files:
                    raise SystemExit(f"Prompt frame {p['frame']} not in {src}")
                fi = files.index(p["frame"])
                predictor.add_new_points_or_box(inference_state=state, frame_idx=fi, obj_id=p["obj"],
                                                box=np.array(p["box"], dtype=np.float32))
                first_seen[p["obj"]] = min(fi, first_seen.get(p["obj"], fi))
                prompted.add(fi)
            for fi, obj_ids, logits in tqdm(predictor.propagate_in_video(state),
                                            total=len(files) - min(first_seen.values()),
                                            desc="  tracking", unit="frame", dynamic_ncols=True):
                for k, obj in enumerate(obj_ids):
                    if fi < first_seen[obj]:
                        continue
                    poly = mask_to_polygon((logits[k, 0] > 0).cpu().numpy(), args.min_area)
                    if poly is not None:
                        polys[fi][obj] = (obj_cls[obj], poly)
            predictor.reset_state(state)
        del state
        torch.cuda.empty_cache()

        # ---- labels ----
        lab_dir = os.path.join(args.out, "labels_auto", clip)
        shutil.rmtree(lab_dir, ignore_errors=True)
        os.makedirs(lab_dir)
        for f, d in zip(files, polys):
            lines = [yolo_seg_line(cls, poly, W, H) for obj, (cls, poly) in sorted(d.items())]
            with open(os.path.join(lab_dir, os.path.splitext(f)[0] + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))

        # ---- review ----
        chk = os.path.join(args.out, "check")
        sheet_dir = os.path.join(chk, clip)
        shutil.rmtree(sheet_dir, ignore_errors=True)
        os.makedirs(sheet_dir)
        vw = cv2.VideoWriter(os.path.join(chk, f"{clip}_review.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.review_fps, (W, H))
        thumbs, tw, th = [], W // 3, H // 3
        stats = []
        for idx, (f, d) in enumerate(zip(files, polys)):
            objs = [(o, c, p) for o, (c, p) in sorted(d.items())]
            img = draw(cv2.imread(os.path.join(src, f)), objs, idx, idx in prompted)
            vw.write(img)
            thumbs.append(cv2.resize(img, (tw, th)))
            if len(thumbs) == 24 or idx == len(files) - 1:
                thumbs += [np.zeros((th, tw, 3), np.uint8)] * (24 - len(thumbs))
                cv2.imwrite(os.path.join(sheet_dir, f"sheet_{idx // 24:02d}.jpg"),
                            np.vstack([np.hstack(thumbs[r * 6:(r + 1) * 6]) for r in range(4)]))
                thumbs = []
            st = {}
            for o, c, p in objs:
                x, y, w, h = cv2.boundingRect(p)
                st[o] = (c, cv2.contourArea(p), x + w / 2, y + h / 2, w)
            stats.append(st)
        vw.release()

        sus = find_suspicious(stats)
        with open(os.path.join(chk, f"{clip}_suspicious.txt"), "w") as fh:
            for i, o, why in sus:
                fh.write(f"#{i}  {files[i]}  object #{o} ({C.CLASSES[obj_cls[o]]}): {why}\n")
        n_masks = [sum(1 for d in polys for o, (c, _) in d.items() if c == k) for k in range(len(C.CLASSES))]
        line = (f"{clip:22s} frames {len(files):4d} | objects {len(obj_cls):3d} | masks: " +
                ", ".join(f"{C.CLASSES[k]} {n_masks[k]}" for k in range(len(C.CLASSES))) +
                f" | empty frames {sum(not d for d in polys)} | suspicious {len(sus)}")
        print("  " + line)
        summary.append(line)

    os.makedirs(os.path.join(args.out, "check"), exist_ok=True)
    open(os.path.join(args.out, "check", "summary.txt"), "w").write("\n".join(summary) + "\n")
    print("\nDone.\n  " + "\n  ".join(summary))


if __name__ == "__main__":
    main()