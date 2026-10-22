"""
select_frames.py - pick the sharpest frames from a video for Nerfstudio / COLMAP.

The video is split into short time windows (e.g. 1/3 s each). From every window
the SHARPEST frame is kept (sharpness = variance of the Laplacian: blurry frames
have few strong edges -> low value). This gives evenly spaced frames along the
whole path while skipping motion-blurred ones.

Usage:
    python select_frames.py --video warehouse.mp4 --out frames --fps 3
    python select_frames.py --video warehouse.mp4 --out frames --fps 3 --max-width 1920

Options:
    --fps        frames kept per second of video (more = more overlap). 2-4 is typical.
                 For 14 min:  2 fps -> ~1700 frames, 3 fps -> ~2500 frames
    --max-width  resize saved frames to this width (keeps aspect ratio). 1920 is
                 plenty for Nerfstudio and makes COLMAP much faster than 4K.
    --min-sharp  also drop the kept frame if its window was entirely blurry
                 (value relative to the median sharpness; 0 = keep all)
Outputs:
    <out>/frame_00000.jpg ...
    <out>/sharpness.csv      time, sharpness of every kept frame
"""
import argparse
import csv
import os

import cv2
import numpy as np


def sharpness(frame, small_width=640):
    h, w = frame.shape[:2]
    if w > small_width:
        frame = cv2.resize(frame, (small_width, int(h * small_width / w)),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--min-sharp", type=float, default=0.35)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window = max(1, int(round(vfps / args.fps)))
    print(f"Video: {W}x{H} @ {vfps:.2f} fps, {total} frames ({total / vfps / 60:.1f} min)")
    print(f"Keeping the sharpest frame of every {window} frames -> ~{total // window} frames")

    os.makedirs(args.out, exist_ok=True)

    def resize(img):
        if args.max_width and img.shape[1] > args.max_width:
            h = int(img.shape[0] * args.max_width / img.shape[1])
            img = cv2.resize(img, (args.max_width, h), interpolation=cv2.INTER_AREA)
        return img

    # Pass 1: read the video once; save the sharpest frame of each window to
    # disk immediately (resized), so memory use stays small even for 8K video.
    kept = []                       # (file, time, sharpness)
    best, best_s, best_i, i = None, -1.0, 0, 0

    def flush():
        name = f"frame_{len(kept):05d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), resize(best), [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append((name, best_i / vfps, best_s))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        s = sharpness(frame)
        if s > best_s:
            best, best_s, best_i = frame, s, i
        i += 1
        if i % window == 0:
            flush()
            best, best_s = None, -1.0
            if len(kept) % 100 == 0:
                print(f"  {i}/{total} frames read ({100 * i / max(total, 1):.0f}%), "
                      f"{len(kept)} frames saved", flush=True)
    if best is not None:
        flush()
    cap.release()

    # Pass 2: delete windows where even the best frame was very blurry,
    # then renumber the remaining files consecutively.
    med = float(np.median([k[2] for k in kept]))
    limit = args.min_sharp * med
    saved, dropped = 0, 0
    with open(os.path.join(args.out, "sharpness.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "time_s", "sharpness"])
        for name, t, s in kept:
            path = os.path.join(args.out, name)
            if s < limit:
                os.remove(path)
                dropped += 1
                continue
            new_name = f"frame_{saved:05d}.jpg"
            if new_name != name:
                os.rename(path, os.path.join(args.out, new_name))
            wr.writerow([new_name, f"{t:.2f}", f"{s:.1f}"])
            saved += 1

    print(f"\nSaved {saved} frames to {args.out}  (dropped {dropped} very blurry windows)")
    print(f"Median sharpness {med:.1f}; drop limit {limit:.1f}")
    if dropped > 0.1 * len(kept):
        print("[!] More than 10% of the video was very blurry - check sharpness.csv for "
              "the time ranges; consider re-recording those parts more slowly.")


if __name__ == "__main__":
    main()