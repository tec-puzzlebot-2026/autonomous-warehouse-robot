"""
prompt_tool.py - draw the prompt boxes SAM 2 needs (run on your LAPTOP).

For each frame folder you mark where each logo is, on the first frame where it
appears. SAM 2 then tracks it through the rest of the folder by itself.
Add another box later in the folder only if the logo leaves the view and comes
back, or to correct the tracking.

Usage (one folder at a time; all results go into the same prompts.json):
    python3 prompt_tool.py --frames <path>/frames --folder logo_1_train
    python3 prompt_tool.py --frames <path>/frames --folder logo_1_val
    ... (6 folders)

Controls (click the image window first so it receives the keys):
    mouse drag      draw a box for the CURRENT class on the CURRENT frame
    1 / 2 / 3       current class: 1=yinyang  2=emezon  3=walnut
    d / a           next / previous frame
    D / A           jump +10 / -10 frames      (or  e / q  if Shift does not work)
    u               undo the last box drawn on this frame
    s               save
    ESC             save and quit

Rules:
  * Box EVERY one of the 3 logos that is visible on its first frame, even
    logos that belong to "another" folder. Unlabeled logos teach YOLO that
    they are background.
  * Draw the box tight around the logo sign.
  * One physical logo per class is assumed (the same class = the same object
    that SAM 2 tracks). A second box of the same class on a later frame is a
    correction for that object.
"""
import argparse
import json
import os

import cv2

CLASSES = ["yinyang", "emezon", "walnut"]
COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0)]   # BGR per class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder that contains logo_1_train/ etc.")
    ap.add_argument("--folder", required=True, help="e.g. logo_1_train")
    ap.add_argument("--out", default="prompts.json")
    ap.add_argument("--scale", type=float, default=1.5, help="display zoom")
    args = ap.parse_args()

    folder_path = os.path.join(args.frames, args.folder)
    files = sorted(f for f in os.listdir(folder_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"No images in {folder_path}")

    prompts = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            prompts = json.load(f)
    entries = prompts.setdefault(args.folder, [])

    # Default class = the folder's own logo (logo_2_* -> emezon)
    cur_cls = 0
    for i in range(3):
        if args.folder.startswith(f"logo_{i + 1}"):
            cur_cls = i
    idx = 0
    drag = {"start": None, "end": None}
    win = f"prompt_tool - {args.folder}"

    def save():
        prompts[args.folder] = sorted(entries, key=lambda e: (e["frame"], e["cls"]))
        with open(args.out, "w") as f:
            json.dump(prompts, f, indent=1)
        print(f"saved {len(entries)} boxes for {args.folder} -> {os.path.abspath(args.out)}")

    def on_mouse(event, x, y, flags, param):
        x, y = int(x / args.scale), int(y / args.scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"], drag["end"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
            (x1, y1), (x2, y2) = drag["start"], (x, y)
            drag["start"] = drag["end"] = None
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return
            # replace an existing box of the same class on this frame
            entries[:] = [e for e in entries
                          if not (e["frame"] == files[idx] and e["cls"] == cur_cls)]
            entries.append({"frame": files[idx], "cls": cur_cls,
                            "box": [x1, y1, x2, y2]})

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__.split("Controls")[1].split("Rules")[0])

    while True:
        img = cv2.imread(os.path.join(folder_path, files[idx]))
        h, w = img.shape[:2]
        # boxes on this frame
        for e in entries:
            if e["frame"] == files[idx]:
                x1, y1, x2, y2 = e["box"]
                c = COLORS[e["cls"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, CLASSES[e["cls"]], (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if drag["start"]:
            cv2.rectangle(img, drag["start"], drag["end"], COLORS[cur_cls], 1)
        n_here = sum(e["frame"] == files[idx] for e in entries)
        prompted = sorted({e["frame"] for e in entries})
        status = (f"{idx + 1}/{len(files)}  class: {cur_cls + 1}={CLASSES[cur_cls]}  "
                  f"boxes here: {n_here}  frames with boxes: {len(prompted)}")
        cv2.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(img, status, (5, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLORS[cur_cls], 1)
        cv2.imshow(win, cv2.resize(img, None, fx=args.scale, fy=args.scale))

        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        ch = chr(k)
        if ch in "123":
            cur_cls = int(ch) - 1
        elif ch == "d":
            idx = min(idx + 1, len(files) - 1)
        elif ch == "a":
            idx = max(idx - 1, 0)
        elif ch in "De":
            idx = min(idx + 10, len(files) - 1)
        elif ch in "Aq":
            idx = max(idx - 10, 0)
        elif ch == "u":
            here = [e for e in entries if e["frame"] == files[idx]]
            if here:
                entries.remove(here[-1])
        elif ch == "s":
            save()
        elif k == 27:   # ESC
            save()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
