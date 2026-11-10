"""
prompt_tool_seg.py - draw the prompt boxes SAM 2 needs, for SEVERAL boxes per class.
Run on your LAPTOP (it needs a screen). One clip (frames folder) at a time:

    python3 prompt_tool_seg.py --frames <path>/frames --clip bag_boxes_train_1
    python3 prompt_tool_seg.py --frames <path>/frames --clip bag_boxes_test_1
    ...
All clips are saved into the same prompts.json.

Every physical box is one OBJECT with its own number (#1, #2, ...).
SAM 2 tracks each object through the clip by itself.

Controls (click the image window first):
    n            start a NEW object (new number) with the current class
    1 / 2        current class: 1 = box_plain, 2 = box_drinks
    mouse drag   draw the box of the CURRENT object on the CURRENT frame
    [ / ]        select the previous / next EXISTING object
                 (use this when a box that left the view comes back)
    d / a        next / previous frame
    D / A        +10 / -10 frames      (e / q also work)
    u            undo the last box drawn on this frame
    s            save            ESC   save and quit

Typical use on the first frame:  n, 1, draw box  ->  n, 1, draw box  ->  n, 2, draw box ...
Then step through the clip; when a NEW box appears press n and draw it; when an
OLD box comes back select it with [ ] and draw it again.
Rule: every visible box must belong to an object - unlabeled boxes are learned as background.
"""
import argparse
import json
import os

import cv2

CLASSES = ["box_plain", "box_drinks"]
COLORS = [(0, 200, 255), (60, 220, 60)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder that contains one folder per clip")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="prompts.json")
    ap.add_argument("--scale", type=float, default=1.5)
    args = ap.parse_args()

    clip_dir = os.path.join(args.frames, args.clip)
    files = sorted(f for f in os.listdir(clip_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"No images in {clip_dir}")

    prompts = json.load(open(args.out)) if os.path.exists(args.out) else {}
    entries = prompts.setdefault(args.clip, [])

    def objects():
        """obj id -> class (from existing prompts)"""
        return {e["obj"]: e["cls"] for e in entries}

    cur_cls = 0
    cur_obj = max(objects(), default=0) or None
    if cur_obj:
        cur_cls = objects()[cur_obj]
    idx = 0
    drag = {"start": None, "end": None}
    win = f"prompt_tool_seg - {args.clip}"

    def save():
        prompts[args.clip] = sorted(entries, key=lambda e: (e["frame"], e["obj"]))
        json.dump(prompts, open(args.out, "w"), indent=1)
        print(f"saved {len(entries)} prompts, {len(objects())} objects -> {os.path.abspath(args.out)}")

    def on_mouse(event, x, y, flags, param):
        nonlocal cur_obj
        x, y = int(x / args.scale), int(y / args.scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"] = drag["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
            (x1, y1), (x2, y2) = drag["start"], (x, y)
            drag["start"] = drag["end"] = None
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return
            if cur_obj is None:                       # first box ever: create object 1
                cur_obj = 1
            entries[:] = [e for e in entries if not (e["frame"] == files[idx] and e["obj"] == cur_obj)]
            entries.append({"frame": files[idx], "obj": cur_obj, "cls": cur_cls,
                            "box": [x1, y1, x2, y2]})

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__.split("Controls")[1].split("Typical")[0])

    while True:
        img = cv2.imread(os.path.join(clip_dir, files[idx]))
        h, w = img.shape[:2]
        for e in entries:
            if e["frame"] == files[idx]:
                x1, y1, x2, y2 = e["box"]
                c = COLORS[e["cls"]]
                th = 3 if e["obj"] == cur_obj else 1
                cv2.rectangle(img, (x1, y1), (x2, y2), c, th)
                cv2.putText(img, f"#{e['obj']} {CLASSES[e['cls']]}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)
        if drag["start"]:
            cv2.rectangle(img, drag["start"], drag["end"], COLORS[cur_cls], 1)
        n_obj = len(objects())
        status = (f"{idx + 1}/{len(files)} | current: #{cur_obj or '-'} {CLASSES[cur_cls]} | "
                  f"objects: {n_obj} | n=new [ ]=select")
        cv2.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(img, status, (5, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[cur_cls], 1)
        cv2.imshow(win, cv2.resize(img, None, fx=args.scale, fy=args.scale))

        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        ch = chr(k)
        objs = objects()
        if ch == "n":
            cur_obj = max(objs, default=0) + 1
            print(f"new object #{cur_obj} ({CLASSES[cur_cls]}) - draw its box")
        elif ch in "12":
            cur_cls = int(ch) - 1
            if cur_obj in objs:                       # existing object keeps its class
                print(f"#{cur_obj} already is {CLASSES[objs[cur_obj]]}; press n for a new object")
                cur_cls = objs[cur_obj]
        elif ch in "[]" and objs:
            ids = sorted(objs)
            pos = ids.index(cur_obj) if cur_obj in ids else 0
            cur_obj = ids[(pos + (1 if ch == "]" else -1)) % len(ids)]
            cur_cls = objs[cur_obj]
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
        elif k == 27:
            save()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()