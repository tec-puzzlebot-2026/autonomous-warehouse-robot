"""
train_yolo_seg.py - train YOLO11n-seg (or YOLO11n detect) on the box dataset, evaluate it,
export it for the Jetson. Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/train_yolo_seg.py                       # ROI model (Jetson)
    ~/venvs/sem9-vision/bin/python vision_m2/train_yolo_seg.py --model yolo11n.pt --name boxes_roi_det
    ~/venvs/sem9-vision/bin/python vision_m2/train_yolo_seg.py --full --imgsz 640 --name boxes_v2   # old style

ROI model (default): trains on yolo_dataset_roi/ (frames already cut to config_m2.ROI by
build_dataset_seg.py) at imgsz = NET_W (320). A 640x384 crop is scaled to 320x192 in
training exactly as on the robot, so the boxes have the same pixel size in both.
The ONNX export is RECTANGULAR (NET_H x NET_W = 192x320): a fixed 640x640 export would pad
the strip back to a square and throw away most of the speed-up.

--model yolo11n.pt trains a detection-only model on the same polygon labels (Ultralytics
turns them into boxes). About a third fewer FLOPs than -seg; use it if the robot only needs
the box rectangle, not its outline.

Results go to vision_m2/runs/<name>/ :
    weights/best.pt / best.onnx   the model
    weights/near_box.yaml         ROI, network size, thresholds -> copy WITH the model to the Jetson
    results.png                   losses + metrics per epoch
    confusion_matrix.png          box_plain vs box_drinks confusions
    predictions/                  30 validation crops with predictions drawn
"""
import argparse
import glob
import math
import os
import random
import sys

import cv2
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


def dataset_frame_size(data_yaml):
    d = yaml.safe_load(open(data_yaml))
    root = d.get("path") or os.path.dirname(data_yaml)
    for split in ("val", "train"):
        imgs = sorted(glob.glob(os.path.join(root, d[split], "*.jpg")))
        if imgs:
            h, w = cv2.imread(imgs[0]).shape[:2]
            return w, h, imgs if split == "val" else []
    raise SystemExit(f"No images found for {data_yaml}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full-frame dataset (yolo_dataset/), no ROI")
    ap.add_argument("--data", help="default: yolo_dataset_roi/data.yaml (yolo_dataset/ with --full)")
    ap.add_argument("--model", default="yolo11n-seg.pt", help="yolo11n-seg.pt (masks) or yolo11n.pt (boxes)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, help=f"default {C.NET_W} (ROI) / 640 (--full)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", help="default boxes_roi_v1 (ROI) / boxes_v2 (--full)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.data = args.data or os.path.join(C.DATASET if args.full else C.DATASET_ROI, "data.yaml")
    args.imgsz = args.imgsz or (640 if args.full else max(C.NET_W, C.NET_H))
    args.name = args.name or ("boxes_v2" if args.full else "boxes_roi_v1")

    # ---- network input the model will be exported with (and the Jetson will use) ----
    settings = C.near_box_settings()
    dw, dh, val_imgs = dataset_frame_size(args.data)
    if args.full:
        settings["roi"] = [0, 0, C.FRAME_W, C.FRAME_H]
        s = args.imgsz / float(max(C.FRAME_W, C.FRAME_H))
    else:
        rw, rh = C.ROI[2] - C.ROI[0], C.ROI[3] - C.ROI[1]
        if abs(dw / float(dh) - rw / float(rh)) > 0.02:
            raise SystemExit(f"Dataset images are {dw}x{dh} but config_m2.ROI is {rw}x{rh}. Rebuild the "
                             "dataset with build_dataset_seg.py after changing the ROI (or use --full).")
        s = args.imgsz / float(max(rw, rh))
        if (round(rw * s), round(rh * s)) != (C.NET_W, C.NET_H):
            print(f"[!] imgsz {args.imgsz} scales the ROI to {rw * s:.0f}x{rh * s:.0f}, config NET is "
                  f"{C.NET_W}x{C.NET_H}. near_box.yaml gets the size used here.")
    x0, y0, x1, y1 = settings["roi"]
    net_w = int(math.ceil((x1 - x0) * s / 32.0) * 32)
    net_h = int(math.ceil((y1 - y0) * s / 32.0) * 32)
    settings["net"] = [net_w, net_h]
    print(f"Dataset {dw}x{dh} | train imgsz {args.imgsz} | export {net_w}x{net_h} "
          f"({640 * 640 / float(net_w * net_h):.1f}x fewer pixels than 640x640)")

    model = YOLO(args.model)        # COCO-pretrained
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, cache="ram", patience=30,
        project=C.RUNS, name=args.name, exist_ok=True, seed=0,
        # geometric augmentation (YOLO moves the polygons with the image)
        fliplr=0.0,                 # printed pictures: a mirrored box is not a real box
        degrees=0.0, translate=0.1, scale=0.5, perspective=0.0005,
        mosaic=1.0, close_mosaic=10,
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
        overlap_mask=True,          # boxes in a rack touch each other (ignored by detect models)
        mask_ratio=4,
    )

    run_dir = os.path.join(C.RUNS, args.name)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))

    print("\n=== Final evaluation of best.pt on the validation set ===")
    m = best.val(data=args.data, imgsz=args.imgsz, device=args.device, project=C.RUNS,
                 name=f"{args.name}_val", exist_ok=True)
    print(f"BOX   mAP50 {m.box.map50:.3f}   mAP50-95 {m.box.map:.3f}")
    seg = getattr(m, "seg", None)
    if seg is not None:
        print(f"MASK  mAP50 {seg.map50:.3f}   mAP50-95 {seg.map:.3f}")
    part = seg if seg is not None else m.box
    for i, c in enumerate(part.ap_class_index):
        print(f"  {best.names[int(c)]:11s} {'mask' if seg is not None else 'box'}: precision {part.p[i]:.3f}  "
              f"recall {part.r[i]:.3f}  mAP50 {part.ap50[i]:.3f}")

    random.Random(0).shuffle(val_imgs)
    if val_imgs:
        best.predict(val_imgs[:30], imgsz=[net_h, net_w], conf=0.25, device=args.device, save=True,
                     project=run_dir, name="predictions", exist_ok=True, verbose=False)

    # rectangular, static-shape ONNX: TensorRT on the Nano builds its engine from this
    onnx_path = best.export(format="onnx", imgsz=[net_h, net_w], opset=12, simplify=True,
                            dynamic=False, device=args.device)
    settings["classes"] = [best.names[i] for i in range(len(best.names))]
    yaml_path = os.path.join(run_dir, "weights", "near_box.yaml")
    with open(yaml_path, "w") as f:
        f.write("# near_box.py settings - keep next to best.onnx / best.engine\n")
        yaml.safe_dump(settings, f, sort_keys=False, default_flow_style=None)

    print(f"\nDone.\n  PyTorch:  {run_dir}/weights/best.pt\n  ONNX:     {onnx_path}  (input 1x3x{net_h}x{net_w})"
          f"\n  Settings: {yaml_path}")
    print("\nOn the Jetson (copy best.onnx + near_box.yaml + near_box.py into one folder):\n"
          "  /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --workspace=1024\n"
          "  python3 near_box.py --model best.engine --source csi --bench 300")


if __name__ == "__main__":
    main()