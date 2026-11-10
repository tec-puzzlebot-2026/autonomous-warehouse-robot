"""
train_yolo_seg.py - train YOLO11n-seg on the box dataset, evaluate it, export it.
Run on the workstation:

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision_m2/train_yolo_seg.py
    ~/venvs/sem9-vision/bin/python vision_m2/train_yolo_seg.py --name boxes_v2 --epochs 150

Results go to vision_m2/runs/<name>/ :
    weights/best.pt / best.onnx   the model to use
    results.png                   losses + box and mask metrics per epoch
    confusion_matrix.png          box_plain vs box_drinks confusions
    predictions/                  masks drawn on 30 validation frames
"""
import argparse
import glob
import os
import random
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(C.DATASET, "data.yaml"))
    ap.add_argument("--model", default="yolo11n-seg.pt", help="pretrained SEGMENTATION model")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="boxes_v1")
    args = ap.parse_args()

    model = YOLO(args.model)        # COCO-pretrained, already predicts masks
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=8, cache="ram", patience=30,
        project=C.RUNS, name=args.name, exist_ok=True, seed=0,
        # geometric augmentation (YOLO moves the polygons with the image)
        fliplr=0.0,                 # printed pictures: a mirrored box is not a real box
        degrees=0.0, translate=0.1, scale=0.5, perspective=0.0005,
        mosaic=1.0, close_mosaic=10,
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
        overlap_mask=True,          # boxes in a rack touch each other
        mask_ratio=4,
    )

    run_dir = os.path.join(C.RUNS, args.name)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))

    print("\n=== Final evaluation of best.pt on the validation set ===")
    m = best.val(data=args.data, imgsz=args.imgsz, device=0, project=C.RUNS,
                 name=f"{args.name}_val", exist_ok=True)
    print(f"BOX   mAP50 {m.box.map50:.3f}   mAP50-95 {m.box.map:.3f}")
    print(f"MASK  mAP50 {m.seg.map50:.3f}   mAP50-95 {m.seg.map:.3f}")
    for i, c in enumerate(m.seg.ap_class_index):
        print(f"  {best.names[int(c)]:11s} mask: precision {m.seg.p[i]:.3f}  recall {m.seg.r[i]:.3f}  "
              f"mAP50 {m.seg.ap50[i]:.3f}")

    val_imgs = sorted(glob.glob(os.path.join(os.path.dirname(args.data), "images/val/*.jpg")))
    random.Random(0).shuffle(val_imgs)
    best.predict(val_imgs[:30], conf=0.25, device=0, save=True,
                 project=run_dir, name="predictions", exist_ok=True, verbose=False)

    onnx_path = best.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    print(f"\nDone.\n  PyTorch: {run_dir}/weights/best.pt\n  ONNX:    {onnx_path}")


if __name__ == "__main__":
    main()