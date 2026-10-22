"""
train_yolo.py - train YOLO11n on the logo dataset, evaluate it, and export it.
Run on the workstation (the GPU):

    cd ~/data/cosmostests/sem9_team
    ~/venvs/sem9-vision/bin/python vision/train_yolo.py
    ~/venvs/sem9-vision/bin/python vision/train_yolo.py --epochs 150 --name try2   # another run

Results go to vision/runs/<name>/ :
    weights/best.pt      the best model (highest validation mAP)  <- this is the one you use
    weights/last.pt      the model after the last epoch
    weights/best.onnx    best.pt exported to ONNX (for the Jetson / TensorRT)
    results.png          loss + metric curves per epoch
    confusion_matrix.png which logo gets confused with which
    val_batch*_pred.jpg  validation images with the model's predictions
    predictions/         predictions drawn on a sample of validation frames
"""
import argparse
import glob
import os
import random

from ultralytics import YOLO

TEAM = os.path.expanduser("~/data/cosmostests/sem9_team")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{TEAM}/vision/yolo_dataset/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained starting point")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="yolo11n_logos")
    args = ap.parse_args()
    project = f"{TEAM}/vision/runs"

    # Start from YOLO11n pre-trained on COCO (80 everyday objects): it already
    # knows edges, shapes and textures, so it only has to learn our 3 logos
    # ("transfer learning"). Much better than starting from random weights.
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,          # passes over the whole training set
        imgsz=args.imgsz,            # images resized to 640 (our frames are 640x480)
        batch=args.batch,            # images per training step
        device=0,                    # the RTX 6000
        workers=8,                   # CPU processes that load/augment images
        cache="ram",                 # keep the dataset in RAM (avoids the slow disk)
        patience=30,                 # stop early if val mAP has not improved for 30 epochs
        project=project,
        name=args.name,
        exist_ok=True,
        seed=0,
        # ---- geometric augmentation done by YOLO (boxes are moved correctly) ----
        fliplr=0.0,                  # NO mirror flips: a mirrored logo is not our logo
        degrees=0.0,                 # no rotation (camera is always upright)
        translate=0.1,               # shift up to 10%
        scale=0.5,                   # zoom in/out up to 50%  (different distances)
        perspective=0.0005,          # slight viewpoint change
        mosaic=1.0,                  # combine 4 images into 1 (more variety per step)
        close_mosaic=10,             # last 10 epochs without mosaic (fine-tune on real-looking images)
        # ---- colour augmentation (on top of our photometric augmentation) ----
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
    )

    run_dir = f"{project}/{args.name}"
    best = YOLO(f"{run_dir}/weights/best.pt")

    print("\n=== Final evaluation of best.pt on the validation set ===")
    m = best.val(data=args.data, imgsz=args.imgsz, device=0, project=project,
                 name=f"{args.name}_val", exist_ok=True)
    names = best.names
    print(f"mAP50     (all): {m.box.map50:.3f}")
    print(f"mAP50-95  (all): {m.box.map:.3f}")
    for i, c in enumerate(m.box.ap_class_index):
        print(f"  {names[int(c)]:8s}  precision {m.box.p[i]:.3f}  recall {m.box.r[i]:.3f}  "
              f"mAP50 {m.box.ap50[i]:.3f}")

    print("\n=== Drawing predictions on 30 random validation frames ===")
    val_imgs = sorted(glob.glob(os.path.join(os.path.dirname(args.data), "images/val/*.jpg")))
    random.Random(0).shuffle(val_imgs)
    best.predict(val_imgs[:30], conf=0.25, device=0, save=True,
                 project=run_dir, name="predictions", exist_ok=True, verbose=False)

    print("\n=== Exporting to ONNX (for the Jetson / TensorRT) ===")
    onnx_path = best.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    print(f"\nDone.\n  PyTorch model: {run_dir}/weights/best.pt\n  ONNX model:    {onnx_path}")


if __name__ == "__main__":
    main()
