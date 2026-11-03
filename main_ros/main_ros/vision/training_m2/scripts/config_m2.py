"""
config_m2.py - shared settings for the box segmentation model (model 2).
All scripts of training_m2 import this file. Change things here, not in each script.
"""
import os

# ---- classes (YOLO class id = position in this list) ----
CLASSES = ["box_plain", "box_drinks"]          # 0 = white/brown box, 1 = drinks box
COLORS = [(0, 200, 255), (60, 220, 60)]          # BGR colour per class (review images)

# ---- folders on the WORKSTATION ----
TEAM = os.path.expanduser("~/data/cosmostests/sem9_team")
WORK = os.path.join(TEAM, "vision_m2")           # everything for model 2 lives here
FRAMES = os.path.join(WORK, "frames")            # frames/<clip>/*.jpg  (one folder per bag)
PROMPTS = os.path.join(WORK, "prompts.json")
LABELS = os.path.join(WORK, "labels_auto")       # SAM 2 -> YOLO-seg labels
CHECK = os.path.join(WORK, "check")              # review videos / sheets
DATASET = os.path.join(WORK, "yolo_dataset")
RUNS = os.path.join(WORK, "runs")

SAM2_CKPT = os.path.expanduser("~/data/weights/sam2_hiera_large.pt")
SAM2_CFG = "sam2_hiera_l.yaml"


def is_val(clip_name):
    """Clips whose name contains 'val' or 'test' are validation, the rest training."""
    n = clip_name.lower()
    return "val" in n or "test" in n