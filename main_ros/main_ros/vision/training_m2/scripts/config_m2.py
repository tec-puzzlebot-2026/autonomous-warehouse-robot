"""
config_m2.py - shared settings for the box segmentation model (model 2).
All scripts of training_m2 import this file. Change things here, not in each script.

The NEAR-BOX block at the bottom is copied next to the exported model as near_box.yaml
(train_yolo_seg.py does it), so the Jetson uses exactly the same ROI and size filter
as training and evaluation.
"""
import os

# ---- classes (YOLO class id = position in this list) ----
CLASSES = ["box_plain", "box_drinks"]          # 0 = white/brown box, 1 = drinks box
COLORS = [(0, 200, 255), (60, 220, 60)]          # BGR colour per class (review images)

# ---- folders on the WORKSTATION ----
TEAM = os.path.expanduser("~/data/cosmostests/sem9_team")
WORK = os.path.join(TEAM, "vision_m2")           # everything for model 2 lives here
FRAMES = os.path.join(WORK, "frames")            # frames/<clip>/*.jpg  (one folder per bag)
BAGS = os.path.join(WORK, "bags")  # bags/<clip>/   raw ros2 bag recordings, pre-extraction
PROMPTS = os.path.join(WORK, "prompts.json")
LABELS = os.path.join(WORK, "labels_auto")       # SAM 2 -> YOLO-seg labels (FULL frames, all boxes)
CHECK = os.path.join(WORK, "check")              # review videos / sheets
DATASET = os.path.join(WORK, "yolo_dataset")     # full-frame dataset (old)
DATASET_ROI = os.path.join(WORK, "yolo_dataset_roi")   # ROI-cropped dataset (for the Jetson)
RUNS = os.path.join(WORK, "runs")

SAM2_CKPT = os.path.expanduser("~/data/weights/sam2_hiera_large.pt")
SAM2_CFG = "sam2_hiera_l.yaml"
# Camera topic frames are recorded from (extract_rosbag_frames.py, record_clip.py) and that
# box_target_node.py reads from at runtime - keep these the same, or the robot runs the model
# on different-looking images than it was trained on. /camera/image_rect is colour-corrected
# AND rectified; it needs vision.launch.py running, not just camera.launch.py.
CAMERA_TOPIC = "/camera/image_rect"

def is_val(clip_name):
    """Clips whose name contains 'val' or 'test' are validation, the rest training."""
    n = clip_name.lower()
    return "val" in n or "test" in n


# ============================================================================
# NEAR-BOX runtime settings (Jetson Nano)
# ============================================================================
# Frame the camera publishes (camera_node default: 640x480).
FRAME_W, FRAME_H = 640, 480

# Region of interest in FRAME pixels: x0, y0, x1, y1. Everything outside is cut away
# before the network. The camera is ~9 cm above the floor and looks straight ahead, so a
# box 10-30 cm away spans the image CENTRE and continues to the bottom edge; it does not
# stay in the bottom half (a 15 cm tall box at 20 cm reaches row ~80). Far floor boxes
# (1-2 m) sit just below the centre line, rows 250-290, so cropping cannot remove them -
# the size filter does. Run roi_from_labels.py to pick y0 from your labels.
ROI = (0, 96, 640, 480)                          # 640 x 384

# Network input (width, height). Multiples of 32. The ROI is scaled down by 2:
# near boxes are >= ~150 px wide in the frame, still >= 75 px at the network, which is
# easy for YOLO. 320x192 = 61k pixels vs 409k for the old 640x640 export (6.7x fewer).
# Keep NET aspect == ROI aspect (no padding). y0 choices that keep it exact with /2:
# 32, 96, 160, 224  (ROI height 448, 384, 320, 256 -> NET height 224, 192, 160, 128).
NET_W, NET_H = 320, 192

# Size filter, in FRAME pixels. size = sqrt(box width * box height).
# A 10 cm box at 30 cm is ~170 px; the same box at 1 m is ~50 px.
MIN_SIZE_PX = 100
CONF = 0.35                                      # detection confidence used at runtime
NMS_IOU = 0.5                                    # class-agnostic NMS (one physical box = one detection)
MAX_DET = 10
# Keep the previous target if it is still >= STICKY x the area of the largest box
# (avoids flicker between two similar boxes). 1.0 = always take the largest.
STICKY = 0.8


def near_box_settings():
    """Dict written to near_box.yaml next to the exported model."""
    return {
        "classes": list(CLASSES),
        "frame": [FRAME_W, FRAME_H],
        "roi": list(ROI),
        "net": [NET_W, NET_H],
        "min_size_px": MIN_SIZE_PX,
        "conf": CONF,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
        "sticky": STICKY,
    }