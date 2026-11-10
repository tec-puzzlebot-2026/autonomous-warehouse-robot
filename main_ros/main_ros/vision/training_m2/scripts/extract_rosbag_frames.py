#!/usr/bin/env python3
"""
extract_rosbag_frames.py - ROS 2 bags -> frames/<clip>/*.jpg for the box segmentation model.
Run on the WORKSTATION, with the ROS-sourced python3 (needs rclpy/rosbag2_py/cv_bridge -
NOT the sem9-vision venv, which has torch/ultralytics/sam2 instead):

    source ~/ros2_ws/install/setup.bash
    cd ~/data/cosmostests/sem9_team
    python3 vision_m2/extract_rosbag_frames.py                               # the 4 original bags
    python3 vision_m2/extract_rosbag_frames.py --bags bag_boxes_near_1 bag_boxes_near_val_1
    python3 vision_m2/extract_rosbag_frames.py --bags bag_boxes_near_1 --fps 8 --topic /camera/image_rect

Record a bag per clip first (on the ROBOT, with vision.launch.py running so
/camera/image_rect exists):

    ros2 bag record -o ~/bags/bag_boxes_near_1 /camera/image_rect
    # ... drive / park / jitter the robot ...  Ctrl+C to stop
    scp -r puzzlebot:~/bags/bag_boxes_near_1 <this workstation>/vision_m2/bags/

Bag folder name = clip name = output frames folder name. Add 'test' or 'val' to the name
to make it a VALIDATION clip (config_m2.is_val checks for that substring).

Output per clip: vision_m2/frames/<clip>/<clip>_NNNNNN.jpg + manifest.csv (source topic and
timestamp of every saved frame - check this if you're ever unsure which topic old frames
came from). Default topic is config_m2.CAMERA_TOPIC (/camera/image_rect): keep this the SAME
topic box_target_node.py reads from at runtime, or the model sees different-looking images
in training than it does on the robot.

Supports sensor_msgs/msg/Image and sensor_msgs/msg/CompressedImage.
"""
import csv
import os
import sys
from pathlib import Path

import argparse
import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_m2 as C

BAGS_DIR = Path(C.BAGS)
FRAMES_DIR = Path(C.FRAMES) # frames/<clip>/*.jpg  (one folder per bag)

# The 4 original bags (used when --bags is not given, for backward compatibility).
# For new clips, just pass --bags <name> [<name> ...] instead of editing this.
BAG_MAPPING = {
    "bag_boxes_train_1": "bag_boxes_train_1",
    "bag_boxes_train_2": "bag_boxes_train_2",
    "bag_boxes_test_1": "bag_boxes_test_1",
    "bag_boxes_test_2": "bag_boxes_test_2",
}

SUPPORTED_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}


def find_camera_topics(topic_types):
    return {t.name: t.type for t in topic_types if t.type in SUPPORTED_TYPES}


def decode_image(message, message_type, bridge):
    if message_type == "sensor_msgs/msg/CompressedImage":
        return cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if message_type == "sensor_msgs/msg/Image":
        return bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
    return None


def open_bag(bag_path):
    metadata = rosbag2_py.Info().read_metadata(str(bag_path), "")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=metadata.storage_identifier),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    return reader


def pick_topic(available_topics, wanted):
    """Prefer `wanted` if the bag has it; otherwise the only image topic; otherwise ask."""
    if wanted in available_topics:
        return wanted
    if len(available_topics) == 1:
        only = next(iter(available_topics))
        print(f"[WARN] '{wanted}' not in this bag; using the only image topic found: {only}")
        return only
    print(f"[ERROR] '{wanted}' not in this bag, and it has more than one image topic:")
    for t in available_topics:
        print(f"  {t}")
    print("Pass --topic to pick one.")
    return None


def extract_bag(bag_name, output_name, fps, bags_dir, frames_dir, wanted_topic, jpeg_quality=95):
    bag_path, output_path = bags_dir / bag_name, frames_dir / output_name
    if not bag_path.exists():
        print(f"[ERROR] Bag not found: {bag_path}")
        return
    output_path.mkdir(parents=True, exist_ok=True)
    if not C.is_val(output_name):
        print(f"[!] '{output_name}' has no 'val'/'test' in it - this clip becomes TRAINING data.")

    print("\n" + "=" * 60 + f"\n[BAG] {bag_name}\n[OUTPUT] {output_name}\n" + "=" * 60)
    try:
        reader = open_bag(bag_path)
    except Exception as error:
        print(f"[ERROR] Could not open bag: {error}")
        return

    available_topics = find_camera_topics(reader.get_all_topics_and_types())
    if not available_topics:
        print("[ERROR] No supported camera topic found.")
        return
    selected_topic = pick_topic(available_topics, wanted_topic)
    if selected_topic is None:
        return
    selected_type = available_topics[selected_topic]
    print(f"[CAMERA] {selected_topic}\n[TYPE] {selected_type}\n[FPS] {fps}")

    message_class = get_message(selected_type)
    bridge = CvBridge()
    interval_ns = round(1_000_000_000 / fps)
    last_saved_timestamp = None
    saved_frames = read_frames = 0
    manifest_rows = []

    while reader.has_next():
        topic, raw_data, timestamp = reader.read_next()
        if topic != selected_topic:
            continue
        read_frames += 1
        if last_saved_timestamp is not None and timestamp - last_saved_timestamp < interval_ns:
            continue
        try:
            frame = decode_image(deserialize_message(raw_data, message_class), selected_type, bridge)
        except Exception as error:
            print(f"[WARNING] Could not decode frame {read_frames}: {error}")
            continue
        if frame is None or frame.size == 0:
            continue

        filename = f"{output_name}_{saved_frames:06d}.jpg"
        if not cv2.imwrite(str(output_path / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            print(f"[WARNING] Failed to save {filename}")
            continue
        manifest_rows.append({"filename": filename, "bag": bag_name, "topic": selected_topic,
                              "timestamp_ns": timestamp})
        saved_frames += 1
        last_saved_timestamp = timestamp
        if saved_frames % 100 == 0:
            print(f"[PROGRESS] Saved {saved_frames} frames")

    with open(output_path / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "bag", "topic", "timestamp_ns"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"[DONE] Camera messages read: {read_frames}\n[DONE] Frames saved: {saved_frames}"
          f"\n[DONE] Output: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract training frames from ROS 2 bags.")
    ap.add_argument("--bags", nargs="+", help="bag folder names under --bags-dir (default: the "
                    "original 4 bags). Output goes to <frames-dir>/<bag name>/")
    ap.add_argument("--bags-dir", type=Path, default=BAGS_DIR, help=f"default: {BAGS_DIR}")
    ap.add_argument("--frames-dir", type=Path, default=FRAMES_DIR, help=f"default: {FRAMES_DIR}")
    ap.add_argument("--fps", type=float, default=5.0, help="target extraction FPS (default: 5.0)")
    ap.add_argument("--topic", default=C.CAMERA_TOPIC, help=f"default: {C.CAMERA_TOPIC} "
                    "(config_m2.CAMERA_TOPIC - keep in sync with box_target_node.py's image_topic)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()
    if args.fps <= 0:
        ap.error("--fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        ap.error("--jpeg-quality must be between 1 and 100")

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    mapping = {b: b for b in args.bags} if args.bags else BAG_MAPPING
    for bag_name, output_name in mapping.items():
        extract_bag(bag_name, output_name, args.fps, args.bags_dir, args.frames_dir, args.topic,
                   args.jpeg_quality)
    print("\n[FINISHED] All bags processed.")


if __name__ == "__main__":
    main()