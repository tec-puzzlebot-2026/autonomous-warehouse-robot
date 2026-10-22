#!/usr/bin/env python3

"""
Extract camera frames from ROS 2 bags for logo classification.

Supports:
    sensor_msgs/msg/Image
    sensor_msgs/msg/CompressedImage

Output:
    frames/logo_1_train/
    frames/logo_1_val/
    frames/logo_2_train/
    frames/logo_2_val/
    frames/logo_3_train/
    frames/logo_3_val/
"""

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np

import rosbag2_py

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


BASE_DIR = Path(__file__).resolve().parent

BAGS_DIR = BASE_DIR / "data" / "raw_bags"
FRAMES_DIR = BASE_DIR / "frames"

BAG_MAPPING = {
    "bag_logo_1": "logo_1_train",
    "bag_logo_1_1": "logo_1_val",

    "bag_logo_2": "logo_2_train",
    "bag_logo_2_1": "logo_2_val",

    "bag_logo_3": "logo_3_train",
    "bag_logo_3_1": "logo_3_val",
}

SUPPORTED_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}


def find_camera_topics(topic_types):
    """Find image topics in a ROS 2 bag."""

    return {
        topic.name: topic.type
        for topic in topic_types
        if topic.type in SUPPORTED_TYPES
    }


def decode_image(message, message_type, bridge):
    """Convert a ROS image message to an OpenCV BGR image."""

    if message_type == "sensor_msgs/msg/CompressedImage":

        buffer = np.frombuffer(
            message.data,
            dtype=np.uint8,
        )

        return cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

    if message_type == "sensor_msgs/msg/Image":

        return bridge.imgmsg_to_cv2(
            message,
            desired_encoding="bgr8",
        )

    return None


def open_bag(bag_path):
    """Open a ROS 2 bag using its storage backend."""

    metadata_reader = rosbag2_py.Info()

    metadata = metadata_reader.read_metadata(
        str(bag_path),
        "",
    )

    storage_id = metadata.storage_identifier

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(
        storage_options,
        converter_options,
    )

    return reader


def extract_bag(
    bag_name,
    output_name,
    fps,
    camera_topic=None,
    jpeg_quality=95,
):

    bag_path = BAGS_DIR / bag_name
    output_path = FRAMES_DIR / output_name

    if not bag_path.exists():
        print(f"[ERROR] Bag not found: {bag_path}")
        return

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(f"[BAG] {bag_name}")
    print(f"[OUTPUT] {output_name}")
    print("=" * 60)

    try:
        reader = open_bag(bag_path)

    except Exception as error:
        print(f"[ERROR] Could not open bag: {error}")
        return

    available_topics = find_camera_topics(
        reader.get_all_topics_and_types()
    )

    if not available_topics:
        print("[ERROR] No supported camera topic found.")
        return

    if camera_topic is not None:

        if camera_topic not in available_topics:
            print(
                f"[ERROR] Camera topic '{camera_topic}' "
                "not found in this bag."
            )
            print("Available image topics:", available_topics)
            return

        selected_topic = camera_topic

    elif len(available_topics) == 1:

        selected_topic = next(iter(available_topics))

    else:

        print("[ERROR] Multiple camera topics found:")
        for topic in available_topics:
            print(f"  {topic}")

        print(
            "Specify the desired topic using --topic."
        )
        return

    selected_type = available_topics[selected_topic]

    print(f"[CAMERA] {selected_topic}")
    print(f"[TYPE] {selected_type}")
    print(f"[FPS] {fps}")

    message_class = get_message(selected_type)

    bridge = CvBridge()

    interval_ns = round(1_000_000_000 / fps)

    last_saved_timestamp = None

    saved_frames = 0
    read_frames = 0

    manifest_rows = []

    while reader.has_next():

        topic, raw_data, timestamp = reader.read_next()

        if topic != selected_topic:
            continue

        read_frames += 1

        if last_saved_timestamp is not None:

            elapsed_ns = timestamp - last_saved_timestamp

            if elapsed_ns < interval_ns:
                continue

        try:

            message = deserialize_message(
                raw_data,
                message_class,
            )

            frame = decode_image(
                message,
                selected_type,
                bridge,
            )

        except Exception as error:

            print(
                f"[WARNING] Could not decode frame "
                f"{read_frames}: {error}"
            )
            continue

        if frame is None or frame.size == 0:
            continue

        filename = (
            f"{output_name}_{saved_frames:06d}.jpg"
        )

        filepath = output_path / filename

        success = cv2.imwrite(
            str(filepath),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality,
            ],
        )

        if not success:
            print(f"[WARNING] Failed to save {filepath}")
            continue

        manifest_rows.append(
            {
                "filename": filename,
                "bag": bag_name,
                "topic": selected_topic,
                "timestamp_ns": timestamp,
            }
        )

        saved_frames += 1

        last_saved_timestamp = timestamp

        if saved_frames % 100 == 0:
            print(f"[PROGRESS] Saved {saved_frames} frames")

    manifest_path = output_path / "manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "bag",
                "topic",
                "timestamp_ns",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[DONE] Camera messages read: {read_frames}")
    print(f"[DONE] Frames saved: {saved_frames}")
    print(f"[DONE] Output: {output_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Extract training frames from ROS 2 bags."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target extraction FPS (default: 5.0)",
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Camera topic, if multiple image topics exist",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for bag_name, output_name in BAG_MAPPING.items():

        extract_bag(
            bag_name=bag_name,
            output_name=output_name,
            fps=args.fps,
            camera_topic=args.topic,
            jpeg_quality=args.jpeg_quality,
        )

    print("\n[FINISHED] All bags processed.")


if __name__ == "__main__":
    main()
