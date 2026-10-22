#!/usr/bin/env python3

"""
Extract consecutive frames from a ROS 2 bag.

Example:

python3 extract_consecutive_frames.py \
    --bag data/raw_bags/bag_logo_1 \
    --start 7 \
    --duration 4 \
    --fps 30

Output:
    output/consecutive_frames/bag_logo_1/
        frame_000000.png
        frame_000001.png
        ...
        frame_000119.png

The script:
    1. Reads images from a ROS 2 bag.
    2. Selects a time interval relative to the first camera frame.
    3. Resamples the frames to the requested FPS.
    4. Saves exactly duration * FPS frames.

Supports:
    - sensor_msgs/msg/Image
    - sensor_msgs/msg/CompressedImage
    - SQLite3 ROS bags (.db3)
    - MCAP ROS bags (.mcap), if the storage plugin is installed

Requires ROS 2 Humble.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import rosbag2_py

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT = (
    SCRIPT_DIR / "output" / "consecutive_frames"
)

SUPPORTED_IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}


# ============================================================
# ROS BAG UTILITIES
# ============================================================

def resolve_bag_path(bag_argument):
    """
    Resolve relative paths from the training directory.
    """

    bag_path = Path(bag_argument).expanduser()

    if not bag_path.is_absolute():
        bag_path = SCRIPT_DIR / bag_path

    bag_path = bag_path.resolve()

    if not bag_path.exists():
        raise FileNotFoundError(
            f"ROS bag not found: {bag_path}"
        )

    return bag_path


def detect_storage_id(bag_path):
    """
    Detect rosbag storage format.

    Supports:
        sqlite3
        mcap
    """

    if bag_path.is_file():

        if bag_path.suffix == ".db3":
            return "sqlite3"

        if bag_path.suffix == ".mcap":
            return "mcap"

    if bag_path.is_dir():

        if list(bag_path.glob("*.db3")):
            return "sqlite3"

        if list(bag_path.glob("*.mcap")):
            return "mcap"

    raise RuntimeError(
        "Could not detect ROS bag storage format. "
        "Expected .db3 or .mcap files."
    )


def open_rosbag(bag_path):
    """
    Open a ROS 2 bag for sequential reading.
    """

    storage_id = detect_storage_id(bag_path)

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


def get_image_topics(reader):
    """
    Discover image topics inside the bag.
    """

    topics = reader.get_all_topics_and_types()

    image_topics = {}

    for topic in topics:

        if topic.type in SUPPORTED_IMAGE_TYPES:

            image_topics[topic.name] = topic.type

    return image_topics


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(
    serialized_data,
    message_type,
    bridge,
):
    """
    Convert ROS image messages to OpenCV BGR images.
    """

    if message_type == "sensor_msgs/msg/Image":

        msg = deserialize_message(
            serialized_data,
            Image,
        )

        image = bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

        return image

    if message_type == "sensor_msgs/msg/CompressedImage":

        msg = deserialize_message(
            serialized_data,
            CompressedImage,
        )

        image_array = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        return image

    return None


# ============================================================
# EXTRACT CAMERA FRAMES
# ============================================================

def read_frames(
    reader,
    topic_name,
    topic_type,
    start_seconds,
    duration_seconds,
):
    """
    Read camera frames from the requested interval.

    Time is relative to the first recorded camera message,
    using rosbag recording timestamps.

    Frames are stored in memory for resampling.
    """

    bridge = CvBridge()

    frames = []

    first_timestamp = None

    end_seconds = start_seconds + duration_seconds

    total_messages = 0

    while reader.has_next():

        topic, data, timestamp = reader.read_next()

        if topic != topic_name:
            continue

        total_messages += 1

        if first_timestamp is None:
            first_timestamp = timestamp

        relative_time = (
            timestamp - first_timestamp
        ) / 1e9

        if relative_time < start_seconds:
            continue

        if relative_time > end_seconds:
            break

        try:

            image = decode_image(
                data,
                topic_type,
                bridge,
            )

            if image is None:
                continue

            frames.append(
                (relative_time, image)
            )

        except Exception as error:

            print(
                f"Warning: Could not decode frame "
                f"at {relative_time:.3f}s: {error}"
            )

    print(
        f"Camera messages inspected: {total_messages}"
    )

    print(
        f"Frames found in interval: {len(frames)}"
    )

    return frames


# ============================================================
# FRAME RESAMPLING
# ============================================================

def resample_frames(
    frames,
    start_seconds,
    duration_seconds,
    fps,
):
    """
    Resample to exactly duration * fps frames.

    Uses nearest-neighbor temporal sampling.

    Example:
        start=7
        duration=4
        fps=30

    Output:
        120 frames

    If the original FPS is lower, frames may be repeated.
    """

    target_count = int(
        round(duration_seconds * fps)
    )

    if target_count <= 0:
        raise ValueError(
            "Requested frame count must be positive."
        )

    timestamps = np.array(
        [frame[0] for frame in frames],
        dtype=np.float64,
    )

    images = [
        frame[1]
        for frame in frames
    ]

    target_times = (
        start_seconds
        + np.arange(target_count) / fps
    )

    selected_frames = []

    for target_time in target_times:

        index = np.searchsorted(
            timestamps,
            target_time,
        )

        if index == 0:

            selected_index = 0

        elif index >= len(timestamps):

            selected_index = len(timestamps) - 1

        else:

            before = index - 1
            after = index

            if (
                abs(timestamps[before] - target_time)
                <=
                abs(timestamps[after] - target_time)
            ):

                selected_index = before

            else:

                selected_index = after

        selected_frames.append({
            "target_time": target_time,
            "source_time": timestamps[selected_index],
            "source_index": selected_index,
            "image": images[selected_index],
        })

    return selected_frames


# ============================================================
# SAVE FRAMES
# ============================================================

def save_frames(
    frames,
    output_folder,
):
    """
    Save frames as numbered PNG images.

    Also save a CSV file recording source timestamps.
    """

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_folder / "manifest.csv"
    )

    rows = []

    for index, frame in enumerate(frames):

        filename = (
            f"frame_{index:06d}.png"
        )

        destination = (
            output_folder / filename
        )

        success = cv2.imwrite(
            str(destination),
            frame["image"],
        )

        if not success:

            raise RuntimeError(
                f"Failed to save {destination}"
            )

        rows.append({
            "filename": filename,
            "target_time_sec": (
                f"{frame['target_time']:.6f}"
            ),
            "source_time_sec": (
                f"{frame['source_time']:.6f}"
            ),
            "source_index": frame["source_index"],
        })

    with manifest_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "target_time_sec",
                "source_time_sec",
                "source_index",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved {len(frames)} frames to:"
    )

    print(output_folder)

    print(
        f"Manifest saved to: {manifest_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract consecutive frames from a ROS 2 bag."
        )
    )

    parser.add_argument(
        "--bag",
        required=True,
        help=(
            "ROS bag path, e.g. "
            "data/raw_bags/bag_logo_1"
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        required=True,
        help=(
            "Start time in seconds relative "
            "to the first camera frame."
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Duration in seconds.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output frames per second (default: 30).",
    )

    parser.add_argument(
        "--topic",
        default=None,
        help=(
            "Camera topic. Automatically selected "
            "if the bag contains one image topic."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory.",
    )

    args = parser.parse_args()

    if args.start < 0:
        parser.error(
            "--start must be >= 0"
        )

    if args.duration <= 0:
        parser.error(
            "--duration must be > 0"
        )

    if args.fps <= 0:
        parser.error(
            "--fps must be > 0"
        )

    bag_path = resolve_bag_path(
        args.bag
    )

    print("\n" + "=" * 60)
    print("ROS 2 CONSECUTIVE FRAME EXTRACTION")
    print("=" * 60)

    print(f"Bag: {bag_path}")
    print(f"Start: {args.start} seconds")
    print(f"Duration: {args.duration} seconds")
    print(f"End: {args.start + args.duration} seconds")
    print(f"FPS: {args.fps}")

    reader = open_rosbag(
        bag_path
    )

    image_topics = get_image_topics(
        reader
    )

    if not image_topics:

        raise RuntimeError(
            "No Image or CompressedImage topics "
            "found in this ROS bag."
        )

    if args.topic is None:

        if len(image_topics) == 1:

            topic_name = next(
                iter(image_topics)
            )

        else:

            print("\nAvailable camera topics:")

            for topic in image_topics:

                print(
                    f"  {topic}: "
                    f"{image_topics[topic]}"
                )

            raise RuntimeError(
                "Multiple image topics found. "
                "Specify one using --topic."
            )

    else:

        topic_name = args.topic

        if topic_name not in image_topics:

            raise RuntimeError(
                f"Topic not found: {topic_name}\n"
                f"Available: {list(image_topics)}"
            )

    topic_type = image_topics[
        topic_name
    ]

    print(f"Camera topic: {topic_name}")
    print(f"Message type: {topic_type}")

    frames = read_frames(
        reader=reader,
        topic_name=topic_name,
        topic_type=topic_type,
        start_seconds=args.start,
        duration_seconds=args.duration,
    )

    if not frames:

        raise RuntimeError(
            "No camera frames found in the requested "
            "time interval. Check --start and --duration."
        )

    selected_frames = resample_frames(
        frames=frames,
        start_seconds=args.start,
        duration_seconds=args.duration,
        fps=args.fps,
    )

    if args.output:

        output_folder = Path(
            args.output
        ).expanduser()

        if not output_folder.is_absolute():
            output_folder = (
                SCRIPT_DIR / output_folder
            )

    else:

        output_folder = (
            DEFAULT_OUTPUT
            / bag_path.stem
            / f"start_{args.start:g}_duration_{args.duration:g}"
        )

    save_frames(
        selected_frames,
        output_folder,
    )

    unique_sources = len({
        frame["source_index"]
        for frame in selected_frames
    })

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Requested frames: "
        f"{len(selected_frames)}"
    )

    print(
        f"Unique source frames: "
        f"{unique_sources}"
    )

    print(
        f"Repeated frames: "
        f"{len(selected_frames) - unique_sources}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()