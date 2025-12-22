#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

# ROS msg types
from sensor_msgs.msg import Image, CompressedImage


def _to_bgr_from_image(msg: Image) -> np.ndarray:
    # Common encodings: "bgr8", "rgb8", "mono8", etc.
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()

    # Build numpy view (copy only when needed)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8",):
        frame = data.reshape((h, w, 3))
        return frame
    if enc in ("rgb8",):
        frame = data.reshape((h, w, 3))
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if enc in ("mono8",):
        frame = data.reshape((h, w))
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Fallback: try to interpret as 3-channel
    # (If you have 16-bit depth images etc., you’ll want a custom path.)
    raise ValueError(f"Unsupported Image encoding: {msg.encoding}")


def _to_bgr_from_compressed(msg: CompressedImage) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # already BGR
    if frame is None:
        raise ValueError("Failed to decode CompressedImage")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to rosbag2 directory")
    ap.add_argument("--topic", required=True, help="Image topic name")
    ap.add_argument("--out", default="out.mp4", help="Output video file (mp4/avi/etc.)")
    ap.add_argument("--fps", type=float, default=30.0, help="Output FPS (constant)")
    ap.add_argument("--start", type=float, default=None, help="Start time (sec, relative to bag start)")
    ap.add_argument("--end", type=float, default=None, help="End time (sec, relative to bag start)")
    ap.add_argument("--max_frames", type=int, default=None, help="Stop after N frames")
    ap.add_argument("--codec", default="mp4v", help="FourCC codec (mp4v, avc1, XVID, etc.)")
    args = ap.parse_args()

    bag_path = str(Path(args.bag).expanduser())
    if not os.path.isdir(bag_path):
        raise FileNotFoundError(f"Bag directory not found: {bag_path}")

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    # Map topic -> type
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in topics:
        raise ValueError(f"Topic not found in bag: {args.topic}\nAvailable: {list(topics.keys())}")

    topic_type_str = topics[args.topic]
    msg_cls = get_message(topic_type_str)

    # Timing: bag timestamps are in nanoseconds since epoch-like; we just use relative windowing
    first_t = None
    written = 0
    writer = None

    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != args.topic:
            continue

        if first_t is None:
            first_t = t
        rel_sec = (t - first_t) / 1e9

        if args.start is not None and rel_sec < args.start:
            continue
        if args.end is not None and rel_sec > args.end:
            break

        msg = deserialize_message(data, msg_cls)

        # Convert to BGR frame
        if isinstance(msg, Image):
            frame = _to_bgr_from_image(msg)
        elif isinstance(msg, CompressedImage):
            frame = _to_bgr_from_compressed(msg)
        else:
            # Some bags store as Image but you didn't import class; handle by type string:
            if topic_type_str.endswith("CompressedImage"):
                frame = _to_bgr_from_compressed(msg)
            elif topic_type_str.endswith("Image"):
                frame = _to_bgr_from_image(msg)
            else:
                raise ValueError(f"Unsupported message type for video: {topic_type_str}")

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {args.out} (codec={args.codec})")

        writer.write(frame)
        written += 1

        if args.max_frames is not None and written >= args.max_frames:
            break

    if writer is not None:
        writer.release()

    print(f"Done. Wrote {written} frames to {args.out}")


if __name__ == "__main__":
    main()
