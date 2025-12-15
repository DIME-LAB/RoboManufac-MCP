#!/usr/bin/env python3
"""
ROS2-based Assembly Pose Detector.

Detects ArUco markers for all available models, estimates object poses with Kalman
filtering, and overlays wireframes. Shares the same pose math and filtering logic
as the object pose estimators.
"""

import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from core.kalman_filter import (
    QuaternionKalman,
    MAX_MOVEMENT_THRESHOLD,
    HOLD_REQUIRED_FRAMES,
    GHOST_TRACKING_FRAMES,
    BLEND_FACTOR,
)
from core.model_io import (
    load_wireframe_data,
    load_aruco_annotations,
    get_available_models,
)
from core.pose_math import estimate_object_pose_from_marker, rotation_matrix_to_euler
from core.mesh_ops import transform_mesh_to_camera_frame, project_vertices_to_image, draw_wireframe
from object_pose_estimator_camera import estimate_pose_with_kalman


# Camera parameters (used if camera info not available)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV = 69.4  # degrees
CAMERA_VFOV = 42.5  # degrees

# Default data directory (repo_root/data)
DATA_DIRECTORY_DEFAULT = Path(__file__).resolve().parent.parent.parent / "data"


class AssemblyPoseDetectorROS2(Node):
    """ROS2 node for assembly pose detection with Kalman filtering."""

    def __init__(self, camera_topic: str, data_dir: Path):
        super().__init__("assembly_pose_detector_ros2")
        self.data_dir = Path(data_dir)

        self.bridge = CvBridge()

        # Load all models and markers
        self.model_data, self.aruco_dict_names = self.load_all_models()

        # Camera parameters
        self.setup_camera_parameters()

        # ArUco detectors (one per dictionary)
        self.detectors = self.setup_aruco_detectors(self.aruco_dict_names)

        # Kalman tracking state
        self.kalman_filters = {}
        self.marker_stabilities = {}
        self.last_seen_frames = {}
        self.current_frame = 0
        self.should_shutdown = False

        # ROS2 subscription
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)

        # Precompute marker→model lookup and dictionary grouping
        self.marker_lookup, self.marker_ids_by_dict = self.build_marker_lookup(self.model_data)

        self.get_logger().info(f"Assembly Pose Detector initialized. Camera topic: {camera_topic}")
        self.get_logger().info(f"Loaded models: {list(self.model_data.keys())}")
        self.get_logger().info(f"Using ArUco dictionaries: {sorted(self.aruco_dict_names)}")

    def setup_camera_parameters(self):
        fx = CAMERA_WIDTH / (2 * np.tan(np.deg2rad(CAMERA_HFOV / 2)))
        fy = CAMERA_HEIGHT / (2 * np.tan(np.deg2rad(CAMERA_VFOV / 2)))
        self.camera_matrix = np.array(
            [
                [fx, 0, CAMERA_WIDTH / 2],
                [0, fy, CAMERA_HEIGHT / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def setup_aruco_detectors(self, dict_names):
        detectors = {}
        for dict_name in dict_names:
            try:
                dictionary_id = getattr(aruco, dict_name)
            except AttributeError:
                self.get_logger().warning(f"Unknown dictionary '{dict_name}', skipping")
                continue
            dictionary = aruco.getPredefinedDictionary(dictionary_id)
            parameters = aruco.DetectorParameters()
            detectors[dict_name] = aruco.ArucoDetector(dictionary, parameters)
        if not detectors:
            self.get_logger().warning("No valid ArUco dictionaries found; defaulting to DICT_4X4_50")
            detectors["DICT_4X4_50"] = aruco.ArucoDetector(
                aruco.getPredefinedDictionary(aruco.DICT_4X4_50), aruco.DetectorParameters()
            )
        return detectors

    def load_all_models(self):
        available_models = get_available_models(self.data_dir)
        if not available_models:
            raise RuntimeError(f"No models found in data directory: {self.data_dir}")

        model_data = {}
        aruco_dict_names = set()

        for model_name in available_models:
            wireframe_file = self.data_dir / "wireframe" / f"{model_name}_wireframe.json"
            aruco_annotations_file = self.data_dir / "aruco" / f"{model_name}_aruco.json"

            vertices, edges = load_wireframe_data(wireframe_file)
            (
                aruco_annotations,
                base_marker_size,
                border_width_percent,
                aruco_dict_name,
            ) = load_aruco_annotations(aruco_annotations_file)

            aruco_dict_names.add(aruco_dict_name)
            marker_annotations = {ann["aruco_id"]: ann for ann in aruco_annotations}

            model_data[model_name] = {
                "vertices": vertices,
                "edges": edges,
                "marker_annotations": marker_annotations,
                "base_marker_size": base_marker_size,
                "border_width_percent": border_width_percent,
                "aruco_dict_name": aruco_dict_name,
            }

            self.get_logger().info(
                f"Loaded {model_name}: {len(vertices)} vertices, {len(edges)} edges, {len(marker_annotations)} markers"
            )

        return model_data, aruco_dict_names

    @staticmethod
    def build_marker_lookup(model_data):
        # Key by (dict_name, marker_id) to avoid collisions across models sharing IDs.
        lookup = {}
        marker_ids_by_dict = {}
        for model_name, data in model_data.items():
            dict_name = data["aruco_dict_name"]
            marker_ids_by_dict.setdefault(dict_name, set())
            for marker_id, annotation in data["marker_annotations"].items():
                key = (dict_name, marker_id)
                lookup[key] = {
                    "model": model_name,
                    "annotation": annotation,
                    "base_marker_size": data["base_marker_size"],
                    "border_width_percent": data["border_width_percent"],
                    "vertices": data["vertices"],
                    "edges": data["edges"],
                    "aruco_dict_name": dict_name,
                }
                marker_ids_by_dict[dict_name].add(marker_id)
        return lookup, marker_ids_by_dict

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc}")
            return

        self.current_frame += 1
        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        successful_detections = []

        # Run detection for each dictionary; this allows mixed 4x4 / 5x5 models
        for dict_name, detector in self.detectors.items():
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue

            aruco.drawDetectedMarkers(display_frame, corners, ids)

            for idx, marker_id in enumerate(ids.flatten()):
                key = (dict_name, marker_id)
                # Ensure the marker belongs to this dictionary and lookup
                if key not in self.marker_lookup:
                    continue
                entry = self.marker_lookup[key]

                target_corners = corners[idx]

                base_marker_size = entry["base_marker_size"]
                border_width_percent = entry["border_width_percent"]
                border_width = base_marker_size * border_width_percent
                marker_size = base_marker_size - 2 * border_width

                tvec, rvec, _, is_confirmed = estimate_pose_with_kalman(
                    frame,
                    [target_corners],
                    [marker_id],
                    self.camera_matrix,
                    self.dist_coeffs,
                    marker_size,
                    self.kalman_filters,
                    self.marker_stabilities,
                    self.last_seen_frames,
                    self.current_frame,
                )

                if tvec is None or rvec is None:
                    continue

                annotation = entry["annotation"]
                vertices = entry["vertices"]
                edges = entry["edges"]

                obj_tvec, obj_rvec = estimate_object_pose_from_marker((tvec, rvec), annotation)

                successful_detections.append(
                    {
                        "marker_id": marker_id,
                        "model": entry["model"],
                        "distance": np.linalg.norm(tvec.flatten()),
                        "confirmed": is_confirmed,
                        "obj_pose": (obj_tvec, obj_rvec),
                        "vertices": vertices,
                        "edges": edges,
                    }
                )

        if successful_detections:
            # Select the best (closest) detection per model, preferring confirmed
            best_per_model = {}
            for det in successful_detections:
                model = det["model"]
                current = best_per_model.get(model)
                if current is None:
                    best_per_model[model] = det
                else:
                    # Prefer confirmed; tie-breaker: closer distance
                    if (not current["confirmed"] and det["confirmed"]) or (
                        current["confirmed"] == det["confirmed"] and det["distance"] < current["distance"]
                    ):
                        best_per_model[model] = det

            # Draw only the best per model
            for det in best_per_model.values():
                transformed_vertices = transform_mesh_to_camera_frame(det["vertices"], det["obj_pose"])
                projected_vertices = project_vertices_to_image(transformed_vertices, self.camera_matrix, self.dist_coeffs)
                draw_wireframe(display_frame, projected_vertices, det["edges"], color=(0, 255, 0), thickness=2)

            # Text overlay: list per-model best detections with object pose
            y0 = 30
            dy = 25
            line = 0
            confirmed = [d for d in best_per_model.values() if d["confirmed"]]
            display_list = confirmed if confirmed else list(best_per_model.values())
            for det in sorted(display_list, key=lambda d: d["distance"])[:5]:
                obj_tvec, obj_rvec = det["obj_pose"]
                rot_mtx, _ = cv2.Rodrigues(obj_rvec)
                rpy = rotation_matrix_to_euler(rot_mtx)
                text = (
                    f"{det['model']}: pos ({obj_tvec[0]:.3f},{obj_tvec[1]:.3f},{obj_tvec[2]:.3f}) "
                    f"rpy ({np.degrees(rpy[0]):.1f},{np.degrees(rpy[1]):.1f},{np.degrees(rpy[2]):.1f})"
                )
                cv2.putText(
                    display_frame,
                    text,
                    (10, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if det["confirmed"] else (0, 165, 255),
                    2,
                )
                line += 1
        else:
            cv2.putText(
                display_frame,
                "No markers detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Assembly Pose Detector (ROS2)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("Quit signal received.")
            self.should_shutdown = True


def main():
    parser = argparse.ArgumentParser(description="ROS2 Assembly Pose Detector")
    parser.add_argument(
        "--camera-topic",
        "-c",
        type=str,
        default="/camera/image_raw",
        help="ROS2 camera image topic",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=None,
        help="Path to data directory containing wireframe/aruco (default: repo_root/data)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIRECTORY_DEFAULT

    rclpy.init()
    node = AssemblyPoseDetectorROS2(camera_topic=args.camera_topic, data_dir=data_dir)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if getattr(node, "should_shutdown", False):
                break
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

