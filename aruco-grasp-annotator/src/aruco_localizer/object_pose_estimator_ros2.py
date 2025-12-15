#!/usr/bin/env python3
"""
ROS2-based Object Pose Estimator with Kalman Filtering

This script subscribes to a single ROS2 camera image topic and estimates 6D object pose from ArUco markers
using Kalman filtering for smooth tracking.

Usage:
    python3 object_pose_estimator_ros2.py --model fork_orange_scaled70 --camera-topic /camera/image_raw

Dependencies:
    - rclpy
    - sensor_msgs
    - geometry_msgs
    - std_msgs
    - cv_bridge
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Camera parameters (used if camera info not available)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV = 82.4  # degrees
CAMERA_VFOV = 52.4  # degrees

# Camera orientation (world ← camera). This assumes the camera is facing the marker
# from above (top-down) so we can lift the camera-frame pose into a nominal world frame.
CAMERA_QUAT_WORLD = np.array([0.0, 1.0, 0.0, 0.0])  # [x, y, z, w]

# ArUco marker parameters
MARKER_SIZE = 0.021  # meters - physical size of your ArUco markers
ARUCO_DICTIONARY = aruco.DICT_4X4_50

# No scaling factor needed - wireframe should match actual object size

# ROS2 topic names
CAMERA_IMAGE_TOPIC = "/camera/image_raw"  # Change this to your camera topic
OBJECT_POSE_TOPIC = "/object_pose"        # Published object pose topic
MARKER_POSE_TOPIC = "/marker_poses"       # Published marker poses topic
STATUS_TOPIC = "/object_pose_status"      # Published status topic
DEBUG_IMAGE_TOPIC = "/object_pose_debug"  # Published debug image topic (optional)

# Data directory path (default: repo_root/data)
DATA_DIRECTORY_DEFAULT = Path(__file__).resolve().parent.parent.parent / "data"

from core.kalman_filter import (
    QuaternionKalman,
    MAX_MOVEMENT_THRESHOLD,
    HOLD_REQUIRED_FRAMES,
    GHOST_TRACKING_FRAMES,
    BLEND_FACTOR,
)
from core.pose_math import (
    rvec_to_quat,
    quat_to_rvec,
    slerp_quat,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    estimate_object_pose_from_marker,
    pose_to_world,
)
from core.model_io import load_wireframe_data, load_aruco_annotations, get_available_models
from core.mesh_ops import (
    transform_mesh_to_camera_frame,
    project_vertices_to_image,
    draw_wireframe,
)
from object_pose_estimator_camera import estimate_pose_with_kalman

def estimate_object_pose_from_marker_no_scaling(marker_pose, aruco_annotation):
    """
    Estimate the 6D pose of the object center from ArUco marker pose.
    Uses homogeneous transformation matrices to compute position and orientation together.
    This function is an alias for the updated estimate_object_pose_from_marker function.
    
    Args:
        marker_pose: (tvec, rvec) - ArUco marker pose in camera frame
        aruco_annotation: ArUco annotation data containing marker-to-object transform
    
    Returns:
        object_pose: (object_tvec, object_rvec) - Object center pose in camera frame
    """
    # Use the updated implementation from object_pose_estimator_kalman
    return estimate_object_pose_from_marker(marker_pose, aruco_annotation)

def transform_mesh_to_camera_frame_no_scaling(vertices, object_pose):
    """Transform mesh vertices from object center frame to camera frame.
    This function is an alias for the updated transform_mesh_to_camera_frame function.
    """
    # Use the updated implementation from object_pose_estimator_kalman
    from object_pose_estimator_kalman import transform_mesh_to_camera_frame
    return transform_mesh_to_camera_frame(vertices, object_pose)

class ObjectPoseEstimatorROS2(Node):
    """ROS2 Node for object pose estimation with Kalman filtering"""
    
    def __init__(self, model_name, camera_topic, data_dir):
        super().__init__('object_pose_estimator_ros2')
        
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        
        # Initialize CV bridge for ROS2 image conversion
        self.bridge = CvBridge()
        
        # Load model data
        self.load_model_data()
        
        # Initialize camera parameters using defaults
        self.setup_camera_parameters()
        
        # Initialize ArUco detector
        self.setup_aruco_detector()
        
        # Initialize Kalman filters and tracking
        self.kalman_filters = {}
        self.marker_stabilities = {}
        self.last_seen_frames = {}
        self.current_frame = 0
        
        # Setup ROS2 publishers and subscribers
        self.setup_ros2_interface(camera_topic)
        
        # Initialize pose tracking
        self.last_object_pose = None
        self.pose_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]

        # Shutdown flag to allow graceful exit from main loop
        self.should_shutdown = False
        
        self.get_logger().info(f"Object Pose Estimator ROS2 initialized for model: {model_name}")
        self.get_logger().info(f"Camera topic: {camera_topic}")
    
    def setup_camera_parameters(self):
        """Setup camera parameters using default values"""
        # Calculate camera matrix from field of view
        fx = CAMERA_WIDTH / (2 * np.tan(np.deg2rad(CAMERA_HFOV / 2)))
        fy = CAMERA_HEIGHT / (2 * np.tan(np.deg2rad(CAMERA_VFOV / 2)))
        self.camera_matrix = np.array([[fx, 0, CAMERA_WIDTH / 2],
                                      [0, fy, CAMERA_HEIGHT / 2],
                                      [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        self.get_logger().info(f"Camera matrix: fx={fx:.1f}, fy={fy:.1f}")
        self.get_logger().info(f"FOV settings: HFOV={CAMERA_HFOV}°, VFOV={CAMERA_VFOV}°")
        self.get_logger().info(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        
    def load_model_data(self):
        """Load wireframe and ArUco annotation data for the selected model"""
        # Construct file paths
        wireframe_file = self.data_dir / "wireframe" / f"{self.model_name}_wireframe.json"
        aruco_annotations_file = self.data_dir / "aruco" / f"{self.model_name}_aruco.json"
        
        # Load wireframe data
        try:
            self.vertices, self.edges = load_wireframe_data(wireframe_file)
            self.get_logger().info(f"Loaded wireframe: {len(self.vertices)} vertices, {len(self.edges)} edges")
        except Exception as e:
            self.get_logger().error(f"Error loading wireframe data: {e}")
            raise
        
        # Load ArUco annotations
        try:
            (
                aruco_annotations,
                base_marker_size,
                border_width_percent,
                aruco_dict_name,
            ) = load_aruco_annotations(aruco_annotations_file)
            self.get_logger().info(f"Loaded {len(aruco_annotations)} ArUco annotations")
            self.get_logger().info(f"Marker size: {base_marker_size}m, border width: {border_width_percent}")
            self.get_logger().info(f"ArUco dictionary: {aruco_dict_name}")
        except Exception as e:
            self.get_logger().error(f"Error loading ArUco annotations: {e}")
            raise
        
        # Store marker size info for later use
        self.base_marker_size = base_marker_size
        self.border_width_percent = border_width_percent
        self.aruco_dict_name = aruco_dict_name
        
        # Create marker annotations dictionary
        self.marker_annotations = {}
        for annotation in aruco_annotations:
            marker_id = annotation['aruco_id']
            self.marker_annotations[marker_id] = annotation
            self.get_logger().info(f"Loaded annotation for marker ID {marker_id}: face={annotation['face_type']}")
        
        self.target_ids = list(self.marker_annotations.keys())
        self.get_logger().info(f"Target marker IDs: {self.target_ids}")
    
    def setup_aruco_detector(self):
        """Setup ArUco marker detector"""
        try:
            dictionary_id = getattr(aruco, getattr(self, "aruco_dict_name", "DICT_4X4_50"))
        except AttributeError:
            self.get_logger().warning(
                f"Unknown dictionary '{getattr(self, 'aruco_dict_name', None)}', falling back to DICT_4X4_50"
            )
            dictionary_id = ARUCO_DICTIONARY
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary, parameters)
    
    def setup_ros2_interface(self, camera_topic):
        """Setup ROS2 subscribers"""
        # Only subscribe to camera image topic
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
    
    def image_callback(self, msg):
        """Main callback for processing camera images"""
        try:
            # Convert ROS2 image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        self.current_frame += 1
        
        # Process the frame and show in OpenCV window
        self.process_frame(frame)
    
    def process_frame(self, frame):
        """Process a single frame for object pose estimation and show in OpenCV window"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Process detections
        object_pose_detected = False
        best_marker_pose = None
        best_marker_id = None
        
        if ids is not None:
            # Draw all detected markers
            aruco.drawDetectedMarkers(display_frame, corners, ids)
            
            # Check for target markers
            detected_targets = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.target_ids:
                    detected_targets.append((i, marker_id))
            
            if detected_targets:
                # Process all target markers
                successful_detections = []
                for target_idx, marker_id in detected_targets:
                    target_corners = corners[target_idx]
                    marker_annotation = self.marker_annotations[marker_id]
                    
                    # Calculate actual ArUco pattern size (border is INSIDE the total marker size)
                    # Match external aruco_camera_localizer: TOTAL_MARKER_SIZE = MARKER_SIZE - 2 * BORDER_WIDTH
                    border_width = self.base_marker_size * self.border_width_percent
                    marker_size_with_border = self.base_marker_size - 2 * border_width  # Actual ArUco pattern size
                    
                    # Estimate pose with Kalman filtering
                    tvec, rvec, filtered_marker_id, is_confirmed = estimate_pose_with_kalman(
                        frame, [target_corners], [marker_id], self.camera_matrix, self.dist_coeffs,
                        marker_size_with_border, self.kalman_filters, self.marker_stabilities, 
                        self.last_seen_frames, self.current_frame
                    )
                    
                    if tvec is not None and rvec is not None:
                        position = tvec.flatten()
                        distance = np.linalg.norm(position)
                        successful_detections.append({
                            'marker_id': marker_id,
                            'marker_annotation': marker_annotation,
                            'rvec': rvec,
                            'tvec': tvec,
                            'position': position,
                            'distance': distance,
                            'is_confirmed': is_confirmed,
                            'marker_size': marker_size_with_border
                        })
                
                # Find the best marker (closest confirmed detection)
                confirmed_detections = [d for d in successful_detections if d['is_confirmed']]
                if confirmed_detections:
                    best_detection = min(confirmed_detections, key=lambda x: x['distance'])
                    best_marker_pose = (best_detection['tvec'], best_detection['rvec'])
                    best_marker_id = best_detection['marker_id']
                    best_marker_annotation = best_detection['marker_annotation']
                    object_pose_detected = True
                
        # Estimate and draw object pose if detected
        if object_pose_detected and best_marker_pose is not None:
            self.estimate_and_draw_object_pose(
                best_marker_pose, best_marker_annotation, display_frame
            )
        else:
            # Show no detection message
            cv2.putText(display_frame, "No object detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show controls
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow("ROS2 Object Pose Estimator", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested by user")
            self.should_shutdown = True
    
    def estimate_and_draw_object_pose(self, marker_pose, marker_annotation, display_frame):
        """Estimate object pose from marker pose and draw on display frame"""
        try:
            # Estimate object pose from marker pose (no scaling)
            object_tvec, object_rvec = estimate_object_pose_from_marker_no_scaling(marker_pose, marker_annotation)
            
            # Draw mesh overlay using the same transform as aruco_mesh_overlay
            self.draw_mesh_overlay(display_frame, marker_pose, marker_annotation)
            
            # Draw object center coordinate axes
            cv2.drawFrameAxes(display_frame, self.camera_matrix, self.dist_coeffs, 
                            object_rvec, object_tvec, 0.05)  # 5cm axes
            
            # Print pose info to console
            rotation_matrix, _ = cv2.Rodrigues(object_rvec)
            rpy = rotation_matrix_to_euler(rotation_matrix)
            quat_cam = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w] in camera frame

            # Compute world-frame pose using shared helper.
            object_tvec_world, world_rvec, quat_world, world_rpy = pose_to_world(
                object_tvec, object_rvec, CAMERA_QUAT_WORLD
            )
            
            # Console output: world-frame only. Camera-frame lines retained below as comments.
            # print(
            #     f"\rObject Pose (cam): Pos=({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f}) | "
            #     f"RPY_cam=({np.degrees(rpy[0]):.1f}°, {np.degrees(rpy[1]):.1f}°, {np.degrees(rpy[2]):.1f}°) | "
            #     f"Quat_cam=({quat_cam[0]:.3f}, {quat_cam[1]:.3f}, {quat_cam[2]:.3f}, {quat_cam[3]:.3f})",
            #     end="", flush=True
            # )
            print(
                f"\rObject Pose (world): Pos=({object_tvec_world[0]:.3f}, {object_tvec_world[1]:.3f}, {object_tvec_world[2]:.3f}) | "
                f"RPY_world=({np.degrees(world_rpy[0]):.1f}°, {np.degrees(world_rpy[1]):.1f}°, {np.degrees(world_rpy[2]):.1f}°) | "
                f"Quat_world=({quat_world[0]:.3f}, {quat_world[1]:.3f}, {quat_world[2]:.3f}, {quat_world[3]:.3f})",
                end="", flush=True
            )

            # Overlay object pose text on the image
            y0 = 40
            face = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 255)
            marker_id = marker_annotation.get('aruco_id', 'N/A')
            face_type = marker_annotation.get('face_type', 'N/A')
            cv2.putText(display_frame, f"Marker {marker_id} ({face_type}) | Dict: {self.aruco_dict_name}", (10, y0), face, 0.7, (0, 255, 0), 2)
            # Overlay: world-frame only. Camera-frame overlay kept commented for reference.
            # cv2.putText(display_frame, f"Pos_cam: ({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f})",
            #             (10, y0 + 30), face, 0.6, color, 2)
            # cv2.putText(display_frame, f"RPY_cam: ({np.degrees(rpy[0]):.1f}, {np.degrees(rpy[1]):.1f}, {np.degrees(rpy[2]):.1f}) deg",
            #             (10, y0 + 60), face, 0.6, color, 2)
            # cv2.putText(display_frame, f"Quat_cam: ({quat_cam[0]:.3f}, {quat_cam[1]:.3f}, {quat_cam[2]:.3f}, {quat_cam[3]:.3f})",
            #             (10, y0 + 90), face, 0.6, color, 2)
            cv2.putText(display_frame, f"Pos_world: ({object_tvec_world[0]:.3f}, {object_tvec_world[1]:.3f}, {object_tvec_world[2]:.3f})",
                        (10, y0 + 30), face, 0.6, color, 2)
            cv2.putText(display_frame, f"RPY_world: ({np.degrees(world_rpy[0]):.1f}, {np.degrees(world_rpy[1]):.1f}, {np.degrees(world_rpy[2]):.1f}) deg",
                        (10, y0 + 60), face, 0.6, color, 2)
            cv2.putText(display_frame, f"Quat_world: ({quat_world[0]:.3f}, {quat_world[1]:.3f}, {quat_world[2]:.3f}, {quat_world[3]:.3f})",
                        (10, y0 + 90), face, 0.6, color, 2)
            
            return (object_tvec, object_rvec)
            
        except Exception as e:
            self.get_logger().error(f"Error estimating object pose: {e}")
            return None
    
    
    def draw_debug_info(self, frame, successful_detections, confirmed_detections):
        """(Debug overlay removed)"""
        return
    
    def draw_mesh_overlay(self, frame, marker_pose, marker_annotation):
        """Draw mesh wireframe overlay on the frame using shared mesh ops."""
        try:
            object_pose = estimate_object_pose_from_marker(marker_pose, marker_annotation)
            transformed_vertices = transform_mesh_to_camera_frame(self.vertices, object_pose)
            projected_vertices = project_vertices_to_image(
                transformed_vertices, self.camera_matrix, self.dist_coeffs
            )
            if len(projected_vertices) > 0:
                draw_wireframe(frame, projected_vertices, self.edges, color=(0, 255, 0), thickness=2)
        except Exception as e:
            self.get_logger().warning(f"Error drawing mesh overlay: {e}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="ROS2 Object Pose Estimator with Kalman Filtering"
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Model name to use (e.g., 'fork_orange_scaled70')")
    parser.add_argument("--camera-topic", "-c", type=str, default=CAMERA_IMAGE_TOPIC,
                       help="ROS2 camera image topic")
    parser.add_argument("--list-models", "-l", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                       help="Path to data directory containing wireframe/aruco (default: repo_root/data)")
    
    args = parser.parse_args()
    
    # Get available models
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIRECTORY_DEFAULT
    available_models = get_available_models(data_dir)
    
    if args.list_models:
        print(f"Available models in {data_dir}:")
        if available_models:
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
        else:
            print("  No models found!")
        return
    
    if not available_models:
        print(f"No models found in data directory: {data_dir}")
        return
    
    if args.model not in available_models:
        print(f"Model '{args.model}' not found in available models: {available_models}")
        return
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create and run the node
        node = ObjectPoseEstimatorROS2(
            model_name=args.model,
            camera_topic=args.camera_topic,
            data_dir=data_dir
        )
        
        print(f"Starting ROS2 Object Pose Estimator for model: {args.model}")
        print(f"Camera topic: {args.camera_topic}")
        print("Press 'q' in the OpenCV window to quit")
        
        # Spin with periodic check for shutdown flag to handle 'q' key presses
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass  # Ignore if context already shutdown


if __name__ == "__main__":
    main()
