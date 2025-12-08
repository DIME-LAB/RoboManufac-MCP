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

# ArUco marker parameters
MARKER_SIZE = 0.021  # meters - physical size of your ArUco markers
ARUCO_DICTIONARY = aruco.DICT_4X4_50

# No scaling factor needed - wireframe should match actual object size

# Kalman filter parameters
MAX_MOVEMENT_THRESHOLD = 0.05  # meters - maximum allowed movement between frames
HOLD_REQUIRED_FRAMES = 2       # frames - required stable detections before confirmation
GHOST_TRACKING_FRAMES = 15     # frames - continue tracking when marker lost
BLEND_FACTOR = 0.99            # 0.0-1.0 - trust in measurements vs predictions

# Process noise parameters
PROCESS_NOISE_POSITION = 1e-4   # Process noise for position (x,y,z)
PROCESS_NOISE_QUATERNION = 1e-3 # Process noise for quaternion (qx,qy,qz,qw)
PROCESS_NOISE_VELOCITY = 1e-4   # Process noise for velocity (vx,vy,vz)
MEASUREMENT_NOISE_POSITION = 1e-4 # Measurement noise for position
MEASUREMENT_NOISE_QUATERNION = 1e-4 # Measurement noise for quaternion

# ROS2 topic names
CAMERA_IMAGE_TOPIC = "/camera/image_raw"  # Change this to your camera topic
OBJECT_POSE_TOPIC = "/object_pose"        # Published object pose topic
MARKER_POSE_TOPIC = "/marker_poses"       # Published marker poses topic
STATUS_TOPIC = "/object_pose_status"      # Published status topic
DEBUG_IMAGE_TOPIC = "/object_pose_debug"  # Published debug image topic (optional)

# Data directory path
DATA_DIRECTORY = "../../data"  # Path to your data directory with wireframe and aruco files

# Import the Kalman filter and utility functions from the original script
from object_pose_estimator_kalman import (
    QuaternionKalman, rvec_to_quat, quat_to_rvec, slerp_quat,
    load_wireframe_data, load_aruco_annotations, get_available_models,
    estimate_object_pose_from_marker, project_vertices_to_image, draw_wireframe, 
    estimate_pose_with_kalman, euler_to_rotation_matrix, rotation_matrix_to_euler
)

# Import functions without scaling
from object_pose_estimator_kalman import (
    transform_mesh_to_camera_frame as original_transform_mesh_to_camera_frame
)

def estimate_object_pose_from_marker_no_scaling(marker_pose, aruco_annotation):
    """
    Estimate the 6D pose of the object center from ArUco marker pose without scaling.
    
    Args:
        marker_pose: (tvec, rvec) - ArUco marker pose in camera frame
        aruco_annotation: ArUco annotation data containing marker-to-object transform
    
    Returns:
        object_pose: (object_tvec, object_rvec) - Object center pose in camera frame
    """
    # Get marker position and rotation
    marker_tvec, marker_rvec = marker_pose
    
    # Convert marker rotation vector to rotation matrix
    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    
    # Get the marker's pose relative to CAD center from annotation
    marker_relative_pose = aruco_annotation['pose_relative_to_cad_center']
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Get marker position relative to object center (in object frame) - NO SCALING
    marker_pos_in_object = np.array([
        marker_relative_pose['position']['x'],
        marker_relative_pose['position']['y'], 
        marker_relative_pose['position']['z']
    ])
    
    # Apply coordinate transformation only (no scaling)
    marker_pos_in_object = coord_transform @ marker_pos_in_object
    
    # Get marker orientation relative to object center
    marker_rot = marker_relative_pose['rotation']
    marker_rotation_in_object = euler_to_rotation_matrix(
        marker_rot['roll'], marker_rot['pitch'], marker_rot['yaw']
    )
    
    # Apply coordinate system transformation to the rotation matrix
    marker_rotation_in_object = coord_transform @ marker_rotation_in_object @ coord_transform.T
    
    # Calculate object center position in camera frame
    # The object center is at the origin of the object frame
    # We need to transform the origin (0,0,0) from object frame to camera frame
    object_origin_in_marker_frame = marker_rotation_in_object.T @ (-marker_pos_in_object)
    object_tvec = marker_tvec.flatten() + marker_rotation_matrix @ object_origin_in_marker_frame
    
    # Calculate object center orientation in camera frame
    # The object orientation is the marker orientation composed with the marker-to-object rotation
    object_rotation_matrix = marker_rotation_matrix @ marker_rotation_in_object.T
    
    # Convert back to rotation vector
    object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)
    
    return object_tvec, object_rvec

def transform_mesh_to_camera_frame_no_scaling(vertices, object_pose):
    """Transform mesh vertices from object center frame to camera frame without scaling"""
    object_tvec, object_rvec = object_pose
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(object_rvec)
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Transform vertices from object center frame to camera frame
    transformed_vertices = []
    for vertex in vertices:
        # Apply coordinate system transformation only (no scaling)
        vertex_transformed = coord_transform @ np.array(vertex)
        
        # Transform from object frame to camera frame
        vertex_cam = rotation_matrix @ vertex_transformed + object_tvec
        transformed_vertices.append(vertex_cam)
    
    return np.array(transformed_vertices)

class ObjectPoseEstimatorROS2(Node):
    """ROS2 Node for object pose estimation with Kalman filtering"""
    
    def __init__(self, model_name, camera_topic):
        super().__init__('object_pose_estimator_ros2')
        
        self.model_name = model_name
        self.data_dir = Path(DATA_DIRECTORY)
        
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
            aruco_annotations = load_aruco_annotations(aruco_annotations_file)
            self.get_logger().info(f"Loaded {len(aruco_annotations)} ArUco annotations")
        except Exception as e:
            self.get_logger().error(f"Error loading ArUco annotations: {e}")
            raise
        
        # Create marker annotations dictionary
        self.marker_annotations = {}
        for annotation in aruco_annotations:
            marker_id = annotation['aruco_id']
            self.marker_annotations[marker_id] = annotation
            self.get_logger().info(f"Loaded annotation for marker ID {marker_id}: size={annotation['size']}m, border={annotation['border_width']}m, face={annotation['face_type']}")
        
        self.target_ids = list(self.marker_annotations.keys())
        self.get_logger().info(f"Target marker IDs: {self.target_ids}")
    
    def setup_aruco_detector(self):
        """Setup ArUco marker detector"""
        dictionary = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
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
                    
                    # Calculate the inner pattern size for ArUco detection
                    # The border is INSIDE the total size, so we need to subtract it
                    total_marker_size = MARKER_SIZE
                    border_percentage = marker_annotation['border_width']
                    border_width = total_marker_size * border_percentage
                    marker_size = total_marker_size - 2 * border_width  # Inner pattern area
                    
                    # TODO: Update this when border convention changes to outside approach
                    # TODO: Use absolute border values in meters instead of percentages
                    
                    # Estimate pose with Kalman filtering
                    tvec, rvec, filtered_marker_id, is_confirmed = estimate_pose_with_kalman(
                        frame, [target_corners], [marker_id], self.camera_matrix, self.dist_coeffs,
                        marker_size, self.kalman_filters, self.marker_stabilities, 
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
                            'marker_size': marker_size
                        })
                
                # Find the best marker (closest confirmed detection)
                confirmed_detections = [d for d in successful_detections if d['is_confirmed']]
                if confirmed_detections:
                    best_detection = min(confirmed_detections, key=lambda x: x['distance'])
                    best_marker_pose = (best_detection['tvec'], best_detection['rvec'])
                    best_marker_id = best_detection['marker_id']
                    best_marker_annotation = best_detection['marker_annotation']
                    object_pose_detected = True
                
                # Draw debug information
                self.draw_debug_info(display_frame, successful_detections, confirmed_detections)
        
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
            rclpy.shutdown()
    
    def estimate_and_draw_object_pose(self, marker_pose, marker_annotation, display_frame):
        """Estimate object pose from marker pose and draw on display frame"""
        try:
            # Estimate object pose from marker pose (no scaling)
            object_tvec, object_rvec = estimate_object_pose_from_marker_no_scaling(marker_pose, marker_annotation)
            
            # Draw mesh overlay
            self.draw_mesh_overlay(display_frame, object_tvec, object_rvec)
            
            # Draw object center coordinate axes
            cv2.drawFrameAxes(display_frame, self.camera_matrix, self.dist_coeffs, 
                            object_rvec, object_tvec, 0.05)  # 5cm axes
            
            # Print pose info to console
            rotation_matrix, _ = cv2.Rodrigues(object_rvec)
            rpy = rotation_matrix_to_euler(rotation_matrix)
            
            print(f"\rObject Pose: Pos=({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f}) | "
                  f"RPY=({np.degrees(rpy[0]):.1f}°, {np.degrees(rpy[1]):.1f}°, {np.degrees(rpy[2]):.1f}°)", 
                  end="", flush=True)
            
            return (object_tvec, object_rvec)
            
        except Exception as e:
            self.get_logger().error(f"Error estimating object pose: {e}")
            return None
    
    
    def draw_debug_info(self, frame, successful_detections, confirmed_detections):
        """Draw debug information on the frame"""
        for i, detection in enumerate(successful_detections):
            marker_id = detection['marker_id']
            position = detection['position']
            distance = detection['distance']
            is_confirmed = detection['is_confirmed']
            marker_annotation = detection['marker_annotation']
            face_type = marker_annotation['face_type']
            
            # Position text based on marker index
            y_offset = 30 + i * 120
            
            # Color coding
            if is_confirmed:
                text_color = (0, 255, 0)  # Green for confirmed
                marker_status = " (CONFIRMED)"
            else:
                text_color = (255, 255, 0)  # Yellow for holding
                marker_status = " (HOLDING)"
            
            # Draw marker information
            cv2.putText(frame, f"Marker ID: {marker_id} ({face_type}){marker_status}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, f"Pos: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})", 
                       (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(frame, f"Distance: {distance:.3f}m", 
                       (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw coordinate axes
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                            detection['rvec'], detection['tvec'], detection['marker_size'])
    
    def draw_mesh_overlay(self, frame, object_tvec, object_rvec):
        """Draw mesh wireframe overlay on the frame"""
        try:
            # Transform mesh vertices to camera frame (no scaling)
            transformed_vertices = transform_mesh_to_camera_frame_no_scaling(
                self.vertices, (object_tvec, object_rvec)
            )
            
            # Project vertices to image coordinates
            projected_vertices = project_vertices_to_image(
                transformed_vertices, self.camera_matrix, self.dist_coeffs
            )
            
            # Draw wireframe
            if len(projected_vertices) > 0:
                draw_wireframe(frame, projected_vertices, self.edges, 
                             color=(0, 255, 0), thickness=2)
            
        except Exception as e:
            self.get_logger().warn(f"Error drawing mesh overlay: {e}")


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
    
    args = parser.parse_args()
    
    # Get available models
    data_dir = Path(DATA_DIRECTORY)
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
            camera_topic=args.camera_topic
        )
        
        print(f"Starting ROS2 Object Pose Estimator for model: {args.model}")
        print(f"Camera topic: {args.camera_topic}")
        print("Press 'q' in the OpenCV window to quit")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
