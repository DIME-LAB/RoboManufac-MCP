#!/usr/bin/env python3
"""
Grasp Points Publisher
Reads object poses from topic and publishes grasp points to topic.
Uses grasp points data from JSON files and transforms them using object poses.

Supports two modes:
- sim: Uses /objects_poses_sim and /grasp_points_sim topics
- real: Uses /objects_poses_real and /grasp_points_real topics

Usage:
    python3 grasp_points_publisher.py [--mode sim|real]
"""

import sys

# Check Python version - ROS2 Humble requires Python 3.10
if sys.version_info[:2] != (3, 10):
    print("Error: ROS2 Humble requires Python 3.10")
    print(f"Current Python version: {sys.version}")
    print("\nSolutions:")
    print("1. Deactivate conda environment: conda deactivate")
    print("2. Use python3.10 directly: python3.10 src/grasp_candidates/grasp_points_publisher.py")
    print("3. Source ROS2 setup.bash which should set the correct Python")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import grasp points message type
try:
    from max_camera_msgs.msg import GraspPointArray, GraspPoint
except ImportError:
    print("Error: max_camera_msgs not found. Please install the max_camera_msgs package.")
    print("This script requires max_camera_msgs.msg.GraspPointArray and GraspPoint")
    raise


class GraspPointsPublisher(Node):
    """ROS2 node that publishes grasp points based on object poses"""
    
    # Object symmetry configuration
    # Maps object names (topic names, without _scaled70) to symmetry configuration
    # Each entry is a dict with:
    #   - 'symmetry_planes': List of planes the object is symmetric about (mirror/reflection symmetry)
    #     - For 1 plane: ['xy'], ['yz'], or ['xz']
    #     - For 2 planes: ['xy', 'yz'], ['xy', 'xz'], or ['yz', 'xz']
    #     - For 3 planes: ['xy', 'yz', 'xz'] (all three planes - full symmetry)
    # 
    # For a gripper approaching from above (Z direction):
    # - 'xy' plane symmetry: Object is symmetric about XY plane (the approach plane).
    #   X and Y directions in the approach plane are equivalent (full freedom in XY plane).
    # - 'yz' plane symmetry: Object is symmetric about YZ plane (perpendicular to X).
    #   The X direction is constrained - X-axis should align with X direction (perpendicular to YZ plane).
    # - 'xz' plane symmetry: Object is symmetric about XZ plane (perpendicular to Y).
    #   The Y direction is constrained, but X can be arbitrary in the XY plane.
    # 
    # The approach_vector from JSON is used as a reference, but plane symmetries constrain
    # which orientations are valid for the gripper.
    # Objects not listed default to {'symmetry_planes': ['xy', 'yz', 'xz']} (backward compatible)
    OBJECT_SYMMETRY_AXES = {
        # Objects with 2 planes of symmetry
        "fork_orange": {'symmetry_planes': ['xy', 'yz']},  # Symmetric about XY and YZ planes
        "fork_yellow": {'symmetry_planes': ['xy', 'yz']},  # Symmetric about XY and YZ planes
        
        # Objects with 3 planes of symmetry (symmetric about all three planes)
        "line_brown": {'symmetry_planes': ['xy', 'yz', 'xz']},  # Symmetric about XY, YZ, and XZ planes
        "line_red": {'symmetry_planes': ['xy', 'yz', 'xz']},    # Symmetric about XY, YZ, and XZ planes
    }
    
    def __init__(self, objects_poses_topic=None, 
                 grasp_points_topic=None,
                 data_dir=None,
                 mode='sim'):
        super().__init__('grasp_points_publisher')
        
        self.mode = mode  # 'sim' or 'real'
        
        # Set default topics based on mode if not provided
        if objects_poses_topic is None:
            if self.mode == 'sim':
                self.objects_poses_topic = "/objects_poses_sim"
            else:
                self.objects_poses_topic = "/objects_poses_real"
        else:
            self.objects_poses_topic = objects_poses_topic
        
        if grasp_points_topic is None:
            if self.mode == 'sim':
                self.grasp_points_topic = "/grasp_points_sim"
            else:
                self.grasp_points_topic = "/grasp_points_real"
        else:
            self.grasp_points_topic = grasp_points_topic
        
        # Set up data directory
        if data_dir is None:
            # Default to data/grasp relative to this file
            script_dir = Path(__file__).parent.parent.parent
            self.data_dir = Path("/home/aaugus11/Projects/aruco-grasp-annotator/data/grasp")
        else:
            self.data_dir = Path(data_dir)
        
        # Load all grasp points JSON files
        self.grasp_data: Dict[str, dict] = {}
        # Map from topic object names (e.g., "fork_yellow") to JSON object names (e.g., "fork_yellow_scaled70")
        self.object_name_map: Dict[str, str] = {}
        self.load_grasp_data()
        
        # Store latest object poses
        self.object_poses: Dict[str, dict] = {}
        
        # Create subscription to object poses
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.pose_sub = self.create_subscription(
            TFMessage,
            self.objects_poses_topic,
            self.objects_poses_callback,
            qos_profile
        )
        
        # Create publisher for grasp points
        self.grasp_pub = self.create_publisher(
            GraspPointArray,
            self.grasp_points_topic,
            qos_profile
        )
        
        # Timer to publish grasp points periodically
        self.publish_timer = self.create_timer(0.1, self.publish_grasp_points)  # 10 Hz
        
        self.get_logger().info(f"🤖 Grasp Points Publisher started")
        self.get_logger().info(f"📥 Subscribing to: {self.objects_poses_topic}")
        self.get_logger().info(f"📤 Publishing to: {self.grasp_points_topic}")
        self.get_logger().info(f"🔧 Mode: {self.mode.upper()}")
        self.get_logger().info(f"📁 Data directory: {self.data_dir}")
        self.get_logger().info(f"📦 Loaded grasp data for {len(self.grasp_data)} objects")
    
    def load_grasp_data(self):
        """Load all grasp points JSON files from data directory"""
        if not self.data_dir.exists():
            self.get_logger().warn(f"Data directory does not exist: {self.data_dir}")
            return
        
        # Find all grasp points JSON files
        pattern = "*_grasp_points_all_markers.json"
        for grasp_file in self.data_dir.glob(pattern):
            try:
                with open(grasp_file, 'r') as f:
                    data = json.load(f)
                    object_name_json = data.get('object_name')
                    if object_name_json:
                        # Store with full name
                        self.grasp_data[object_name_json] = data
                        
                        # Create mapping: topic name (without _scaled70) -> JSON name (with _scaled70)
                        # Also try direct match and without _scaled70
                        topic_name = object_name_json.replace('_scaled70', '')
                        self.object_name_map[topic_name] = object_name_json
                        # Also allow direct match
                        self.object_name_map[object_name_json] = object_name_json
                        
                        self.get_logger().info(f"  ✓ Loaded: {object_name_json} ({data.get('total_grasp_points', 0)} grasp points)")
                        self.get_logger().debug(f"    Mapped topic name '{topic_name}' -> JSON name '{object_name_json}'")
            except Exception as e:
                self.get_logger().error(f"Error loading {grasp_file}: {e}")
    
    def objects_poses_callback(self, msg: TFMessage):
        """Handle incoming object poses from TFMessage"""
        for transform in msg.transforms:
            object_name = transform.child_frame_id
            
            # Extract pose information
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            # Store the pose
            self.object_poses[object_name] = {
                'translation': np.array([trans.x, trans.y, trans.z]),
                'quaternion': np.array([rot.x, rot.y, rot.z, rot.w]),
                'header': transform.header
            }
    
    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees
        
        Handles gimbal lock cases (when pitch is near ±90°) gracefully.
        Scipy automatically sets the third angle to zero when gimbal lock is detected.
        """
        # Use scipy for robust conversion
        # Note: gimbal lock warnings are expected when pitch ≈ ±90°
        # Scipy handles this by setting roll to 0, which is the standard approach
        r = R.from_quat([x, y, z, w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return roll, pitch, yaw
    
    def is_axis_symmetric(self, axis, symmetry_axes_list):
        """Check if a given axis is in the symmetry axes list"""
        return axis.lower() in [a.lower() for a in symmetry_axes_list]
    
    def has_full_symmetry(self, symmetry_axes_list):
        """Check if object has full symmetry (all three axes)"""
        return len(symmetry_axes_list) == 3 and set(symmetry_axes_list) == {'x', 'y', 'z'}
    
    def planes_to_constrained_axes(self, symmetry_planes):
        """
        Determine which axes in the approach plane (XY) are constrained by plane symmetries.
        
        For a gripper approaching from above (Z direction):
        - The approach plane is XY (perpendicular to Z)
        - X-axis is the gripper opening direction (in XY plane)
        - Y-axis is also in the approach plane
        
        Plane symmetry interpretation:
        - 'xy' plane symmetry: Object is symmetric about XY plane (the approach plane).
          This means X and Y directions in the approach plane are equivalent.
        - 'yz' plane symmetry: Object is symmetric about YZ plane (perpendicular to X).
          This means the X direction can be flipped, so X-axis orientation is constrained.
        - 'xz' plane symmetry: Object is symmetric about XZ plane (perpendicular to Y).
          This means the Y direction can be flipped, so Y-axis orientation is constrained.
        
        For determining X-axis (gripper opening direction):
        - If 'yz' plane is symmetric: X direction is constrained (should align with YZ plane normal)
        - If 'xz' plane is symmetric: Y direction is constrained (X can be arbitrary in XY plane)
        - If 'xy' plane is symmetric: Both X and Y are equivalent (full freedom in XY plane)
        
        Args:
            symmetry_planes: List of plane names like ['xy', 'yz'] or ['xy', 'yz', 'xz']
        
        Returns:
            Dict with:
            - 'x_constrained': bool - whether X-axis direction is constrained
            - 'y_constrained': bool - whether Y-axis direction is constrained
            - 'xy_free': bool - whether X and Y are free (XY plane symmetry)
        """
        x_constrained = False
        y_constrained = False
        xy_free = False
        
        for plane in symmetry_planes:
            plane_lower = plane.lower()
            if plane_lower == 'xy':
                # XY plane symmetry: approach plane has symmetry, X and Y are equivalent
                xy_free = True
            elif plane_lower == 'yz':
                # YZ plane symmetry: X direction is constrained (perpendicular to YZ plane)
                x_constrained = True
            elif plane_lower == 'xz':
                # XZ plane symmetry: Y direction is constrained (perpendicular to XZ plane)
                y_constrained = True
        
        return {
            'x_constrained': x_constrained,
            'y_constrained': y_constrained,
            'xy_free': xy_free
        }
    
    def transform_grasp_point(self, grasp_point_local, object_pose, symmetry_config=None):
        """
        Transform grasp point from CAD center frame to base frame using object pose.
        Implements quaternion computation logic based on approach vector and object symmetry.

        Args:
            grasp_point_local: Dict with position (x, y, z) relative to CAD center
            object_pose: Dict with translation and quaternion of object in base frame
            symmetry_config: Dict with symmetry configuration:
                          - 'axes': Number of axes of symmetry (1, 2, or 3)
                          - 'axis': Which axis object is symmetric about ('x', 'y', or 'z')
                          - 3: X-axis can be arbitrary (full rotational symmetry)
                          - 2: Use approach_vector from JSON if available, otherwise arbitrary
                          - 1: Must use approach_vector from JSON (no symmetry)
                          - None: Default to {'axes': 3, 'axis': 'z'} (backward compatible)
        
        Returns:
            Transformed position and orientation in base frame
        """
        # Local position (relative to CAD center)
        pos_local = np.array([
            grasp_point_local['position']['x'],
            grasp_point_local['position']['y'],
            grasp_point_local['position']['z']
        ])
        
        # Object pose in base frame
        obj_translation = object_pose['translation']
        obj_quaternion = object_pose['quaternion']
        
        # Create rotation matrix from quaternion
        r_object_world = R.from_quat(obj_quaternion)
        rot_matrix = r_object_world.as_matrix()
        
        # Coordinate system transformation matrix (same as wireframe)
        coord_transform = np.array([
            [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
            [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
            [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
        ])
        
        # Apply coordinate system transformation (same as wireframe)
        grasp_pos_transformed = coord_transform @ pos_local
        
        # Transform to world frame
        pos_base = obj_translation + rot_matrix @ grasp_pos_transformed
        
        # Quaternion computation logic with symmetry handling
        # Default to full symmetry (all three planes) for backward compatibility
        if symmetry_config is None:
            symmetry_config = {'symmetry_planes': ['xy', 'yz', 'xz']}
        
        # Get plane symmetry constraints
        symmetry_planes = symmetry_config.get('symmetry_planes', ['xy', 'yz', 'xz'])
        symmetry_constraints = self.planes_to_constrained_axes(symmetry_planes)
        
        # For backward compatibility, also compute symmetry_axes_list
        # This is used by has_full_symmetry() and other checks
        # Full symmetry means all three planes are symmetric
        has_full_symmetry = len(symmetry_planes) == 3 and set([p.lower() for p in symmetry_planes]) == {'xy', 'yz', 'xz'}
        symmetry_axes_list = ['x', 'y', 'z'] if has_full_symmetry else []
        
        # Check if grasp point has approach_vector
        if 'approach_vector' in grasp_point_local:
            # Define upward direction in world frame (Z-up)
            upward_world = np.array([0.0, 0.0, 1.0])
            
            # Transform upward direction from world frame to object frame
            # This ensures the approach vector always points upward in world frame
            # regardless of object orientation
            approach_vec_transformed = rot_matrix.T @ upward_world
            
            # Generate full orientation from approach vector (in object frame)
            # The approach vector becomes the Z-axis of the gripper frame
            approach_norm = np.linalg.norm(approach_vec_transformed)
            if approach_norm < 1e-6:  # Avoid division by zero
                # Use default orientation if approach vector is invalid
                grasp_quat_object = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
            else:
                z_axis = approach_vec_transformed / approach_norm
                
                # Compute X-axis based on symmetry axes list
                # The approach_vector from JSON is relative to one face, so we transform it
                # to world frame accounting for the object's current orientation.
                # For axes in symmetry_axes_list, we can rotate the approach_vector around those axes.
                approach_vec_json = grasp_point_local.get('approach_vector')
                if approach_vec_json is not None:
                    # Get approach_vector from JSON (in object local frame)
                    approach_vec_local = np.array([
                        approach_vec_json.get('x', 0.0),
                        approach_vec_json.get('y', 0.0),
                        approach_vec_json.get('z', 0.0)
                    ])
                    # Apply coordinate system transformation (CAD to OpenCV)
                    approach_vec_transformed_local = coord_transform @ approach_vec_local
                    approach_vec_norm = np.linalg.norm(approach_vec_transformed_local)
                    
                    if approach_vec_norm > 1e-6:
                        # Transform approach_vector from object frame to world frame
                        # This accounts for the object's current orientation
                        approach_vec_world = rot_matrix @ (approach_vec_transformed_local / approach_vec_norm)
                        
                        # Transform back to object frame for projection
                        approach_vec_object = rot_matrix.T @ approach_vec_world
                        
                        # Project approach_vector onto plane perpendicular to z_axis (in object frame)
                        # The approach_vector can be rotated around symmetric axes
                        x_axis_candidate = approach_vec_object
                        # Remove component along z_axis
                        x_axis = x_axis_candidate - np.dot(x_axis_candidate, z_axis) * z_axis
                        x_axis_norm = np.linalg.norm(x_axis)
                        
                        if x_axis_norm > 1e-6:
                            x_axis = x_axis / x_axis_norm
                            
                            # Adjust X-axis based on plane symmetry constraints
                            # The approach_vector gives us a reference direction, but plane symmetries
                            # constrain which orientations are valid
                            
                            if symmetry_constraints['x_constrained']:
                                # YZ plane symmetry: X direction is constrained
                                # Align X-axis with X direction (perpendicular to YZ plane)
                                x_axis_candidate = np.array([1.0, 0.0, 0.0])
                                # Project onto plane perpendicular to z_axis
                                x_axis = x_axis_candidate - np.dot(x_axis_candidate, z_axis) * z_axis
                                x_axis_norm = np.linalg.norm(x_axis)
                                if x_axis_norm > 1e-6:
                                    x_axis = x_axis / x_axis_norm
                                # If X is parallel to Z, keep approach_vector-based x_axis
                            elif symmetry_constraints['xy_free']:
                                # XY plane symmetry: X and Y are equivalent, use approach_vector as-is
                                # (full freedom in XY plane)
                                pass  # Keep x_axis from approach_vector
                            elif symmetry_constraints['y_constrained']:
                                # XZ plane symmetry: Y direction is constrained, X can be arbitrary
                                # Use approach_vector as-is (X is free)
                                pass  # Keep x_axis from approach_vector
                            else:
                                # No specific constraints, use approach_vector as-is
                                pass  # Keep x_axis from approach_vector
                        else:
                            # Fallback: approach_vector is parallel to z_axis
                            # This is common for top-down grasps where approach_vector points upward
                            # Use plane symmetry constraints to determine X-axis
                            if symmetry_constraints['xy_free'] or has_full_symmetry:
                                # Full symmetry or XY plane symmetry: use arbitrary direction (no warning needed)
                                if abs(z_axis[0]) < 0.9:
                                    x_axis = np.array([1.0, 0.0, 0.0])
                                else:
                                    x_axis = np.array([0.0, 1.0, 0.0])
                                x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                                x_axis = x_axis / np.linalg.norm(x_axis)
                            elif symmetry_constraints['x_constrained']:
                                # YZ plane symmetry: align X-axis with X direction
                                x_axis_candidate = np.array([1.0, 0.0, 0.0])
                                x_axis = x_axis_candidate - np.dot(x_axis_candidate, z_axis) * z_axis
                                x_axis_norm = np.linalg.norm(x_axis)
                                if x_axis_norm > 1e-6:
                                    x_axis = x_axis / x_axis_norm
                                else:
                                    # X parallel to Z, use arbitrary
                                    if abs(z_axis[0]) < 0.9:
                                        x_axis = np.array([1.0, 0.0, 0.0])
                                    else:
                                        x_axis = np.array([0.0, 1.0, 0.0])
                                    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                                    x_axis = x_axis / np.linalg.norm(x_axis)
                            else:
                                # Other cases: use arbitrary direction
                                if abs(z_axis[0]) < 0.9:
                                    x_axis = np.array([1.0, 0.0, 0.0])
                                else:
                                    x_axis = np.array([0.0, 1.0, 0.0])
                                x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                                x_axis = x_axis / np.linalg.norm(x_axis)
                    else:
                        # Fallback: approach_vector is invalid
                        if symmetry_constraints['xy_free'] or has_full_symmetry:
                            # Full symmetry or XY plane symmetry: use arbitrary direction
                            if abs(z_axis[0]) < 0.9:
                                x_axis = np.array([1.0, 0.0, 0.0])
                            else:
                                x_axis = np.array([0.0, 1.0, 0.0])
                            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                            x_axis = x_axis / np.linalg.norm(x_axis)
                        elif symmetry_constraints['x_constrained']:
                            # YZ plane symmetry: align X-axis with X direction
                            x_axis_candidate = np.array([1.0, 0.0, 0.0])
                            x_axis = x_axis_candidate - np.dot(x_axis_candidate, z_axis) * z_axis
                            x_axis_norm = np.linalg.norm(x_axis)
                            if x_axis_norm > 1e-6:
                                x_axis = x_axis / x_axis_norm
                            else:
                                # X parallel to Z, use arbitrary
                                if abs(z_axis[0]) < 0.9:
                                    x_axis = np.array([1.0, 0.0, 0.0])
                                else:
                                    x_axis = np.array([0.0, 1.0, 0.0])
                                x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                                x_axis = x_axis / np.linalg.norm(x_axis)
                        else:
                            # Other cases: warn and use fallback
                            self.get_logger().warn(
                                f"Grasp point {grasp_point_local.get('id')} has invalid approach_vector. "
                                f"Using fallback."
                            )
                            if abs(z_axis[0]) < 0.9:
                                x_axis = np.array([1.0, 0.0, 0.0])
                            else:
                                x_axis = np.array([0.0, 1.0, 0.0])
                            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                            x_axis = x_axis / np.linalg.norm(x_axis)
                else:
                    # No approach_vector in JSON
                    if symmetry_constraints['xy_free'] or has_full_symmetry:
                        # Full symmetry or XY plane symmetry: use arbitrary direction (full rotational freedom)
                        if abs(z_axis[0]) < 0.9:  # If not pointing along X
                            x_axis = np.array([1.0, 0.0, 0.0])
                        else:  # If pointing along X, use Y as reference
                            x_axis = np.array([0.0, 1.0, 0.0])
                        
                        # Make X-axis perpendicular to Z-axis
                        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                        x_axis_norm = np.linalg.norm(x_axis)
                        if x_axis_norm < 1e-6:  # Avoid division by zero
                            # Fallback: use a different reference vector
                            if abs(z_axis[1]) < 0.9:
                                x_axis = np.array([0.0, 1.0, 0.0])
                            else:
                                x_axis = np.array([0.0, 0.0, 1.0])
                            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                            x_axis = x_axis / np.linalg.norm(x_axis)
                        else:
                            x_axis = x_axis / x_axis_norm
                    elif symmetry_constraints['x_constrained']:
                        # YZ plane symmetry: align X-axis with X direction
                        x_axis_candidate = np.array([1.0, 0.0, 0.0])
                        x_axis = x_axis_candidate - np.dot(x_axis_candidate, z_axis) * z_axis
                        x_axis_norm = np.linalg.norm(x_axis)
                        if x_axis_norm > 1e-6:
                            x_axis = x_axis / x_axis_norm
                        else:
                            # X parallel to Z, use arbitrary
                            if abs(z_axis[0]) < 0.9:
                                x_axis = np.array([1.0, 0.0, 0.0])
                            else:
                                x_axis = np.array([0.0, 1.0, 0.0])
                            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                            x_axis = x_axis / np.linalg.norm(x_axis)
                    else:
                        # Not full symmetry: approach_vector should be provided, but use fallback
                        self.get_logger().warn(
                            f"Grasp point {grasp_point_local.get('id')} missing approach_vector. "
                            f"Using arbitrary direction."
                        )
                        if abs(z_axis[0]) < 0.9:
                            x_axis = np.array([1.0, 0.0, 0.0])
                        else:
                            x_axis = np.array([0.0, 1.0, 0.0])
                        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                        x_axis = x_axis / np.linalg.norm(x_axis)
                
                # Y-axis is cross product of Z and X
                y_axis = np.cross(z_axis, x_axis)
                y_axis_norm = np.linalg.norm(y_axis)
                if y_axis_norm < 1e-6:  # Avoid division by zero
                    grasp_quat_object = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
                else:
                    y_axis = y_axis / y_axis_norm
                    
                    # Construct rotation matrix
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                    
                    # Convert to quaternion (in object frame)
                    grasp_quat_object = R.from_matrix(rotation_matrix).as_quat()
            
            # Transform orientation from object frame to world frame
            # world_orientation = object_orientation * grasp_orientation_object
            r_grasp_object = R.from_quat(grasp_quat_object)
            r_object_world = R.from_quat(obj_quaternion)
            r_grasp_world = r_object_world * r_grasp_object
            quat_base = r_grasp_world.as_quat()
            
        else:
            # Default orientation (identity in world frame)
            quat_base = obj_quaternion  # Use object orientation as default
        
        return pos_base, quat_base
    
    def publish_grasp_points(self):
        """Publish grasp points for all objects with known poses"""
        if not self.object_poses or not self.grasp_data:
            return
        
        # Create GraspPointArray message
        grasp_array = GraspPointArray()
        
        # Use current time for header
        now = self.get_clock().now()
        grasp_array.header.stamp = now.to_msg()
        grasp_array.header.frame_id = "base"
        
        # Process each object that has both pose and grasp data
        for object_name_topic, object_pose in self.object_poses.items():
            # Map topic object name to JSON object name
            object_name_json = self.object_name_map.get(object_name_topic)
            if object_name_json is None:
                # Try direct match
                if object_name_topic not in self.grasp_data:
                    continue
                object_name_json = object_name_topic
            elif object_name_json not in self.grasp_data:
                continue
            
            grasp_data = self.grasp_data[object_name_json]
            grasp_points_local = grasp_data.get('grasp_points', [])
            
            # Get symmetry configuration from object configuration (defaults to full symmetry if not specified)
            symmetry_config = self.OBJECT_SYMMETRY_AXES.get(object_name_topic, {'symmetry_planes': ['xy', 'yz', 'xz']})
            
            # Transform each grasp point
            for gp_local in grasp_points_local:
                try:
                    # Use object-level symmetry configuration (per-grasp-point override not supported yet)
                    
                    # Transform grasp point to base frame
                    pos_base, quat_base = self.transform_grasp_point(gp_local, object_pose, symmetry_config)
                    
                    # Calculate roll, pitch, yaw
                    roll, pitch, yaw = self.quaternion_to_rpy(
                        quat_base[0], quat_base[1], quat_base[2], quat_base[3]
                    )
                    
                    # Create GraspPoint message
                    grasp_point = GraspPoint()
                    
                    # Header
                    grasp_point.header.stamp = now.to_msg()
                    grasp_point.header.frame_id = "base"
                    
                    # Object info - use topic name (without _scaled70) for consistency
                    grasp_point.object_name = object_name_topic
                    grasp_point.grasp_id = gp_local.get('id', 0)
                    grasp_point.grasp_type = gp_local.get('type', 'center_point')
                    
                    # Pose
                    grasp_point.pose.position.x = float(pos_base[0])
                    grasp_point.pose.position.y = float(pos_base[1])
                    grasp_point.pose.position.z = float(pos_base[2])
                    grasp_point.pose.orientation.x = float(quat_base[0])
                    grasp_point.pose.orientation.y = float(quat_base[1])
                    grasp_point.pose.orientation.z = float(quat_base[2])
                    grasp_point.pose.orientation.w = float(quat_base[3])
                    
                    # Euler angles
                    grasp_point.roll = float(roll)
                    grasp_point.pitch = float(pitch)
                    grasp_point.yaw = float(yaw)
                    
                    grasp_array.grasp_points.append(grasp_point)
                    
                except Exception as e:
                    self.get_logger().error(f"Error transforming grasp point {gp_local.get('id')} for {object_name_topic}: {e}")
        
        # Publish if we have any grasp points
        if len(grasp_array.grasp_points) > 0:
            self.grasp_pub.publish(grasp_array)
            self.get_logger().debug(f"Published {len(grasp_array.grasp_points)} grasp points")


def main(args=None):
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grasp Points Publisher Node')
    parser.add_argument('--objects-poses-topic', type=str, default=None,
                       help='Topic name for object poses subscription (default: based on mode)')
    parser.add_argument('--grasp-points-topic', type=str, default=None,
                       help='Topic name for grasp points publication (default: based on mode)')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Mode: "sim" for simulation (uses /objects_poses_sim, /grasp_points_sim), "real" for real robot (uses /objects_poses_real, /grasp_points_real). Default: sim')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing grasp points JSON files (default: data/grasp relative to project root)')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    
    node = GraspPointsPublisher(
        objects_poses_topic=args.objects_poses_topic,
        grasp_points_topic=args.grasp_points_topic,
        data_dir=args.data_dir,
        mode=args.mode
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
