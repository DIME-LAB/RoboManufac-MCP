#!/usr/bin/env python3
"""
Verify Final Assembly Pose - Checks if object is in correct position and orientation relative to base

The algorithm:
1. Get current object pose and base pose
2. Calculate relative position and orientation of object relative to base
3. Compare with target position and orientation from JSON
4. Check if within tolerance
5. Return success if match, failure if not
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped, TransformStamped
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os

# Configuration
ASSEMBLY_JSON_FILE = "/home/aaugus11/Projects/aruco-grasp-annotator/data/fmb_assembly.json"
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"

# Tolerance thresholds
POSITION_TOLERANCE = 0.01  # 1cm tolerance for position
ORIENTATION_TOLERANCE_DEG = 5.0  # 5 degrees tolerance for orientation

class VerifyFinalAssemblyPose(Node):
    def __init__(self, base_topic=BASE_TOPIC, object_topic=OBJECT_TOPIC, ee_topic=EE_TOPIC):
        super().__init__('verify_final_assembly_pose')
        
        # Load assembly configuration
        self.assembly_config = self.load_assembly_config()
        
        # Subscribers for pose data
        self.base_sub = self.create_subscription(TFMessage, base_topic, self.base_callback, 10)
        self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)
        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, 10)
        
        # Store current poses
        self.current_poses = {}
        self.current_ee_pose = None
        
        self.get_logger().info("VerifyFinalAssemblyPose node initialized")
        self.get_logger().info(f"Assembly config loaded with {len(self.assembly_config.get('components', []))} components")
        self.get_logger().info(f"Position tolerance: {POSITION_TOLERANCE}m, Orientation tolerance: {ORIENTATION_TOLERANCE_DEG}°")
    
    def load_assembly_config(self):
        """Load the assembly configuration from JSON file"""
        try:
            with open(ASSEMBLY_JSON_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"Assembly file not found: {ASSEMBLY_JSON_FILE}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing assembly JSON: {e}")
            return {}
    
    def base_callback(self, msg):
        """Callback for base poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
    
    def object_callback(self, msg):
        """Callback for object poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
    
    def ee_callback(self, msg):
        """Callback for end-effector pose"""
        self.current_ee_pose = msg
    
    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 transformation matrix"""
        t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    def pose_to_matrix(self, pose):
        """Convert ROS Pose to 4x4 transformation matrix"""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    def canonicalize_euler(self, orientation):
        """Canonicalize Euler angles"""
        roll, pitch, yaw = orientation
        if abs(pitch) < 5 and abs(abs(roll) - 180) < 5:
            return np.array([0.0, 180.0, (yaw % 360) - 180])
        else:
            return orientation
    
    def matrix_to_rpy(self, T):
        """Convert 4x4 transformation matrix to position and RPY (degrees)"""
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        rpy_rad = r.as_euler('xyz')
        rpy_deg = np.degrees(rpy_rad)
        rpy_deg = self.canonicalize_euler(rpy_deg)
        return position, rpy_deg
    
    def get_object_target_position(self, object_name):
        """Get target position for object from assembly configuration (relative to base)"""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                position = component.get('position', {})
                return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        return None
    
    def get_object_target_orientation(self, object_name):
        """Get target orientation for object from assembly configuration (relative to base, in radians)"""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                rotation = component.get('rotation', {})
                return np.array([rotation.get('x', 0), rotation.get('y', 0), rotation.get('z', 0)])
        return None
    
    def verify_assembly_pose(self, object_name, base_name):
        """
        Verify if object is in correct position and orientation relative to base
        
        Algorithm:
        1. Get current object pose and base pose
        2. Calculate relative position and orientation of object relative to base
        3. Compare with target position and orientation from JSON
        4. Check if within tolerance
        
        Args:
            object_name: Name of the object to verify
            base_name: Name of the base object
            
        Returns:
            True if object is in correct pose, False otherwise
        """
        self.get_logger().info(f"Verifying assembly pose for {object_name} relative to {base_name}")
        
        # Wait for pose data
        if not self.current_poses:
            self.get_logger().error("No pose data available")
            return False
        
        # Check if object exists
        original_object_name = object_name
        if object_name not in self.current_poses:
            object_name = f"{object_name}_scaled70"
            if object_name not in self.current_poses:
                self.get_logger().error(f"Object {original_object_name} not found in poses")
                return False
        
        # Check if base exists
        original_base_name = base_name
        if base_name not in self.current_poses:
            base_name = f"{base_name}_scaled70"
            if base_name not in self.current_poses:
                self.get_logger().error(f"Base {original_base_name} not found in poses")
                return False
        
        # Convert poses to matrices
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)
        
        # Calculate relative transformation: T_object_relative = T_base^(-1) * T_object
        T_object_relative = np.linalg.inv(T_base_current) @ T_object_current
        
        # Extract relative position and orientation
        object_relative_position = T_object_relative[:3, 3]
        object_relative_rotation = R.from_matrix(T_object_relative[:3, :3])
        object_relative_rpy_rad = object_relative_rotation.as_euler('xyz')
        
        # Get target position and orientation from JSON (relative to base)
        target_position_relative = self.get_object_target_position(original_object_name)
        target_orientation_relative = self.get_object_target_orientation(original_object_name)
        
        if target_position_relative is None:
            self.get_logger().error(f"No target position found for {original_object_name} in assembly config")
            return False
        
        if target_orientation_relative is None:
            self.get_logger().error(f"No target orientation found for {original_object_name} in assembly config")
            return False
        
        # Calculate position error
        position_error = np.linalg.norm(object_relative_position - target_position_relative)
        
        # Calculate orientation error
        target_rotation_relative = R.from_euler('xyz', target_orientation_relative)
        orientation_error_rad = (object_relative_rotation.inv() * target_rotation_relative).magnitude()
        orientation_error_deg = np.degrees(orientation_error_rad)
        
        # Log the verification details
        self.get_logger().info("=" * 80)
        self.get_logger().info("ASSEMBLY POSE VERIFICATION:")
        self.get_logger().info("")
        self.get_logger().info("CURRENT RELATIVE POSE:")
        self.get_logger().info(f"  Position (relative to base): {object_relative_position}")
        self.get_logger().info(f"  Orientation (relative to base, RPY): {np.degrees(object_relative_rpy_rad)}°")
        self.get_logger().info("")
        self.get_logger().info("TARGET RELATIVE POSE (from JSON):")
        self.get_logger().info(f"  Position (relative to base): {target_position_relative}")
        self.get_logger().info(f"  Orientation (relative to base, RPY): {np.degrees(target_orientation_relative)}°")
        self.get_logger().info("")
        self.get_logger().info("ERRORS:")
        self.get_logger().info(f"  Position error: {position_error:.6f}m (tolerance: {POSITION_TOLERANCE}m)")
        self.get_logger().info(f"  Orientation error: {orientation_error_deg:.2f}° (tolerance: {ORIENTATION_TOLERANCE_DEG}°)")
        self.get_logger().info("")
        
        # Check if within tolerance
        position_ok = position_error <= POSITION_TOLERANCE
        orientation_ok = orientation_error_deg <= ORIENTATION_TOLERANCE_DEG
        
        if position_ok and orientation_ok:
            self.get_logger().info("✅ VERIFICATION SUCCESSFUL: Object is in correct assembly pose")
            self.get_logger().info("=" * 80)
            return True
        else:
            self.get_logger().error("❌ VERIFICATION FAILED: Object is NOT in correct assembly pose")
            if not position_ok:
                self.get_logger().error(f"  - Position error ({position_error:.6f}m) exceeds tolerance ({POSITION_TOLERANCE}m)")
            if not orientation_ok:
                self.get_logger().error(f"  - Orientation error ({orientation_error_deg:.2f}°) exceeds tolerance ({ORIENTATION_TOLERANCE_DEG}°)")
            self.get_logger().info("=" * 80)
            return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Verify Final Assembly Pose - Check if object is in correct position')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to verify')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object')
    args = parser.parse_args()
    
    rclpy.init()
    node = VerifyFinalAssemblyPose()
    
    try:
        # Wait for pose data (wait indefinitely until received)
        node.get_logger().info(f"Waiting for pose data for object: {args.object_name} and base: {args.base_name}")
        start_time = time.time()
        last_log_time = start_time
        
        while not node.current_poses:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
            
            # Log every 5 seconds to show we're still waiting
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - start_time
                node.get_logger().info(f"Still waiting for pose data... ({elapsed:.1f}s elapsed)")
                last_log_time = current_time
        
        elapsed = time.time() - start_time
        node.get_logger().info(f"Received pose data for {len(node.current_poses)} objects (waited {elapsed:.1f}s)")
        
        # Verify assembly pose
        success = node.verify_assembly_pose(
            args.object_name,
            args.base_name
        )
        
        if success:
            node.get_logger().info("Assembly pose verification: SUCCESS")
        else:
            node.get_logger().error("Assembly pose verification: FAILED - Placement failed")
        
        # Exit with appropriate code
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()

