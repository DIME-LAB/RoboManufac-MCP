#!/usr/bin/env python3
"""
Grasp Points Publisher
Reads object poses from /objects_poses_sim topic and publishes grasp points to /grasp_points_sim topic.
Uses grasp points data from JSON files and transforms them using object poses.
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
    
    def __init__(self, objects_poses_topic="/objects_poses_sim", 
                 grasp_points_topic="/grasp_points_sim",
                 data_dir=None):
        super().__init__('grasp_points_publisher')
        
        self.objects_poses_topic = objects_poses_topic
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
            objects_poses_topic,
            self.objects_poses_callback,
            qos_profile
        )
        
        # Create publisher for grasp points
        self.grasp_pub = self.create_publisher(
            GraspPointArray,
            grasp_points_topic,
            qos_profile
        )
        
        # Timer to publish grasp points periodically
        self.publish_timer = self.create_timer(0.1, self.publish_grasp_points)  # 10 Hz
        
        self.get_logger().info(f"🤖 Grasp Points Publisher started")
        self.get_logger().info(f"📥 Subscribing to: {objects_poses_topic}")
        self.get_logger().info(f"📤 Publishing to: {grasp_points_topic}")
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
        """Convert quaternion to roll, pitch, yaw in degrees"""
        # Use scipy for robust conversion
        r = R.from_quat([x, y, z, w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return roll, pitch, yaw
    
    def transform_grasp_point(self, grasp_point_local, object_pose):
        """
        Transform grasp point from CAD center frame to base frame using object pose.
        
        Args:
            grasp_point_local: Dict with position (x, y, z) relative to CAD center
            object_pose: Dict with translation and quaternion of object in base frame
        
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
        r = R.from_quat(obj_quaternion)
        rotation_matrix = r.as_matrix()
        
        # Transform: base_position = object_position + rotation * local_position
        pos_base = obj_translation + rotation_matrix @ pos_local
        
        # For orientation, we need to combine object orientation with approach vector
        # Get approach vector (default to [0, 0, 1] if not specified)
        approach = grasp_point_local.get('approach_vector', {'x': 0.0, 'y': 0.0, 'z': 1.0})
        approach_vec = np.array([approach['x'], approach['y'], approach['z']])
        approach_vec = approach_vec / np.linalg.norm(approach_vec)  # Normalize
        
        # Rotate approach vector by object orientation
        approach_base = rotation_matrix @ approach_vec
        
        # Create quaternion from approach vector (assuming approach is Z-axis direction)
        # We need to construct a full rotation matrix
        # For simplicity, we'll use the object's orientation and adjust based on approach
        # For now, use object orientation directly (can be refined later)
        quat_base = obj_quaternion
        
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
            
            # Transform each grasp point
            for gp_local in grasp_points_local:
                try:
                    # Transform grasp point to base frame
                    pos_base, quat_base = self.transform_grasp_point(gp_local, object_pose)
                    
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
    parser.add_argument('--objects-poses-topic', type=str, default="/objects_poses_sim",
                       help='Topic name for object poses subscription (default: /objects_poses_sim)')
    parser.add_argument('--grasp-points-topic', type=str, default="/grasp_points_sim",
                       help='Topic name for grasp points publication (default: /grasp_points_sim)')
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
        data_dir=args.data_dir
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
