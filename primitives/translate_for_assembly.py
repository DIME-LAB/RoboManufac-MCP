#!/usr/bin/env python3
"""
Translate for Assembly - Moves object to target position from assembly configuration

The algorithm:
1. Read current base pose and target position from JSON
2. Calculate target EE position to place object at target location
3. Move to hover height (0.25m) above target first
4. Move down to final height accounting for base height and object height
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped, TransformStamped
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time

# Import action libraries for trajectory generation
from action_libraries import move, move_robust

# Configuration
ASSEMBLY_JSON_FILE = "/home/aaugus11/Projects/aruco-grasp-annotator/data/fmb_assembly.json"
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"
HOVER_HEIGHT = 0.25  # Height to hover above base before descending

class TranslateForAssembly(Node):
    def __init__(self, base_topic=BASE_TOPIC, object_topic=OBJECT_TOPIC, ee_topic=EE_TOPIC):
        super().__init__('translate_for_assembly')
        
        # Load assembly configuration
        self.assembly_config = self.load_assembly_config()
        
        # Subscribers for pose data
        self.base_sub = self.create_subscription(TFMessage, base_topic, self.base_callback, 10)
        self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)
        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, 10)
        
        # Store current poses
        self.current_poses = {}
        self.current_ee_pose = None
        
        # Action client for trajectory execution
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.action_client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        
        self.get_logger().info("TranslateForAssembly node initialized")
        self.get_logger().info(f"Assembly config loaded with {len(self.assembly_config.get('components', []))} components")
        self.get_logger().info(f"Hover height set to: {HOVER_HEIGHT}m")
    
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
        """Get target position for object from assembly configuration"""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                position = component.get('position', {})
                return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        return None
    
    def translate_for_target(self, object_name, base_name, execute_trajectory=False, duration=10.0, use_robust_ik=False):
        """
        Calculate and execute EE translation to place object at target position
        
        Algorithm:
        1. Get current base pose (T_base)
        2. Get target object position from JSON (relative to base)
        3. Calculate grasp transformation (T_grasp)
        4. Calculate target EE position to place object at target location
        5. Move to hover height first
        6. Move down to final height
        
        Args:
            object_name: Name of the object being held
            base_name: Name of the base object (e.g., 'base')
            execute_trajectory: If True, execute the calculated trajectory
            duration: Duration for trajectory execution
            use_robust_ik: If True, use robust IK solver
        """
        self.get_logger().info(f"Calculating translation for {object_name} relative to {base_name}")
        
        if use_robust_ik:
            self.get_logger().info("Using ROBUST IK solver")
        
        # Wait for pose data
        if not self.current_poses or self.current_ee_pose is None:
            self.get_logger().error("No pose data available")
            return None
        
        # Get current EE pose
        if self.current_ee_pose is None:
            self.get_logger().error("End-effector pose not available")
            return None
        
        # Check if object exists
        if object_name not in self.current_poses:
            object_name = f"{object_name}_scaled70"
            if object_name not in self.current_poses:
                self.get_logger().error(f"Object {object_name} not found")
                return None
        
        # Check if base exists
        if base_name not in self.current_poses:
            base_name = f"{base_name}_scaled70"
            if base_name not in self.current_poses:
                self.get_logger().error(f"Base {base_name} not found")
                return None
        
        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)
        
        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current
        
        # Get current positions
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        object_current_position, object_current_rpy = self.matrix_to_rpy(T_object_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        grasp_position, grasp_rpy = self.matrix_to_rpy(T_grasp)
        
        self.get_logger().info(f"DEBUG: Grasp transformation: Position={grasp_position} RPY={grasp_rpy}")
        
        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.get_logger().error(f"No target position found for {object_name}")
            return None
        
        # Calculate absolute target object position (base position + relative position)
        target_object_position_abs = base_current_position + target_position_relative
        
        # Create target object transformation (keep current orientation for now)
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = T_object_current[:3, :3]  # Keep current orientation
        T_object_target[:3, 3] = target_object_position_abs
        
        # Calculate required EE position to place object at target
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)
        ee_target_position, ee_target_rpy = self.matrix_to_rpy(T_EE_target)
        
        # Create hover position (same XY as target, but at HOVER_HEIGHT above base)
        hover_position = ee_target_position.copy()
        hover_position[2] = base_current_position[2] + HOVER_HEIGHT
        
        # Log the calculations
        self.get_logger().info("=" * 80)
        self.get_logger().info("TRANSLATION CALCULATION:")
        self.get_logger().info("")
        self.get_logger().info("CURRENT STATE:")
        self.get_logger().info(f"  Current EE: Position={ee_current_position} RPY={ee_current_rpy}")
        self.get_logger().info(f"  Current Object: Position={object_current_position}")
        self.get_logger().info(f"  Current Base: Position={base_current_position}")
        self.get_logger().info("")
        self.get_logger().info("TARGET STATE:")
        self.get_logger().info(f"  Target Position (relative to base): {target_position_relative}")
        self.get_logger().info(f"  Target Object Position (absolute): {target_object_position_abs}")
        self.get_logger().info(f"  Hover Position: {hover_position}")
        self.get_logger().info(f"  Final Target EE Position: {ee_target_position} RPY={ee_target_rpy}")
        self.get_logger().info("=" * 80)
        
        if execute_trajectory:
            # Step 1: Move to hover position
            self.get_logger().info("Step 1: Moving to hover position...")
            
            if use_robust_ik:
                hover_trajectory = move_robust(hover_position.tolist(), ee_target_rpy.tolist(), duration)
            else:
                hover_trajectory = move(hover_position.tolist(), ee_target_rpy.tolist(), duration)
            
            if not hover_trajectory:
                self.get_logger().error("Failed to generate hover trajectory")
                return False
            
            success = self.execute_trajectory({"traj1": hover_trajectory})
            if not success:
                self.get_logger().error("Failed to reach hover position")
                return False
            
            self.get_logger().info("Reached hover position")
            time.sleep(0.5)
            
            # Step 2: Move down to final position
            self.get_logger().info("Step 2: Moving down to final position...")
            
            if use_robust_ik:
                final_trajectory = move_robust(ee_target_position.tolist(), ee_target_rpy.tolist(), duration)
            else:
                final_trajectory = move(ee_target_position.tolist(), ee_target_rpy.tolist(), duration)
            
            if not final_trajectory:
                self.get_logger().error("Failed to generate final trajectory")
                return False
            
            success = self.execute_trajectory({"traj1": final_trajectory})
            
            if success:
                # Wait for poses to update
                time.sleep(0.5)
                
                # Log final state
                if object_name in self.current_poses and self.current_ee_pose is not None:
                    T_EE_final = self.pose_to_matrix(self.current_ee_pose.pose)
                    T_object_final = self.transform_to_matrix(self.current_poses[object_name].transform)
                    
                    ee_final_position, ee_final_rpy = self.matrix_to_rpy(T_EE_final)
                    object_final_position, object_final_rpy = self.matrix_to_rpy(T_object_final)
                    
                    self.get_logger().info("")
                    self.get_logger().info("=" * 80)
                    self.get_logger().info("FINAL STATE (after execution):")
                    self.get_logger().info(f"  Final EE: Position={ee_final_position} RPY={ee_final_rpy}")
                    self.get_logger().info(f"  Final Object: Position={object_final_position}")
                    self.get_logger().info("")
                    self.get_logger().info("COMPARISON:")
                    self.get_logger().info(f"  Object Position Change: {object_final_position - object_current_position}")
                    self.get_logger().info(f"  Target Object Position: {target_object_position_abs}")
                    self.get_logger().info(f"  Final Object Position: {object_final_position}")
                    self.get_logger().info(f"  Position Error: {object_final_position - target_object_position_abs}")
                    self.get_logger().info("=" * 80)
            
            return success
        
        return True
    
    def execute_trajectory(self, trajectory):
        """Execute trajectory and wait for completion"""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                return False
            
            point = trajectory['traj1'][0]
            positions = point['positions']
            duration = point['time_from_start'].sec
            
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            
            traj_point = JointTrajectoryPoint()
            traj_point.positions = positions
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = Duration(sec=duration)
            traj_msg.points.append(traj_point)
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            future = self.action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error("Trajectory goal rejected")
                return False
            
            self.get_logger().info("Trajectory goal accepted, waiting for completion...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            
            if result.status == 4:  # SUCCEEDED
                self.get_logger().info("Trajectory completed successfully")
                return True
            else:
                self.get_logger().error(f"Trajectory failed with status: {result.status}")
                return False
        except Exception as e:
            self.get_logger().error(f"Trajectory execution error: {e}")
            return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Translate for Assembly - Move object to target position')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object being held')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object')
    parser.add_argument('--duration', type=float, default=10.0, help='Movement duration in seconds')
    parser.add_argument('--execute', action='store_true', help='Execute trajectory')
    parser.add_argument('--robust-ik', action='store_true', help='Use robust IK solver')
    args = parser.parse_args()
    
    rclpy.init()
    node = TranslateForAssembly()
    
    node.get_logger().info("Waiting for action server...")
    node.action_client.wait_for_server()
    node.get_logger().info("Action server available!")
    
    try:
        # Wait for pose data
        node.get_logger().info(f"Waiting for pose data for object: {args.object_name} and base: {args.base_name}")
        max_wait_time = 10
        start_time = time.time()
        
        while (not node.current_poses or node.current_ee_pose is None) and (time.time() - start_time) < max_wait_time:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
        
        if not node.current_poses or node.current_ee_pose is None:
            node.get_logger().error("No pose data received")
            return
        
        node.get_logger().info(f"Received pose data for {len(node.current_poses)} objects")
        
        # Execute translation
        success = node.translate_for_target(
            args.object_name,
            args.base_name,
            execute_trajectory=args.execute,
            duration=args.duration,
            use_robust_ik=args.robust_ik
        )
        
        if success:
            node.get_logger().info("Translation successful!")
        else:
            node.get_logger().error("Translation failed")
            
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

