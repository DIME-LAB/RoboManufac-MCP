#!/usr/bin/env python3
"""
Reorient for Assembly - Properly calculates EE orientation needed to achieve target object orientation

The key insight: 
- JSON contains target OBJECT orientation
- We need to calculate target EE orientation based on current grasp
- T_EE_target = T_object_target * T_grasp^(-1)
where T_grasp = T_EE_current^(-1) * T_object_current
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

class ReorientForAssembly(Node):
    def __init__(self, base_topic=BASE_TOPIC, object_topic=OBJECT_TOPIC, ee_topic=EE_TOPIC):
        super().__init__('reorient_for_assembly')
        
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
        
        self.get_logger().info("ReorientForAssembly node initialized")
        self.get_logger().info(f"Assembly config loaded with {len(self.assembly_config.get('components', []))} components")
    
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
        """Canonicalize Euler angles - relaxed threshold for IK solver"""
        roll, pitch, yaw = orientation
        # Relax threshold to 5 degrees to handle small pitch variations
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
        # Apply canonicalization to match get_ee_pose.py
        rpy_deg = self.canonicalize_euler(rpy_deg)
        return position, rpy_deg
    
    def get_object_target_orientation(self, object_name):
        """Get target orientation for object from assembly configuration (in Euler angles radians)"""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                rotation = component.get('rotation', {})
                return np.array([rotation.get('x', 0), rotation.get('y', 0), rotation.get('z', 0)])
        return None
    
    def reorient_for_target(self, object_name, base_name, execute_trajectory=False, duration=10.0, use_robust_ik=False):
        """
        Calculate and execute EE reorientation to achieve target object orientation
        
        ONLY changes orientation, keeps current position!
        Target orientation from JSON is relative to base frame.
        
        Algorithm:
        1. T_EE_current = current end-effector pose
        2. T_object_current = current object pose  
        3. T_base_current = current base pose
        4. T_grasp = T_EE_current^(-1) * T_object_current  (how EE is holding the object)
        5. Get target orientation from JSON (relative to base)
        6. R_object_target_world = R_base_current * R_target_relative  (transform to world frame)
        7. T_object_target = current object position + target orientation in world frame
        8. T_EE_target = T_object_target * T_grasp^(-1)  (required EE pose)
        
        Args:
            object_name: Name of the object to reorient
            base_name: Name of the base object
            execute_trajectory: If True, execute the calculated trajectory
            duration: Duration for trajectory execution
            use_robust_ik: If True, use robust IK solver with multiple seed configurations
        
        Note:
            The initial grasp relationship (relative rotation between gripper and object) is preserved.
            The object's absolute orientation changes to match the target, but the way the object is held
            relative to the gripper remains unchanged throughout the reorientation.
            This shouldnt be the case ideally- need to fix this after fixing the physics of the object in arm simulation
            and after fixing the real world object pose detection.
        """
        self.get_logger().info(f"Calculating reorientation for {object_name} relative to {base_name}")
        
        if use_robust_ik:
            self.get_logger().info("Using ROBUST IK solver (can handle arbitrary orientations)")
            self.get_logger().warn("WARNING: Testing enhanced IK solver with quaternion-based error metric")
        
        # Wait for pose data
        if not self.current_poses or self.current_ee_pose is None:
            self.get_logger().error("No pose data available")
            return None
        
        # Get current EE pose
        if object_name not in self.current_poses:
            self.get_logger().error(f"Object {object_name} not found in poses")
            # Try with _scaled70 suffix
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
        
        # Get target object orientation from JSON (relative to base frame)
        target_euler_relative = self.get_object_target_orientation(object_name)
        if target_euler_relative is None:
            self.get_logger().error(f"No target orientation found for {object_name}")
            return None
        
        # Transform target orientation from base frame to world frame
        # R_object_target_world = R_base_current * R_target_relative
        R_base_current = T_base_current[:3, :3]
        R_target_relative = R.from_euler('xyz', target_euler_relative).as_matrix()
        R_object_target_world = R_base_current @ R_target_relative
        
        # Create target object transformation matrix (keep current position, apply world-frame orientation)
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = R_object_target_world
        T_object_target[:3, 3] = T_object_current[:3, 3]  # Keep current object position
        
        # Calculate required EE pose
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)
        
        # Convert to position and RPY (already canonicalized by matrix_to_rpy)
        ee_target_position, ee_target_rpy = self.matrix_to_rpy(T_EE_target)
        object_target_position, object_target_rpy = self.matrix_to_rpy(T_object_target)
        
        self.get_logger().info(f"DEBUG: Target EE RPY before sending to IK: {ee_target_rpy}")
        
        # Log the calculations
        self.get_logger().info("=" * 80)
        self.get_logger().info("REORIENTATION CALCULATION:")
        self.get_logger().info("")
        self.get_logger().info("CURRENT STATE:")
        self.get_logger().info(f"  Current Base: Position={base_current_position} RPY={base_current_rpy}")
        self.get_logger().info(f"  Current EE: Position={ee_current_position} RPY={ee_current_rpy}")
        self.get_logger().info(f"  Current Object: Position={object_current_position} RPY={object_current_rpy}")
        self.get_logger().info("")
        self.get_logger().info("TARGET STATE:")
        self.get_logger().info(f"  Target Object Orientation (from JSON, relative to base): {np.degrees(target_euler_relative)} degrees")
        self.get_logger().info(f"  Target Object Orientation (world frame): RPY={object_target_rpy}")
        self.get_logger().info(f"  Target Object: Position={object_target_position} RPY={object_target_rpy}")
        self.get_logger().info(f"  Calculated Target EE: Position={ee_target_position} RPY={ee_target_rpy}")
        self.get_logger().info("=" * 80)
        
        if execute_trajectory:
            # Generate trajectory
            self.get_logger().info("Generating trajectory...")
            
            # Choose IK solver based on flag
            if use_robust_ik:
                trajectory_points = move_robust(ee_target_position.tolist(), ee_target_rpy.tolist(), duration)
            else:
                trajectory_points = move(ee_target_position.tolist(), ee_target_rpy.tolist(), duration)
            
            if trajectory_points:
                trajectory = {"traj1": trajectory_points}
                self.get_logger().info("Executing trajectory...")
                success = self.execute_trajectory(trajectory)
                
                if success:
                    # Wait a moment for poses to update
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
                        self.get_logger().info(f"  Final Object: Position={object_final_position} RPY={object_final_rpy}")
                        self.get_logger().info("")
                        self.get_logger().info("COMPARISON:")
                        self.get_logger().info(f"  Object Position Change: {object_final_position - object_current_position}")
                        self.get_logger().info(f"  Object Orientation Change: {object_final_rpy - object_current_rpy}")
                        self.get_logger().info(f"  Target Object Orientation: {object_target_rpy}")
                        self.get_logger().info(f"  Final Object Orientation: {object_final_rpy}")
                        self.get_logger().info(f"  Orientation Error: {object_final_rpy - object_target_rpy}")
                        self.get_logger().info("=" * 80)
                
                return success
            else:
                self.get_logger().error("Failed to generate trajectory")
                return False
        
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
                self.get_logger().error("❌ Trajectory goal rejected")
                return False
            
            self.get_logger().info("✅ Trajectory goal accepted, waiting for completion...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            
            if result.status == 4:  # SUCCEEDED
                self.get_logger().info("✅ Trajectory completed successfully")
                return True
            else:
                self.get_logger().error(f"❌ Trajectory failed with status: {result.status}")
                return False
        except Exception as e:
            self.get_logger().error(f"❌ Trajectory execution error: {e}")
            return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Reorient for Assembly - ONLY changes orientation, keeps position')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to reorient')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object (for orientation reference)')
    parser.add_argument('--duration', type=float, default=10.0, help='Movement duration in seconds')
    parser.add_argument('--execute', action='store_true', help='Execute trajectory')
    parser.add_argument('--robust-ik', action='store_true', help='Use robust IK solver (can handle arbitrary orientations)')
    args = parser.parse_args()
    
    rclpy.init()
    node = ReorientForAssembly()
    
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
        
        # Execute reorientation
        success = node.reorient_for_target(
            args.object_name,
            args.base_name,
            execute_trajectory=args.execute, 
            duration=args.duration,
            use_robust_ik=args.robust_ik
        )
        
        if success:
            node.get_logger().info("Reorientation successful!")
        else:
            node.get_logger().error("Reorientation failed")
            
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

