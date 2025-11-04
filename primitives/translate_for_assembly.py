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
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import argparse
import time
import sys
import os

# Add custom libraries to Python path for IK solver
custom_lib_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries"
if custom_lib_path not in sys.path:
    sys.path.append(custom_lib_path)

try:
    from ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
except ImportError as e:
    print(f"Failed to import IK solver: {e}")
    sys.exit(1)

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
        
        # Subscriber for joint states to get current joint angles (use as IK seed)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Store current poses
        self.current_poses = {}
        self.current_ee_pose = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False
        
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
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state data"""
        # Extract joint angles in the correct order
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            # Map joint names to positions in correct order
            ordered_positions = []
            for joint_name in self.joint_names:
                if joint_name in joint_dict:
                    ordered_positions.append(joint_dict[joint_name])
            
            if len(ordered_positions) == 6:
                self.current_joint_angles = np.array(ordered_positions)
                self.joint_angles_received = True
    
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
    
    def read_current_joint_angles(self):
        """Read current joint angles using ROS2 subscriber"""
        self.get_logger().info("Reading current joint angles...")
        
        # Reset the flag
        self.joint_angles_received = False
        
        # Wait for joint angles to arrive (with timeout)
        timeout_count = 0
        max_timeout = 100  # 10 seconds (100 * 0.1s)
        
        while rclpy.ok() and not self.joint_angles_received and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            
            if timeout_count % 10 == 0:  # Log every second
                self.get_logger().info(f"Waiting for joint angles... ({timeout_count * 0.1:.1f}s)")
        
        if not self.joint_angles_received:
            self.get_logger().error("Timeout waiting for joint angles message")
            return None
        
        if self.current_joint_angles is None:
            self.get_logger().error("Joint angles data is None")
            return None
        
        self.get_logger().info(f"Successfully read joint angles: {self.current_joint_angles}")
        return self.current_joint_angles.copy()
    
    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        """
        Compute IK using current joint angles as seed (similar to move_down.py)
        
        Args:
            target_position: [x, y, z] target position
            target_quat: [x, y, z, w] target orientation quaternion
            max_tries: Number of position perturbations to try
            dx: Position perturbation step size
            
        Returns:
            Joint angles if successful, None otherwise
        """
        # Convert quaternion to rotation matrix
        target_rotation = R.from_quat(target_quat)
        target_rot_matrix = target_rotation.as_matrix()
        
        # Create target pose
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rot_matrix
        
        # Use current joint angles as seed
        if self.current_joint_angles is None:
            self.get_logger().error("Current joint angles not available! Cannot compute IK.")
            return None
        
        q_guess = self.current_joint_angles.copy()
        self.get_logger().info(f"Using current joint angles from joint state as seed: {q_guess}")
        
        # Try IK with current joint angles and position perturbations
        solution_found = False
        best_result = None
        best_cost = float('inf')
        
        for i in range(max_tries):
            if solution_found:
                break
                
            # Try small x-shift each iteration (helps with workspace boundaries)
            perturbed_position = np.array(target_position).copy()
            perturbed_position[0] += i * dx
            
            perturbed_pose = target_pose.copy()
            perturbed_pose[:3, 3] = perturbed_position
            
            joint_bounds = [(-np.pi, np.pi)] * 6
            
            # Use quaternion-based objective
            result = minimize(ik_objective_quaternion, q_guess, args=(perturbed_pose,), 
                            method='L-BFGS-B', bounds=joint_bounds)
            
            if result.success:
                cost = ik_objective_quaternion(result.x, perturbed_pose)
                
                # Check if this is a good solution
                if cost < 0.01:
                    self.get_logger().info(f"Quaternion-based IK succeeded with current joint angles seed (perturbation {i}), cost={cost:.6f}")
                    
                    # Verify orientation accuracy
                    T_result = forward_kinematics(dh_params, result.x)
                    orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                    self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                    
                    return result.x
                
                # Keep track of best solution
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.x
        
        # If we found any reasonable solution, use it
        if best_result is not None and best_cost < 0.1:
            self.get_logger().info(f"Using best quaternion-based IK solution with cost={best_cost:.6f}")
            
            # Verify orientation accuracy
            T_result = forward_kinematics(dh_params, best_result)
            orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
            self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            return best_result
        
        # Fallback: Try multiple predefined seeds if current seed failed
        self.get_logger().warn("IK failed with current joint angles as seed. Trying multiple predefined seeds...")
        
        # Convert target quaternion to RPY for seed generation
        target_rpy_deg = R.from_matrix(target_rot_matrix).as_euler('xyz', degrees=True)
        target_rpy_deg = target_rpy_deg.tolist()
        
        # Generate diverse seed configurations (similar to compute_ik_robust)
        seed_configs = [
            # Standard seeds
            np.radians([85, -80, 90, -90, -90, -(np.mod(target_rpy_deg[2] + 180, 360) - 180)]),
            np.radians([90, -90, 90, -90, -90, target_rpy_deg[2]]),
            np.radians([0, -90, 90, -90, -90, target_rpy_deg[2]]),
            np.radians([180, -90, 90, -90, -90, target_rpy_deg[2]]),
            # Elbow-up configurations
            np.radians([85, -100, 120, -110, -90, target_rpy_deg[2]]),
            np.radians([85, -60, 60, -90, -90, target_rpy_deg[2]]),
            # Wrist variations
            np.radians([85, -80, 90, -90, 0, target_rpy_deg[2]]),
            np.radians([85, -80, 90, -90, -180, target_rpy_deg[2]]),
            # Additional variations for pitch
            np.radians([85, -70, 80, -100, -90, target_rpy_deg[2]]),
            np.radians([85, -90, 100, -100, -90, target_rpy_deg[2]]),
        ]
        
        self.get_logger().info(f"Trying {len(seed_configs)} alternative seed configurations...")
        
        best_result = None
        best_cost = float('inf')
        
        for seed_idx, q_guess_fallback in enumerate(seed_configs):
            for i in range(max_tries):
                # Try small x-shift each iteration
                perturbed_position = np.array(target_position).copy()
                perturbed_position[0] += i * dx
                
                perturbed_pose = target_pose.copy()
                perturbed_pose[:3, 3] = perturbed_position
                
                joint_bounds = [(-np.pi, np.pi)] * 6
                
                # Use quaternion-based objective
                result = minimize(ik_objective_quaternion, q_guess_fallback, args=(perturbed_pose,), 
                                method='L-BFGS-B', bounds=joint_bounds)
                
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed_pose)
                    
                    # Check if this is a good solution
                    if cost < 0.01:
                        self.get_logger().info(f"IK succeeded with fallback seed {seed_idx+1}/{len(seed_configs)} (perturbation {i}), cost={cost:.6f}")
                        
                        # Verify orientation accuracy
                        T_result = forward_kinematics(dh_params, result.x)
                        orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                        self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                        
                        return result.x
                    
                    # Keep track of best solution
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result.x
        
        # If we found any reasonable solution with fallback seeds, use it
        if best_result is not None and best_cost < 0.1:
            self.get_logger().info(f"Using best fallback IK solution with cost={best_cost:.6f}")
            
            # Verify orientation accuracy
            T_result = forward_kinematics(dh_params, best_result)
            orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
            self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            return best_result
        
        self.get_logger().error("IK failed: couldn't find solution even with multiple seeds")
        return None
    
    def translate_for_target(self, object_name, base_name, duration=10.0):
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
            duration: Duration for trajectory execution
        """
        self.get_logger().info(f"Calculating translation for {object_name} relative to {base_name}")
        
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
        
        # Transform target position from base frame to world frame
        # If base is rotated, we need to apply base rotation to relative position
        # target_position_world = base_position + R_base @ target_position_relative
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        
        # Create target object transformation (keep current orientation for now)
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = T_object_current[:3, :3]  # Keep current orientation
        T_object_target[:3, 3] = target_object_position_abs
        
        # Calculate required EE position to place object at target
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)
        
        # Extract target position and quaternion
        ee_target_position = T_EE_target[:3, 3]
        ee_target_rot_matrix = T_EE_target[:3, :3]
        ee_target_rotation = R.from_matrix(ee_target_rot_matrix)
        ee_target_quat = ee_target_rotation.as_quat()  # [x, y, z, w]
        
        # Convert to position and RPY for logging
        ee_target_position_rpy, ee_target_rpy = self.matrix_to_rpy(T_EE_target)
        
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
        
        # Read current joint angles before computing IK
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.get_logger().error("Could not read current joint angles")
                return False
        
        # Step 1: Move to hover position
        self.get_logger().info("Step 1: Moving to hover position...")
        
        # Compute IK for hover position using current joint angles as seed
        hover_computed_joint_angles = self.compute_ik_with_current_seed(
            hover_position.tolist(),
            ee_target_quat.tolist(),
            max_tries=5,
            dx=0.001
        )
        
        if hover_computed_joint_angles is None:
            self.get_logger().error("Failed to compute IK for hover position")
            return False
        
        self.get_logger().info(f"Computed joint angles for hover: {hover_computed_joint_angles}")
        
        # Create hover trajectory
        hover_trajectory = [{
            "positions": [float(x) for x in hover_computed_joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]
        
        success = self.execute_trajectory({"traj1": hover_trajectory})
        if not success:
            self.get_logger().error("Failed to reach hover position")
            return False
        
        self.get_logger().info("Reached hover position")
        time.sleep(0.5)
        
        # Update current joint angles after hover movement
        # Read joint angles again to use as seed for final position
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.get_logger().error("Could not read current joint angles after hover")
                return False
        
        # Step 2: Move down to final position
        self.get_logger().info("Step 2: Moving down to final position...")
        
        # Compute IK for final position using current joint angles as seed
        final_computed_joint_angles = self.compute_ik_with_current_seed(
            ee_target_position.tolist(),
            ee_target_quat.tolist(),
            max_tries=5,
            dx=0.001
        )
        
        if final_computed_joint_angles is None:
            self.get_logger().error("Failed to compute IK for final position")
            return False
        
        self.get_logger().info(f"Computed joint angles for final: {final_computed_joint_angles}")
        
        # Create final trajectory
        final_trajectory = [{
            "positions": [float(x) for x in final_computed_joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]
        
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
            duration=args.duration
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

