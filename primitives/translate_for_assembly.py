#!/usr/bin/env python3
"""
Translate for Assembly - Step 1: Move to hover position (translation only, no rotation)

The algorithm:
1. Read current base pose and target position from JSON (or accept as arguments)
2. Calculate target EE position to place object at target location
3. Keep current EE orientation unchanged (translation only)
4. Move to hover height (0.25m) above target position

Note: This is step 1 only. Step 2 (moving down to final position+force feedback) is handled separately.
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
        # self.get_logger().info(f"Assembly config loaded with {len(self.assembly_config.get('components', []))} components")
        # self.get_logger().info(f"Hover height set to: {HOVER_HEIGHT}m")
    
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
        # self.get_logger().info("Reading current joint angles...")
        
        # Reset the flag
        self.joint_angles_received = False
        
        # Wait for joint angles to arrive (with timeout)
        timeout_count = 0
        max_timeout = 100  # 10 seconds (100 * 0.1s)
        
        while rclpy.ok() and not self.joint_angles_received and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            
            # if timeout_count % 10 == 0:  # Log every second
            #     self.get_logger().info(f"Waiting for joint angles... ({timeout_count * 0.1:.1f}s)")
        
        if not self.joint_angles_received:
            self.get_logger().error("Timeout waiting for joint angles message")
            return None
        
        if self.current_joint_angles is None:
            self.get_logger().error("Joint angles data is None")
            return None
        
        # self.get_logger().info(f"Successfully read joint angles: {self.current_joint_angles}")
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
        # self.get_logger().info(f"Using current joint angles from joint state as seed: {q_guess}")
        
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
                    # self.get_logger().info(f"Quaternion-based IK succeeded with current joint angles seed (perturbation {i}), cost={cost:.6f}")
                    
                    # Verify orientation accuracy
                    # T_result = forward_kinematics(dh_params, result.x)
                    # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                    # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                    
                    return result.x
                
                # Keep track of best solution
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.x
        
        # If we found any reasonable solution, use it
        if best_result is not None and best_cost < 0.1:
            # self.get_logger().info(f"Using best quaternion-based IK solution with cost={best_cost:.6f}")
            
            # Verify orientation accuracy
            # T_result = forward_kinematics(dh_params, best_result)
            # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
            # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            return best_result
        
        # Fallback: Try multiple predefined seeds if current seed failed
        # self.get_logger().warn("IK failed with current joint angles as seed. Trying multiple predefined seeds...")
        
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
        
        # self.get_logger().info(f"Trying {len(seed_configs)} alternative seed configurations...")
        
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
                        # self.get_logger().info(f"IK succeeded with fallback seed {seed_idx+1}/{len(seed_configs)} (perturbation {i}), cost={cost:.6f}")
                        
                        # Verify orientation accuracy
                        # T_result = forward_kinematics(dh_params, result.x)
                        # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                        # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                        
                        return result.x
                    
                    # Keep track of best solution
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result.x
        
        # If we found any reasonable solution with fallback seeds, use it
        if best_result is not None and best_cost < 0.1:
            # self.get_logger().info(f"Using best fallback IK solution with cost={best_cost:.6f}")
            
            # Verify orientation accuracy
            # T_result = forward_kinematics(dh_params, best_result)
            # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
            # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            return best_result
        
        self.get_logger().error("IK failed: couldn't find solution even with multiple seeds")
        return None
    
    def translate_for_target(self, object_name, base_name, duration=20.0, 
                            final_base_pos=None, final_base_orientation=None,
                            current_object_pos=None, current_object_orientation=None,
                            target_object_pos=None):
        """
        Calculate and execute EE translation to hover position (translation only, no rotation).
        This is step 1 only - moves to hover height above target position.
        
        Algorithm:
        1. Get current base pose (T_base) - either from topic or provided
        2. Get target object position from JSON (relative to base) or provided directly
        3. Calculate grasp transformation (T_grasp)
        4. Calculate hover EE position (target XY, hover height above base)
        5. Keep current EE orientation unchanged (translation only)
        6. Move to hover position
        
        Args:
            object_name: Name of the object being held
            base_name: Name of the base object (e.g., 'base')
            duration: Duration for trajectory execution
            final_base_pos: Optional [x, y, z] final base position (overrides topic)
            final_base_orientation: Optional [x, y, z, w] final base orientation quaternion (overrides topic)
            current_object_pos: Optional [x, y, z] current object position (overrides topic)
            current_object_orientation: Optional [x, y, z, w] current object orientation quaternion (overrides topic)
            target_object_pos: Optional [x, y, z] target object position in world frame (overrides JSON)
        """
        # self.get_logger().info(f"Calculating translation for {object_name} relative to {base_name}")
        
        # Get current EE pose (always needed from topic)
        if self.current_ee_pose is None:
            self.get_logger().error("End-effector pose not available")
            return None
        
        # Use provided positions or get from topics
        if current_object_pos is not None:
            # Use provided object position
            if current_object_orientation is not None:
                # Create pose from provided position and orientation
                object_pose = PoseStamped()
                object_pose.pose.position.x = current_object_pos[0]
                object_pose.pose.position.y = current_object_pos[1]
                object_pose.pose.position.z = current_object_pos[2]
                object_pose.pose.orientation.x = current_object_orientation[0]
                object_pose.pose.orientation.y = current_object_orientation[1]
                object_pose.pose.orientation.z = current_object_orientation[2]
                object_pose.pose.orientation.w = current_object_orientation[3]
                T_object_current = self.pose_to_matrix(object_pose.pose)
                self.get_logger().info(f"Using provided object position and orientation: {current_object_pos}")
            else:
                # Try to get orientation from topics if position is provided but orientation is not
                if self.current_poses:
                    obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
                    if obj_key in self.current_poses:
                        # Get orientation from topic, use provided position
                        T_object_from_topic = self.transform_to_matrix(self.current_poses[obj_key].transform)
                        T_object_current = T_object_from_topic.copy()
                        T_object_current[:3, 3] = np.array(current_object_pos)  # Override position with provided
                        self.get_logger().info(f"Using provided object position, orientation from topic: {current_object_pos}")
                    else:
                        # Fallback: use identity orientation
                        T_object_current = np.eye(4)
                        T_object_current[:3, 3] = np.array(current_object_pos)
                        self.get_logger().warn(f"Using provided object position with identity orientation (object {object_name} not found in topics): {current_object_pos}")
                else:
                    # Fallback: use identity orientation
                    T_object_current = np.eye(4)
                    T_object_current[:3, 3] = np.array(current_object_pos)
                    self.get_logger().warn(f"Using provided object position with identity orientation (no topic data): {current_object_pos}")
        else:
            # Get from topics
            if not self.current_poses:
                self.get_logger().error("No pose data available")
                return None
            
            # Check if object exists
            obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
            if obj_key not in self.current_poses:
                self.get_logger().error(f"Object {object_name} not found")
                return None
            
            T_object_current = self.transform_to_matrix(self.current_poses[obj_key].transform)
        
        if final_base_pos is not None:
            # Use provided base position
            if final_base_orientation is not None:
                # Create pose from provided position and orientation
                base_pose = PoseStamped()
                base_pose.pose.position.x = final_base_pos[0]
                base_pose.pose.position.y = final_base_pos[1]
                base_pose.pose.position.z = final_base_pos[2]
                base_pose.pose.orientation.x = final_base_orientation[0]
                base_pose.pose.orientation.y = final_base_orientation[1]
                base_pose.pose.orientation.z = final_base_orientation[2]
                base_pose.pose.orientation.w = final_base_orientation[3]
                T_base_current = self.pose_to_matrix(base_pose.pose)
            else:
                # Use provided position with identity orientation
                T_base_current = np.eye(4)
                T_base_current[:3, 3] = np.array(final_base_pos)
            self.get_logger().info(f"Using provided base position: {final_base_pos}")
        else:
            # Get from topics
            if not self.current_poses:
                self.get_logger().error("No pose data available")
                return None
            
            # Check if base exists
            if base_name not in self.current_poses:
                base_name = f"{base_name}_scaled70"
                if base_name not in self.current_poses:
                    self.get_logger().error(f"Base {base_name} not found")
                    return None
            
            T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)
        
        # Convert EE pose to matrix
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        
        # Get current positions
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        object_current_position, object_current_rpy = self.matrix_to_rpy(T_object_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        
        # Get target object position - either provided directly or from JSON
        if target_object_pos is not None:
            # Use provided target position directly (assumed to be in world frame)
            target_object_position_abs = np.array(target_object_pos)
            self.get_logger().info(f"Using provided target object position: {target_object_pos}")
        else:
            # Get target object position from JSON (relative to base)
            target_position_relative = self.get_object_target_position(object_name)
            if target_position_relative is None:
                self.get_logger().error(f"No target position found for {object_name}")
                return None
            
            # Transform target position from base frame to world frame
            # If base is rotated, we need to apply base rotation to relative position
            R_base_current = T_base_current[:3, :3]
            target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        
        # Translation-only mode: keep current EE orientation unchanged
        ee_target_quat = R.from_matrix(T_EE_current[:3, :3]).as_quat()
        self.get_logger().info("Translation-only mode: keeping current EE orientation unchanged")
        
        # For translation-only: calculate position offset and apply to EE
        # Position offset = target_object_pos - current_object_pos
        position_offset = target_object_position_abs - object_current_position
        ee_target_position = ee_current_position + position_offset
        
        # Create hover position (same XY as target, but at HOVER_HEIGHT above base)
        hover_position = ee_target_position.copy()
        hover_position[2] = base_current_position[2] + HOVER_HEIGHT
        
        # Read current joint angles before computing IK
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.get_logger().error("Could not read current joint angles")
                return False
        
        # Step 1: Move to hover position (translation only, no rotation)
        self.get_logger().info(f"Moving to hover position: {hover_position} (keeping current EE orientation)")
        
        # Compute IK for hover position using current joint angles as seed
        # Keep current EE orientation unchanged (translation only)
        hover_computed_joint_angles = self.compute_ik_with_current_seed(
            hover_position.tolist(),
            ee_target_quat.tolist(),
            max_tries=5,
            dx=0.001
        )
        
        if hover_computed_joint_angles is None:
            self.get_logger().error("Failed to compute IK for hover position")
            return False
        
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
        
        self.get_logger().info("Reached hover position (step 1 complete)")
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
            
            # self.get_logger().info("Trajectory goal accepted, waiting for completion...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            
            if result.status == 4:  # SUCCEEDED
                # self.get_logger().info("Trajectory completed successfully")
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
    parser.add_argument('--duration', type=float, default=5.0, help='Movement duration in seconds')
    
    # Optional: Specify final base position and current object position directly
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'), 
                       help='Final base position [x, y, z] in meters (overrides topic)')
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                       help='Final base orientation quaternion [x, y, z, w] (overrides topic, optional)')
    parser.add_argument('--current-object-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Current object position [x, y, z] in meters (overrides topic)')
    parser.add_argument('--current-object-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                       help='Current object orientation quaternion [x, y, z, w] (overrides topic, optional)')
    parser.add_argument('--target-object-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Target object position [x, y, z] in world frame (overrides JSON)')
    
    args = parser.parse_args()
    
    rclpy.init()
    node = TranslateForAssembly()
    
    # node.get_logger().info("Waiting for action server...")
    node.action_client.wait_for_server()
    # node.get_logger().info("Action server available!")
    
    try:
        # Wait for pose data (wait indefinitely until received)
        # node.get_logger().info(f"Waiting for pose data for object: {args.object_name} and base: {args.base_name}")
        # start_time = time.time()
        # last_log_time = start_time
        
        # Only wait for poses if not provided via command line
        # Always need EE pose from topic
        need_base_from_topic = args.final_base_pos is None
        need_object_from_topic = args.current_object_pos is None
        
        # Wait for EE pose (always needed) and base/object poses if not provided
        while node.current_ee_pose is None or (need_base_from_topic or need_object_from_topic) and not node.current_poses:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
            
            # Log every 5 seconds to show we're still waiting
            # current_time = time.time()
            # if current_time - last_log_time >= 5.0:
            #     elapsed = current_time - start_time
            #     node.get_logger().info(f"Still waiting for pose data... ({elapsed:.1f}s elapsed)")
            #     last_log_time = current_time
        
        # elapsed = time.time() - start_time
        # node.get_logger().info(f"Received pose data for {len(node.current_poses)} objects (waited {elapsed:.1f}s)")
        
        # Execute translation (step 1 only: hover position, translation only, no rotation)
        success = node.translate_for_target(
            args.object_name,
            args.base_name,
            duration=args.duration,
            final_base_pos=args.final_base_pos,
            final_base_orientation=args.final_base_orientation,
            current_object_pos=args.current_object_pos,
            current_object_orientation=args.current_object_orientation,
            target_object_pos=args.target_object_pos
        )
        
        if success:
            node.get_logger().info("Translation successful!")
        else:
            node.get_logger().error("Translation failed")
        
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

