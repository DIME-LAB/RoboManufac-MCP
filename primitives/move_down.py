#!/usr/bin/env python3
"""
Move Down Primitive for UR5e - ROS2 Version
Moves robot down in Z-axis with force monitoring on all three axes (X, Y, Z).
Stops when any axis force exceeds -10N (like in simulation).
Supports both simulation and real robot modes via --mode argument.
"""

import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import sys
import os
import re
import json
import yaml
import argparse
import threading

# Add custom libraries to Python path
custom_lib_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries"
if custom_lib_path not in sys.path:
    sys.path.append(custom_lib_path)

try:
    from ik_solver import compute_ik, compute_ik_robust
except ImportError as e:
    print(f"Failed to import IK solver: {e}")
    sys.exit(1)

# Add primitives directory to path for action_libraries
primitives_path = os.path.dirname(os.path.abspath(__file__))
if primitives_path not in sys.path:
    sys.path.append(primitives_path)

try:
    from action_libraries import move_robust
except ImportError:
    # Fallback if action_libraries not in path
    move_robust = None

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
GRIPPER_FORCE_THRESHOLD = 100.0  # Stop when gripper force exceeds this value (for sim mode)
FORCE_THRESHOLD = -20.0  # Stop when any axis force exceeds (is less than) -10N (for real mode)
MOVEMENT_DURATION = 15.0  # Duration for smooth movement in seconds
FORCE_CHECK_INTERVAL = 0.02  # Check force every 20ms during movement
# =============================================================================

class MoveDown(Node):
    def __init__(self, target_height=None, mode='real'):
        super().__init__('move_down')
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Mode: 'sim' or 'real'
        self.mode = mode
        
        # Action client for trajectory control
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Target height (if None, will use current height - 0.2m as default)
        self.target_height = target_height
        
        # Force monitoring - different for sim vs real
        if self.mode == 'sim':
            # Sim mode: gripper force (single value)
            self.current_gripper_force = 0.0
        else:
            # Real mode: force readings for all three axes
            self.current_force_x = 0.0
            self.current_force_y = 0.0
            self.current_force_z = 0.0
        self.force_threshold_reached = False
        
        # Current goal handle for cancellation
        self._current_goal_handle = None
        
        # Timer for force monitoring during movement
        self.force_monitor_timer = None
        
        # Movement state
        self.moving = False
        
        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False
        
        # Subscriber for force/torque data - use different topics and message types for sim vs real
        if self.mode == 'sim':
            # Simulation mode: use gripper force topic (Float64)
            self.gripper_force_sub = self.create_subscription(
                Float64,
                '/gripper_force',
                self.gripper_force_callback,
                10
            )
            self.get_logger().info(f"Using mode: {self.mode}, gripper force topic: /gripper_force")
        else:
            # Real mode: use force/torque sensor topic (WrenchStamped)
            self.force_sub = self.create_subscription(
                WrenchStamped,
                '/force_torque_sensor_broadcaster/wrench',
                self.force_callback,
                10
            )
            self.get_logger().info(f"Using mode: {self.mode}, force/torque topic: /force_torque_sensor_broadcaster/wrench")
        
        # Subscriber for EE pose data (using same QoS as get_ee_pose.py)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            qos_profile
        )
        
        # Subscriber for joint states to get current joint angles (use as IK seed)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info("Waiting for action server...")
        self.action_client.wait_for_server()
        
        # Execute movement
        self.move_down()
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose data"""
        self.ee_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.ee_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        self.ee_pose_received = True
    
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

    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees - same as other primitives"""
        import math
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))
        else:
            pitch = math.degrees(math.asin(sinp))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        
        return [roll, pitch, yaw]

    def read_current_ee_pose(self):
        """Read current end-effector pose and joint angles using ROS2 subscriber"""
        self.get_logger().info("Reading current end-effector pose and joint angles...")
        
        # Reset the flags
        self.ee_pose_received = False
        self.joint_angles_received = False
        
        # Wait for both pose and joint angles to arrive (with timeout)
        timeout_count = 0
        max_timeout = 100  # 10 seconds (100 * 0.1s)
        
        while rclpy.ok() and (not self.ee_pose_received or not self.joint_angles_received) and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            
            if timeout_count % 10 == 0:  # Log every second
                status = []
                if not self.ee_pose_received:
                    status.append("EE pose")
                if not self.joint_angles_received:
                    status.append("joint angles")
                self.get_logger().info(f"Waiting for {' and '.join(status)}... ({timeout_count * 0.1:.1f}s)")
        
        if not self.ee_pose_received:
            self.get_logger().error("Timeout waiting for EE pose message")
            return None
        
        if not self.joint_angles_received:
            self.get_logger().error("Timeout waiting for joint angles message")
            return None
        
        if self.ee_position is None or self.ee_quat is None:
            self.get_logger().error("EE pose data is None")
            return None
        
        if self.current_joint_angles is None:
            self.get_logger().error("Joint angles data is None")
            return None
        
        # Extract position and orientation
        position = self.ee_position.tolist()
        orientation = self.ee_quat.tolist()
        
        self.get_logger().info(f"Successfully read pose: position={position}, orientation={orientation}")
        self.get_logger().info(f"Successfully read joint angles: {self.current_joint_angles}")
        
        return {
            'position': position,
            'orientation': orientation
        }

    def gripper_force_callback(self, msg: Float64):
        """Callback for gripper force data (sim mode)"""
        self.current_gripper_force = msg.data
        
        # Check if force threshold is exceeded
        if self.current_gripper_force > GRIPPER_FORCE_THRESHOLD and not self.force_threshold_reached:
            self.get_logger().warn(f"Gripper force threshold reached! Force: {self.current_gripper_force:.2f} (threshold: {GRIPPER_FORCE_THRESHOLD})")
            self.force_threshold_reached = True
            self.emergency_stop()
    
    def force_callback(self, msg: WrenchStamped):
        """Callback for force/torque sensor data - monitors all three axes (real mode)"""
        # Extract force components for all three axes
        self.current_force_x = msg.wrench.force.x
        self.current_force_y = msg.wrench.force.y
        self.current_force_z = msg.wrench.force.z
        
        # Check if any axis force exceeds threshold (is less than -10N)
        if not self.force_threshold_reached:
            if self.current_force_x <= FORCE_THRESHOLD:
                self.get_logger().warn(f"Force threshold reached on X axis! Force: {self.current_force_x:.2f}N (threshold: {FORCE_THRESHOLD}N)")
                self.force_threshold_reached = True
                self.emergency_stop()
            elif self.current_force_y <= FORCE_THRESHOLD:
                self.get_logger().warn(f"Force threshold reached on Y axis! Force: {self.current_force_y:.2f}N (threshold: {FORCE_THRESHOLD}N)")
                self.force_threshold_reached = True
                self.emergency_stop()
            elif self.current_force_z <= FORCE_THRESHOLD:
                self.get_logger().warn(f"Force threshold reached on Z axis! Force: {self.current_force_z:.2f}N (threshold: {FORCE_THRESHOLD}N)")
                self.force_threshold_reached = True
                self.emergency_stop()

    def start_force_monitoring(self):
        """Start monitoring force during trajectory execution"""
        if self.force_monitor_timer is None:
            self.force_monitor_timer = self.create_timer(FORCE_CHECK_INTERVAL, self.check_force_during_execution)
            if self.mode == 'sim':
                self.get_logger().info("Started gripper force monitoring during execution")
            else:
                self.get_logger().info("Started force monitoring during execution (all axes)")

    def stop_force_monitoring(self):
        """Stop force monitoring"""
        if self.force_monitor_timer:
            self.force_monitor_timer.cancel()
            self.force_monitor_timer = None
            self.get_logger().info("Stopped force monitoring")

    def check_force_during_execution(self):
        """Check force during trajectory execution and cancel if threshold exceeded"""
        if not self.moving:
            return
        
        if self.mode == 'sim':
            # Sim mode: check gripper force
            if self.current_gripper_force > GRIPPER_FORCE_THRESHOLD:
                self.get_logger().error(f"EMERGENCY STOP: Gripper force threshold exceeded during movement! Force: {self.current_gripper_force:.2f}")
                self.force_threshold_reached = True
                self.emergency_stop()
            else:
                # Debug: log force values during movement
                self.get_logger().debug(f"Force monitoring: Gripper force = {self.current_gripper_force:.2f} (threshold: {GRIPPER_FORCE_THRESHOLD})")
        else:
            # Real mode: check all three axes
            if self.current_force_x <= FORCE_THRESHOLD:
                self.get_logger().error(f"EMERGENCY STOP: Force threshold exceeded on X axis during movement! Force: {self.current_force_x:.2f}N")
                self.force_threshold_reached = True
                self.emergency_stop()
            elif self.current_force_y <= FORCE_THRESHOLD:
                self.get_logger().error(f"EMERGENCY STOP: Force threshold exceeded on Y axis during movement! Force: {self.current_force_y:.2f}N")
                self.force_threshold_reached = True
                self.emergency_stop()
            elif self.current_force_z <= FORCE_THRESHOLD:
                self.get_logger().error(f"EMERGENCY STOP: Force threshold exceeded on Z axis during movement! Force: {self.current_force_z:.2f}N")
                self.force_threshold_reached = True
                self.emergency_stop()
            else:
                # Debug: log force values during movement
                self.get_logger().debug(f"Force monitoring: X={self.current_force_x:.2f}N, Y={self.current_force_y:.2f}N, Z={self.current_force_z:.2f}N (threshold: {FORCE_THRESHOLD}N)")

    def emergency_stop(self):
        """Emergency stop - cancel current trajectory and stop immediately, then exit"""
        self.moving = False
        self.get_logger().error("EMERGENCY STOP: Force threshold exceeded. Exiting...")
        
        # Stop force monitoring
        self.stop_force_monitoring()
        
        # Cancel the current goal if it exists
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
                self.get_logger().info("Trajectory cancellation requested")
            except Exception as e:
                self.get_logger().error(f"Failed to cancel trajectory: {e}")
        
        # Force immediate exit - use a timer to shutdown after brief delay to allow logging
        def delayed_shutdown():
            try:
                rclpy.shutdown()
            except:
                pass
            # Force exit if rclpy.shutdown() doesn't work
            os._exit(0)
        
        # Start shutdown in a separate thread after 100ms to allow log messages to flush
        shutdown_timer = threading.Timer(0.1, delayed_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

    def move_down(self):
        """Move down while maintaining current position and orientation"""
        # Read current end-effector pose
        self.get_logger().info("Reading current end-effector pose...")
        pose_data = self.read_current_ee_pose()
        
        if pose_data is None:
            self.get_logger().error("Could not read current end-effector pose")
            rclpy.shutdown()
            return
            
        current_pos = pose_data['position']
        current_quat = pose_data['orientation']
        
        # Convert quaternion directly to rotation matrix to avoid precision loss from RPY conversion
        from scipy.spatial.transform import Rotation as Rot
        
        # Keep the current orientation (don't change it, just move down)
        # Convert quaternion directly to rotation matrix for more accurate IK
        target_rotation = Rot.from_quat(current_quat)
        target_rot_matrix = target_rotation.as_matrix()
        
        # Also compute RPY for logging purposes
        current_rpy = self.quaternion_to_rpy(
            current_quat[0], current_quat[1], 
            current_quat[2], current_quat[3]
        )
        
        self.get_logger().info(f"Current EE position: {current_pos}")
        self.get_logger().info(f"Current EE quaternion: {current_quat}")
        self.get_logger().info(f"Current EE RPY (deg): {current_rpy}")
        self.get_logger().info(f"Target orientation: keeping current quaternion (no RPY conversion)")

        # Create target position - move down (decrease Z)
        target_position = current_pos.copy()
        
        if self.target_height is not None:
            target_position[2] = self.target_height
            self.get_logger().info(f"Using specified target height: {self.target_height}m")
        else:
            # Default: move down 0.2m from current position
            target_position[2] = current_pos[2] - 0.2
            self.get_logger().info(f"Using default: moving down 0.2m to {target_position[2]:.3f}m")
        
        self.get_logger().info(f"Target position: {target_position}")
        if self.mode == 'sim':
            self.get_logger().info(f"Gripper force threshold: {GRIPPER_FORCE_THRESHOLD}")
        else:
            self.get_logger().info(f"Force threshold: {FORCE_THRESHOLD}N (any axis)")

        # Compute inverse kinematics for target pose
        # Use quaternion directly converted to rotation matrix for more accurate orientation preservation
        try:
            from scipy.optimize import minimize
            from scipy.spatial.transform import Rotation as Rot
            from ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
            
            # Create target pose with quaternion-derived rotation matrix
            target_pose = np.eye(4)
            target_pose[:3, 3] = target_position
            target_pose[:3, :3] = target_rot_matrix
            
            self.get_logger().info(f"Computing IK for position: {target_position}")
            self.get_logger().info(f"Using quaternion-derived rotation matrix directly (NO RPY conversion)")
            
            # Use quaternion-based IK directly - no RPY conversion at all!
            # Since we're only moving down, current joint angles should be very close to the solution
            joint_angles = None
            best_result = None
            best_cost = float('inf')
            max_tries = 5
            dx = 0.001
            
            # Primary seed: use current joint angles from joint state subscription
            # This is the best seed since we're only moving down (small Z change)
            if self.current_joint_angles is None:
                self.get_logger().error("Current joint angles not available! Cannot compute IK.")
                rclpy.shutdown()
                return
            
            q_guess = self.current_joint_angles.copy()
            self.get_logger().info(f"Using current joint angles from joint state as seed: {q_guess}")
            
            # Try IK with current joint angles and position perturbations
            solution_found = False
            for i in range(max_tries):
                if solution_found:
                    break
                    
                # Try small x-shift each iteration (helps with workspace boundaries)
                perturbed_position = np.array(target_position).copy()
                perturbed_position[0] += i * dx
                
                perturbed_pose = target_pose.copy()
                perturbed_pose[:3, 3] = perturbed_position
                
                joint_bounds = [(-np.pi, np.pi)] * 6
                
                # Use quaternion-based objective directly - NO RPY conversion!
                result = minimize(ik_objective_quaternion, q_guess, args=(perturbed_pose,), 
                                method='L-BFGS-B', bounds=joint_bounds)
                
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed_pose)
                    
                    # Check if this is a good solution
                    if cost < 0.01:
                        self.get_logger().info(f"Quaternion-based IK succeeded with current joint angles seed (perturbation {i}), cost={cost:.6f}")
                        joint_angles = result.x
                        solution_found = True
                        
                        # Verify orientation accuracy
                        T_result = forward_kinematics(dh_params, joint_angles)
                        orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                        self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                        break
                    
                    # Keep track of best solution
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result.x
            
            # If we found any reasonable solution, use it
            if joint_angles is None and best_result is not None and best_cost < 0.1:
                self.get_logger().info(f"Using best quaternion-based IK solution with cost={best_cost:.6f}")
                joint_angles = best_result
                
                # Verify orientation accuracy
                T_result = forward_kinematics(dh_params, joint_angles)
                orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            if joint_angles is None:
                self.get_logger().error("IK failed: couldn't compute move down position")
                rclpy.shutdown()
                return
                
            self.get_logger().info(f"Computed joint angles: {joint_angles}")
            
            # Create trajectory point directly from joint angles (we already computed them from quaternion-based IK)
            # No need for move_robust since we have the exact joint angles
            point = JointTrajectoryPoint(
                positions=[float(x) for x in joint_angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=int(MOVEMENT_DURATION))
            )
            
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = [point]
            
            # Create and send trajectory
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.get_logger().info("Sending trajectory to move down...")
            self.moving = True
            
            # Start force monitoring
            self.start_force_monitoring()
            
            # Send goal
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.get_logger().error(f"Failed to compute IK: {e}")
            rclpy.shutdown()

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.stop_force_monitoring()
            self.moving = False
            rclpy.shutdown()
            return

        self._current_goal_handle = goal_handle
        if self.mode == 'sim':
            self.get_logger().info("Move down trajectory accepted. Monitoring gripper force...")
        else:
            self.get_logger().info("Move down trajectory accepted. Monitoring force on all axes...")
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        try:
            result = future.result()
            self.stop_force_monitoring()
            self.moving = False
            self._current_goal_handle = None
            
            if result.status == 1:  # SUCCEEDED
                if self.force_threshold_reached:
                    self.get_logger().info("Movement completed: Force threshold reached on one or more axes")
                else:
                    self.get_logger().info("Movement completed successfully")
            elif result.status == 5:  # PREEMPTED (cancelled)
                self.get_logger().info("Trajectory was preempted (likely due to force threshold)")
                if self.force_threshold_reached:
                    self.get_logger().info("Force threshold reached. Movement stopped.")
            elif result.status == 4:  # ABORTED
                self.get_logger().warn("Trajectory was aborted by controller")
                if self.force_threshold_reached:
                    self.get_logger().info("Force threshold reached. Movement stopped.")
            else:
                self.get_logger().error(f"Trajectory failed with status: {result.status}")
                
        except Exception as e:
            self.get_logger().error(f"Error in goal result callback: {e}")
        finally:
            self.stop_force_monitoring()
            self.moving = False
            rclpy.shutdown()

def main(args=None):
    parser = argparse.ArgumentParser(description='Move Down Primitive with force monitoring on all axes')
    parser.add_argument('--height', type=float, default=None,
                       help='Target height in meters (optional, defaults to current height - 0.2m if not provided)')
    parser.add_argument('--mode', type=str, default='real', choices=['sim', 'real'],
                       help='Mode: "sim" for simulation, "real" for real robot (default: real)')
    
    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()
    
    rclpy.init(args=args)
    node = MoveDown(known_args.height, known_args.mode)
    
    try:
        # Use regular spin() like move_down_yoloe.py - rclpy.shutdown() will cause spin to exit
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Move down interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Error in move down: {e}")
    finally:
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass  # Context already shut down

if __name__ == '__main__':
    main()

