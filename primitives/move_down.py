#!/usr/bin/env python3
"""
Move Down Primitive for UR5e - ROS2 Async Version
Moves robot down in Z-axis with real-time force monitoring and immediate trajectory cancellation.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import sys
import yaml

# Add path to your ur_asu package
main_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main"
if main_path not in sys.path:
    sys.path.append(main_path)

from ur_asu.custom_libraries.actionlibraries import move
from ur_asu.custom_libraries.ik_solver import compute_ik

# =============================================================================
# CONFIGURABLE PARAMETERS - CHANGE THESE AS NEEDED
# =============================================================================
FORCE_Z_THRESHOLD = -5.0  # Force threshold in Z direction (Newtons, negative = downward force)
FINAL_Z_POSITION = 0.148  # Final Z position in meters
MOVEMENT_DURATION = 5.0   # Duration for smooth movement in seconds
FORCE_CHECK_INTERVAL = 0.02  # Check force every 20ms during movement - more responsive
# =============================================================================

class MoveDown(Node):
    def __init__(self):
        super().__init__('move_down')
        
        # Robot control setup
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Action client for trajectory control
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Publisher for direct joint commands (for emergency stop)
        self.joint_pub = self.create_publisher(
            Float64MultiArray, 
            '/forward_position_controller/commands', 
            10
        )
        
        # Subscriber for force/torque data
        self.force_sub = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.force_callback,
            10
        )
        
        # Subscriber for current end-effector pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.pose_callback,
            10
        )
        
        # Current force reading
        self.current_force_z = 0.0
        self.force_detected = False
        
        # Current end-effector pose
        self.current_ee_pose = None
        self.pose_received = False
        
        # Movement tracking
        self.current_position = None
        self.start_position = None
        self.total_movement = 0.0
        self.moving = False
        
        # Current goal handle for cancellation
        self._current_goal_handle = None
        
        # Timer for force monitoring during movement
        self.force_monitor_timer = None
        
        # State management
        self.sequence_started = False
        self.start_timer = None
        self.retry_count = 0
        self.max_retries = 3  # Maximum retries per position
        
        # Pose timeout management
        self.pose_timeout_count = 0
        self.max_pose_timeout = 50  # 5 seconds timeout (50 * 0.1s)
        
        # Wait for action server
        self.get_logger().info("Waiting for action server...")
        self.action_client.wait_for_server()
        
        # Start the move down sequence after a brief delay
        self.start_timer = self.create_timer(1.0, self.start_move_down)
    
    def force_callback(self, msg: WrenchStamped):
        """Callback for force/torque sensor data"""
        # Extract Z force component
        self.current_force_z = msg.wrench.force.z
        
        # Check if force threshold is exceeded
        if self.current_force_z <= FORCE_Z_THRESHOLD and not self.force_detected:
            self.get_logger().warn(f"Force threshold reached! Z force: {self.current_force_z:.2f}N (threshold: {FORCE_Z_THRESHOLD}N)")
            self.force_detected = True
            self.emergency_stop()
    
    def pose_callback(self, msg: PoseStamped):
        """Callback for current end-effector pose"""
        self.current_ee_pose = msg.pose
        self.pose_received = True
    
    def check_pose_and_start(self):
        """Check if pose is received and start movement with timeout"""
        self.pose_timeout_count += 1
        
        if self.pose_received:
            # Cancel this timer
            if hasattr(self, '_pose_check_timer'):
                self._pose_check_timer.cancel()
            # Start the movement
            self.start_move_down()
        elif self.pose_timeout_count >= self.max_pose_timeout:
            # Timeout reached, try fallback method
            self.get_logger().warn("Pose timeout reached. Trying fallback method...")
            if hasattr(self, '_pose_check_timer'):
                self._pose_check_timer.cancel()
            self.get_pose_fallback()
        else:
            # Still waiting, log progress
            if self.pose_timeout_count % 10 == 0:  # Log every second
                self.get_logger().info(f"Still waiting for pose... ({self.pose_timeout_count * 0.1:.1f}s)")
    
    def get_pose_fallback(self):
        """Fallback method to get current pose using ROS2 command line"""
        try:
            import subprocess
            
            self.get_logger().info("Attempting to get pose using ROS2 command line...")
            
            # Use ros2 topic echo to get pose data
            result = subprocess.run([
                'bash', '-c', 
                'source /opt/ros/humble/setup.bash && ros2 topic echo /tcp_pose_broadcaster/pose --once'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.get_logger().error(f"Failed to get pose via command line: {result.stderr}")
                rclpy.shutdown()
                return
            
            # Parse the YAML output
            data = yaml.safe_load(result.stdout)
            
            if 'pose' not in data:
                self.get_logger().error("No pose data found in command output")
                rclpy.shutdown()
                return
            
            pose_data = data['pose']
            
            # Create a mock pose object
            class MockPose:
                def __init__(self, pos_data, ori_data):
                    self.position = type('obj', (object,), {
                        'x': pos_data['x'],
                        'y': pos_data['y'], 
                        'z': pos_data['z']
                    })()
                    self.orientation = type('obj', (object,), {
                        'x': ori_data['x'],
                        'y': ori_data['y'],
                        'z': ori_data['z'],
                        'w': ori_data['w']
                    })()
            
            self.current_ee_pose = MockPose(pose_data['position'], pose_data['orientation'])
            self.pose_received = True
            
            self.get_logger().info(f"Successfully got pose via fallback method:")
            self.get_logger().info(f"Position: x={self.current_ee_pose.position.x:.3f}, y={self.current_ee_pose.position.y:.3f}, z={self.current_ee_pose.position.z:.3f}")
            
            # Start the movement
            self.start_move_down()
            
        except Exception as e:
            self.get_logger().error(f"Failed to get pose via fallback method: {e}")
            rclpy.shutdown()
    
    def emergency_stop(self):
        """Emergency stop - cancel current trajectory and stop immediately"""
        self.moving = False
        self.get_logger().error("EMERGENCY STOP: Cancelling current trajectory")
        
        # Stop force monitoring
        if self.force_monitor_timer:
            self.force_monitor_timer.cancel()
            self.force_monitor_timer = None
        
        # Cancel the current goal if it exists
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
                self.get_logger().info("Trajectory cancellation requested")
            except Exception as e:
                self.get_logger().error(f"Failed to cancel trajectory: {e}")
        
        # Send immediate stop command to hold current position
        self.send_immediate_stop()
    
    def send_immediate_stop(self):
        """Send immediate stop command to hold current position"""
        try:
            # Get current joint positions from joint states (approximate)
            # For now, use a safe position
            safe_position = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
            
            stop_msg = Float64MultiArray()
            stop_msg.data = safe_position
            self.joint_pub.publish(stop_msg)
            
            self.get_logger().info("Immediate stop command sent")
        except Exception as e:
            self.get_logger().error(f"Failed to send immediate stop: {e}")
    
    def start_move_down(self):
        """Start the move down sequence"""
        # Prevent multiple starts
        if self.sequence_started:
            return
        
        self.sequence_started = True
        
        # Cancel the start timer
        if self.start_timer:
            self.start_timer.cancel()
            self.start_timer = None
        
        self.get_logger().info("Starting move down sequence...")
        self.get_logger().info(f"Target force threshold: {FORCE_Z_THRESHOLD}N in Z direction")
        self.get_logger().info(f"Final Z position: {FINAL_Z_POSITION}m")
        self.get_logger().info(f"Movement duration: {MOVEMENT_DURATION}s")
        self.get_logger().info(f"Force check interval: {FORCE_CHECK_INTERVAL}s")
        
        # Wait for current pose to be received
        if not self.pose_received:
            self.get_logger().info("Waiting for current end-effector pose...")
            # Create a timer to check for pose and start movement (with timeout)
            self._pose_check_timer = self.create_timer(0.1, self.check_pose_and_start)
            return
        
        # Use actual current position from pose broadcaster
        self.current_position = [
            self.current_ee_pose.position.x,
            self.current_ee_pose.position.y,
            self.current_ee_pose.position.z
        ]
        self.start_position = self.current_position.copy()
        
        self.get_logger().info(f"Starting from actual position: {self.current_position}")
        self.get_logger().info(f"Target final position: Z = {FINAL_Z_POSITION}m")
        
        # Calculate target position (keep X and Y, set Z to final position)
        target_position = [
            self.current_position[0],
            self.current_position[1], 
            FINAL_Z_POSITION
        ]
        
        self.get_logger().info(f"Moving directly to target position: {target_position}")
        
        # Move directly to target position with force monitoring
        self.move_to_position_with_force_monitoring(target_position)
    
    
    def move_to_position_with_force_monitoring(self, position):
        """Move to position with real-time force monitoring and cancellation"""
        self.moving = True
        
        # Use IK to get joint angles
        rpy = [0, 180, 0]  # HOME_POSE orientation from actionlibraries
        self.get_logger().info(f"Computing IK for position: {position} with orientation: {rpy}")
        joint_angles = compute_ik(position, rpy)
        
        if joint_angles is None:
            self.get_logger().error(f"Failed to compute IK for position {position}")
            self.moving = False
            rclpy.shutdown()
            return
        
        self.get_logger().info(f"IK computed successfully. Joint angles: {[f'{angle:.3f}' for angle in joint_angles]}")
        
        # Create trajectory
        self.get_logger().info(f"Creating trajectory for position: {position}")
        trajectory_points = move(position, rpy, MOVEMENT_DURATION)  # Use configurable duration for smooth movement
        
        if not trajectory_points:
            self.get_logger().error(f"Failed to generate trajectory for position {position}")
            self.moving = False
            rclpy.shutdown()
            return
        
        self.get_logger().info(f"Trajectory created successfully with {len(trajectory_points)} points")
        
        # Execute trajectory with real-time force monitoring
        self.execute_trajectory_with_force_monitoring(trajectory_points, position)
    
    def execute_trajectory_with_force_monitoring(self, trajectory_points, target_position):
        """Execute trajectory with real-time force monitoring and cancellation"""
        # Create trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Convert trajectory points to JointTrajectoryPoint format
        for point_data in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = point_data["positions"]
            point.velocities = point_data["velocities"]
            point.time_from_start = point_data["time_from_start"]
            trajectory.points.append(point)
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=1)
        
        # Send goal
        self.get_logger().info("Sending trajectory goal...")
        self._send_goal_future = self.action_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(
            lambda future: self.goal_response_callback(future, target_position)
        )
        
        # Start force monitoring during execution
        self.start_force_monitoring()
    
    def start_force_monitoring(self):
        """Start monitoring force during trajectory execution"""
        if self.force_monitor_timer is None:
            self.force_monitor_timer = self.create_timer(FORCE_CHECK_INTERVAL, self.check_force_during_execution)
            self.get_logger().info("Started force monitoring during execution")
    
    def stop_force_monitoring(self):
        """Stop force monitoring"""
        if self.force_monitor_timer:
            self.force_monitor_timer.cancel()
            self.force_monitor_timer = None
            self.get_logger().info("Stopped force monitoring")
    
    def check_force_during_execution(self):
        """Check force during trajectory execution and cancel if threshold exceeded"""
        if self.moving and self.current_force_z <= FORCE_Z_THRESHOLD:
            self.get_logger().error(f"EMERGENCY STOP: Force threshold exceeded during movement! Z force: {self.current_force_z:.2f}N")
            self.force_detected = True
            self.emergency_stop()
        elif self.moving:
            # Debug: log force values during movement
            self.get_logger().debug(f"Force monitoring: Z force = {self.current_force_z:.2f}N (threshold: {FORCE_Z_THRESHOLD}N)")
    
    def goal_response_callback(self, future, target_position):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected!")
            self.stop_force_monitoring()
            self.moving = False
            rclpy.shutdown()
            return
        
        self._current_goal_handle = goal_handle
        self.get_logger().info("Trajectory goal accepted. Monitoring force...")
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(
            lambda future: self.goal_result_callback(future, target_position)
        )
    
    def goal_result_callback(self, future, target_position):
        """Handle goal completion"""
        try:
            result = future.result()
            self.stop_force_monitoring()
            self.moving = False
            self._current_goal_handle = None
            
            if result.status == 1:  # SUCCEEDED
                self.get_logger().info("Trajectory completed successfully")
                
                # Update current position
                self.current_position = target_position
                self.total_movement = self.start_position[2] - self.current_position[2]
                
                self.get_logger().info(f"Movement completed. Final position: {self.current_position}")
                self.get_logger().info(f"Total Z movement: {self.total_movement:.3f}m")
                
                if self.force_detected:
                    self.get_logger().info("Force threshold reached during movement.")
                else:
                    self.get_logger().info("Movement completed without force threshold being reached.")
                
                rclpy.shutdown()
            elif result.status == 5:  # PREEMPTED (cancelled)
                self.get_logger().info("Trajectory was preempted (likely due to force threshold)")
                if self.force_detected:
                    self.get_logger().info("Force threshold reached. Movement complete.")
                else:
                    self.get_logger().info("Trajectory was cancelled for unknown reason.")
                rclpy.shutdown()
            elif result.status == 4:  # ABORTED
                self.get_logger().warn("Trajectory was aborted by controller")
                if self.force_detected:
                    self.get_logger().info("Force threshold reached. Movement complete.")
                else:
                    self.get_logger().error("Trajectory was aborted without force threshold being reached.")
                rclpy.shutdown()
            else:
                self.get_logger().error(f"Trajectory failed with status: {result.status}")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"Error in goal result callback: {e}")
            self.stop_force_monitoring()
            self.moving = False
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MoveDown()
    
    try:
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
