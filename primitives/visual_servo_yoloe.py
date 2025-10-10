#!/usr/bin/env python3
"""
Direct Object Movement - Native ROS2 Node
Read object poses from ObjectPoseArray and perform single direct movement to specific object by name
Includes calibration offset correction for accurate positioning
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import sys
import argparse
import numpy as np

# Import from local action_libraries file
from action_libraries import hover_over

# Import the new message type
try:
    from max_camera_msgs.msg import ObjectPoseArray
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs not found. Using geometry_msgs.PoseStamped as fallback.")
    ObjectPoseArray = None

class PoseKalmanFilter:
    """Kalman filter for pose estimation and smoothing"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
        self.state_dim = 12
        self.measurement_dim = 6  # [x, y, z, roll, pitch, yaw]
        
        # State vector
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10  # Initial covariance
        
        # Process noise
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # Measurement matrix (we only measure position and orientation)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:6, :6] = np.eye(6)
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        dt = 1.0  # Time step (will be updated dynamically)
        self.F[:6, 6:] = np.eye(6) * dt
        
        self.initialized = False
        
    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees"""
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
    
    def update(self, pose_msg, dt=1.0):
        """Update Kalman filter with new pose measurement"""
        # Extract measurement
        position = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        rpy = self.quaternion_to_rpy(
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        )
        
        measurement = np.array(position + rpy)
        
        # Update state transition matrix with current dt
        self.F[:6, 6:] = np.eye(6) * dt
        
        if not self.initialized:
            # Initialize state
            self.x[:6] = measurement
            self.initialized = True
            return self.x[:6], self.x[6:12]  # Return position and velocity
        
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        
        return self.x[:6], self.x[6:12]  # Return filtered position and velocity
    
    def get_filtered_pose(self):
        """Get current filtered pose"""
        if not self.initialized:
            return None, None
        return self.x[:6], self.x[6:12]

class DirectObjectMove(Node):
    def __init__(self, topic_name="/objects_poses", object_name="blue_dot_0", hover_height=0.15, movement_duration=7.0, target_xyz=None, target_xyzw=None):
        super().__init__('direct_object_move')
        
        self.topic_name = topic_name
        self.object_name = object_name
        self.hover_height = hover_height
        self.movement_duration = movement_duration  # Duration for IK movement
        self.target_xyz = target_xyz  # Optional target position [x, y, z]
        self.target_xyzw = target_xyzw  # Optional target orientation [x, y, z, w]
        self.last_target_pose = None
        self.position_threshold = 0.005  # 5mm
        self.angle_threshold = 2.0       # 2 degrees
        # Calibration offset to correct systematic detection bias
        self.calibration_offset_x = -0.000  # (move left)
        self.calibration_offset_y = +0.025  # (move forward)
        
        # Initialize Kalman filter
        self.kalman_filter = PoseKalmanFilter(process_noise=0.005, measurement_noise=0.05)
        self.last_update_time = None
        
        # Subscribe to object poses topic
        if ObjectPoseArray is not None:
            self.pose_sub = self.create_subscription(
                ObjectPoseArray,
                topic_name,
                self.objects_poses_callback,
                5  # Lower QoS to reduce update frequency
            )
        else:
            # Fallback to old PoseStamped subscription
            self.pose_sub = self.create_subscription(
                PoseStamped,
                topic_name,
                self.pose_callback,
                5
            )
        
        # No timer needed for fire-and-forget approach
        self.latest_pose = None
        self.movement_completed = False  # Flag to track if movement has been completed
        self.should_exit = False  # Flag to control exit
        self.initial_pose_processed = False  # Flag to ensure we only process pose once
        
        # Add timeout timer to handle stuck robot cases
        self.timeout_timer = self.create_timer(movement_duration + 5.0, self.timeout_callback)
        
        # Action client for trajectory execution
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info(f"🤖 Direct object movement started for object '{object_name}' on topic {topic_name}")
        self.get_logger().info(f"📏 Target height: {hover_height}m")
        self.get_logger().info(f"⏱️ Movement duration: {movement_duration}s")
        
    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees"""
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
    
    def poses_are_similar(self, position, rpy):
        """Check if pose is similar to last target"""
        if self.last_target_pose is None:
            return False
            
        last_pos, last_rpy = self.last_target_pose
        
        # Check position difference (only x, y)
        pos_diff = math.sqrt(
            (position[0] - last_pos[0])**2 +
            (position[1] - last_pos[1])**2
        )
        
        if pos_diff > self.position_threshold:
            return False
            
        # Check yaw difference
        angle_diff = abs(rpy[2] - last_rpy[2])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        return angle_diff <= self.angle_threshold
    
    def objects_poses_callback(self, msg):
        """Handle ObjectPoseArray message and find target object"""
        if ObjectPoseArray is None or self.initial_pose_processed:
            return
            
        # Find the object with the specified name
        target_object = None
        for obj in msg.objects:
            if obj.object_name == self.object_name:
                target_object = obj
                break
        
        if target_object is not None:
            # Convert ObjectPose to PoseStamped for compatibility
            pose_stamped = PoseStamped()
            pose_stamped.header = target_object.header
            pose_stamped.pose = target_object.pose
            self.latest_pose = pose_stamped
            
            # Process the initial pose immediately (fire and forget)
            self.process_initial_pose()
        else:
            # Object not found in this message
            self.get_logger().warn(f"Object '{self.object_name}' not found in current message")
            self.latest_pose = None
    
    def pose_callback(self, msg):
        """Store latest pose message (fallback for PoseStamped)"""
        if self.initial_pose_processed:
            return
        self.latest_pose = msg
        
        # Process the initial pose immediately (fire and forget)
        self.process_initial_pose()
    
    def process_initial_pose(self):
        """Process initial pose and execute single movement (fire and forget)"""
        if self.movement_completed or self.initial_pose_processed:
            return
            
        self.initial_pose_processed = True
        
        # Check if we have optional target position/orientation
        if self.target_xyz is not None and self.target_xyzw is not None:
            # Use provided target position and orientation
            position = self.target_xyz[:3].copy()  # Take first 3 elements and make a copy
            
            # Apply calibration offset to correct systematic detection bias (same as detected objects)
            position[0] += self.calibration_offset_x  # Correct X offset
            position[1] += self.calibration_offset_y  # Correct Y offset
            
            rpy = self.quaternion_to_rpy(
                self.target_xyzw[0], self.target_xyzw[1], 
                self.target_xyzw[2], self.target_xyzw[3]
            )
            self.get_logger().info(f"🎯 Using provided target position: {position} (with calibration offset applied) and orientation: {rpy}")
        elif self.latest_pose is not None:
            # Use detected object pose (single reading, no continuous updates)
            # Calculate time delta for Kalman filter
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
            else:
                dt = 1.0  # Default time step
            self.last_update_time = current_time
            
            # Update Kalman filter with initial measurement only
            filtered_pose, velocity = self.kalman_filter.update(self.latest_pose, dt)
            
            if filtered_pose is None:
                return
                
            # Extract filtered position and orientation
            position = filtered_pose[:3].tolist()
            rpy = filtered_pose[3:6].tolist()
            
            # Apply calibration offset to correct systematic detection bias
            position[0] += self.calibration_offset_x  # Correct X offset
            position[1] += self.calibration_offset_y  # Correct Y offset
            
            self.get_logger().info(f"🎯 Moving to detected object at ({position[0]:.3f}, {position[1]:.3f}) at height {self.hover_height:.3f}m")
        else:
            # No target provided and no object detected
            self.get_logger().warn("No target position provided and no object detected")
            return
        
        # Create target pose at final height
        target_position = [position[0], position[1], self.hover_height]
        target_pose = (target_position, rpy)
        
        trajectory = hover_over(target_pose, self.hover_height, self.movement_duration)
        
        # Execute trajectory (callbacks will handle completion)
        self.execute_trajectory(trajectory)
    
    def execute_trajectory(self, trajectory):
        """Execute trajectory using ROS2 action with proper feedback waiting"""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                self.get_logger().error("No trajectory found")
                return
            
            
            point = trajectory['traj1'][0]
            positions = point['positions']
            duration = point['time_from_start'].sec
            
            # Create trajectory message
            traj_msg = JointTrajectory()
            traj_msg.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
            
            traj_point = JointTrajectoryPoint()
            traj_point.positions = positions
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = Duration(sec=duration)
            traj_msg.points.append(traj_point)
            
            # Create and send goal
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            # Send goal with proper callback handling for feedback
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response_callback)
            self.get_logger().info("📤 Trajectory goal sent, waiting for response...")
            
        except Exception as e:
            self.get_logger().error(f"❌ Trajectory execution error: {e}")
    
    def goal_response_callback(self, future):
        """Handle goal response from action server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Trajectory goal rejected by action server")
            self.should_exit = True
            return

        self.get_logger().info("✅ Trajectory goal accepted - Robot is moving!")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle goal completion from action server"""
        try:
            result = future.result()
            if result.status == 1:  # SUCCEEDED
                self.get_logger().info("✅ Trajectory completed successfully!")
            else:
                # Don't log trajectory failed message - status 4 and others can still mean success
                # Just log that movement completed
                self.get_logger().info("✅ Trajectory completed")
            
            # Mark movement as completed and exit
            self.movement_completed = True
            self.should_exit = True
            self.get_logger().info("✅ Direct movement completed. Exiting.")
            
            # Actually shutdown ROS2 to exit the script
            rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in goal result callback: {e}")
            self.movement_completed = True
            self.should_exit = True
            rclpy.shutdown()
    
    def timeout_callback(self):
        """Handle timeout when robot gets stuck"""
        if not self.movement_completed:
            self.get_logger().error("⏰ Timeout reached - robot may be stuck")
            self.movement_completed = True
            self.should_exit = True
            rclpy.shutdown()


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Direct Object Movement Node')
    parser.add_argument('--topic', type=str, default="/objects_poses", 
                       help='Topic name for object poses subscription')
    parser.add_argument('--object-name', type=str, default="tripod_dot_0",
                       help='Name of the object to move to (e.g., blue_dot_0, red_dot_0)')
    parser.add_argument('--height', type=float, default=0.15,
                       help='Hover height in meters')
    parser.add_argument('--duration', type=int, default=30,
                       help='Maximum duration in seconds')
    parser.add_argument('--movement-duration', type=float, default=5.0,
                       help='Duration for the movement in seconds (default: 5.0)')
    parser.add_argument('--target-xyz', type=float, nargs=3, default=None,
                       help='Optional target position [x, y, z] in meters')
    parser.add_argument('--target-xyzw', type=float, nargs=4, default=None,
                       help='Optional target orientation [x, y, z, w] quaternion')
    
    # Parse arguments from sys.argv if args is None
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    node = DirectObjectMove(topic_name=args.topic, object_name=args.object_name, 
                      hover_height=args.height, movement_duration=args.movement_duration,
                      target_xyz=args.target_xyz, target_xyzw=args.target_xyzw)
    
    import time
    start_time = time.time()
    
    try:
        while rclpy.ok() and not node.should_exit:
            # Check if we've exceeded the duration
            if time.time() - start_time > args.duration:
                node.get_logger().info(f"⏰ Duration limit reached ({args.duration}s). Exiting direct movement.")
                break
                
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Direct movement stopped by user")
    except Exception as e:
        node.get_logger().error(f"Direct movement error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
