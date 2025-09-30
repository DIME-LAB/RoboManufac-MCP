#!/usr/bin/env python3
"""
Simple Visual Servoing - Native ROS2 Node
Just read pose, convert to RPY, and hover
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

class VisualServo(Node):
    def __init__(self, topic_name="/object_poses/jenga_3", hover_height=0.15, movement_duration=7.0):
        super().__init__('visual_servo')
        
        self.topic_name = topic_name
        self.hover_height = hover_height
        self.movement_duration = movement_duration  # Duration for IK movement
        self.last_target_pose = None
        self.position_threshold = 0.005  # 5mm
        self.angle_threshold = 2.0       # 2 degrees
        
        # Initialize Kalman filter
        self.kalman_filter = PoseKalmanFilter(process_noise=0.005, measurement_noise=0.05)
        self.last_update_time = None
        
        # Subscribe to pose topic
        self.pose_sub = self.create_subscription(
            PoseStamped,
            topic_name,
            self.pose_callback,
            5  # Lower QoS to reduce update frequency
        )
        
        # Add timer to control update frequency (every 2 seconds = 0.5Hz)
        self.update_timer = self.create_timer(3.0, self.timer_callback)
        self.latest_pose = None
        self.stable_count = 0  # Count consecutive stable readings
        self.stable_threshold = 1  # Exit after 3 consecutive stable readings (9 seconds total)
        self.should_exit = False  # Flag to control exit
        
        # Action client for trajectory execution
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info(f"🤖 Visual servo started for {topic_name}")
        self.get_logger().info(f"📏 Hover height: {hover_height}m")
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
    
    def pose_callback(self, msg):
        """Store latest pose message"""
        self.latest_pose = msg
    
    def timer_callback(self):
        """Process pose at controlled frequency with Kalman filtering"""
        if self.latest_pose is None:
            return
        
        # Calculate time delta for Kalman filter
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
        else:
            dt = 1.0  # Default time step
        self.last_update_time = current_time
        
        # Update Kalman filter with new measurement
        filtered_pose, velocity = self.kalman_filter.update(self.latest_pose, dt)
        
        if filtered_pose is None:
            return
            
        # Extract filtered position and orientation
        position = filtered_pose[:3].tolist()
        rpy = filtered_pose[3:6].tolist()
        
        # Check if we need to move
        if not self.poses_are_similar(position, rpy):
            self.get_logger().info(f"🎯 Moving to hover over: ({position[0]:.3f}, {position[1]:.3f}) "
                                  f"at height {self.hover_height:.3f}m, yaw={rpy[2]:.1f}°")
            
            # Reset stable count when movement is needed
            self.stable_count = 0
            
            # Use hover_over function with custom duration
            target_pose = (position, rpy)
            trajectory = hover_over(target_pose, self.hover_height, self.movement_duration)
            
            # Execute trajectory
            self.execute_trajectory(trajectory)
            self.last_target_pose = (position, rpy)
        else:
            self.stable_count += 1
            self.get_logger().info(f"🔄 Tracking: ({position[0]:.3f}, {position[1]:.3f}) "
                                  f"yaw={rpy[2]:.1f}° (stable {self.stable_count}/{self.stable_threshold})")
            
            # Exit if we've been stable for enough consecutive readings
            if self.stable_count >= self.stable_threshold:
                self.get_logger().info("✅ Successfully aligned and stable. Exiting visual servo.")
                self.should_exit = True
    
    def execute_trajectory(self, trajectory):
        """Execute trajectory using ROS2 action"""
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
            
            self.action_client.send_goal_async(goal)
            self.get_logger().info("✅ Trajectory sent successfully")
            
        except Exception as e:
            self.get_logger().error(f"❌ Trajectory execution error: {e}")


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visual Servo Node')
    parser.add_argument('--topic', type=str, default="/object_poses/jenga_4", 
                       help='Topic name for pose subscription')
    parser.add_argument('--height', type=float, default=0.15,
                       help='Hover height in meters')
    parser.add_argument('--duration', type=int, default=30,
                       help='Maximum duration in seconds')
    parser.add_argument('--movement-duration', type=float, default=7.0,
                       help='Duration for each IK movement in seconds (default: 7.0)')
    
    # Parse arguments from sys.argv if args is None
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    node = VisualServo(topic_name=args.topic, hover_height=args.height, movement_duration=args.movement_duration)
    
    import time
    start_time = time.time()
    
    try:
        while rclpy.ok() and not node.should_exit:
            # Check if we've exceeded the duration
            if time.time() - start_time > args.duration:
                node.get_logger().info(f"⏰ Duration limit reached ({args.duration}s). Exiting visual servo.")
                break
                
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Visual servo stopped by user")
    except Exception as e:
        node.get_logger().error(f"Visual servo error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
