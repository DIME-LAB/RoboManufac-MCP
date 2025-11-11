#!/usr/bin/env python3
"""
Direct Object Movement - Native ROS2 Node
Read object poses from ObjectPoseArray and perform single direct movement to specific object by name
Includes calibration offset correction for accurate positioning
Supports grasp point selection from /grasp_points topic
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import from local action_libraries file
from action_libraries import hover_over_grasp

# Import the new message types
try:
    from max_camera_msgs.msg import ObjectPoseArray
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs not found. Using geometry_msgs.PoseStamped as fallback.")
    ObjectPoseArray = None

# Import grasp points message type
try:
    from max_camera_msgs.msg import GraspPointArray, GraspPoint
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs GraspPointArray not found. Using geometry_msgs.PoseStamped as fallback.")
    GraspPointArray = None
    GraspPoint = None

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
    def __init__(self, topic_name="/objects_poses_sim", object_name="blue_dot_0", height=None, movement_duration=10.0, target_xyz=None, target_xyzw=None, grasp_points_topic="/grasp_points", grasp_id=None):
        super().__init__('direct_object_move')
        
        self.topic_name = topic_name
        self.object_name = object_name
        self.height = height  # None means use offset, otherwise use exact height
        self.movement_duration = movement_duration  # Duration for IK movement
        self.target_xyz = target_xyz  # Optional target position [x, y, z]
        self.target_xyzw = target_xyzw  # Optional target orientation [x, y, z, w]
        self.grasp_points_topic = grasp_points_topic  # Topic for grasp points
        self.grasp_id = grasp_id  # Specific grasp point ID to use
        self.last_target_pose = None
        self.position_threshold = 0.005  # 5mm
        self.angle_threshold = 2.0       # 2 degrees
        # Calibration offset to correct systematic detection bias
        self.calibration_offset_x = -0.0  # -0mm correction (move left)
        self.calibration_offset_y = -0.0  # +0mm correction (move forward)
        # Object to end-effector offset distance (maintain this distance from object/grasp point)
        self.object_to_ee_offset = 0.120  # 0.12m = 12cm
        
        # Initialize Kalman filter
        self.kalman_filter = PoseKalmanFilter(process_noise=0.005, measurement_noise=0.05)
        self.last_update_time = None
        
        # Store latest grasp points
        self.latest_grasp_points = None
        self.selected_grasp_point = None
        
        # Store current end-effector pose
        self.current_ee_pose = None
        self.ee_pose_received = False
        
        # Subscribe to object poses topic
        # Use TFMessage (for /objects_poses_real topic which publishes TFMessage)
        self.pose_sub = self.create_subscription(
            TFMessage,
            topic_name,
            self.tf_message_callback,
            5  # Lower QoS to reduce update frequency
        )
        
        # Subscribe to end-effector pose topic
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
        
        # Note: ObjectPoseArray subscription removed - topic publishes TFMessage format
        # If you need ObjectPoseArray support, use a different topic name
        
        # Subscribe to grasp points topic if grasp_id is provided
        if self.grasp_id is not None and GraspPointArray is not None:
            self.grasp_points_sub = self.create_subscription(
                GraspPointArray,
                grasp_points_topic,
                self.grasp_points_callback,
                5
            )
            self.get_logger().info(f"🎯 Grasp point mode: Looking for grasp_id {grasp_id} on topic {grasp_points_topic}")
        else:
            self.grasp_points_sub = None
            if self.grasp_id is not None:
                self.get_logger().warn(f"⚠️ Grasp point mode requested but GraspPointArray not available. Falling back to object center.")
        
        # Add timer to control update frequency (every 2 seconds = 0.5Hz)
        self.update_timer = self.create_timer(3.0, self.timer_callback)
        self.latest_pose = None
        self.movement_completed = False  # Flag to track if movement has been completed
        self.should_exit = False  # Flag to control exit
        self.trajectory_in_progress = False  # Flag to track if trajectory is executing
        
        # Action client for trajectory execution
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info(f"🤖 Direct object movement started for object '{object_name}' on topic {topic_name}")
        if height is not None:
            self.get_logger().info(f"📏 Target height: {height}m (offset will be ignored)")
        else:
            self.get_logger().info(f"📏 Using {self.object_to_ee_offset*100:.1f}cm offset from object/grasp point")
        self.get_logger().info(f"⏱️ Movement duration: {movement_duration}s")
        if self.grasp_id is not None:
            self.get_logger().info(f"🎯 Grasp point mode: Using grasp_id {grasp_id} from topic {grasp_points_topic}")
        else:
            self.get_logger().info(f"🎯 Object center mode: Moving to object center")
        
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
        if ObjectPoseArray is None:
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
        else:
            # Object not found in this message
            self.get_logger().warn(f"Object '{self.object_name}' not found in current message")
            self.latest_pose = None
    
    def tf_message_callback(self, msg):
        """Handle TFMessage and find target object by child_frame_id"""
        # Find the transform with matching child_frame_id (object name)
        target_transform = None
        for transform in msg.transforms:
            if transform.child_frame_id == self.object_name:
                target_transform = transform
                break
        
        if target_transform is not None:
            # Convert TransformStamped to PoseStamped
            pose_stamped = PoseStamped()
            pose_stamped.header = target_transform.header
            pose_stamped.pose.position.x = target_transform.transform.translation.x
            pose_stamped.pose.position.y = target_transform.transform.translation.y
            pose_stamped.pose.position.z = target_transform.transform.translation.z
            pose_stamped.pose.orientation.x = target_transform.transform.rotation.x
            pose_stamped.pose.orientation.y = target_transform.transform.rotation.y
            pose_stamped.pose.orientation.z = target_transform.transform.rotation.z
            pose_stamped.pose.orientation.w = target_transform.transform.rotation.w
            self.latest_pose = pose_stamped
        else:
            # Object not found in this message
            self.latest_pose = None
    
    def pose_callback(self, msg):
        """Store latest pose message (fallback for PoseStamped)"""
        self.latest_pose = msg
    
    def grasp_points_callback(self, msg):
        """Handle GraspPointArray message and find target grasp point"""
        if GraspPointArray is None:
            return
        
        # Store all grasp points
        self.latest_grasp_points = msg
        
        # Find the grasp point with the specified ID and object name
        target_grasp_point = None
        for grasp_point in msg.grasp_points:
            if (grasp_point.grasp_id == self.grasp_id and 
                grasp_point.object_name == self.object_name):
                target_grasp_point = grasp_point
                break
        
        if target_grasp_point is not None:
            self.selected_grasp_point = target_grasp_point
            self.get_logger().info(f"🎯 Found grasp point {self.grasp_id} for object '{self.object_name}'")
            # Unsubscribe after getting the grasp point once (simulation data is accurate)
            if self.grasp_points_sub is not None:
                self.destroy_subscription(self.grasp_points_sub)
                self.grasp_points_sub = None
        else:
            # Grasp point not found in this message
            self.get_logger().warn(f"Grasp point {self.grasp_id} for object '{self.object_name}' not found in current message")
            self.selected_grasp_point = None
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose data"""
        self.current_ee_pose = msg
        self.ee_pose_received = True
    
    def timer_callback(self):
        """Process pose and perform single direct movement to object"""
        if self.movement_completed:
            return
        
        # Don't send new trajectory if one is already in progress
        if self.trajectory_in_progress:
            self.get_logger().debug("Trajectory already in progress, skipping...")
            return
        
        # Wait for end-effector pose if not received yet
        if not self.ee_pose_received or self.current_ee_pose is None:
            self.get_logger().warn("Waiting for end-effector pose...")
            return
        
        # Get current end-effector position
        current_ee_position = np.array([
            self.current_ee_pose.pose.position.x,
            self.current_ee_pose.pose.position.y,
            self.current_ee_pose.pose.position.z
        ])
        
        # Check if we have optional target position/orientation
        if self.target_xyz is not None and self.target_xyzw is not None:
            # Use provided target position and orientation
            object_position = np.array(self.target_xyz[:3])  # Take first 3 elements
            
            # Apply calibration offset to correct systematic detection bias (same as detected objects)
            object_position[0] += self.calibration_offset_x  # Correct X offset
            object_position[1] += self.calibration_offset_y  # Correct Y offset
            
            rpy = self.quaternion_to_rpy(
                self.target_xyzw[0], self.target_xyzw[1], 
                self.target_xyzw[2], self.target_xyzw[3]
            )
            self.get_logger().info(f"🎯 Using provided target position: {object_position} (with calibration offset applied) and orientation: {rpy}")
        elif self.selected_grasp_point is not None:
            # Use only grasp point position (ignore orientation)
            grasp_point_position = np.array([
                self.selected_grasp_point.pose.position.x,
                self.selected_grasp_point.pose.position.y,
                self.selected_grasp_point.pose.position.z
            ])
            
            # Apply calibration offset to correct systematic detection bias
            grasp_point_position[0] += self.calibration_offset_x  # Correct X offset
            grasp_point_position[1] += self.calibration_offset_y  # Correct Y offset
            
            # Set object position to grasp point position for distance calculation
            object_position = grasp_point_position
            
            # Set orientation to face down (roll=0, pitch=180, yaw=0)
            rpy = [0.0, 180.0, 0.0]
            
            self.get_logger().info(f"🎯 Using grasp point {self.grasp_id} position: {grasp_point_position} (with calibration offset applied)")
            self.get_logger().info(f"🎯 Gripper orientation: face down (RPY: [0.0, 180.0, 0.0])")
        elif self.latest_pose is not None:
            # Use detected object pose
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
            object_position = np.array(filtered_pose[:3])
            rpy = filtered_pose[3:6].tolist()
            
            # Apply calibration offset to correct systematic detection bias
            object_position[0] += self.calibration_offset_x  # Correct X offset
            object_position[1] += self.calibration_offset_y  # Correct Y offset
            
            self.get_logger().info(f"🎯 Detected object at ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        else:
            # No target provided and no object detected
            self.get_logger().warn("No target position provided and no object detected")
            return
        
        # Calculate direction vector from object to current end-effector
        direction_vector = current_ee_position - object_position
        current_distance = np.linalg.norm(direction_vector)
        
        self.get_logger().info(f"📏 Current distance between object and EE: {current_distance*100:.2f} cm")
        self.get_logger().info(f"📍 Current EE position: ({current_ee_position[0]:.3f}, {current_ee_position[1]:.3f}, {current_ee_position[2]:.3f})")
        self.get_logger().info(f"📍 Object position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        
        # Calculate target end-effector position
        if self.selected_grasp_point is not None:
            # For grasp points: place EE above the grasp point (in Z direction)
            target_ee_position = np.array([
                object_position[0],
                object_position[1],
                object_position[2] + self.object_to_ee_offset
            ])
            self.get_logger().info(f"📏 Placing EE {self.object_to_ee_offset*100:.1f}cm above grasp point at Z={target_ee_position[2]:.3f}m")
        elif self.height is not None:
            # If height is explicitly specified, use that exact height
            target_ee_position = np.array([
                object_position[0],
                object_position[1],
                self.height
            ])
            self.get_logger().info(f"📏 Using specified height={self.height:.3f}m")
        else:
            # Determine approach direction for non-grasp-point cases
            approach_direction = None
            if current_distance > 1e-6:
                # Use direction from object to current EE
                approach_direction = direction_vector / current_distance
                self.get_logger().info(f"🎯 Using current EE direction: ({approach_direction[0]:.3f}, {approach_direction[1]:.3f}, {approach_direction[2]:.3f})")
            else:
                # Default to upward direction
                approach_direction = np.array([0.0, 0.0, 1.0])
                self.get_logger().warn("Current distance is very small, using default upward direction")
            
            # Calculate target position by offsetting from object along approach direction
            target_ee_position = object_position + approach_direction * self.object_to_ee_offset
            self.get_logger().info(f"📏 Maintaining {self.object_to_ee_offset*100:.1f}cm offset from object")
        
        # Verify the target distance
        calculated_distance = np.linalg.norm(target_ee_position - object_position)
        self.get_logger().info(f"✅ Calculated target distance: {calculated_distance*100:.2f} cm")
        
        self.get_logger().info(f"🎯 Final target EE position: ({target_ee_position[0]:.3f}, {target_ee_position[1]:.3f}, {target_ee_position[2]:.3f})")
        
        # Create target pose with calculated position
        target_position = target_ee_position.tolist()
        target_pose = (target_position, rpy)
        
        # Use the calculated z-coordinate (which is either the specified height or the auto-calculated one)
        trajectory = hover_over_grasp(target_pose, target_ee_position[2], self.movement_duration)
        
        # Execute trajectory
        self.trajectory_in_progress = True  # Mark trajectory as in progress
        self.execute_trajectory(trajectory)
        # Don't set movement_completed or should_exit here - wait for trajectory completion
    
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
            
            self.get_logger().info("Sending trajectory...")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.get_logger().error(f"❌ Trajectory execution error: {e}")
            self.trajectory_in_progress = False  # Clear flag on error
            self.movement_completed = True
            self.should_exit = True

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            # Set exit flags if goal is rejected
            self.trajectory_in_progress = False
            self.movement_completed = True
            self.should_exit = True
            return

        self.get_logger().info("Trajectory goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        result = future.result()
        self.trajectory_in_progress = False  # Clear trajectory in progress flag
        
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("✅ Trajectory completed successfully")
        else:
            self.get_logger().error(f"Trajectory failed with status: {result.status}")
        
        # Set exit flags after trajectory completes
        self.movement_completed = True
        self.should_exit = True
        self.get_logger().info("✅ Direct movement completed. Exiting.")


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Direct Object Movement Node')
    parser.add_argument('--topic', type=str, default="/objects_poses_real", 
                       help='Topic name for object poses subscription')
    parser.add_argument('--object-name', type=str, default="fork_orange_scaled70",
                       help='Name of the object to move to (e.g., blue_dot_0, red_dot_0)')
    parser.add_argument('--height', type=float, default=None,
                       help='Hover height in meters (if not specified, will use 5.5cm offset from object/grasp point)')
    parser.add_argument('--movement-duration', type=float, default=10.0,
                       help='Duration for the movement in seconds (default: 10.0)')
    parser.add_argument('--target-xyz', type=float, nargs=3, default=None,
                       help='Optional target position [x, y, z] in meters')
    parser.add_argument('--target-xyzw', type=float, nargs=4, default=None,
                       help='Optional target orientation [x, y, z, w] quaternion')
    parser.add_argument('--grasp-points-topic', type=str, default="/grasp_points",
                       help='Topic name for grasp points subscription')
    parser.add_argument('--grasp-id', type=int, default=None,
                       help='Specific grasp point ID to use (if provided, will use grasp point instead of object center)')
    
    # Parse arguments from sys.argv if args is None
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    node = DirectObjectMove(topic_name=args.topic, object_name=args.object_name, 
                      height=args.height, movement_duration=args.movement_duration,
                      target_xyz=args.target_xyz, target_xyzw=args.target_xyzw,
                      grasp_points_topic=args.grasp_points_topic, grasp_id=args.grasp_id)
    
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Direct movement stopped by user")
    except Exception as e:
        node.get_logger().error(f"Direct movement error: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
