import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import sys
import os
import re
import json
import yaml

# Add custom libraries to Python path
custom_lib_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries"
if custom_lib_path not in sys.path:
    sys.path.append(custom_lib_path)

try:
    from ik_solver import compute_ik, compute_ik_robust, compute_ik_quaternion_robust
except ImportError as e:
    print(f"Failed to import IK solver: {e}")
    sys.exit(1)

class MoveToSafeHeight(Node):
    def __init__(self):
        super().__init__('move_to_safe_height')
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
        
        # Safe height target (lowered to be more reachable with various orientations)
        self.safe_height = 0.40
        
        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        
        # Shutdown flag
        self.should_exit = False
        self.joint_angles_received = False
        
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
        self.move_to_safe_height()
    
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

    def move_to_safe_height(self):
        """Move to safe height while maintaining current position and orientation"""
        # Read current end-effector pose using MCP read_topic
        self.get_logger().info("Reading current end-effector pose...")
        pose_data = self.read_current_ee_pose()
        
        if pose_data is None:
            self.get_logger().error("Could not read current end-effector pose")
            self.should_exit = True
            return
            
        current_pos = pose_data['position']
        current_quat = pose_data['orientation']
        
        # Keep the current orientation (don't change it, just move to safe height)
        # Also compute RPY for logging purposes
        current_rpy = self.quaternion_to_rpy(
            current_quat[0], current_quat[1], 
            current_quat[2], current_quat[3]
        )
        
        self.get_logger().info(f"Current EE position: {current_pos}")
        self.get_logger().info(f"Current EE quaternion: {current_quat}")
        self.get_logger().info(f"Current EE RPY (deg): {current_rpy}")
        self.get_logger().info(f"Target orientation: keeping current quaternion")

        # Create target position with safe height (same x,y but z=0.481)
        # Convert to numpy array first to ensure we can modify it
        target_position = np.array(current_pos).copy()
        target_position[2] = self.safe_height  # Set z to safe height
        
        self.get_logger().info(f"Target position: {target_position}")

        # Compute inverse kinematics by trying progressively lower heights
        # Start with target height, then try lower heights if needed
        try:
            self.get_logger().info(f"Computing IK while maintaining current orientation")
            self.get_logger().info(f"Using quaternion: {current_quat}")
            
            # Get current joint angles as seed for better convergence
            q_guess = self.current_joint_angles if self.current_joint_angles is not None else None
            if q_guess is not None:
                self.get_logger().info(f"Using current joint angles as seed: {q_guess}")
            
            # Prepare RPY in degrees for potential fallback use
            current_rpy_deg = current_rpy  # Already in degrees
            
            # Try heights from target down to find the highest reachable height
            # Heights to try: target, then 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10
            heights_to_try = [self.safe_height, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
            joint_angles = None
            
            for test_height in heights_to_try:
                if test_height <= current_pos[2]:
                    # Don't try heights lower than current position
                    continue
                
                test_position = np.array(current_pos).copy()
                test_position[2] = test_height
                
                if test_height == self.safe_height:
                    self.get_logger().info(f"Trying target height {test_height}m...")
                else:
                    self.get_logger().info(f"Trying height {test_height}m...")
                
                # Try quaternion IK first
                joint_angles = compute_ik_quaternion_robust(
                    test_position.tolist(),
                    current_quat,
                    max_tries=5,
                    dx=0.001,
                    multiple_seeds=True,
                    q_guess=q_guess
                )
                
                if joint_angles is not None:
                    if test_height == self.safe_height:
                        self.get_logger().info(f"Successfully computed IK at target height {test_height}m")
                    else:
                        self.get_logger().info(f"Successfully computed IK at height {test_height}m (lower than target {self.safe_height}m)")
                    target_position = test_position  # Update target position
                    break
                
                # If quaternion IK fails, try RPY-based IK as fallback
                if joint_angles is None:
                    joint_angles = compute_ik_robust(
                        test_position.tolist(),
                        current_rpy_deg,
                        max_tries=5,
                        dx=0.001,
                        multiple_seeds=True
                    )
                    
                    if joint_angles is not None:
                        if test_height == self.safe_height:
                            self.get_logger().info(f"Successfully computed IK at target height {test_height}m using RPY-based IK")
                        else:
                            self.get_logger().info(f"Successfully computed IK at height {test_height}m using RPY-based IK (lower than target {self.safe_height}m)")
                        target_position = test_position  # Update target position
                        break
            
            if joint_angles is None:
                self.get_logger().error("IK failed: couldn't compute any safe height position while maintaining current orientation. The orientation may not be reachable at any reasonable height above current position.")
                self.should_exit = True
                return
                
            self.get_logger().info(f"Computed joint angles: {joint_angles}")
            
            # Create trajectory point
            point = JointTrajectoryPoint(
                positions=[float(x) for x in joint_angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=30)  # 30 seconds movement
            )
            
            # Create and send trajectory
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = [point]
            
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.get_logger().info("Sending trajectory to safe height...")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.get_logger().error(f"Failed to compute IK: {e}")
            self.should_exit = True

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.should_exit = True
            return

        self.get_logger().info("Safe height trajectory accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Successfully moved to safe height")
        else:
            self.get_logger().error(f"Trajectory failed with status: {result.status}")
        self.should_exit = True

def main(args=None):
    rclpy.init(args=args)
    node = MoveToSafeHeight()
    
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Move to safe height stopped by user")
    except Exception as e:
        node.get_logger().error(f"Move to safe height error: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
