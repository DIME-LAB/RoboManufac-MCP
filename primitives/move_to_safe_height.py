import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
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
    from ik_solver import compute_ik
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
        
        # Safe height target
        self.safe_height = 0.481
        
        self.get_logger().info("Waiting for action server...")
        self.action_client.wait_for_server()
        
        # Execute movement
        self.move_to_safe_height()

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
        """Read current end-effector pose using YAML parsing like in move_down.py"""
        try:
            import subprocess
            import re
            
            self.get_logger().info("Reading current end-effector pose...")
            
            # Use ros2 topic echo to get pose data (same as move_down.py)
            result = subprocess.run([
                'bash', '-c', 
                'source /opt/ros/humble/setup.bash && ros2 topic echo /tcp_pose_broadcaster/pose --once'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.get_logger().error(f"Failed to get pose via command line: {result.stderr}")
                return None
            
            # Clean the output to remove special characters and escape sequences
            cleaned_output = result.stdout
            
            # Remove ANSI escape sequences and other special characters
            cleaned_output = re.sub(r'\x1b\[[0-9;]*m', '', cleaned_output)  # Remove ANSI color codes
            cleaned_output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', cleaned_output)  # Remove other escape sequences
            cleaned_output = re.sub(r'[^\x20-\x7E\n\r]', '', cleaned_output)  # Keep only printable ASCII and newlines
            
            # Find the YAML content between the --- markers
            yaml_start = cleaned_output.find('---')
            yaml_end = cleaned_output.rfind('---')
            
            if yaml_start != -1 and yaml_end != -1 and yaml_end > yaml_start:
                yaml_content = cleaned_output[yaml_start:yaml_end].strip()
            else:
                yaml_content = cleaned_output.strip()
            
            self.get_logger().info(f"Cleaned YAML content length: {len(yaml_content)}")
            self.get_logger().info(f"YAML content preview: {yaml_content[:300]}...")
            
            # Parse the YAML output - handle multiple documents
            data = None
            try:
                # Try to parse as single document first
                data = yaml.safe_load(yaml_content)
                if data is None:
                    raise yaml.YAMLError("Empty YAML document")
            except yaml.YAMLError as e:
                self.get_logger().info(f"Single document parsing failed: {e}")
                # If that fails, try parsing multiple documents and take the last one
                self.get_logger().info("Trying to parse multiple YAML documents...")
                try:
                    documents = list(yaml.safe_load_all(yaml_content))
                    self.get_logger().info(f"Found {len(documents)} YAML documents")
                    if documents:
                        # Find the last non-None document
                        for doc in reversed(documents):
                            if doc is not None:
                                data = doc
                                break
                        if data is None:
                            self.get_logger().error("All YAML documents are None")
                            return None
                    else:
                        self.get_logger().error("No YAML documents found")
                        return None
                except Exception as parse_error:
                    self.get_logger().error(f"Failed to parse YAML documents: {parse_error}")
                    return None
            
            if data is None:
                self.get_logger().error("Failed to parse any YAML content")
                return None
            
            if 'pose' not in data:
                self.get_logger().error("No pose data found in command output")
                return None
            
            pose_data = data['pose']
            
            # Extract position and orientation
            position = [
                pose_data['position']['x'],
                pose_data['position']['y'], 
                pose_data['position']['z']
            ]
            
            orientation = [
                pose_data['orientation']['x'],
                pose_data['orientation']['y'],
                pose_data['orientation']['z'],
                pose_data['orientation']['w']
            ]
            
            self.get_logger().info(f"Successfully read pose: position={position}, orientation={orientation}")
            
            return {
                'position': position,
                'orientation': orientation
            }
                
        except Exception as e:
            self.get_logger().error(f"Failed to read current EE pose: {e}")
            return None

    def move_to_safe_height(self):
        """Move to safe height while maintaining current position and orientation"""
        # Read current end-effector pose using MCP read_topic
        self.get_logger().info("Reading current end-effector pose...")
        pose_data = self.read_current_ee_pose()
        
        if pose_data is None:
            self.get_logger().error("Could not read current end-effector pose")
            rclpy.shutdown()
            return
            
        current_pos = pose_data['position']
        current_quat = pose_data['orientation']
        
        # Convert quaternion to RPY using the same method as other primitives
        current_rpy = self.quaternion_to_rpy(
            current_quat[0], current_quat[1], 
            current_quat[2], current_quat[3]
        )
        
        # Create target orientation: [0, 180, current_yaw] as requested
        current_roll, current_pitch, current_yaw = current_rpy
        
        # Set target orientation exactly as requested: roll=0, pitch=180, yaw=current
        target_rpy = [0, 180, 0]
        self.get_logger().info(f"Setting target orientation: [0, 180, {current_yaw:.2f}] as requested")
        
        self.get_logger().info(f"Current EE position: {current_pos}")
        self.get_logger().info(f"Current EE RPY (deg): {current_rpy}")
        self.get_logger().info(f"Target RPY (deg): {target_rpy}")

        # Create target position with safe height (same x,y but z=0.481)
        target_position = current_pos.copy()
        target_position[2] = self.safe_height  # Set z to safe height
        
        self.get_logger().info(f"Target position: {target_position}")

        # Compute inverse kinematics for target pose
        try:
            joint_angles = compute_ik(target_position, target_rpy)
            
            if joint_angles is None:
                self.get_logger().error("IK failed: couldn't compute safe height position")
                rclpy.shutdown()
                return
                
            self.get_logger().info(f"Computed joint angles: {joint_angles}")
            
            # Create trajectory point
            point = JointTrajectoryPoint(
                positions=[float(x) for x in joint_angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=3)  # 3 seconds movement
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
            rclpy.shutdown()

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            rclpy.shutdown()
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
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MoveToSafeHeight()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
