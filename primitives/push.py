#!/usr/bin/env python3
"""
Visual Push Primitive for UR5e - ROS2 Version
Reads object pose from topic and performs complete push sequence:
  1. Move to approach position (above object)
  2. Move to push start position (before object in push direction)
  3. Execute push to final position
  
Uses joint trajectory controller with IK (same as move_home.py).
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import argparse
import sys
from pytransform3d.rotations import quaternion_from_euler

# Add path to IK solver
main_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries"
if main_path not in sys.path:
    sys.path.append(main_path)

try:
    from ik_solver import compute_ik
except ImportError as e:
    print(f"Failed to import IK solver: {e}")
    sys.exit(1)

# Import the new message type
try:
    from max_camera_msgs.msg import ObjectPoseArray
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs not found. Using geometry_msgs.PoseStamped as fallback.")
    ObjectPoseArray = None

# ANSI color codes for terminal output
def colorize(color_code, message):
    return f"\033[{color_code}m{message}\033[0m"

class VisualPush(Node):
    def __init__(self, topic_name="/objects_poses", object_name="blue block_dot_0", 
                 target_x=0.0, target_y=-0.5, target_z=0.15, 
                 approach_height=0.25, push_offset=0.05,
                 stage_duration=5.0, target_angle=None,
                 calibration_offset_x=-0.013, calibration_offset_y=0.028):
        super().__init__('visual_push')
        
        self.topic_name = topic_name
        self.object_name = object_name
        self.target_position = [target_x, target_y, target_z]
        self.approach_height = approach_height  # Height above object for approach
        self.push_offset = push_offset  # Distance before object to start push
        self.stage_duration = stage_duration  # Duration for each stage movement
        self.target_angle = target_angle  # Desired final angle in degrees (None = auto-calculate from push direction)
        
        # Calibration offset to correct systematic detection bias (same as visual_servo_yoloe.py)
        self.calibration_offset_x = calibration_offset_x  # Default: -13mm correction (move left)
        self.calibration_offset_y = calibration_offset_y  # Default: +28mm correction (move forward)
        
        # Joint names for UR5e
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Action client for joint trajectory control (like move_home.py)
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Subscribe to object poses topic
        if ObjectPoseArray is not None:
            self.pose_sub = self.create_subscription(
                ObjectPoseArray,
                topic_name,
                self.objects_poses_callback,
                5
            )
        else:
            # Fallback to old PoseStamped subscription
            self.pose_sub = self.create_subscription(
                PoseStamped,
                topic_name,
                self.pose_callback,
                5
            )
        
        self.get_logger().info("⏳ Waiting for action server...")
        self.action_client.wait_for_server()
        self.get_logger().info("✅ Action server ready")
        
        # State management
        self.latest_pose = None
        self.pose_captured = False
        self.current_stage = 0  # 0=waiting, 1=approach, 2=push, 3=done
        
        self.get_logger().info(colorize(36, f"🤖 Visual push started for object '{object_name}' on topic {topic_name}"))
        self.get_logger().info(colorize(36, f"🎯 Target position: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"))
        if target_angle is not None:
            self.get_logger().info(colorize(36, f"🎯 Target angle: {target_angle:.1f}° (user-specified)"))
        else:
            self.get_logger().info(colorize(36, f"🎯 Target angle: auto-calculate from push direction"))
        self.get_logger().info(colorize(36, f"📏 Approach height: {approach_height}m, Push offset: {push_offset}m"))
        self.get_logger().info(colorize(36, f"⏱️ Stage duration: {stage_duration}s"))

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
        
        if target_object is not None and not self.pose_captured:
            # Convert ObjectPose to PoseStamped for compatibility
            pose_stamped = PoseStamped()
            pose_stamped.header = target_object.header
            pose_stamped.pose = target_object.pose
            self.latest_pose = pose_stamped
            self.pose_captured = True
            self.get_logger().info(colorize(32, f"📡 Captured object '{self.object_name}' at: ({pose_stamped.pose.position.x:.3f}, {pose_stamped.pose.position.y:.3f}, {pose_stamped.pose.position.z:.3f})"))
            # Start the push sequence
            self.start_push_sequence()
        elif target_object is None and not self.pose_captured:
            self.get_logger().warn(f"⚠️  Object '{self.object_name}' not found in current message")

    def pose_callback(self, msg):
        """Store latest pose message (fallback for PoseStamped)"""
        if not self.pose_captured:
            self.latest_pose = msg
            self.pose_captured = True
            self.get_logger().info(colorize(32, f"📡 Captured pose: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"))
            # Start the push sequence
            self.start_push_sequence()

    def start_push_sequence(self):
        """Initialize and execute complete push sequence using joint trajectory control"""
        if self.latest_pose is None:
            self.get_logger().warn("⚠️  No pose data available yet")
            return
            
        # Get object position and apply calibration offset
        object_pos = [
            self.latest_pose.pose.position.x + self.calibration_offset_x,  # Apply X offset correction
            self.latest_pose.pose.position.y + self.calibration_offset_y,  # Apply Y offset correction
            self.latest_pose.pose.position.z
        ]
        
        self.get_logger().info(colorize(95, f"📍 Raw detected position: ({self.latest_pose.pose.position.x:.3f}, {self.latest_pose.pose.position.y:.3f}, {self.latest_pose.pose.position.z:.3f})"))
        self.get_logger().info(colorize(95, f"📍 Calibrated position: ({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}) [offset: X{self.calibration_offset_x:+.3f}, Y{self.calibration_offset_y:+.3f}]"))
        
        # Get object's CURRENT orientation from detected pose
        object_rpy = self.quaternion_to_rpy(
            self.latest_pose.pose.orientation.x,
            self.latest_pose.pose.orientation.y,
            self.latest_pose.pose.orientation.z,
            self.latest_pose.pose.orientation.w
        )
        object_yaw = object_rpy[2]  # Extract yaw angle
        
        self.get_logger().info(colorize(95, f"📐 Object current orientation: Roll={object_rpy[0]:.1f}°, Pitch={object_rpy[1]:.1f}°, Yaw={object_yaw:.1f}°"))
        
        # Calculate push direction (from object to target)
        push_direction = [
            self.target_position[0] - object_pos[0],
            self.target_position[1] - object_pos[1],
            0  # No Z component
        ]
        
        # Normalize direction
        direction_magnitude = math.sqrt(push_direction[0]**2 + push_direction[1]**2)
        if direction_magnitude < 0.001:
            self.get_logger().error("❌ Object is already at target position!")
            rclpy.shutdown()
            return
            
        push_direction[0] /= direction_magnitude
        push_direction[1] /= direction_magnitude
        
        # Determine FINAL target angle (for stage 3)
        if self.target_angle is not None:
            # Use user-specified angle
            final_angle_deg = self.target_angle
            self.get_logger().info(colorize(95, f"🎯 Target final angle: {final_angle_deg:.1f}° (user-specified)"))
        else:
            # Auto-calculate from push direction
            push_angle_rad = math.atan2(push_direction[1], push_direction[0])
            final_angle_deg = math.degrees(push_angle_rad)
            self.get_logger().info(colorize(95, f"🎯 Target final angle: {final_angle_deg:.1f}° (calculated from push direction)"))
        
        # Stage 1 & 2: Match OBJECT's current orientation (approach and pre-push)
        # Fixed orientation: Roll=180°, Pitch=0° (pointing down), Yaw=object_yaw
        # This matches the original move3d_PUSH.py convention
        rpy_stages_1_2 = [0, 180, object_yaw]
        
        # Stage 3: Use FINAL target orientation
        rpy_stage_3 = [0, 180, final_angle_deg]
        
        # Stage 1: Approach position (above object)
        stage1_pos = [object_pos[0], object_pos[1], self.approach_height]
        
        # Stage 2: Push start position (before object, at push height)
        stage2_pos = [
            object_pos[0] - push_direction[0] * self.push_offset,
            object_pos[1] - push_direction[1] * self.push_offset,
            self.target_position[2]
        ]
        
        # Stage 3: Push end position (at target position)
        stage3_pos = [self.target_position[0], self.target_position[1], self.target_position[2]]
        
        self.get_logger().info(colorize(93, "╔════════════════════════════════════════╗"))
        self.get_logger().info(colorize(93, "║       PUSH SEQUENCE INITIALIZED        ║"))
        self.get_logger().info(colorize(93, "╚════════════════════════════════════════╝"))
        self.get_logger().info(colorize(36, f"📍 Object position:    ({object_pos[0]:+.3f}, {object_pos[1]:+.3f}, {object_pos[2]:+.3f})"))
        self.get_logger().info(colorize(36, f"📍 Stage 1 (approach): ({stage1_pos[0]:+.3f}, {stage1_pos[1]:+.3f}, {stage1_pos[2]:+.3f}) @ {object_yaw:.1f}° (object angle)"))
        self.get_logger().info(colorize(36, f"📍 Stage 2 (pre-push): ({stage2_pos[0]:+.3f}, {stage2_pos[1]:+.3f}, {stage2_pos[2]:+.3f}) @ {object_yaw:.1f}° (object angle)"))
        self.get_logger().info(colorize(36, f"📍 Stage 3 (push end): ({stage3_pos[0]:+.3f}, {stage3_pos[1]:+.3f}, {stage3_pos[2]:+.3f}) @ {final_angle_deg:.1f}° (target angle)"))
        self.get_logger().info(colorize(36, f"🔄 Orientation change: {object_yaw:.1f}° → {final_angle_deg:.1f}° (Δ{final_angle_deg - object_yaw:.1f}°)"))
        
        # Execute all 3 stages in one trajectory with different orientations
        self.execute_push_trajectory(stage1_pos, stage2_pos, stage3_pos, rpy_stages_1_2, rpy_stage_3)
        
    def execute_push_trajectory(self, stage1_pos, stage2_pos, stage3_pos, rpy_stages_1_2, rpy_stage_3):
        """Execute complete push trajectory with IK and joint trajectory control"""
        # Compute IK for all three stages with their respective orientations
        self.get_logger().info("🔧 Computing IK for all stages...")
        
        # Stage 1: Match object orientation
        stage1_joints = compute_ik(stage1_pos, rpy_stages_1_2)
        if stage1_joints is None:
            self.get_logger().error("❌ IK failed for stage 1 (approach)")
            rclpy.shutdown()
            return
        
        # Stage 2: Match object orientation    
        stage2_joints = compute_ik(stage2_pos, rpy_stages_1_2)
        if stage2_joints is None:
            self.get_logger().error("❌ IK failed for stage 2 (pre-push)")
            rclpy.shutdown()
            return
        
        # Stage 3: Use target final orientation
        stage3_joints = compute_ik(stage3_pos, rpy_stage_3)
        if stage3_joints is None:
            self.get_logger().error("❌ IK failed for stage 3 (push end)")
            rclpy.shutdown()
            return
        
        self.get_logger().info("✅ IK computed successfully for all stages")
        self.get_logger().info(colorize(36, f"   Stage 1 & 2 orientation: {rpy_stages_1_2}"))
        self.get_logger().info(colorize(36, f"   Stage 3 orientation: {rpy_stage_3}"))
        
        # Create trajectory with all 3 points
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Point 1: Approach (above object)
        point1 = JointTrajectoryPoint()
        point1.positions = [float(x) for x in stage1_joints]
        point1.velocities = [0.0] * 6
        point1.time_from_start = Duration(sec=int(self.stage_duration))
        trajectory.points.append(point1)
        
        # Point 2: Pre-push (before object)
        point2 = JointTrajectoryPoint()
        point2.positions = [float(x) for x in stage2_joints]
        point2.velocities = [0.0] * 6
        point2.time_from_start = Duration(sec=int(self.stage_duration * 2))
        trajectory.points.append(point2)
        
        # Point 3: Push end (target position)
        point3 = JointTrajectoryPoint()
        point3.positions = [float(x) for x in stage3_joints]
        point3.velocities = [0.0] * 6
        point3.time_from_start = Duration(sec=int(self.stage_duration * 3))
        trajectory.points.append(point3)
        
        # Create and send goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=1)
        
        self.get_logger().info("📤 Sending 3-stage push trajectory to robot...")
        self._send_goal_future = self.action_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(self.goal_response)
    
    def goal_response(self, future):
        """Handle goal response from action server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Push trajectory goal rejected")
            rclpy.shutdown()
            return

        self.get_logger().info(colorize(32, "✅ Push trajectory accepted - Robot is moving!"))
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result from action server"""
        result = future.result()
        if result.status == 1:  # SUCCEEDED
            self.get_logger().info(colorize(42, "✅ Push sequence completed successfully!"))
        else:
            self.get_logger().error(colorize(31, f"❌ Push sequence failed with status: {result.status}"))
        rclpy.shutdown()


def main(args=None):
    """
    Main entry point for visual push primitive.
    
    Example usage:
        python push_visual.py --object-name "blue block_dot_0" --target-x 0.2 --target-y -0.3 --target-z 0.15
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Visual Push Primitive - Complete 3-stage push sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push blue block to position (0.2, -0.3, 0.15) - auto-calculated angle
  python push_visual.py --object-name "blue block_dot_0" --target-x 0.2 --target-y -0.3 --target-z 0.15
  
  # Push with specific final angle (e.g., 45 degrees)
  python push_visual.py --object-name "blue block_dot_0" --target-x 0.2 --target-y -0.3 --target-angle 45
  
  # Push with custom approach height and offset
  python push_visual.py --object-name "red block_dot_0" --target-x 0.1 --target-y -0.4 --approach-height 0.3 --push-offset 0.08
  
  # Fast push with shorter stage duration
  python push_visual.py --object-name "blue block_dot_0" --target-x 0.0 --target-y -0.5 --stage-duration 3.0
        """
    )
    parser.add_argument('--topic', type=str, default="/objects_poses", 
                       help='Topic name for object poses subscription (default: /objects_poses)')
    parser.add_argument('--object-name', type=str, default="blue block_dot_0",
                       help='Name of the object to push (default: "blue block_dot_0")')
    parser.add_argument('--target-x', type=float, required=True,
                       help='Target X position in meters [REQUIRED]')
    parser.add_argument('--target-y', type=float, required=True,
                       help='Target Y position in meters [REQUIRED]')
    parser.add_argument('--target-z', type=float, default=0.15,
                       help='Target Z height in meters (default: 0.15)')
    parser.add_argument('--target-angle', type=float, default=None,
                       help='Target angle for final block orientation in degrees (default: auto-calculate from push direction)')
    parser.add_argument('--approach-height', type=float, default=0.25,
                       help='Height above object for safe approach in meters (default: 0.25)')
    parser.add_argument('--push-offset', type=float, default=0.05,
                       help='Distance before object to start push in meters (default: 0.05)')
    parser.add_argument('--stage-duration', type=float, default=5.0,
                       help='Duration for each stage movement in seconds (default: 5.0)')
    parser.add_argument('--calibration-offset-x', type=float, default=-0.013,
                       help='X-axis calibration offset in meters (default: -0.013)')
    parser.add_argument('--calibration-offset-y', type=float, default=0.028,
                       help='Y-axis calibration offset in meters (default: 0.028)')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Initialize ROS2
    rclpy.init(args=None)
    
    # Create node with parsed arguments
    node = VisualPush(
        topic_name=parsed_args.topic, 
        object_name=parsed_args.object_name,
        target_x=parsed_args.target_x,
        target_y=parsed_args.target_y,
        target_z=parsed_args.target_z,
        approach_height=parsed_args.approach_height,
        push_offset=parsed_args.push_offset,
        stage_duration=parsed_args.stage_duration,
        target_angle=parsed_args.target_angle,
        calibration_offset_x=parsed_args.calibration_offset_x,
        calibration_offset_y=parsed_args.calibration_offset_y
    )
    
    # Spin the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(colorize(93, "\n⚠️  Push stopped by user"))
    except Exception as e:
        node.get_logger().error(colorize(31, f"❌ Error: {e}"))
        import traceback
        traceback.print_exc()
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        print("\n✓ Node shutdown complete")

if __name__ == '__main__':
    main()
