#!/usr/bin/env python3
"""
Simulation Push Primitive for UR5e - ROS2 Version
Performs complete push sequence using predefined initial and final positions:
  1. Move to approach position (above initial object position)
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

# ANSI color codes for terminal output
def colorize(color_code, message):
    return f"\033[{color_code}m{message}\033[0m"

class SimulationPush(Node):
    def __init__(self, 
                 # Initial Jenga block position (x, y, z, yaw)
                 initial_x=0.0, initial_y=0.0, initial_z=0.15, initial_yaw=0.0,
                 # Final Jenga block position (x, y, z, yaw)  
                 final_x=0.2, final_y=-0.3, final_z=0.15, final_yaw=45.0,
                 # End-effector height (constant throughout motion)
                 ee_height=0.25,
                 # Motion parameters
                 initial_offset=0.05, stage_duration=7.0, final_offset=0.02351):
        super().__init__('simulation_push')
        
        # Initial and final positions
        self.initial_position = [initial_x, initial_y, initial_z]
        self.initial_yaw = initial_yaw
        self.final_position = [final_x, final_y, final_z] 
        self.final_yaw = final_yaw
        
        # End-effector height (constant throughout motion)
        self.ee_height = ee_height
        
        # Motion parameters
        self.initial_offset = initial_offset  # Distance before object to start push
        self.stage_duration = stage_duration  # Duration for each stage movement
        self.final_offset = final_offset  # Final offset for push end position
        
        # Joint names for UR5e
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Action client for joint trajectory control
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info("⏳ Waiting for action server...")
        self.action_client.wait_for_server()
        self.get_logger().info("✅ Action server ready")
        
        self.get_logger().info(colorize(36, f"🤖 Simulation push started"))
        self.get_logger().info(colorize(36, f"📍 Initial position: ({initial_x:.3f}, {initial_y:.3f}, {initial_z:.3f}) @ {initial_yaw:.1f}°"))
        self.get_logger().info(colorize(36, f"📍 Final position: ({final_x:.3f}, {final_y:.3f}, {final_z:.3f}) @ {final_yaw:.1f}°"))
        self.get_logger().info(colorize(36, f"📏 EE height: {ee_height}m, Initial offset: {initial_offset}m"))
        self.get_logger().info(colorize(36, f"⏱️ Stage duration: {stage_duration}s"))
        
        # Start the push sequence immediately
        self.start_push_sequence()

    def start_push_sequence(self):
        """Initialize and execute complete push sequence using joint trajectory control"""
        
        # Calculate push direction (from initial to final position)
        push_direction = [
            self.final_position[0] - self.initial_position[0],
            self.final_position[1] - self.initial_position[1],
            0  # No Z component
        ]
        
        # Normalize direction
        direction_magnitude = math.sqrt(push_direction[0]**2 + push_direction[1]**2)
        if direction_magnitude < 0.001:
            self.get_logger().error("❌ Initial and final positions are too close!")
            rclpy.shutdown()
            return
            
        push_direction[0] /= direction_magnitude
        push_direction[1] /= direction_magnitude
        
        self.get_logger().info(colorize(95, f"📐 Push direction: ({push_direction[0]:.3f}, {push_direction[1]:.3f})"))
        self.get_logger().info(colorize(95, f"📐 Initial yaw: {self.initial_yaw:.1f}°, Final yaw: {self.final_yaw:.1f}°"))
        
        # Stage 1-3: Use initial object orientation (roll=0, pitch=180, yaw=initial_yaw)
        # Fixed orientation: Roll=0°, Pitch=180° (pointing down), Yaw=initial_yaw
        rpy_stages_1_3 = [0, 180, self.initial_yaw+90]
        
        # Stage 4: Use final target orientation (roll=0, pitch=180, yaw=final_yaw)
        rpy_stage_4 = [0, 180, self.final_yaw+90]
        
        # Stage 1: Approach position (above initial object position at fixed height 0.25m)
        stage1_pos = [self.initial_position[0], self.initial_position[1], 0.25]
        
        # Stage 2: Push start position (before object, at user-specified height)
        stage2_pos = [
            self.initial_position[0] - push_direction[0] * self.initial_offset,
            self.initial_position[1] - push_direction[1] * self.initial_offset,
            self.ee_height
        ]
        
        # Stage 3: Move down to user-specified height above initial position
        stage3_pos = [self.initial_position[0], self.initial_position[1], self.ee_height]
        
        # Stage 4: Push end position (at final position + final offset, at user-specified height)
        stage4_pos = [
            self.final_position[0] + push_direction[0] * self.final_offset,
            self.final_position[1] + push_direction[1] * self.final_offset,
            self.ee_height
        ]
        
        self.get_logger().info(colorize(93, "╔════════════════════════════════════════╗"))
        self.get_logger().info(colorize(93, "║       PUSH SEQUENCE INITIALIZED        ║"))
        self.get_logger().info(colorize(93, "╚════════════════════════════════════════╝"))
        self.get_logger().info(colorize(36, f"📍 Initial position:    ({self.initial_position[0]:+.3f}, {self.initial_position[1]:+.3f}, {self.initial_position[2]:+.3f})"))
        self.get_logger().info(colorize(36, f"📍 Stage 1 (approach):  ({stage1_pos[0]:+.3f}, {stage1_pos[1]:+.3f}, {stage1_pos[2]:+.3f}) @ {self.initial_yaw:.1f}° (fixed 0.25m height)"))
        self.get_logger().info(colorize(36, f"📍 Stage 2 (pre-push):  ({stage2_pos[0]:+.3f}, {stage2_pos[1]:+.3f}, {stage2_pos[2]:+.3f}) @ {self.initial_yaw:.1f}° (user height {self.ee_height:.3f}m)"))
        self.get_logger().info(colorize(36, f"📍 Stage 3 (descend):   ({stage3_pos[0]:+.3f}, {stage3_pos[1]:+.3f}, {stage3_pos[2]:+.3f}) @ {self.initial_yaw:.1f}° (user height {self.ee_height:.3f}m)"))
        self.get_logger().info(colorize(36, f"📍 Stage 4 (push end):  ({stage4_pos[0]:+.3f}, {stage4_pos[1]:+.3f}, {stage4_pos[2]:+.3f}) @ {self.final_yaw:.1f}° (user height {self.ee_height:.3f}m)"))
        self.get_logger().info(colorize(36, f"🔄 Orientation change: {self.initial_yaw:.1f}° → {self.final_yaw:.1f}° (Δ{self.final_yaw - self.initial_yaw:.1f}°)"))
        
        # Execute all 4 stages in one trajectory with different orientations
        self.execute_push_trajectory(stage1_pos, stage2_pos, stage3_pos, stage4_pos, rpy_stages_1_3, rpy_stage_4)
        
    def execute_push_trajectory(self, stage1_pos, stage2_pos, stage3_pos, stage4_pos, rpy_stages_1_3, rpy_stage_4):
        """Execute complete push trajectory with IK and joint trajectory control"""
        # Compute IK for all four stages with their respective orientations
        self.get_logger().info("🔧 Computing IK for all stages...")
        
        # Stage 1: Use initial orientation (approach at fixed height)
        stage1_joints = compute_ik(stage1_pos, rpy_stages_1_3)
        if stage1_joints is None:
            self.get_logger().error("❌ IK failed for stage 1 (approach)")
            rclpy.shutdown()
            return
        
        # Stage 2: Use initial orientation (pre-push position)
        stage2_joints = compute_ik(stage2_pos, rpy_stages_1_3)
        if stage2_joints is None:
            self.get_logger().error("❌ IK failed for stage 2 (pre-push)")
            rclpy.shutdown()
            return
        
        # Stage 3: Use initial orientation (descend to user height)
        stage3_joints = compute_ik(stage3_pos, rpy_stages_1_3)
        if stage3_joints is None:
            self.get_logger().error("❌ IK failed for stage 3 (descend)")
            rclpy.shutdown()
            return
        
        # Stage 4: Use final target orientation (push end)
        stage4_joints = compute_ik(stage4_pos, rpy_stage_4)
        if stage4_joints is None:
            self.get_logger().error("❌ IK failed for stage 4 (push end)")
            rclpy.shutdown()
            return
        
        self.get_logger().info("✅ IK computed successfully for all stages")
        self.get_logger().info(colorize(36, f"   Stage 1-3 orientation: {rpy_stages_1_3}"))
        self.get_logger().info(colorize(36, f"   Stage 4 orientation: {rpy_stage_4}"))
        
        # Create trajectory with all 4 points
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Point 1: Approach (above initial position at fixed height)
        point1 = JointTrajectoryPoint()
        point1.positions = [float(x) for x in stage1_joints]
        point1.velocities = [0.0] * 6
        point1.time_from_start = Duration(sec=int(self.stage_duration))
        trajectory.points.append(point1)
        
        # Point 2: Pre-push (before object at user height)
        point2 = JointTrajectoryPoint()
        point2.positions = [float(x) for x in stage2_joints]
        point2.velocities = [0.0] * 6
        point2.time_from_start = Duration(sec=int(self.stage_duration * 2))
        trajectory.points.append(point2)
        
        # Point 3: Descend (to user-specified height)
        point3 = JointTrajectoryPoint()
        point3.positions = [float(x) for x in stage3_joints]
        point3.velocities = [0.0] * 6
        point3.time_from_start = Duration(sec=int(self.stage_duration * 3))
        trajectory.points.append(point3)
        
        # Point 4: Push end (final position at user height) - 10 seconds duration
        point4 = JointTrajectoryPoint()
        point4.positions = [float(x) for x in stage4_joints]
        point4.velocities = [0.0] * 6
        point4.time_from_start = Duration(sec=int(self.stage_duration * 3 + 10))  # 3 stages + 10 seconds
        trajectory.points.append(point4)
        
        # Create and send goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=1)
        
        self.get_logger().info("📤 Sending 4-stage push trajectory to robot...")
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
    Main entry point for simulation push primitive.
    
    Example usage:
        python push_sim.py --initial-x 0.0 --initial-y 0.0 --initial-z 0.15 --initial-yaw 0.0 \
                          --final-x 0.2 --final-y -0.3 --final-z 0.15 --final-yaw 45.0 \
                          --ee-height 0.25
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Simulation Push Primitive - Complete 3-stage push sequence with predefined positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic push from (0,0) to (0.2,-0.3) with 45° rotation
  python push_sim.py --initial-x 0.0 --initial-y 0.0 --initial-z 0.15 --initial-yaw 0.0 \\
                    --final-x 0.2 --final-y -0.3 --final-z 0.15 --final-yaw 45.0
  
  # Push with custom EE height and timing
  python push_sim.py --initial-x 0.1 --initial-y 0.1 --initial-z 0.15 --initial-yaw 30.0 \\
                    --final-x 0.3 --final-y -0.2 --final-z 0.15 --final-yaw 90.0 \\
                    --ee-height 0.3 --stage-duration 3.0
        """
    )
    
    # Initial position arguments
    parser.add_argument('--initial-x', type=float, required=True,
                       help='Initial X position in meters [REQUIRED]')
    parser.add_argument('--initial-y', type=float, required=True,
                       help='Initial Y position in meters [REQUIRED]')
    parser.add_argument('--initial-z', type=float, default=0.15,
                       help='Initial Z height in meters (default: 0.15)')
    parser.add_argument('--initial-yaw', type=float, default=0.0,
                       help='Initial yaw angle in degrees (default: 0.0)')
    
    # Final position arguments
    parser.add_argument('--final-x', type=float, required=True,
                       help='Final X position in meters [REQUIRED]')
    parser.add_argument('--final-y', type=float, required=True,
                       help='Final Y position in meters [REQUIRED]')
    parser.add_argument('--final-z', type=float, default=0.15,
                       help='Final Z height in meters (default: 0.15)')
    parser.add_argument('--final-yaw', type=float, default=0.0,
                       help='Final yaw angle in degrees (default: 0.0)')
    
    # Motion parameters
    parser.add_argument('--ee-height', type=float, default=0.25,
                       help='End-effector height during push in meters (default: 0.25)')
    parser.add_argument('--initial-offset', type=float, default=0.05,
                       help='Distance before object to start push in meters (default: 0.05)')
    parser.add_argument('--stage-duration', type=float, default=7.0,
                       help='Duration for each stage movement in seconds (default: 7.0)')
    parser.add_argument('--final-offset', type=float, default=-0.02351,
                       help='Final offset distance in meters for push end position (default: -0.02351)')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Initialize ROS2
    rclpy.init(args=None)
    
    # Create node with parsed arguments
    node = SimulationPush(
        initial_x=parsed_args.initial_x,
        initial_y=parsed_args.initial_y,
        initial_z=parsed_args.initial_z,
        initial_yaw=parsed_args.initial_yaw,
        final_x=parsed_args.final_x,
        final_y=parsed_args.final_y,
        final_z=parsed_args.final_z,
        final_yaw=parsed_args.final_yaw,
        ee_height=parsed_args.ee_height,
        initial_offset=parsed_args.initial_offset,
        stage_duration=parsed_args.stage_duration,
        final_offset=parsed_args.final_offset
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
