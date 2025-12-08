#!/usr/bin/env python3
"""
URSim-like UI for controlling TCP orientation using rotation vectors.
Based on recorded button press data.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import json
import time
import threading
from action_libraries import home_quaternion, compute_ik_quaternion_robust

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter not available. Please install it: sudo apt-get install python3-tk")
    exit(1)


class TCPControlNode(Node):
    def __init__(self):
        super().__init__('tcp_control_node')
        
        # Joint names
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Action client
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Subscribe to TCP pose
        tcp_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.tcp_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.tcp_pose_callback,
            tcp_qos_profile
        )
        
        # Subscribe to joint states for IK seeding
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Current TCP pose
        self.current_pose = None
        self.current_quaternion = None
        self.current_position = None
        self.current_rpy = None
        self.current_rotvec = None
        self.current_joint_angles = None
        
        # TCP offset from end effector (default 0.115 m)
        # This offset extends from the TCP along the end effector's Z-axis
        # The sphere flexure principle applies to this offset point
        self.tcp_offset = 0.115  # meters
        
        # Initial/fixed position for rotation (default to grasp point offset point)
        # This will be the offset point (TCP + offset * Z-axis), not the TCP position
        # Using grasp point 6, direction 2 offset point: (0.250, -0.525, 0.043)
        # Default orientation: [1.0, 0.0, 0.0, 0.0] (RPY: [180°, 0°, 0°]) - pointing straight down
        self.fixed_position = [0.250, -0.525, 0.043]  # Grasp point offset point (fingertips position)
        self.fixed_quaternion = [1.0, 0.0, 0.0, 0.0]  # Default orientation (pointing straight down)
        self.get_logger().info(f"Initialized fixed position (offset point) to grasp point: {self.fixed_position}")
        
        # Button states
        self.button_pressed = None
        self.button_press_start_time = None
        self.rotation_rate = 1.0  # Multiplier for rotation rate (adjustable via slider)
        self.update_interval = 0.05  # seconds between updates
        
        # Target orientation (for discrete movement)
        self.target_quaternion = None  # Target quaternion to move to
        self.accumulated_rotvec = np.array([0.0, 0.0, 0.0])  # Accumulated rotation vector (in radians)
        
        # Rotation vectors for each button (from recorded data)
        # Format: [rx, ry, rz] in radians per update (will be scaled by rotation_rate)
        # Default values based on recorded data analysis
        self.button_rotation_vectors = {
            'RX+': np.array([0.0, -0.05, 0.05]),   # Will be loaded from recorded data
            'RX-': np.array([0.0, -0.02, -0.06]),
            'RY+': np.array([0.0, -0.08, 0.0]),
            'RY-': np.array([0.0, -0.03, 0.0]),
            'RZ+': np.array([0.04, -0.08, 0.0]),
            'RZ-': np.array([0.04, -0.02, 0.0]),
        }
        
        # Load rotation vectors from recorded data if available
        self.load_button_data()
        
        # Control thread
        self.control_thread = None
        self.running = True
        
    def joint_state_callback(self, msg):
        """Update current joint angles for IK seeding"""
        if len(msg.position) >= 6:
            self.current_joint_angles = list(msg.position)
    
    def canonicalize_euler(self, rpy_deg):
        """Canonicalize Euler angles to handle gimbal lock cases"""
        roll, pitch, yaw = rpy_deg
        
        # Handle gimbal lock: when pitch is near 0, roll and yaw are ambiguous
        if abs(pitch) < 1.0:  # Pitch is very close to 0 (within 1 degree)
            # At gimbal lock, we can represent the same orientation with different roll/yaw
            # However, we want to preserve the user's representation if it's already reasonable
            # Only normalize angles to [-180, 180] range, don't change the representation
            roll = ((roll + 180) % 360) - 180
            yaw = ((yaw + 180) % 360) - 180
            return np.array([roll, 0.0, yaw])
        else:
            # Normal case: just normalize angles to [-180, 180]
            roll = ((roll + 180) % 360) - 180
            pitch = ((pitch + 180) % 360) - 180
            yaw = ((yaw + 180) % 360) - 180
            return np.array([roll, pitch, yaw])
    
    def load_button_data(self):
        """Load rotation vector data from recorded button presses"""
        try:
            # Try to find the latest button press data file
            import glob
            import os
            files = glob.glob('button_press_data_*.json')
            if files:
                latest_file = max(files, key=os.path.getmtime)
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                
                buttons = data.get('buttons', {})
                for button_name, button_data in buttons.items():
                    if 'rotation_vector_change_deg' in button_data:
                        rotvec_change_deg = np.array(button_data['rotation_vector_change_deg'])
                        duration = button_data.get('duration', 1.0)
                        
                        # Convert to radians
                        rotvec_change = np.deg2rad(rotvec_change_deg)
                        
                        # Store the normalized direction vector (not per-update)
                        # This represents the direction and relative magnitude of rotation
                        rotvec_magnitude = np.linalg.norm(rotvec_change)
                        if rotvec_magnitude > 0:
                            # Normalize to get direction, but preserve the relative magnitudes
                            # We'll scale this to 10 degrees in button_press
                            rotvec_normalized = rotvec_change / rotvec_magnitude
                            self.button_rotation_vectors[button_name] = rotvec_normalized
                            self.get_logger().info(f"Loaded rotation vector direction for {button_name}: {rotvec_normalized} (magnitude: {rotvec_magnitude:.4f} rad)")
                        else:
                            self.get_logger().warn(f"Zero magnitude rotation vector for {button_name}")
        except Exception as e:
            self.get_logger().warn(f"Could not load button data: {e}. Using defaults.")
    
    def compute_offset_point(self, tcp_position, quaternion):
        """Compute the offset point from TCP position along end effector Z-axis
        
        Args:
            tcp_position: TCP position [x, y, z]
            quaternion: TCP orientation [x, y, z, w]
        
        Returns:
            offset_point: Position of the offset point [x, y, z]
        """
        # Z-axis in gripper frame points from TCP to fingertips (downward)
        # In gripper frame: [0, 0, 1] points from TCP to fingertips
        z_axis_gripper = np.array([0.0, 0.0, 1.0])
        
        # Rotate Z-axis to world frame
        r = Rot.from_quat(quaternion)
        z_axis_world = r.apply(z_axis_gripper)
        
        # Compute offset point: TCP + offset * Z-axis direction
        offset_point = np.array(tcp_position) + self.tcp_offset * z_axis_world
        
        return offset_point.tolist()
    
    def compute_tcp_from_offset_point(self, offset_point, quaternion):
        """Compute TCP position from offset point
        
        Args:
            offset_point: Position of the offset point [x, y, z]
            quaternion: TCP orientation [x, y, z, w]
        
        Returns:
            tcp_position: TCP position [x, y, z]
        """
        # Z-axis in gripper frame points from TCP to fingertips (downward)
        z_axis_gripper = np.array([0.0, 0.0, 1.0])
        
        # Rotate Z-axis to world frame
        r = Rot.from_quat(quaternion)
        z_axis_world = r.apply(z_axis_gripper)
        
        # Compute TCP position: offset_point - offset * Z-axis direction
        tcp_position = np.array(offset_point) - self.tcp_offset * z_axis_world
        
        return tcp_position.tolist()
    
    def tcp_pose_callback(self, msg):
        """Update current TCP pose"""
        self.current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.current_quaternion = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        
        # Initialize fixed position to offset point from current TCP if not set
        # (Note: fixed_position is now pre-set to grasp point, so this may not be called)
        if self.fixed_position is None:
            offset_point = self.compute_offset_point(self.current_position, self.current_quaternion)
            self.fixed_position = offset_point
            self.fixed_quaternion = list(self.current_quaternion)
            self.get_logger().info(f"Initialized fixed position (offset point) to: {self.fixed_position} (TCP: {self.current_position})")
        
        # Initialize target quaternion to current if not set
        if self.target_quaternion is None:
            self.target_quaternion = list(self.current_quaternion)
            # Initialize accumulated rotation vector from current quaternion
            r = Rot.from_quat(self.current_quaternion)
            self.accumulated_rotvec = r.as_rotvec()
        
        # Convert to RPY and rotation vector
        r = Rot.from_quat(self.current_quaternion)
        rpy_raw = r.as_euler('xyz', degrees=True)
        # Normalize angles to [-180, 180] range (don't canonicalize to preserve representation)
        roll, pitch, yaw = rpy_raw
        roll = ((roll + 180) % 360) - 180
        pitch = ((pitch + 180) % 360) - 180
        yaw = ((yaw + 180) % 360) - 180
        self.current_rpy = np.array([roll, pitch, yaw])
        self.current_rotvec = r.as_rotvec()
        
        self.current_pose = {
            'position': self.current_position,
            'quaternion': self.current_quaternion,
            'rpy': self.current_rpy,
            'rotvec': self.current_rotvec
        }
    
    def set_fixed_position(self, position=None):
        """Set the fixed position for rotation (defaults to current offset point)
        
        Note: The fixed_position is the offset point (TCP + offset * Z-axis),
        not the TCP position itself. This is the point around which rotations occur.
        """
        if position is None:
            if self.current_position is not None and self.current_quaternion is not None:
                # Compute offset point from current TCP position
                offset_point = self.compute_offset_point(self.current_position, self.current_quaternion)
                self.fixed_position = list(offset_point)  # Make a copy
                self.fixed_quaternion = list(self.current_quaternion)  # Make a copy
                self.get_logger().info(f"Fixed position (offset point) set to: {self.fixed_position} (TCP: {self.current_position})")
                return True
            else:
                return False
        else:
            # If position is provided, it's assumed to be the offset point
            self.fixed_position = list(position)  # Make a copy
            if self.current_quaternion is not None:
                self.fixed_quaternion = list(self.current_quaternion)  # Make a copy
            self.get_logger().info(f"Fixed position (offset point) set to: {self.fixed_position}")
            return True
    
    def apply_rotation_vector(self, rotvec_delta):
        """Apply a rotation vector increment around the fixed position (offset point)"""
        if self.current_quaternion is None:
            return False
        
        # Get current quaternion
        current_quat = self.current_quaternion
        
        # Create rotation from rotation vector delta
        r_delta = Rot.from_rotvec(rotvec_delta)
        
        # Compose rotations
        r_current = Rot.from_quat(current_quat)
        r_new = r_current * r_delta
        new_quat = r_new.as_quat()
        
        # Use fixed position (offset point) - rotate around this point
        if self.fixed_position is None:
            # Fallback: compute offset point from current TCP position
            if self.current_position is None:
                return False
            offset_point = self.compute_offset_point(self.current_position, self.current_quaternion)
        else:
            offset_point = self.fixed_position
        
        # Compute TCP position from offset point with new orientation
        # This ensures the offset point remains fixed while rotating
        tcp_position = self.compute_tcp_from_offset_point(offset_point, new_quat)
        
        # Compute IK with current joint angles as seed for better convergence
        q_guess = self.current_joint_angles if self.current_joint_angles is not None else None
        
        # Try with seed first
        if q_guess is not None:
            from ik_solver import compute_ik_quaternion
            joint_angles = compute_ik_quaternion(
                tcp_position,
                new_quat,
                q_guess=q_guess,
                max_tries=3,
                dx=0.001
            )
        else:
            joint_angles = None
        
        # If that fails, try robust solver
        if joint_angles is None:
            joint_angles = compute_ik_quaternion_robust(
                tcp_position, 
                new_quat, 
                max_tries=3, 
                dx=0.001, 
                multiple_seeds=True,
                q_guess=q_guess
            )
        
        if joint_angles is None:
            return False
        
        # Send trajectory with 5 second duration to avoid exceeding joint velocity limits
        return self.send_trajectory([joint_angles], 5.0)
    
    def send_trajectory(self, joint_angles_list, duration):
        """Send trajectory to robot
        
        Args:
            joint_angles_list: List of joint angle arrays, or single duration for all waypoints
            duration: If single value, total duration for trajectory. If list, duration per waypoint.
        """
        if not self.action_client.server_is_ready():
            return False
        
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        points = []
        if isinstance(duration, (list, tuple)) and len(duration) == len(joint_angles_list):
            # Duration per waypoint
            cumulative_time = 0.0
            for i, (joint_angles, waypoint_duration) in enumerate(zip(joint_angles_list, duration)):
                cumulative_time += waypoint_duration
                point = JointTrajectoryPoint()
                point.positions = [float(x) for x in joint_angles]
                point.velocities = [0.0] * 6
                point.time_from_start = Duration(sec=int(cumulative_time), nanosec=int((cumulative_time % 1) * 1e9))
                points.append(point)
        else:
            # Single duration - distribute evenly across waypoints
            time_per_waypoint = duration / len(joint_angles_list) if len(joint_angles_list) > 0 else duration
            cumulative_time = 0.0
            for i, joint_angles in enumerate(joint_angles_list):
                cumulative_time += time_per_waypoint
                point = JointTrajectoryPoint()
                point.positions = [float(x) for x in joint_angles]
                point.velocities = [0.0] * 6
                point.time_from_start = Duration(sec=int(cumulative_time), nanosec=int((cumulative_time % 1) * 1e9))
                points.append(point)
        
        traj.points = points
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=1)
        
        # Send goal (fire and forget for continuous control)
        self.action_client.send_goal_async(goal)
        return True
    
    def interpolate_quaternion_trajectory(self, start_quat, target_quat, position, num_waypoints=10, total_duration=5.0):
        """Interpolate between start and target quaternions using SLERP, creating intermediate waypoints
        
        Args:
            start_quat: Starting quaternion
            target_quat: Target quaternion
            position: Offset point position (the point around which rotation occurs)
            num_waypoints: Number of intermediate waypoints
            total_duration: Total duration for trajectory
        """
        if start_quat is None or target_quat is None:
            return None
        
        # Normalize quaternions
        start_quat = np.array(start_quat)
        target_quat = np.array(target_quat)
        start_quat = start_quat / np.linalg.norm(start_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # Use SLERP to interpolate
        trajectory_points = []
        previous_joint_angles = self.current_joint_angles
        last_positions = None
        
        for i in range(num_waypoints + 1):  # +1 to include target
            # Interpolation parameter [0, 1]
            t = i / num_waypoints if num_waypoints > 0 else 1.0
            
            # SLERP interpolation using scipy's Slerp
            from scipy.spatial.transform import Slerp
            times = [0, 1]
            key_rots = Rot.from_quat([start_quat, target_quat])
            slerp = Slerp(times, key_rots)
            r_interp = slerp([t])[0]
            interp_quat = r_interp.as_quat()
            
            # Compute TCP position from offset point with interpolated orientation
            # This ensures the offset point remains fixed while rotating
            tcp_position = self.compute_tcp_from_offset_point(position, interp_quat)
            
            # Compute IK for this waypoint
            q_guess = previous_joint_angles if previous_joint_angles is not None else None
            
            # Try with seed first
            if q_guess is not None:
                from ik_solver import compute_ik_quaternion
                joint_angles = compute_ik_quaternion(
                    tcp_position,
                    interp_quat,
                    q_guess=q_guess,
                    max_tries=3,
                    dx=0.001
                )
            else:
                joint_angles = None
            
            # If that fails, try robust solver
            if joint_angles is None:
                joint_angles = compute_ik_quaternion_robust(
                    tcp_position,
                    interp_quat,
                    max_tries=3,
                    dx=0.001,
                    multiple_seeds=True,
                    q_guess=q_guess
                )
            
            if joint_angles is None:
                self.get_logger().warn(f"IK failed for waypoint {i+1}/{num_waypoints+1}")
                # Skip this waypoint
                continue
            
            # Check for duplicates
            current_positions = tuple(joint_angles)
            if current_positions == last_positions:
                continue
            
            previous_joint_angles = joint_angles
            last_positions = current_positions
            
            trajectory_points.append(joint_angles)
        
        return trajectory_points if trajectory_points else None
    
    def move_to_home(self):
        """Move robot to home position and set it as fixed position"""
        points = home_quaternion()
        if not points:
            return False
        
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        traj.points = [JointTrajectoryPoint(
            positions=pt["positions"],
            velocities=pt["velocities"],
            time_from_start=pt["time_from_start"]
        ) for pt in points]
        
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=1)
        
        future = self.action_client.send_goal_async(goal)
        
        # Set fixed position to home offset point after movement
        from action_libraries import HOME_POSE_QUATERNION
        home_tcp_position = HOME_POSE_QUATERNION[0:3]
        home_quaternion = HOME_POSE_QUATERNION[3:7]
        # Compute offset point from home TCP position
        self.fixed_position = self.compute_offset_point(home_tcp_position, home_quaternion)
        self.fixed_quaternion = list(home_quaternion)
        
        return True
    
    def move_to_fixed_position(self):
        """Move robot to the fixed position (offset point)"""
        if self.fixed_position is None or self.fixed_quaternion is None:
            return False
        
        # Compute TCP position from offset point
        tcp_position = self.compute_tcp_from_offset_point(self.fixed_position, self.fixed_quaternion)
        
        # Compute IK
        joint_angles = compute_ik_quaternion_robust(
            tcp_position,
            self.fixed_quaternion,
            max_tries=5,
            dx=0.001,
            multiple_seeds=True
        )
        
        if joint_angles is None:
            return False
        
        # Send trajectory with 5 second duration to avoid exceeding joint velocity limits
        return self.send_trajectory([joint_angles], 5.0)
    
    def button_press(self, button_name):
        """Handle button press - start continuous rotation"""
        if self.current_quaternion is None:
            return
        
        self.button_pressed = button_name
        self.button_press_start_time = time.time()
        self.get_logger().info(f"Button {button_name} pressed - starting continuous rotation")
    
    def button_release(self):
        """Handle button release - stop continuous rotation"""
        if self.button_pressed is not None:
            self.get_logger().info(f"Button {self.button_pressed} released")
            self.button_pressed = None
            self.button_press_start_time = None
    
    def move_to_target_orientation(self):
        """Move robot to target orientation around fixed position (offset point)"""
        if self.target_quaternion is None:
            self.get_logger().warn("No target orientation set")
            return False
        
        # Get offset point (fixed position)
        if self.fixed_position is None:
            if self.current_position is None:
                self.get_logger().warn("No fixed position or current position available")
                return False
            # Compute offset point from current TCP position
            offset_point = self.compute_offset_point(self.current_position, self.current_quaternion)
        else:
            offset_point = self.fixed_position
        
        # Compute TCP position from offset point with target orientation
        # This ensures the offset point remains fixed while rotating
        tcp_position = self.compute_tcp_from_offset_point(offset_point, self.target_quaternion)
        
        # Compute IK with current joint angles as seed
        q_guess = self.current_joint_angles if self.current_joint_angles is not None else None
        
        # Try with seed first
        if q_guess is not None:
            from ik_solver import compute_ik_quaternion
            joint_angles = compute_ik_quaternion(
                tcp_position,
                self.target_quaternion,
                q_guess=q_guess,
                max_tries=3,
                dx=0.001
            )
        else:
            joint_angles = None
        
        # If that fails, try robust solver
        if joint_angles is None:
            joint_angles = compute_ik_quaternion_robust(
                tcp_position, 
                self.target_quaternion, 
                max_tries=5, 
                dx=0.001, 
                multiple_seeds=True,
                q_guess=q_guess
            )
        
        if joint_angles is None:
            self.get_logger().warn("IK failed for target orientation")
            return False
        
        # Send trajectory with 5 second duration to avoid exceeding joint velocity limits
        success = self.send_trajectory([joint_angles], 5.0)
        if success:
            # Update current quaternion to target after movement
            self.current_quaternion = self.target_quaternion
        return success
    
    def control_loop(self):
        """Continuous control loop - applies rotation while button is held"""
        while self.running:
            if self.button_pressed and self.current_quaternion is not None:
                # Control logic for UR5e with end-effector pointing downward
                # Each button rotates around its axis in the robot's BASE frame
                
                # Rotation rate in radians per update (scaled by rotation_rate slider)
                base_rate = 0.01  # Base rotation rate in rad/update
                rate = base_rate * self.rotation_rate
                
                # Get current orientation
                base_quat = self.target_quaternion if self.target_quaternion is not None else self.current_quaternion
                
                if base_quat is not None:
                    # Convert to RPY to check for gimbal lock
                    r_current = Rot.from_quat(base_quat)
                    rpy_raw = r_current.as_euler('xyz', degrees=True)
                    # Normalize angles to [-180, 180] range
                    roll, pitch, yaw = rpy_raw
                    roll = ((roll + 180) % 360) - 180
                    pitch = ((pitch + 180) % 360) - 180
                    yaw = ((yaw + 180) % 360) - 180
                    pitch_deg = pitch
                    
                    # Check if we're at gimbal lock (pitch near 0)
                    at_gimbal_lock = abs(pitch_deg) < 1.0
                    
                    # Handle rotation around Y axis at gimbal lock specially
                    if at_gimbal_lock and (self.button_pressed == 'RY+' or self.button_pressed == 'RY-'):
                        # At gimbal lock, rotating around Y axis should adjust roll while keeping yaw constant
                        # This creates a new orientation (not just a different representation)
                        # Use the normalized roll, pitch, yaw values
                        
                        # Convert rotation rate to degrees
                        rate_deg = np.degrees(rate)
                        
                        if self.button_pressed == 'RY+':
                            # Decrease roll (e.g., 180 -> -179)
                            roll_new = roll - rate_deg
                        else:  # RY-
                            # Increase roll (e.g., -179 -> 180)
                            roll_new = roll + rate_deg
                        
                        # Normalize roll to [-180, 180]
                        roll_new = ((roll_new + 180) % 360) - 180
                        
                        # Keep pitch at 0 and yaw unchanged
                        # Convert back to quaternion
                        r_new = Rot.from_euler('xyz', [np.deg2rad(roll_new), 0.0, np.deg2rad(yaw)], degrees=False)
                        new_quat = r_new.as_quat()
                    else:
                        # Normal rotation using rotation vectors
                        # Map button to rotation axis in robot's base frame
                        if self.button_pressed == 'RX+':
                            rotvec_delta = np.array([rate, 0.0, 0.0])
                        elif self.button_pressed == 'RX-':
                            rotvec_delta = np.array([-rate, 0.0, 0.0])
                        elif self.button_pressed == 'RY+':
                            rotvec_delta = np.array([0.0, rate, 0.0])
                        elif self.button_pressed == 'RY-':
                            rotvec_delta = np.array([0.0, -rate, 0.0])
                        elif self.button_pressed == 'RZ+':
                            rotvec_delta = np.array([0.0, 0.0, rate])
                        elif self.button_pressed == 'RZ-':
                            rotvec_delta = np.array([0.0, 0.0, -rate])
                        else:
                            time.sleep(self.update_interval)
                            continue
                        
                        # Create rotation from rotation vector delta (in base frame)
                        r_delta = Rot.from_rotvec(rotvec_delta)
                        
                        # Compose rotations: new_orientation = current_orientation * delta_rotation
                        r_new = r_current * r_delta
                        new_quat = r_new.as_quat()
                    
                    # Update target quaternion
                    self.target_quaternion = new_quat
                    
                    # Update accumulated rotation vector from the new quaternion
                    r_new_rotvec = Rot.from_quat(new_quat)
                    self.accumulated_rotvec = r_new_rotvec.as_rotvec()
                    
                    # Only update target, don't move robot yet
                    # Robot will move when "Move to Target Orientation" button is clicked
            
            time.sleep(self.update_interval)
    


class TCPControlUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("URSim TCP Orientation Control")
        self.root.geometry("800x600")
        
        # Button states
        self.button_states = {}
        
        # Create UI
        self.create_ui()
        
        # Start control thread
        self.node.control_thread = threading.Thread(target=self.node.control_loop, daemon=True)
        self.node.control_thread.start()
        
        # Start update loop
        self.update_loop()
    
    def create_ui(self):
        """Create the UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="TCP Orientation Control", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Current pose display
        pose_frame = ttk.LabelFrame(main_frame, text="Current TCP Pose", padding="10")
        pose_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.position_label = ttk.Label(pose_frame, text="Position: [0.000, 0.000, 0.000] m")
        self.position_label.grid(row=0, column=0, sticky=tk.W)
        
        self.rpy_label = ttk.Label(pose_frame, text="RPY: Roll=0.00°, Pitch=0.00°, Yaw=0.00°")
        self.rpy_label.grid(row=1, column=0, sticky=tk.W)
        
        self.rotvec_label = ttk.Label(pose_frame, text="Rotation Vector: [0.000, 0.000, 0.000] rad")
        self.rotvec_label.grid(row=2, column=0, sticky=tk.W)
        
        # Position input and move section
        position_frame = ttk.LabelFrame(main_frame, text="Set TCP Position (Rotation Point)", padding="10")
        position_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Position input fields
        input_frame = ttk.Frame(position_frame)
        input_frame.grid(row=0, column=0, columnspan=4, pady=5)
        
        ttk.Label(input_frame, text="X (m):", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.x_var = tk.StringVar(value="0.250")  # Grasp point offset point X
        x_entry = ttk.Entry(input_frame, textvariable=self.x_var, width=12, font=("Arial", 10))
        x_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Y (m):", font=("Arial", 10)).grid(row=0, column=2, padx=5)
        self.y_var = tk.StringVar(value="-0.525")  # Grasp point offset point Y
        y_entry = ttk.Entry(input_frame, textvariable=self.y_var, width=12, font=("Arial", 10))
        y_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(input_frame, text="Z (m):", font=("Arial", 10)).grid(row=0, column=4, padx=5)
        self.z_var = tk.StringVar(value="0.043")  # Grasp point offset point Z
        z_entry = ttk.Entry(input_frame, textvariable=self.z_var, width=12, font=("Arial", 10))
        z_entry.grid(row=0, column=5, padx=5)
        
        # Buttons to set and move to position
        button_frame_pos = ttk.Frame(position_frame)
        button_frame_pos.grid(row=1, column=0, columnspan=4, pady=10)
        
        # Set to current position button
        set_current_btn = tk.Button(
            button_frame_pos,
            text="Set to Current Position",
            width=18,
            height=2,
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            command=self.set_fixed_to_current
        )
        set_current_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Set to home position button
        set_home_btn = tk.Button(
            button_frame_pos,
            text="Set to Home Position",
            width=18,
            height=2,
            font=("Arial", 10),
            bg="#4CAF50",
            fg="white",
            command=self.set_fixed_to_home
        )
        set_home_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Move to entered position button (main action)
        move_to_position_btn = tk.Button(
            button_frame_pos,
            text="Move to Position",
            width=18,
            height=2,
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            command=self.move_to_entered_position
        )
        move_to_position_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Display current fixed position (offset point)
        self.fixed_position_label = ttk.Label(
            position_frame, 
            text="Fixed Position (Offset Point): [0.250, -0.525, 0.043] m (Rotation Point - Grasp Point)",
            font=("Arial", 10, "italic")
        )
        self.fixed_position_label.grid(row=2, column=0, columnspan=4, pady=5)
        
        # TCP offset control
        offset_control_frame = ttk.Frame(position_frame)
        offset_control_frame.grid(row=3, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(offset_control_frame, text="TCP Offset (m):", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        
        # Entry field for precise input
        self.tcp_offset_var = tk.DoubleVar(value=self.node.tcp_offset)
        offset_entry = ttk.Entry(offset_control_frame, textvariable=self.tcp_offset_var, width=10, font=("Arial", 10))
        offset_entry.grid(row=0, column=1, padx=5)
        offset_entry.bind('<Return>', lambda e: self.update_tcp_offset())
        offset_entry.bind('<FocusOut>', lambda e: self.update_tcp_offset())
        
        # Slider for easy adjustment
        offset_scale = ttk.Scale(
            offset_control_frame, 
            from_=0.0, 
            to=0.3, 
            variable=self.tcp_offset_var, 
            orient=tk.HORIZONTAL, 
            length=200,
            command=lambda v: self.update_tcp_offset_from_slider(float(v))
        )
        offset_scale.grid(row=0, column=2, padx=5)
        
        # Display current offset value
        self.tcp_offset_label = ttk.Label(
            offset_control_frame,
            text=f"Current: {self.node.tcp_offset:.3f} m",
            font=("Arial", 9, "italic")
        )
        self.tcp_offset_label.grid(row=0, column=3, padx=5)
        
        # RPY input and control section
        rpy_frame = ttk.LabelFrame(main_frame, text="RPY Control (Roll-Pitch-Yaw in Degrees)", padding="10")
        rpy_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # RPY input fields
        rpy_input_frame = ttk.Frame(rpy_frame)
        rpy_input_frame.grid(row=0, column=0, columnspan=6, pady=5)
        
        ttk.Label(rpy_input_frame, text="Roll (°):", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.roll_var = tk.StringVar(value="0.0")
        self.roll_entry = ttk.Entry(rpy_input_frame, textvariable=self.roll_var, width=12, font=("Arial", 10))
        self.roll_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(rpy_input_frame, text="Pitch (°):", font=("Arial", 10)).grid(row=0, column=2, padx=5)
        self.pitch_var = tk.StringVar(value="0.0")
        self.pitch_entry = ttk.Entry(rpy_input_frame, textvariable=self.pitch_var, width=12, font=("Arial", 10))
        self.pitch_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(rpy_input_frame, text="Yaw (°):", font=("Arial", 10)).grid(row=0, column=4, padx=5)
        self.yaw_var = tk.StringVar(value="0.0")
        self.yaw_entry = ttk.Entry(rpy_input_frame, textvariable=self.yaw_var, width=12, font=("Arial", 10))
        self.yaw_entry.grid(row=0, column=5, padx=5)
        
        # RPY increment/decrement buttons
        rpy_inc_frame = ttk.Frame(rpy_frame)
        rpy_inc_frame.grid(row=1, column=0, columnspan=6, pady=5)
        
        # Roll controls
        ttk.Label(rpy_inc_frame, text="Roll:", font=("Arial", 9)).grid(row=0, column=0, padx=2)
        roll_dec_btn = tk.Button(rpy_inc_frame, text="-", width=3, height=1, font=("Arial", 9),
                                 command=lambda: self.adjust_rpy('roll', -1))
        roll_dec_btn.grid(row=0, column=1, padx=2)
        roll_inc_btn = tk.Button(rpy_inc_frame, text="+", width=3, height=1, font=("Arial", 9),
                                command=lambda: self.adjust_rpy('roll', 1))
        roll_inc_btn.grid(row=0, column=2, padx=2)
        
        # Pitch controls
        ttk.Label(rpy_inc_frame, text="Pitch:", font=("Arial", 9)).grid(row=0, column=3, padx=2)
        pitch_dec_btn = tk.Button(rpy_inc_frame, text="-", width=3, height=1, font=("Arial", 9),
                                  command=lambda: self.adjust_rpy('pitch', -1))
        pitch_dec_btn.grid(row=0, column=4, padx=2)
        pitch_inc_btn = tk.Button(rpy_inc_frame, text="+", width=3, height=1, font=("Arial", 9),
                                 command=lambda: self.adjust_rpy('pitch', 1))
        pitch_inc_btn.grid(row=0, column=5, padx=2)
        
        # Yaw controls
        ttk.Label(rpy_inc_frame, text="Yaw:", font=("Arial", 9)).grid(row=0, column=6, padx=2)
        yaw_dec_btn = tk.Button(rpy_inc_frame, text="-", width=3, height=1, font=("Arial", 9),
                               command=lambda: self.adjust_rpy('yaw', -1))
        yaw_dec_btn.grid(row=0, column=7, padx=2)
        yaw_inc_btn = tk.Button(rpy_inc_frame, text="+", width=3, height=1, font=("Arial", 9),
                               command=lambda: self.adjust_rpy('yaw', 1))
        yaw_inc_btn.grid(row=0, column=8, padx=2)
        
        # RPY step size
        ttk.Label(rpy_inc_frame, text="Step:", font=("Arial", 9)).grid(row=0, column=9, padx=5)
        self.rpy_step_var = tk.DoubleVar(value=1.0)
        rpy_step_scale = ttk.Scale(rpy_inc_frame, from_=0.1, to=10.0, variable=self.rpy_step_var, 
                                   orient=tk.HORIZONTAL, length=100)
        rpy_step_scale.grid(row=0, column=10, padx=2)
        self.rpy_step_label = ttk.Label(rpy_inc_frame, text="1.0°", font=("Arial", 8))
        self.rpy_step_label.grid(row=0, column=11, padx=2)
        rpy_step_scale.configure(command=lambda v: self.rpy_step_label.config(text=f"{float(v):.1f}°"))
        
        # RPY action buttons
        rpy_button_frame = ttk.Frame(rpy_frame)
        rpy_button_frame.grid(row=2, column=0, columnspan=6, pady=10)
        
        set_current_rpy_btn = tk.Button(
            rpy_button_frame,
            text="Set to Current RPY",
            width=18,
            height=2,
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            command=self.set_rpy_to_current
        )
        set_current_rpy_btn.grid(row=0, column=0, padx=5, pady=5)
        
        move_to_rpy_btn = tk.Button(
            rpy_button_frame,
            text="Move to RPY",
            width=18,
            height=2,
            font=("Arial", 12, "bold"),
            bg="#9C27B0",
            fg="white",
            command=self.move_to_rpy
        )
        move_to_rpy_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Button frame with editable rotation vectors
        button_frame = ttk.LabelFrame(main_frame, text="TCP Orientation Buttons", padding="10")
        button_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Create buttons in a grid with editable rotation vector values
        buttons = [
            ('RX+', 0, 0), ('RX-', 0, 1),
            ('RY+', 1, 0), ('RY-', 1, 1),
            ('RZ+', 2, 0), ('RZ-', 2, 1),
        ]
        
        self.buttons = {}
        self.rotvec_vars = {}  # Store rotation vector input variables
        
        for button_name, row, col in buttons:
            # Button
            btn = tk.Button(
                button_frame,
                text=button_name,
                width=12,
                height=2,
                font=("Arial", 11, "bold"),
                bg="#4CAF50" if "+" in button_name else "#f44336",
                fg="white",
                relief=tk.RAISED,
                bd=3
            )
            btn.grid(row=row*2, column=col, padx=5, pady=5)
            
            # Bind events
            btn.bind('<Button-1>', lambda e, name=button_name: self.on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=button_name: self.on_button_release(name))
            
            self.buttons[button_name] = btn
            self.button_states[button_name] = False
            
            # Rotation vector input fields (below each button)
            rotvec_frame = ttk.Frame(button_frame)
            rotvec_frame.grid(row=row*2+1, column=col, padx=5, pady=2)
            
            # Get default values
            default_rotvec = self.node.button_rotation_vectors.get(button_name, np.array([0.0, 0.0, 0.0]))
            
            ttk.Label(rotvec_frame, text="RX:", font=("Arial", 8)).grid(row=0, column=0, padx=1)
            rx_var = tk.StringVar(value=f"{default_rotvec[0]:.4f}")
            ttk.Entry(rotvec_frame, textvariable=rx_var, width=8, font=("Arial", 8)).grid(row=0, column=1, padx=1)
            
            ttk.Label(rotvec_frame, text="RY:", font=("Arial", 8)).grid(row=0, column=2, padx=1)
            ry_var = tk.StringVar(value=f"{default_rotvec[1]:.4f}")
            ttk.Entry(rotvec_frame, textvariable=ry_var, width=8, font=("Arial", 8)).grid(row=0, column=3, padx=1)
            
            ttk.Label(rotvec_frame, text="RZ:", font=("Arial", 8)).grid(row=0, column=4, padx=1)
            rz_var = tk.StringVar(value=f"{default_rotvec[2]:.4f}")
            ttk.Entry(rotvec_frame, textvariable=rz_var, width=8, font=("Arial", 8)).grid(row=0, column=5, padx=1)
            
            # Store variables
            self.rotvec_vars[button_name] = {'rx': rx_var, 'ry': ry_var, 'rz': rz_var}
            
            # Update button to save rotation vector values
            update_btn = tk.Button(
                rotvec_frame,
                text="Update",
                width=6,
                height=1,
                font=("Arial", 7),
                command=lambda name=button_name: self.update_button_rotvec(name)
            )
            update_btn.grid(row=0, column=6, padx=2)
        
        # Target orientation display
        target_frame = ttk.LabelFrame(main_frame, text="Target Orientation", padding="10")
        target_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.target_rpy_label = ttk.Label(target_frame, text="Target RPY: Not set", font=("Arial", 10))
        self.target_rpy_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.target_rotvec_label = ttk.Label(target_frame, text="Target Rotation Vector: Not set", font=("Arial", 10))
        self.target_rotvec_label.grid(row=1, column=0, sticky=tk.W, padx=5)
        
        # Move button
        move_btn = tk.Button(
            target_frame,
            text="Move to Target Orientation",
            width=25,
            height=2,
            font=("Arial", 12, "bold"),
            bg="#FF5722",
            fg="white",
            command=self.move_to_target
        )
        move_btn.grid(row=0, column=1, rowspan=2, padx=10, pady=5)
        
        # Reset target button
        reset_target_btn = tk.Button(
            target_frame,
            text="Reset to Current",
            width=15,
            height=2,
            font=("Arial", 10),
            bg="#607D8B",
            fg="white",
            command=self.reset_target
        )
        reset_target_btn.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Rotation rate control
        rate_frame = ttk.Frame(control_frame)
        rate_frame.grid(row=0, column=0, padx=10)
        
        ttk.Label(rate_frame, text="Rotation Rate:", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.rate_var = tk.DoubleVar(value=1.0)
        rate_scale = ttk.Scale(rate_frame, from_=0.1, to=5.0, variable=self.rate_var, orient=tk.HORIZONTAL, length=200)
        rate_scale.grid(row=0, column=1, padx=5)
        self.rate_label = ttk.Label(rate_frame, text="1.0x", font=("Arial", 10))
        self.rate_label.grid(row=0, column=2, padx=5)
        rate_scale.configure(command=lambda v: self.rate_label.config(text=f"{float(v):.1f}x"))
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Status: Ready", font=("Arial", 10))
        self.status_label.grid(row=8, column=0, columnspan=3, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
    def on_button_press(self, button_name):
        """Handle button press - update target orientation"""
        self.button_states[button_name] = True
        self.buttons[button_name].config(relief=tk.SUNKEN, bg="#2E7D32" if "+" in button_name else "#C62828")
        self.node.button_press(button_name)
        self.update_target_display()
        self.status_label.config(text=f"Status: {button_name} pressed - target updated")
    
    def on_button_release(self, button_name):
        """Handle button release"""
        self.button_states[button_name] = False
        self.buttons[button_name].config(relief=tk.RAISED, bg="#4CAF50" if "+" in button_name else "#f44336")
        self.node.button_release()
        self.status_label.config(text="Status: Target orientation updated")
    
    def move_to_target(self):
        """Move robot to target orientation"""
        self.status_label.config(text="Status: Moving to target orientation...")
        if self.node.move_to_target_orientation():
            self.status_label.config(text="Status: Movement command sent")
            # Reset target to current after movement
            if self.node.current_quaternion is not None:
                self.node.target_quaternion = list(self.node.current_quaternion)
                # Update accumulated rotation vector from current quaternion after movement
                r = Rot.from_quat(self.node.current_quaternion)
                self.node.accumulated_rotvec = r.as_rotvec()
                self.update_target_display()
        else:
            self.status_label.config(text="Status: Failed to move to target orientation")
    
    def reset_target(self):
        """Reset target orientation to current orientation"""
        if self.node.current_quaternion is not None:
            self.node.target_quaternion = list(self.node.current_quaternion)
            # Reset accumulated rotation vector from current quaternion
            r = Rot.from_quat(self.node.current_quaternion)
            self.node.accumulated_rotvec = r.as_rotvec()
            self.update_target_display()
            self.status_label.config(text="Status: Target reset to current orientation")
        else:
            self.status_label.config(text="Status: No current orientation available")
    
    def update_target_display(self):
        """Update target orientation display"""
        if self.node.target_quaternion is not None:
            # Convert to RPY
            r = Rot.from_quat(self.node.target_quaternion)
            rpy = r.as_euler('xyz', degrees=True)
            
            # Use accumulated rotation vector for display (in radians)
            accumulated_rotvec = self.node.accumulated_rotvec
            
            self.target_rpy_label.config(text=f"Target RPY: Roll={rpy[0]:.2f}°, Pitch={rpy[1]:.2f}°, Yaw={rpy[2]:.2f}°")
            self.target_rotvec_label.config(text=f"Target Rotation Vector: [{accumulated_rotvec[0]:.3f}, {accumulated_rotvec[1]:.3f}, {accumulated_rotvec[2]:.3f}] rad")
        else:
            self.target_rpy_label.config(text="Target RPY: Not set")
            self.target_rotvec_label.config(text="Target Rotation Vector: Not set")
    
    def update_button_rotvec(self, button_name):
        """Update rotation vector for a button from UI input"""
        try:
            rx = float(self.rotvec_vars[button_name]['rx'].get())
            ry = float(self.rotvec_vars[button_name]['ry'].get())
            rz = float(self.rotvec_vars[button_name]['rz'].get())
            
            self.node.button_rotation_vectors[button_name] = np.array([rx, ry, rz])
            self.status_label.config(text=f"Status: Updated {button_name} rotation vector: [{rx:.4f}, {ry:.4f}, {rz:.4f}]")
        except ValueError:
            self.status_label.config(text=f"Status: Invalid rotation vector values for {button_name}")
    
    def update_tcp_offset(self):
        """Update TCP offset from entry field"""
        try:
            new_offset = float(self.tcp_offset_var.get())
            if new_offset < 0.0:
                new_offset = 0.0
            elif new_offset > 0.3:
                new_offset = 0.3
            
            # Update the node's TCP offset
            old_offset = self.node.tcp_offset
            self.node.tcp_offset = new_offset
            
            # Recompute fixed position (offset point) if we have current TCP pose
            if self.node.current_position is not None and self.node.current_quaternion is not None:
                # Recompute offset point with new offset
                new_offset_point = self.node.compute_offset_point(
                    self.node.current_position, 
                    self.node.current_quaternion
                )
                self.node.fixed_position = new_offset_point
                
                # Update the entry field to reflect any clamping
                self.tcp_offset_var.set(new_offset)
                
                # Update display
                self.tcp_offset_label.config(text=f"Current: {new_offset:.3f} m")
                if self.node.fixed_position:
                    fixed_pos = self.node.fixed_position
                    self.fixed_position_label.config(
                        text=f"Fixed Position (Offset Point): [{fixed_pos[0]:.3f}, {fixed_pos[1]:.3f}, {fixed_pos[2]:.3f}] m (Rotation Point)"
                    )
                
                self.status_label.config(text=f"Status: TCP offset updated from {old_offset:.3f} m to {new_offset:.3f} m")
            else:
                # Just update the offset value
                self.tcp_offset_var.set(new_offset)
                self.tcp_offset_label.config(text=f"Current: {new_offset:.3f} m")
                self.status_label.config(text=f"Status: TCP offset set to {new_offset:.3f} m (will apply when TCP pose is available)")
        except ValueError:
            # Invalid input, reset to current value
            self.tcp_offset_var.set(self.node.tcp_offset)
            self.status_label.config(text="Status: Invalid TCP offset value")
    
    def update_tcp_offset_from_slider(self, value):
        """Update TCP offset from slider (called continuously while dragging)"""
        try:
            # Clamp value to valid range
            if value < 0.0:
                value = 0.0
            elif value > 0.3:
                value = 0.3
            
            # Update the node's TCP offset
            self.node.tcp_offset = value
            
            # Recompute fixed position (offset point) if we have current TCP pose
            if self.node.current_position is not None and self.node.current_quaternion is not None:
                # Recompute offset point with new offset
                new_offset_point = self.node.compute_offset_point(
                    self.node.current_position, 
                    self.node.current_quaternion
                )
                self.node.fixed_position = new_offset_point
                
                # Update display
                self.tcp_offset_label.config(text=f"Current: {value:.3f} m")
                if self.node.fixed_position:
                    fixed_pos = self.node.fixed_position
                    self.fixed_position_label.config(
                        text=f"Fixed Position (Offset Point): [{fixed_pos[0]:.3f}, {fixed_pos[1]:.3f}, {fixed_pos[2]:.3f}] m (Rotation Point)"
                    )
        except Exception as e:
            # Silently handle errors during slider drag
            pass
    
    def adjust_rpy(self, axis, direction):
        """Adjust RPY value by step size, wrapping to [-180, 180] range"""
        try:
            step = self.rpy_step_var.get()
            if axis == 'roll':
                current = float(self.roll_var.get())
                new_value = current + (direction * step)
                # Normalize to [-180, 180] range
                new_value = ((new_value + 180) % 360) - 180
                self.roll_var.set(f"{new_value:.2f}")
            elif axis == 'pitch':
                current = float(self.pitch_var.get())
                new_value = current + (direction * step)
                # Normalize to [-180, 180] range
                new_value = ((new_value + 180) % 360) - 180
                self.pitch_var.set(f"{new_value:.2f}")
            elif axis == 'yaw':
                current = float(self.yaw_var.get())
                new_value = current + (direction * step)
                # Normalize to [-180, 180] range
                new_value = ((new_value + 180) % 360) - 180
                self.yaw_var.set(f"{new_value:.2f}")
        except ValueError:
            pass
    
    def set_rpy_to_current(self):
        """Set RPY input fields to current RPY values"""
        if self.node.current_rpy is not None:
            rpy = self.node.current_rpy
            self.roll_var.set(f"{rpy[0]:.2f}")
            self.pitch_var.set(f"{rpy[1]:.2f}")
            self.yaw_var.set(f"{rpy[2]:.2f}")
            self.status_label.config(text="Status: RPY set to current values")
        else:
            self.status_label.config(text="Status: No current RPY available")
    
    def move_to_rpy(self):
        """Move robot to specified RPY orientation"""
        try:
            roll_deg = float(self.roll_var.get())
            pitch_deg = float(self.pitch_var.get())
            yaw_deg = float(self.yaw_var.get())
            
            # Normalize angles to [-180, 180] range
            roll_deg = ((roll_deg + 180) % 360) - 180
            pitch_deg = ((pitch_deg + 180) % 360) - 180
            yaw_deg = ((yaw_deg + 180) % 360) - 180
            
            # Update the input fields with normalized values
            self.roll_var.set(f"{roll_deg:.2f}")
            self.pitch_var.set(f"{pitch_deg:.2f}")
            self.yaw_var.set(f"{yaw_deg:.2f}")
            
            # Convert degrees to radians
            roll_rad = np.deg2rad(roll_deg)
            pitch_rad = np.deg2rad(pitch_deg)
            yaw_rad = np.deg2rad(yaw_deg)
            
            # Create rotation from RPY (xyz convention)
            r = Rot.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad], degrees=False)
            target_quat = r.as_quat()
            
            # Get offset point (use fixed position if set, otherwise compute from current TCP)
            if self.node.fixed_position is not None:
                offset_point = self.node.fixed_position
            elif self.node.current_position is not None and self.node.current_quaternion is not None:
                # Compute offset point from current TCP position
                offset_point = self.node.compute_offset_point(self.node.current_position, self.node.current_quaternion)
            else:
                self.status_label.config(text="Status: No position available")
                return
            
            # Get current quaternion as start point
            start_quat = self.node.current_quaternion
            if start_quat is None:
                self.status_label.config(text="Status: No current orientation available")
                return
            
            # Compute TCP position from offset point with target orientation
            tcp_position = self.node.compute_tcp_from_offset_point(offset_point, target_quat)
            
            # Interpolate between current and target quaternions for smooth trajectory
            self.status_label.config(text="Status: Computing smooth trajectory...")
            trajectory_points = self.node.interpolate_quaternion_trajectory(
                start_quat,
                target_quat,
                offset_point,  # Pass offset point, not TCP position
                num_waypoints=15,  # Number of intermediate waypoints
                total_duration=5.0  # Total duration in seconds
            )
            
            if trajectory_points is None or len(trajectory_points) == 0:
                # Fallback: try direct IK if interpolation fails
                q_guess = self.node.current_joint_angles if self.node.current_joint_angles is not None else None
                
                if q_guess is not None:
                    from ik_solver import compute_ik_quaternion
                    joint_angles = compute_ik_quaternion(
                        tcp_position,
                        target_quat,
                        q_guess=q_guess,
                        max_tries=3,
                        dx=0.001
                    )
                else:
                    joint_angles = None
                
                if joint_angles is None:
                    from action_libraries import compute_ik_quaternion_robust
                    joint_angles = compute_ik_quaternion_robust(
                        tcp_position,
                        target_quat,
                        max_tries=5,
                        dx=0.001,
                        multiple_seeds=True,
                        q_guess=q_guess
                    )
                
                if joint_angles is None:
                    self.status_label.config(text="Status: IK failed for RPY orientation")
                    return
                
                trajectory_points = [joint_angles]
            
            # Send trajectory with interpolated waypoints
            self.status_label.config(text="Status: Moving to RPY orientation...")
            if self.node.send_trajectory(trajectory_points, 5.0):
                # Update target quaternion
                self.node.target_quaternion = list(target_quat)
                # Update accumulated rotation vector
                self.node.accumulated_rotvec = r.as_rotvec()
                self.update_target_display()
                self.status_label.config(text=f"Status: Moving to RPY: Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°")
            else:
                self.status_label.config(text="Status: Failed to send trajectory")
        except ValueError:
            self.status_label.config(text="Status: Invalid RPY values")
    
    def move_to_home(self):
        """Move robot to home position and set offset point as fixed position"""
        self.status_label.config(text="Status: Moving to home...")
        self.node.move_to_home()
        # Update the position fields to show home offset point values
        from action_libraries import HOME_POSE_QUATERNION
        home_tcp_position = HOME_POSE_QUATERNION[0:3]
        home_quaternion = HOME_POSE_QUATERNION[3:7]
        offset_point = self.node.compute_offset_point(home_tcp_position, home_quaternion)
        self.x_var.set(str(offset_point[0]))
        self.y_var.set(str(offset_point[1]))
        self.z_var.set(str(offset_point[2]))
        # Update fixed position label
        self.fixed_position_label.config(text=f"Fixed Position (Offset Point): [{offset_point[0]:.3f}, {offset_point[1]:.3f}, {offset_point[2]:.3f}] m (Rotation Point)")
        self.status_label.config(text="Status: Home command sent")
    
    def set_fixed_to_current(self):
        """Set fixed position (offset point) to current offset point"""
        if self.node.current_position is not None and self.node.current_quaternion is not None:
            self.node.set_fixed_position()
            # Update input fields with offset point (not TCP position)
            offset_point = self.node.fixed_position
            self.x_var.set(str(offset_point[0]))
            self.y_var.set(str(offset_point[1]))
            self.z_var.set(str(offset_point[2]))
            # Update fixed position label
            self.fixed_position_label.config(text=f"Fixed Position (Offset Point): [{offset_point[0]:.3f}, {offset_point[1]:.3f}, {offset_point[2]:.3f}] m (Rotation Point)")
            self.status_label.config(text="Status: Fixed position (offset point) set to current")
        else:
            self.status_label.config(text="Status: No current position available")
    
    def set_fixed_to_home(self):
        """Set fixed position (offset point) to home offset point and move robot there"""
        from action_libraries import HOME_POSE_QUATERNION
        home_tcp_position = HOME_POSE_QUATERNION[0:3]
        home_quaternion = HOME_POSE_QUATERNION[3:7]
        # Compute offset point from home TCP position
        offset_point = self.node.compute_offset_point(home_tcp_position, home_quaternion)
        # Update input fields with offset point
        self.x_var.set(str(offset_point[0]))
        self.y_var.set(str(offset_point[1]))
        self.z_var.set(str(offset_point[2]))
        # Update fixed position label
        self.fixed_position_label.config(text=f"Fixed Position (Offset Point): [{offset_point[0]:.3f}, {offset_point[1]:.3f}, {offset_point[2]:.3f}] m (Rotation Point)")
        # Move robot to home
        self.move_to_home()
    
    def move_to_entered_position(self):
        """Move robot to the entered position (offset point) and set it as fixed position"""
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            offset_point = [x, y, z]
            
            # Set as fixed position (offset point)
            self.node.set_fixed_position(offset_point)
            
            # Move robot to this offset point (with current orientation)
            self.status_label.config(text="Status: Moving to position...")
            if self.node.current_quaternion is not None:
                # Compute TCP position from offset point
                tcp_position = self.node.compute_tcp_from_offset_point(offset_point, self.node.current_quaternion)
                
                # Use current quaternion
                from action_libraries import compute_ik_quaternion_robust
                joint_angles = compute_ik_quaternion_robust(
                    tcp_position,
                    self.node.current_quaternion,
                    max_tries=5,
                    dx=0.001,
                    multiple_seeds=True
                )
                
                if joint_angles is not None:
                    if self.node.send_trajectory([joint_angles], 5.0):
                        self.status_label.config(text=f"Status: Moving to offset point [{x:.3f}, {y:.3f}, {z:.3f}]")
                        # Update fixed position label
                        self.fixed_position_label.config(text=f"Fixed Position (Offset Point): [{x:.3f}, {y:.3f}, {z:.3f}] m (Rotation Point)")
                    else:
                        self.status_label.config(text="Status: Failed to send trajectory")
                else:
                    self.status_label.config(text="Status: IK failed for this position")
            else:
                self.status_label.config(text="Status: No current orientation available")
        except ValueError:
            self.status_label.config(text="Status: Invalid position values")
    
    def update_loop(self):
        """Update UI with current pose"""
        if self.node.current_pose:
            pose = self.node.current_pose
            pos = pose['position']
            rpy = pose['rpy']
            rotvec = pose['rotvec']
            
            self.position_label.config(text=f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
            self.rpy_label.config(text=f"RPY: Roll={rpy[0]:.2f}°, Pitch={rpy[1]:.2f}°, Yaw={rpy[2]:.2f}°")
            self.rotvec_label.config(text=f"Rotation Vector: [{rotvec[0]:.3f}, {rotvec[1]:.3f}, {rotvec[2]:.3f}] rad")
            
            # Don't auto-update RPY input fields - let user control them
            # They can use "Set to Current RPY" button to update them
        
        # Update fixed position display (offset point)
        if self.node.fixed_position:
            fixed_pos = self.node.fixed_position
            self.fixed_position_label.config(text=f"Fixed Position (Offset Point): [{fixed_pos[0]:.3f}, {fixed_pos[1]:.3f}, {fixed_pos[2]:.3f}] m (Rotation Point)")
        
        # Update TCP offset display (sync with node value in case it changed)
        current_offset = self.node.tcp_offset
        if abs(self.tcp_offset_var.get() - current_offset) > 0.001:  # Only update if significantly different
            self.tcp_offset_var.set(current_offset)
        self.tcp_offset_label.config(text=f"Current: {current_offset:.3f} m")
        
        # Update rotation rate
        self.node.rotation_rate = self.rate_var.get() * 0.01
        
        # Update target display
        self.update_target_display()
        
        # Schedule next update
        self.root.after(100, self.update_loop)
    
    def run(self):
        """Run the UI"""
        self.root.mainloop()
        self.node.running = False


def main():
    rclpy.init()
    
    # Create node
    node = TCPControlNode()
    
    # Wait for action server
    print("Waiting for action server...")
    node.action_client.wait_for_server()
    print("✓ Action server ready.")
    
    # Wait for TCP pose
    print("Waiting for TCP pose...")
    timeout = 10.0
    start_time = time.time()
    while node.current_pose is None:
        if time.time() - start_time > timeout:
            print("⚠ Timeout waiting for TCP pose.")
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    
    if node.current_pose is None:
        print("⚠ Could not get TCP pose. Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    print("✓ TCP pose received.")
    
    # Create and run UI
    ui = TCPControlUI(node)
    
    # Spin node in separate thread
    def spin_node():
        while node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    
    spin_thread = threading.Thread(target=spin_node, daemon=True)
    spin_thread.start()
    
    try:
        ui.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()
        print("✓ Cleanup complete.")


if __name__ == '__main__':
    main()

