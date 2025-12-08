from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R
from ik_solver import compute_ik, compute_ik_robust, compute_ik_quaternion, compute_ik_quaternion_robust

HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY calibration offset x = -0.013 y = +0.028
# HOME_POSE = [0.065, -0.385, 0.160, 0, 180, 0] #lower home pose
# PICK_STATION_POSE = [-0.180, -0.385, 0.481, 0, 180, 0]  # XYZRPY - Pick station position
PICK_STATION_POSE = [-0.330, -0.385, 0.404, 0, 180, 0] #calibration offset x = -0.000, y = +0.025

# Home pose using quaternion [x, y, z, qx, qy, qz, qw] format
# Equivalent to RPY [0, 180, 0] degrees
# Quaternion for 180 deg rotation around Y-axis: [0.0, 1.0, 0.0, ~0.0]
HOME_POSE_QUATERNION = [0.065, -0.385, 0.481, 0.0, 1.0, 0.0, 0.0]  # [x, y, z, qx, qy, qz, qw]



def make_point(joint_positions, seconds):
    """Create a trajectory point with proper time handling"""
    # Handle both int and float seconds
    if isinstance(seconds, float):
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
    else:
        sec = int(seconds)
        nanosec = 0
    
    return {
        "positions": [float(x) for x in joint_positions],  # ensures all are float
        "velocities": [0.0] * 6,
        "time_from_start": Duration(sec=sec, nanosec=nanosec),
    }

def home():
    joint_angles = compute_ik(HOME_POSE[0:3], HOME_POSE[3:6])
    if joint_angles is not None:
        return [make_point(joint_angles, 10)]
    return []

def home_quaternion():
    """
    Move to home position using quaternion for end-effector orientation.
    
    Returns:
        List of trajectory points if successful, empty list otherwise
    """
    position = HOME_POSE_QUATERNION[0:3]
    quaternion = HOME_POSE_QUATERNION[3:7]  # [x, y, z, w]
    
    joint_angles = compute_ik_quaternion_robust(position, quaternion, max_tries=5, dx=0.001, multiple_seeds=True)
    if joint_angles is not None:
        return [make_point(joint_angles, 10)]
    return []

def pick():
    joint_angles = compute_ik(PICK_STATION_POSE[0:3], PICK_STATION_POSE[3:6])
    if joint_angles is not None:
        return [make_point(joint_angles, 4)]
    return []

def move(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    # (j1, j2, j3, j4, j5, j6) = joint_angles
    # print(f"{j1:.3f}, {j2:.3f}, {j3:.3f}, {j4:.3f}, {j5:.3f}, {j6:.3f}")
    if joint_angles is not None:
        return [make_point(joint_angles, seconds)]
    return []

def move_robust(position, rpy, seconds):
    """
    Enhanced move function that can handle arbitrary orientations (not just [0, 180, yaw]).
    Uses robust IK solver with multiple seed configurations.
    
    Args:
        position: [x, y, z] target position
        rpy: [roll, pitch, yaw] target orientation in degrees
        seconds: duration for the movement
        
    Returns:
        List of trajectory points if successful, empty list otherwise
    """
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    
    print(f"move_robust: Attempting to reach position={position}, rpy={rpy}")
    
    # Try robust IK solver with multiple seeds
    joint_angles = compute_ik_robust(position, rpy, max_tries=5, dx=0.001, multiple_seeds=True)
    
    if joint_angles is not None:
        print(f"move_robust: IK successful!")
        return [make_point(joint_angles, seconds)]
    else:
        print(f"move_robust: IK failed for position={position}, rpy={rpy}")
        return []

def moveZ(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    if joint_angles is not None:
        return [make_point(joint_angles, seconds)]
    return []

def moveXY(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    if joint_angles is not None:
        return [make_point(joint_angles, seconds)]
    return []

def pick_and_place(block_pose, slot_pose):
    """
    block_pose and slot_pose are each (position, rpy), where position = [x, y, z]
    and EE orienttaion is [r, p, y]
    """
    block_hover = block_pose[0].copy() ## copying positions
    block_hover[2] += 0.1  # hover 10cm above block

    slot_hover = slot_pose[0].copy()
    slot_hover[2] += 0.1  # hover 10cm above slot

    segment_duration = 6 # specify segment_duration

    return {
        "traj0": home(),
        "traj1": move(block_hover,block_pose[1],segment_duration), # hovers on block 
        "traj2": moveZ(block_pose[0],block_pose[1],segment_duration), # descends to grip position, 
        "traj3": moveZ(block_pose[0],block_pose[1],segment_duration), # gripper close
        "traj4": moveZ(block_hover,block_pose[1],segment_duration), # holds block and hovers 
        "traj5": moveXY(slot_hover,slot_pose[1],segment_duration), # holds block and moves in 2D to hover on slot
        "traj6": moveZ(slot_pose[0],slot_pose[1],segment_duration), # holds block and descends into slot,
        "traj7": moveZ(slot_pose[0],slot_pose[1],segment_duration), # gripper open
        "traj8": home() # homing
    }

def spin_around(target_pose, height):
    """
    target_pose is (position, rpy), where position = [x, y, z] and only x, y are considered
    """
    target_position = target_pose[0].copy() # copying positions
    target_position[2] = height  # Set height to given value
    yaws = range(0, 360, 45)
    segment_duration = 3 # specify segment_duration
    return {
        "traj0": move(target_position, [0, 180, yaws[0]], segment_duration),
        "traj1": move(target_position, [0, 180, yaws[1]], segment_duration),
        "traj2": move(target_position, [0, 180, yaws[2]], segment_duration),
        "traj3": move(target_position, [0, 180, yaws[3]], segment_duration),
        "traj4": move(target_position, [0, 180, yaws[4]], segment_duration),
        "traj5": move(target_position, [0, 180, yaws[5]], segment_duration),
        "traj6": move(target_position, [0, 180, yaws[6]], segment_duration),
        "traj7": move(target_position, [0, 180, yaws[7]], segment_duration),
    }

def hover_over(target_pose, height, duration=3.0):
    """
    target_pose is (position, rpy), where position = [x, y, z] and only x, y are considered
    duration: time in seconds for the movement
    """
    target_position = target_pose[0].copy() # copying positions
    target_position[2] = height  # Set height to given value
    fixed_roll = 0
    fixed_pitch = 180
    yaw = target_pose[1][2] + 90  # Add 90 degrees rotation to target yaw
    # null_rot = [0, 180, 0]
    target_rot = [fixed_roll, fixed_pitch, yaw]
    segment_duration = duration # use provided duration

    return {
        # "traj0": move(target_position,null_rot,segment_duration), # hovers over target 
        "traj1": move(target_position,target_rot,segment_duration), # hovers over target, matching angle
    }

def hover_over_grasp(target_pose, height, duration=3.0):
    """
    target_pose is (position, rpy), where position = [x, y, z] and only x, y are considered
    duration: time in seconds for the movement
    """
    target_position = target_pose[0].copy() # copying positions
    target_position[2] = height  # Set height to given value
    fixed_roll = 0
    fixed_pitch = 180
    yaw = target_pose[1][2]
    # null_rot = [0, 180, 0]
    target_rot = [fixed_roll, fixed_pitch, yaw]
    segment_duration = duration # use provided duration

    return {
        # "traj0": move(target_position,null_rot,segment_duration), # hovers over target 
        "traj1": move(target_position,target_rot,segment_duration), # hovers over target, matching angle
    }

def rotate_orientation_axis(position, base_quaternion, axis, angle_start, angle_end, num_waypoints=10, total_duration=15.0, initial_joint_guess=None):
    """
    Rotate TCP orientation around a single axis while keeping position fixed.
    Mimics teaching pendant button behavior.
    
    Args:
        position: [x, y, z] fixed TCP position
        base_quaternion: [x, y, z, w] base orientation quaternion
        axis: 'x', 'y', or 'z' - axis to rotate around
        angle_start: Starting angle in degrees
        angle_end: Ending angle in degrees
        num_waypoints: Number of intermediate waypoints for smooth rotation
        total_duration: Total duration in seconds for the rotation
        initial_joint_guess: Optional initial joint angles to use as seed (e.g., from previous waypoint or current robot state)
        
    Returns:
        List of trajectory points if successful, empty list otherwise
    """
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    if len(base_quaternion) != 4:
        raise ValueError(f"Expected quaternion [x, y, z, w], got {base_quaternion}")
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"Axis must be 'x', 'y', or 'z', got {axis}")
    
    # Convert base quaternion to rotation object
    base_rot = R.from_quat(base_quaternion)
    
    # Generate angles for rotation
    angles = np.linspace(angle_start, angle_end, num_waypoints)
    
    # Map axis to euler index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis]
    
    # Calculate time per waypoint
    time_per_waypoint = total_duration / num_waypoints
    cumulative_time = 0.0
    
    trajectory_points = []
    previous_joint_angles = initial_joint_guess  # Use provided initial guess if available
    last_positions = None  # Track last unique positions to avoid duplicates
    
    for i, angle in enumerate(angles):
        # Get current euler angles from base rotation
        euler = base_rot.as_euler('xyz', degrees=True)
        
        # Update the specified axis angle
        euler[axis_idx] = angle
        
        # Create new rotation from updated euler angles
        new_rot = R.from_euler('xyz', euler, degrees=True)
        new_quaternion = new_rot.as_quat()  # [x, y, z, w]
        
        # Compute IK with previous joint angles as guess for continuity
        if previous_joint_angles is not None:
            # First try with previous joint angles as seed
            joint_angles = compute_ik_quaternion(position, new_quaternion, 
                                                 q_guess=previous_joint_angles, 
                                                 max_tries=3, dx=0.001)
            
            # If that fails, try robust solver with previous joint angles as first seed
            if joint_angles is None:
                joint_angles = compute_ik_quaternion_robust(position, new_quaternion, 
                                                           max_tries=3, dx=0.001, 
                                                           multiple_seeds=True,
                                                           q_guess=previous_joint_angles)
        else:
            joint_angles = compute_ik_quaternion_robust(position, new_quaternion, 
                                                       max_tries=3, dx=0.001, 
                                                       multiple_seeds=True)
        
        if joint_angles is None:
            print(f"Warning: IK failed for waypoint {i+1}/{num_waypoints} at angle {angle:.1f} deg")
            # Try with robust solver, using previous joint angles if available
            joint_angles = compute_ik_quaternion_robust(position, new_quaternion, 
                                                       max_tries=5, dx=0.001, 
                                                       multiple_seeds=True,
                                                       q_guess=previous_joint_angles)
            if joint_angles is None:
                # Skip this waypoint if unreachable - don't add duplicate
                print(f"Warning: Skipping unreachable waypoint {i+1} at angle {angle:.1f} deg")
                cumulative_time += time_per_waypoint  # Still advance time
                continue  # Skip adding this waypoint
        
        # Check if this is a duplicate of the last waypoint
        current_positions = tuple(joint_angles)
        if current_positions == last_positions:
            # Skip duplicate waypoint but advance time
            cumulative_time += time_per_waypoint
            continue
        
        previous_joint_angles = joint_angles
        last_positions = current_positions
        cumulative_time += time_per_waypoint
        
        # Use fractional seconds for smoother timing
        trajectory_points.append(make_point(joint_angles, cumulative_time))
    
    # Filter out duplicate waypoints (same joint positions)
    # This can happen when we skip unreachable orientations
    filtered_points = []
    last_positions = None
    for point in trajectory_points:
        current_positions = tuple(point["positions"])
        if current_positions != last_positions:
            filtered_points.append(point)
            last_positions = current_positions
    
    # Ensure we have at least 2 waypoints (start and end)
    if len(filtered_points) < 2:
        print(f"Warning: Only {len(filtered_points)} waypoints after filtering, need at least 2")
        # If we only have one point, duplicate it with a slight time offset
        if len(filtered_points) == 1:
            point = {
                "positions": filtered_points[0]["positions"].copy(),
                "velocities": filtered_points[0]["velocities"].copy(),
                "time_from_start": Duration(sec=int(filtered_points[0]["time_from_start"].sec) + 1)
            }
            filtered_points.append(point)
    
    return filtered_points

def rotate_all_axes_teaching_pendant(position, base_quaternion, 
                                     roll_range=(-180, 180), 
                                     pitch_range=(-180, 180), 
                                     yaw_range=(-180, 180),
                                     duration_per_axis=15.0,
                                     num_waypoints=10):
    """
    Rotate TCP orientation around all three axes sequentially (X, Y, Z).
    Mimics teaching pendant button behavior - one axis at a time.
    
    Args:
        position: [x, y, z] fixed TCP position
        base_quaternion: [x, y, z, w] starting orientation quaternion
        roll_range: (min, max) roll angle range in degrees
        pitch_range: (min, max) pitch angle range in degrees
        yaw_range: (min, max) yaw angle range in degrees
        duration_per_axis: Duration in seconds for each axis rotation
        num_waypoints: Number of waypoints per axis rotation
        
    Returns:
        Dictionary with trajectory segments for each axis
    """
    trajectories = {}
    
    # Start with home position
    current_quaternion = base_quaternion.copy()
    
    # Get initial joint angles from home position to use as seed
    initial_joint_angles = None
    initial_joint_angles = compute_ik_quaternion_robust(position, base_quaternion, 
                                                       max_tries=5, dx=0.001, 
                                                       multiple_seeds=True)
    
    # Rotate around X-axis (Roll)
    print("Rotating around X-axis (Roll)...")
    roll_traj = rotate_orientation_axis(
        position, current_quaternion, 'x', 
        roll_range[0], roll_range[1], 
        num_waypoints, duration_per_axis,
        initial_joint_guess=initial_joint_angles
    )
    if roll_traj:
        trajectories['roll'] = roll_traj
        # Get last waypoint's joint angles to use as seed for next axis
        if roll_traj:
            last_waypoint = roll_traj[-1]
            previous_joint_angles = last_waypoint['positions']
        else:
            previous_joint_angles = None
        
        # Update current quaternion to end of roll rotation
        base_rot = R.from_quat(current_quaternion)
        euler = base_rot.as_euler('xyz', degrees=True)
        euler[0] = roll_range[1]  # Update roll to end value
        current_rot = R.from_euler('xyz', euler, degrees=True)
        current_quaternion = current_rot.as_quat()
    else:
        print("Warning: Roll rotation failed")
        return {}
    
    # Rotate around Y-axis (Pitch)
    print("Rotating around Y-axis (Pitch)...")
    pitch_traj = rotate_orientation_axis(
        position, current_quaternion, 'y', 
        pitch_range[0], pitch_range[1], 
        num_waypoints, duration_per_axis,
        initial_joint_guess=previous_joint_angles
    )
    if pitch_traj:
        trajectories['pitch'] = pitch_traj
        # Get last waypoint's joint angles to use as seed for next axis
        if pitch_traj:
            last_waypoint = pitch_traj[-1]
            previous_joint_angles = last_waypoint['positions']
        else:
            previous_joint_angles = None
        
        # Update current quaternion to end of pitch rotation
        base_rot = R.from_quat(current_quaternion)
        euler = base_rot.as_euler('xyz', degrees=True)
        euler[1] = pitch_range[1]  # Update pitch to end value
        current_rot = R.from_euler('xyz', euler, degrees=True)
        current_quaternion = current_rot.as_quat()
    else:
        print("Warning: Pitch rotation failed")
        return trajectories
    
    # Rotate around Z-axis (Yaw)
    print("Rotating around Z-axis (Yaw)...")
    yaw_traj = rotate_orientation_axis(
        position, current_quaternion, 'z', 
        yaw_range[0], yaw_range[1], 
        num_waypoints, duration_per_axis,
        initial_joint_guess=previous_joint_angles
    )
    if yaw_traj:
        trajectories['yaw'] = yaw_traj
    else:
        print("Warning: Yaw rotation failed")
    
    return trajectories