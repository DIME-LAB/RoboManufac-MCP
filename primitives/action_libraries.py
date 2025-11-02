from builtin_interfaces.msg import Duration
from ik_solver import compute_ik, compute_ik_robust

HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY calibration offset x = -0.013 y = +0.028
# HOME_POSE = [0.065, -0.385, 0.160, 0, 180, 0] #lower home pose
# PICK_STATION_POSE = [-0.180, -0.385, 0.481, 0, 180, 0]  # XYZRPY - Pick station position
PICK_STATION_POSE = [-0.330, -0.385, 0.404, 0, 180, 0] #calibration offset x = -0.000, y = +0.025



def make_point(joint_positions, seconds):
    return {
        "positions": [float(x) for x in joint_positions],  # ensures all are float
        "velocities": [0.0] * 6,
        "time_from_start": Duration(sec=int(seconds)),
    }

def home():
    joint_angles = compute_ik(HOME_POSE[0:3], HOME_POSE[3:6])
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