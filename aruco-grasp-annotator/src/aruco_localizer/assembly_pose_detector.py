import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# =============================================================================
# KALMAN FILTER CONFIGURATION - ADJUSTABLE VARIABLES
# =============================================================================

# Temporal filtering parameters
MAX_MOVEMENT_THRESHOLD = 0.05  # meters - maximum allowed movement between frames
HOLD_REQUIRED_FRAMES = 2       # frames - required stable detections before confirmation
GHOST_TRACKING_FRAMES = 15     # frames - continue tracking when marker lost
BLEND_FACTOR = 0.5           # 0.0-1.0 - trust in measurements vs predictions

# Kalman filter noise parameters
PROCESS_NOISE_POSITION = 1e-4   # Process noise for position (x,y,z)
PROCESS_NOISE_QUATERNION = 1e-3 # Process noise for quaternion (qx,qy,qz,qw)
PROCESS_NOISE_VELOCITY = 1e-4   # Process noise for velocity (vx,vy,vz)
MEASUREMENT_NOISE_POSITION = 1e-4 # Measurement noise for position
MEASUREMENT_NOISE_QUATERNION = 1e-4 # Measurement noise for quaternion

# Camera parameters
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV = 69.4  # degrees
CAMERA_VFOV = 42.5  # degrees

# ArUco parameters
MARKER_SIZE = 0.021  # meters - adjust based on your markers
ARUCO_DICTIONARY = aruco.DICT_4X4_50

# Orientation validation parameters
ORIENTATION_TOLERANCE_DEGREES = 15.0  # degrees - tolerance for orientation matching

# =============================================================================
# KALMAN FILTER CLASSES
# =============================================================================

class QuaternionKalman:
    """Kalman filter for 6D pose estimation with quaternions"""
    
    def __init__(self):
        # 10 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz]
        self.kf = cv2.KalmanFilter(10, 7)
        
        dt = 1.0  # Time step (assuming 1 frame = 1 time unit)
        
        # A: Transition matrix (10x10)
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        for i in range(3):  # x += vx*dt, y += vy*dt, z += vz*dt
            self.kf.transitionMatrix[i, i+7] = dt
        
        # H: Measurement matrix (7x10) - we measure position and quaternion
        self.kf.measurementMatrix = np.zeros((7, 10), dtype=np.float32)
        self.kf.measurementMatrix[0:7, 0:7] = np.eye(7)
        
        # Q: Process noise covariance
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-6
        for i in range(3):   # position noise
            self.kf.processNoiseCov[i, i] = PROCESS_NOISE_POSITION
        for i in range(3, 7):  # quaternion noise
            self.kf.processNoiseCov[i, i] = PROCESS_NOISE_QUATERNION
        for i in range(7, 10):  # velocity noise
            self.kf.processNoiseCov[i, i] = PROCESS_NOISE_VELOCITY
        
        # R: Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        for i in range(3):   # position measurement noise
            self.kf.measurementNoiseCov[i, i] = MEASUREMENT_NOISE_POSITION
        for i in range(3, 7):  # quaternion measurement noise
            self.kf.measurementNoiseCov[i, i] = MEASUREMENT_NOISE_QUATERNION
        
        # Initial error covariance
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)
        
        # Initial state
        self.kf.statePost = np.zeros((10, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion
    
    def correct(self, tvec, rvec):
        """Update filter with new measurement"""
        quat = rvec_to_quat(rvec)
        measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
        self.kf.correct(measurement)
    
    def predict(self):
        """Predict next state"""
        pred = self.kf.predict()
        pred_tvec = pred[0:3].flatten()
        pred_quat = pred[3:7].flatten()
        # Normalize quaternion to prevent drift
        pred_quat /= np.linalg.norm(pred_quat)
        pred_rvec = quat_to_rvec(pred_quat).flatten()
        return pred_tvec, pred_rvec

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def rvec_to_quat(rvec):
    """Convert OpenCV rotation vector to quaternion [x, y, z, w]"""
    rot, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot).as_quat()

def quat_to_rvec(quat):
    """Convert quaternion [x, y, z, w] to OpenCV rotation vector"""
    rot = R.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rot)
    return rvec

def slerp_quat(q1, q2, blend=0.5):
    """Spherical linear interpolation between two quaternions"""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate dot product
    dot = np.dot(q1, q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close for comfort, linearly interpolate
    if dot > 0.9995:
        result = q1 + blend * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate the angle between the quaternions
    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * blend
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2

def load_wireframe_data(json_file):
    """Load wireframe data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['vertices'], data['edges']

def load_aruco_annotations(json_file):
    """Load ArUco marker annotations from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['markers']

def load_assembly_data(json_file):
    """Load FMB assembly data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def get_available_models(data_dir):
    """Get list of available models from the data directory"""
    wireframe_dir = Path(data_dir) / "wireframe"
    aruco_dir = Path(data_dir) / "aruco"
    
    if not wireframe_dir.exists() or not aruco_dir.exists():
        return []
    
    # Get all wireframe files
    wireframe_files = list(wireframe_dir.glob("*_wireframe.json"))
    aruco_files = list(aruco_dir.glob("*_aruco.json"))
    
    # Extract model names (remove _wireframe.json and _aruco.json suffixes)
    wireframe_models = {f.stem.replace("_wireframe", "") for f in wireframe_files}
    aruco_models = {f.stem.replace("_aruco", "") for f in aruco_files}
    
    # Return intersection (models that have both wireframe and aruco files)
    available_models = wireframe_models.intersection(aruco_models)
    return sorted(list(available_models))

def parse_assembly_components(assembly_data):
    """Parse assembly data to extract component information"""
    components = {}
    markers = {}
    
    for component in assembly_data['components']:
        if component['type'] == 'component':
            # Store component information
            components[component['name']] = {
                'id': component['id'],
                'display_name': component['displayName'],
                'position': component['position'],
                'rotation': component['rotation'],
                'parent_id': component['parentId']
            }
        elif component['type'] == 'marker':
            # Store marker information
            markers[component['name']] = {
                'id': component['id'],
                'display_name': component['displayName'],
                'position': component['position'],
                'rotation': component['rotation'],
                'parent_id': component['parentId']
            }
    
    return components, markers

def estimate_object_pose_from_marker(marker_pose, aruco_annotation):
    """
    Estimate the 6D pose of the object center from ArUco marker pose.
    
    Args:
        marker_pose: (tvec, rvec) - ArUco marker pose in camera frame
        aruco_annotation: ArUco annotation data containing marker-to-object transform
    
    Returns:
        object_pose: (object_tvec, object_rvec) - Object center pose in camera frame
    """
    # Get marker position and rotation
    marker_tvec, marker_rvec = marker_pose
    
    # Convert marker rotation vector to rotation matrix
    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    
    # Get the marker's pose relative to CAD center from annotation
    marker_relative_pose = aruco_annotation['pose_relative_to_cad_center']
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Get marker position relative to object center (in object frame)
    marker_pos_in_object = np.array([
        marker_relative_pose['position']['x'],
        marker_relative_pose['position']['y'], 
        marker_relative_pose['position']['z']
    ])
    
    # Apply scaling and coordinate transformation
    marker_pos_in_object = coord_transform @ (marker_pos_in_object * 1.25)
    
    # Get marker orientation relative to object center
    marker_rot = marker_relative_pose['rotation']
    marker_rotation_in_object = euler_to_rotation_matrix(
        marker_rot['roll'], marker_rot['pitch'], marker_rot['yaw']
    )
    
    # Apply coordinate system transformation to the rotation matrix
    marker_rotation_in_object = coord_transform @ marker_rotation_in_object @ coord_transform.T
    
    # Calculate object center position in camera frame
    # The object center is at the origin of the object frame
    # We need to transform the origin (0,0,0) from object frame to camera frame
    object_origin_in_marker_frame = marker_rotation_in_object.T @ (-marker_pos_in_object)
    object_tvec = marker_tvec.flatten() + marker_rotation_matrix @ object_origin_in_marker_frame
    
    # Calculate object center orientation in camera frame
    # The object orientation is the marker orientation composed with the marker-to-object rotation
    object_rotation_matrix = marker_rotation_matrix @ marker_rotation_in_object.T
    
    # Convert back to rotation vector
    object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)
    
    return object_tvec, object_rvec

def transform_mesh_to_camera_frame(vertices, object_pose):
    """Transform mesh vertices from object center frame to camera frame"""
    object_tvec, object_rvec = object_pose
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(object_rvec)
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Transform vertices from object center frame to camera frame
    transformed_vertices = []
    for vertex in vertices:
        # Apply coordinate system transformation and scaling
        vertex_transformed = coord_transform @ (np.array(vertex) * 1.25)
        
        # Transform from object frame to camera frame
        vertex_cam = rotation_matrix @ vertex_transformed + object_tvec
        transformed_vertices.append(vertex_cam)
    
    return np.array(transformed_vertices)

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix"""
    # Convert to radians if needed (assuming they're already in radians from JSON)
    r, p, y = roll, pitch, yaw
    
    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    
    # Combine rotations (order: Rz * Ry * Rx)
    return Rz @ Ry @ Rx

def rotation_matrix_to_euler(rotation_matrix):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in radians"""
    # Extract RPY angles from rotation matrix
    # Using ZYX convention (yaw-pitch-roll)
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0
    
    return np.array([roll, pitch, yaw])

def check_orientation_match(detected_rotation_matrix, expected_rotation, tolerance_degrees=ORIENTATION_TOLERANCE_DEGREES):
    """
    Check if the detected object orientation matches the expected orientation from assembly.
    
    Args:
        detected_rotation_matrix: 3x3 rotation matrix of detected object
        expected_rotation: Expected rotation from assembly data (dict with x, y, z in radians)
        tolerance_degrees: Tolerance in degrees for orientation matching
    
    Returns:
        bool: True if orientation matches within tolerance, False otherwise
    """
    # Convert expected rotation from assembly to rotation matrix
    expected_rotation_matrix = euler_to_rotation_matrix(
        expected_rotation['x'], 
        expected_rotation['y'], 
        expected_rotation['z']
    )
    
    # Calculate the relative rotation between detected and expected
    relative_rotation = detected_rotation_matrix @ expected_rotation_matrix.T
    
    # Convert relative rotation to axis-angle representation
    relative_rvec, _ = cv2.Rodrigues(relative_rotation)
    
    # Calculate the angle of rotation (magnitude of rotation vector)
    rotation_angle_radians = np.linalg.norm(relative_rvec)
    rotation_angle_degrees = np.degrees(rotation_angle_radians)
    
    # Check if the rotation angle is within tolerance
    # Handle the case where rotation might be close to 2π (360 degrees)
    rotation_angle_degrees = min(rotation_angle_degrees, 360.0 - rotation_angle_degrees)
    
    return rotation_angle_degrees <= tolerance_degrees

def project_vertices_to_image(vertices, camera_matrix, dist_coeffs):
    """Project 3D vertices to 2D image coordinates"""
    if len(vertices) == 0:
        return np.array([])
    
    # Project points to image plane
    projected_points, _ = cv2.projectPoints(
        vertices.astype(np.float32), 
        np.zeros((3, 1)),  # No rotation (already in camera frame)
        np.zeros((3, 1)),  # No translation (already in camera frame)
        camera_matrix, 
        dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(np.int32)

def draw_wireframe(frame, projected_vertices, edges, color=(0, 255, 0), thickness=2):
    """Draw wireframe on the image"""
    if len(projected_vertices) == 0:
        return
    
    # Filter out vertices that are outside the image bounds
    height, width = frame.shape[:2]
    valid_vertices = []
    valid_indices = []
    
    for i, vertex in enumerate(projected_vertices):
        x, y = vertex
        if 0 <= x < width and 0 <= y < height:
            valid_vertices.append(vertex)
            valid_indices.append(i)
    
    if len(valid_vertices) == 0:
        return
    
    # Create mapping from original indices to valid indices
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
    
    # Draw edges
    for edge in edges:
        if len(edge) >= 2:
            start_idx, end_idx = edge[0], edge[1]
            if start_idx in index_map and end_idx in index_map:
                start_point = tuple(valid_vertices[index_map[start_idx]])
                end_point = tuple(valid_vertices[index_map[end_idx]])
                cv2.line(frame, start_point, end_point, color, thickness)
    
    # Draw vertices as small circles
    for vertex in valid_vertices:
        cv2.circle(frame, tuple(vertex), 3, (255, 0, 0), -1)

def estimate_pose_with_kalman(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                             kalman_filters, marker_stabilities, last_seen_frames, current_frame):
    """Estimate pose with Kalman filtering and stability checking"""
    
    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)
            
            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman()
                marker_stabilities[marker_id] = {
                    "last_tvec": None,
                    "last_frame": -1,
                    "confirmed": False,
                    "hold_counter": 0
                }
                last_seen_frames[marker_id] = 0
            
            kalman = kalman_filters[marker_id]
            stability = marker_stabilities[marker_id]
            
            # Prepare points for solvePnP
            image_points = corner[0].reshape(-1, 2)
            half_size = marker_size / 2
            object_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)
            
            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            
            if success:
                tvec_flat = tvec.flatten()
                distance = np.linalg.norm(tvec_flat - stability["last_tvec"]) if stability["last_tvec"] is not None else 0
                movement_ok = distance < MAX_MOVEMENT_THRESHOLD
                
                if movement_ok:
                    stability["hold_counter"] += 1
                else:
                    stability["hold_counter"] = 0
                
                stability["last_tvec"] = tvec_flat
                stability["last_frame"] = current_frame
                
                if stability["hold_counter"] >= HOLD_REQUIRED_FRAMES:
                    stability["confirmed"] = True
                    
                    # Apply Kalman filtering with blending
                    measured_quat = rvec_to_quat(rvec)
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    
                    # Blend measurements with predictions
                    blended_quat = slerp_quat(pred_quat, measured_quat, blend=BLEND_FACTOR)
                    blended_rvec = quat_to_rvec(blended_quat)
                    blended_tvec = BLEND_FACTOR * tvec_flat + (1 - BLEND_FACTOR) * pred_tvec
                    
                    # Update Kalman filter
                    kalman.correct(blended_tvec, blended_rvec)
                    last_seen_frames[marker_id] = current_frame
                    
                    return blended_tvec, blended_rvec, marker_id, True
                else:
                    return tvec_flat, rvec.flatten(), marker_id, False
    
    # Handle ghost tracking for confirmed markers
    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        
        if not stability["confirmed"]:
            continue
        
        if current_frame - last_seen < GHOST_TRACKING_FRAMES:
            pred_tvec, pred_rvec = kalman.predict()
            return pred_tvec, pred_rvec, marker_id, True
        else:
            stability["confirmed"] = False
    
    return None, None, None, False

def detect_assembly_pose_with_kalman_filtering(data_dir=None):
    """Detect all objects in the assembly using ArUco markers with Kalman filtering"""
    
    # Set default data directory if not provided
    if data_dir is None:
        # Assume we're running from aruco_localizer directory, go up to find data
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent.parent / "data"
    
    data_dir = Path(data_dir)
    
    # Load assembly data
    assembly_file = data_dir / "fmb_assembly.json"
    if not assembly_file.exists():
        print(f"Assembly file not found: {assembly_file}")
        return
    
    try:
        assembly_data = load_assembly_data(assembly_file)
        components, markers = parse_assembly_components(assembly_data)
        print(f"Loaded assembly with {len(components)} components and {len(markers)} markers")
    except Exception as e:
        print(f"Error loading assembly data: {e}")
        return
    
    # Get available models
    available_models = get_available_models(data_dir)
    if not available_models:
        print(f"No models found in data directory: {data_dir}")
        return
    
    print(f"Available models: {available_models}")
    
    # Load all model data
    model_data = {}
    for model_name in available_models:
        wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
        aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
        
        try:
            vertices, edges = load_wireframe_data(wireframe_file)
            aruco_annotations = load_aruco_annotations(aruco_annotations_file)
            
            # Create a dictionary mapping marker IDs to their annotations
            marker_annotations = {}
            for annotation in aruco_annotations:
                marker_id = annotation['aruco_id']
                marker_annotations[marker_id] = annotation
            
            model_data[model_name] = {
                'vertices': vertices,
                'edges': edges,
                'marker_annotations': marker_annotations
            }
            
            print(f"Loaded {model_name}: {len(vertices)} vertices, {len(edges)} edges, {len(marker_annotations)} markers")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    if not model_data:
        print("No model data loaded successfully")
        return
    
    # Calculate camera matrix from field of view
    fx = CAMERA_WIDTH / (2 * np.tan(np.deg2rad(CAMERA_HFOV / 2)))
    fy = CAMERA_HEIGHT / (2 * np.tan(np.deg2rad(CAMERA_VFOV / 2)))
    camera_matrix = np.array([[fx, 0, CAMERA_WIDTH / 2],
                              [0, fy, CAMERA_HEIGHT / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    print(f"Camera matrix: fx={fx:.1f}, fy={fy:.1f}")
    
    # Collect all target marker IDs from all models
    all_target_ids = set()
    for model_name, data in model_data.items():
        all_target_ids.update(data['marker_annotations'].keys())
    
    print(f"Looking for markers: {sorted(all_target_ids)}")
    
    # ArUco parameters
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Open camera
    cap = cv2.VideoCapture(8)  # Using camera ID 8
    if not cap.isOpened():
        print("Failed to open camera 8")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("Assembly Pose Detection with Kalman Filtering Started")
    print(f"Looking for markers {sorted(all_target_ids)} to detect all objects in assembly")
    print("Press 'q' to quit, 's' to toggle mesh display")
    print("=" * 60)
    print(f"Kalman Filter Settings:")
    print(f"  Max Movement: {MAX_MOVEMENT_THRESHOLD}m")
    print(f"  Hold Required: {HOLD_REQUIRED_FRAMES} frames")
    print(f"  Ghost Tracking: {GHOST_TRACKING_FRAMES} frames")
    print(f"  Blend Factor: {BLEND_FACTOR}")
    print(f"Orientation Validation:")
    print(f"  Tolerance: {ORIENTATION_TOLERANCE_DEGREES} degrees")
    print(f"  Colors: Red=Detected, Green=Valid Orientation")
    print("=" * 60)
    
    show_mesh = True
    current_frame = 0
    
    # Kalman filter tracking
    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        current_frame += 1
        
        # Clear detected objects for this frame
        detected_objects = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Draw all detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Check for any of our target markers
            detected_targets = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in all_target_ids:
                    detected_targets.append((i, marker_id))
            
            if detected_targets:
                # First pass: collect all successful pose estimations with Kalman filtering
                successful_detections = []
                for target_idx, marker_id in detected_targets:
                    target_corners = corners[target_idx]
                    
                    # Find which model this marker belongs to
                    marker_model = None
                    marker_annotation = None
                    for model_name, data in model_data.items():
                        if marker_id in data['marker_annotations']:
                            marker_model = model_name
                            marker_annotation = data['marker_annotations'][marker_id]
                            break
                    
                    if marker_model is None:
                        continue
                    
                    # Calculate marker size including border
                    base_marker_size = marker_annotation['size']
                    border_percentage = marker_annotation['border_width']
                    marker_size = base_marker_size * (1 + border_percentage)
                    
                    # Estimate pose with Kalman filtering
                    tvec, rvec, filtered_marker_id, is_confirmed = estimate_pose_with_kalman(
                        frame, [target_corners], [marker_id], camera_matrix, dist_coeffs, 
                        marker_size, kalman_filters, marker_stabilities, last_seen_frames, current_frame
                    )
                    
                    if tvec is not None and rvec is not None:
                        position = tvec.flatten()
                        distance = np.linalg.norm(position)
                        
                        # Estimate object pose from marker pose
                        object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), marker_annotation)
                        
                        # Check orientation match with assembly data
                        object_rotation_matrix, _ = cv2.Rodrigues(object_rvec)
                        
                        # Apply coordinate transformation to match assembly coordinate system
                        coord_transform = np.array([
                            [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
                            [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
                            [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
                        ])
                        # Transform the detected rotation matrix to assembly coordinate system
                        object_rotation_matrix_transformed = coord_transform @ object_rotation_matrix @ coord_transform.T
                        
                        orientation_valid = False
                        if marker_model in components:
                            expected_rotation = components[marker_model]['rotation']
                            orientation_valid = check_orientation_match(object_rotation_matrix_transformed, expected_rotation)
                        
                        successful_detections.append({
                            'target_idx': target_idx,
                            'marker_id': marker_id,
                            'marker_model': marker_model,
                            'marker_annotation': marker_annotation,
                            'target_corners': target_corners,
                            'marker_size': marker_size,
                            'rvec': rvec,
                            'tvec': tvec,
                            'position': position,
                            'distance': distance,
                            'object_pose': (object_tvec, object_rvec),
                            'is_confirmed': is_confirmed,
                            'orientation_valid': orientation_valid
                        })
                
                # Group detections by model and find best marker for each model
                model_detections = {}
                for detection in successful_detections:
                    model_name = detection['marker_model']
                    if model_name not in model_detections:
                        model_detections[model_name] = []
                    model_detections[model_name].append(detection)
                
                # Find best marker for each model (closest confirmed marker)
                for model_name, detections in model_detections.items():
                    confirmed_detections = [d for d in detections if d['is_confirmed']]
                    if confirmed_detections:
                        best_detection = min(confirmed_detections, key=lambda x: x['distance'])
                    else:
                        # If no confirmed detections, use the closest one
                        best_detection = min(detections, key=lambda x: x['distance'])
                    
                    # Store the best detection for this model
                    detected_objects[model_name] = {
                        'marker_id': best_detection['marker_id'],
                        'marker_pose': (best_detection['tvec'], best_detection['rvec']),
                        'object_pose': best_detection['object_pose'],
                        'is_confirmed': best_detection['is_confirmed'],
                        'distance': best_detection['distance'],
                        'face_type': best_detection['marker_annotation']['face_type'],
                        'orientation_valid': best_detection['orientation_valid'],
                        'is_best_marker': True
                    }
                
                # Process all detections for display, but only show wireframe for the best ones
                for detection in successful_detections:
                    target_idx = detection['target_idx']
                    marker_id = detection['marker_id']
                    marker_model = detection['marker_model']
                    marker_annotation = detection['marker_annotation']
                    target_corners = detection['target_corners']
                    marker_size = detection['marker_size']
                    rvec = detection['rvec']
                    tvec = detection['tvec']
                    position = detection['position']
                    distance = detection['distance']
                    object_pose = detection['object_pose']
                    is_confirmed = detection['is_confirmed']
                    orientation_valid = detection['orientation_valid']
                    face_type = marker_annotation['face_type']
                    
                    # Check if this is the best marker for this model
                    model_detections_list = model_detections[marker_model]
                    confirmed_detections = [d for d in model_detections_list if d['is_confirmed']]
                    if confirmed_detections:
                        best_detection = min(confirmed_detections, key=lambda x: x['distance'])
                    else:
                        best_detection = min(model_detections_list, key=lambda x: x['distance'])
                    is_best_marker = (detection == best_detection)
                    
                    # Draw coordinate axes for all markers
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size)
                    
                    # Only show wireframe for the best marker of each model
                    if show_mesh and is_best_marker and is_confirmed:
                        model_vertices = model_data[marker_model]['vertices']
                        model_edges = model_data[marker_model]['edges']
                        object_tvec, object_rvec = object_pose
                        
                        # Transform mesh vertices to camera frame using object pose
                        transformed_vertices = transform_mesh_to_camera_frame(model_vertices, (object_tvec, object_rvec))
                        
                        # Project vertices to image coordinates
                        projected_vertices = project_vertices_to_image(transformed_vertices, camera_matrix, dist_coeffs)
                        
                        # Choose color based on orientation validation
                        if orientation_valid:
                            wireframe_color = (0, 255, 0)  # Green for valid orientation
                        else:
                            wireframe_color = (0, 0, 255)  # Red for detected but invalid orientation
                        
                        # Draw wireframe
                        if len(projected_vertices) > 0:
                            draw_wireframe(frame, projected_vertices, model_edges, color=wireframe_color, thickness=2)
                        
                        # Draw object center coordinate axes
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, object_rvec, object_tvec, marker_size * 1.5)
                
                # Display information for all detected objects
                y_offset = 30
                for model_name, obj_data in detected_objects.items():
                    marker_id = obj_data['marker_id']
                    marker_pose = obj_data['marker_pose']
                    object_pose = obj_data['object_pose']
                    is_confirmed = obj_data['is_confirmed']
                    distance = obj_data['distance']
                    face_type = obj_data['face_type']
                    orientation_valid = obj_data['orientation_valid']
                    
                    marker_tvec, marker_rvec = marker_pose
                    object_tvec, object_rvec = object_pose
                    
                    # Use different colors based on confirmation and orientation validation
                    if is_confirmed and orientation_valid:
                        text_color = (0, 255, 0)  # Green for confirmed and valid orientation
                        status = " (CONFIRMED & VALID)"
                    elif is_confirmed:
                        text_color = (0, 0, 255)  # Red for confirmed but invalid orientation
                        status = " (CONFIRMED & INVALID)"
                    else:
                        text_color = (255, 255, 0)  # Yellow for holding markers
                        status = " (HOLDING)"
                    
                    cv2.putText(frame, f"{model_name} - Marker {marker_id} ({face_type}){status}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(frame, f"Marker Pos: ({marker_tvec[0]:.3f}, {marker_tvec[1]:.3f}, {marker_tvec[2]:.3f})", 
                               (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    cv2.putText(frame, f"Object Pos: ({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f})", 
                               (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    cv2.putText(frame, f"Distance: {distance:.3f}m", 
                               (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    
                    y_offset += 100
                
                # Display mesh status
                cv2.putText(frame, f"Mesh: {'ON' if show_mesh else 'OFF'}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Print summary to console only if objects are detected
                if detected_objects:
                    confirmed_count = sum(1 for obj in detected_objects.values() if obj['is_confirmed'])
                    valid_count = sum(1 for obj in detected_objects.values() if obj['is_confirmed'] and obj['orientation_valid'])
                    total_count = len(detected_objects)
                    print(f"\rDetected {total_count} objects ({confirmed_count} confirmed, {valid_count} valid) | Mesh: {'ON' if show_mesh else 'OFF'}", end="", flush=True)
            else:
                cv2.putText(frame, f"Looking for markers {sorted(all_target_ids)}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show controls
        cv2.putText(frame, "Press 'q' to quit, 's' to toggle mesh", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Assembly Pose Detection with Kalman Filtering", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_mesh = not show_mesh
            print(f"\nMesh display: {'ON' if show_mesh else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nAssembly pose detection with Kalman filtering stopped.")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Assembly Pose Detector with Kalman Filtering - Detect all objects in assembly using ArUco markers")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                       help="Path to data directory containing wireframe, aruco subdirectories and fmb_assembly.json")
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    
    # Run the assembly pose detection with Kalman filtering
    detect_assembly_pose_with_kalman_filtering(data_dir=data_dir)

if __name__ == "__main__":
    main()
