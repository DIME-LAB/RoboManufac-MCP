import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from core.kalman_filter import (
    QuaternionKalman,
    MAX_MOVEMENT_THRESHOLD,
    HOLD_REQUIRED_FRAMES,
    GHOST_TRACKING_FRAMES,
    BLEND_FACTOR,
)
from core.pose_math import (
    rvec_to_quat,
    quat_to_rvec,
    slerp_quat,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    estimate_object_pose_from_marker,
)
from core.model_io import (
    load_wireframe_data,
    load_aruco_annotations,
    get_available_models,
    select_model_interactive,
    load_assembly_data,
    parse_assembly_components,
)
from core.mesh_ops import (
    transform_mesh_to_camera_frame,
    project_vertices_to_image,
    draw_wireframe,
)

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

# Shared utilities imported from core; no local duplicates.
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
    assembly_file = data_dir / "fmb_assembly3.json"
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
    aruco_dict_names = set()
    for model_name in available_models:
        wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
        aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
        
        try:
            vertices, edges = load_wireframe_data(wireframe_file)
            (
                aruco_annotations,
                base_marker_size,
                border_width_percent,
                aruco_dict_name,
            ) = load_aruco_annotations(aruco_annotations_file)
            aruco_dict_names.add(aruco_dict_name)
            
            # Create a dictionary mapping marker IDs to their annotations
            marker_annotations = {}
            for annotation in aruco_annotations:
                marker_id = annotation['aruco_id']
                marker_annotations[marker_id] = annotation
            
            model_data[model_name] = {
                'vertices': vertices,
                'edges': edges,
                'marker_annotations': marker_annotations,
                'base_marker_size': base_marker_size,
                'border_width_percent': border_width_percent,
                'aruco_dict_name': aruco_dict_name,
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
    
    # Build marker lookup keyed by (dict_name, marker_id) to avoid collisions
    marker_lookup = {}
    marker_ids_by_dict = {}
    for model_name, data in model_data.items():
        dict_name = data['aruco_dict_name']
        marker_ids_by_dict.setdefault(dict_name, set())
        for marker_id, annotation in data['marker_annotations'].items():
            key = (dict_name, marker_id)
            marker_lookup[key] = {
                'model': model_name,
                'annotation': annotation,
                'base_marker_size': data['base_marker_size'],
                'border_width_percent': data['border_width_percent'],
                'vertices': data['vertices'],
                'edges': data['edges'],
                'aruco_dict_name': dict_name,
            }
            marker_ids_by_dict[dict_name].add(marker_id)
    
    # Create detectors per dictionary
    detectors = {}
    for dict_name in aruco_dict_names:
        if hasattr(aruco, dict_name):
            dictionary_id = getattr(aruco, dict_name)
            detectors[dict_name] = aruco.ArucoDetector(
                aruco.getPredefinedDictionary(dictionary_id), aruco.DetectorParameters()
            )
        else:
            print(f"Warning: Unknown dictionary '{dict_name}', skipping")
    if not detectors:
        print("No valid ArUco dictionaries found; defaulting to DICT_4X4_50")
        detectors["DICT_4X4_50"] = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_4X4_50), aruco.DetectorParameters()
        )
    
    # Open camera
    cap = cv2.VideoCapture(8)  # Using camera ID 8
    if not cap.isOpened():
        print("Failed to open camera 8")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("Assembly Pose Detection with Kalman Filtering Started")
    print(f"Using ArUco dictionaries: {sorted(aruco_dict_names)}")
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
        
        successful_detections = []

        # Detect per dictionary (supports mixed 4x4 / 5x5)
        for dict_name, detector in detectors.items():
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue

            aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                key = (dict_name, marker_id)
                if key not in marker_lookup:
                    continue

                entry = marker_lookup[key]
                target_corners = corners[i]

                base_marker_size = entry['base_marker_size']
                border_percentage = entry['border_width_percent']
                border_width = base_marker_size * border_percentage
                marker_size = base_marker_size - 2 * border_width

                tvec, rvec, _, is_confirmed = estimate_pose_with_kalman(
                    frame, [target_corners], [marker_id], camera_matrix, dist_coeffs,
                    marker_size, kalman_filters, marker_stabilities, last_seen_frames, current_frame
                )

                if tvec is None or rvec is None:
                    continue

                position = tvec.flatten()
                distance = np.linalg.norm(position)

                object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), entry['annotation'])

                # Orientation validation against assembly data
                object_rotation_matrix, _ = cv2.Rodrigues(object_rvec)
                coord_transform = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                object_rotation_matrix_transformed = coord_transform @ object_rotation_matrix @ coord_transform.T
                orientation_valid = False
                if entry['model'] in components:
                    expected_rotation = components[entry['model']]['rotation']
                    orientation_valid = check_orientation_match(object_rotation_matrix_transformed, expected_rotation)

                successful_detections.append({
                    'marker_id': marker_id,
                    'marker_model': entry['model'],
                    'marker_annotation': entry['annotation'],
                    'marker_size': marker_size,
                    'rvec': rvec,
                    'tvec': tvec,
                    'position': position,
                    'distance': distance,
                    'object_pose': (object_tvec, object_rvec),
                    'is_confirmed': is_confirmed,
                    'orientation_valid': orientation_valid,
                })

        if successful_detections:
            # Best per model (prefer confirmed, then nearest)
            best_per_model = {}
            for det in successful_detections:
                model = det['marker_model']
                current = best_per_model.get(model)
                if current is None:
                    best_per_model[model] = det
                else:
                    if (not current['is_confirmed'] and det['is_confirmed']) or (
                        current['is_confirmed'] == det['is_confirmed'] and det['distance'] < current['distance']
                    ):
                        best_per_model[model] = det

            # Draw only best per model
            for det in best_per_model.values():
                object_tvec, object_rvec = det['object_pose']
                model_vertices = model_data[det['marker_model']]['vertices']
                model_edges = model_data[det['marker_model']]['edges']
                transformed_vertices = transform_mesh_to_camera_frame(model_vertices, (object_tvec, object_rvec))
                projected_vertices = project_vertices_to_image(transformed_vertices, camera_matrix, dist_coeffs)
                wireframe_color = (0, 255, 0) if det['orientation_valid'] else (0, 0, 255)
                if show_mesh and len(projected_vertices) > 0:
                    draw_wireframe(frame, projected_vertices, model_edges, color=wireframe_color, thickness=2)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, object_rvec, object_tvec, det['marker_size'] * 1.5)

            # Text overlay: object pose per model
            y_offset = 30
            dy = 25
            for det in sorted(best_per_model.values(), key=lambda d: d['distance']):
                object_tvec, object_rvec = det['object_pose']
                rot_mtx, _ = cv2.Rodrigues(object_rvec)
                rpy = rotation_matrix_to_euler(rot_mtx)
                text = (
                    f"{det['marker_model']}: pos ({object_tvec[0]:.3f},{object_tvec[1]:.3f},{object_tvec[2]:.3f}) "
                    f"rpy ({np.degrees(rpy[0]):.1f},{np.degrees(rpy[1]):.1f},{np.degrees(rpy[2]):.1f})"
                )
                color = (0, 255, 0) if det['is_confirmed'] else (0, 165, 255)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += dy
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
