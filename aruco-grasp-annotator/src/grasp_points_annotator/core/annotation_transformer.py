"""Transform grasp points from single marker to all markers"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def load_grasp_points(grasp_json_path):
    """Load grasp points from JSON file."""
    with open(grasp_json_path, 'r') as f:
        data = json.load(f)
    return data


def load_aruco_annotations(aruco_json_path):
    """Load ArUco marker annotations."""
    with open(aruco_json_path, 'r') as f:
        data = json.load(f)
    return data


def load_wireframe(wireframe_json_path):
    """Load wireframe data."""
    with open(wireframe_json_path, 'r') as f:
        data = json.load(f)
    return data


def adjust_z_to_center(grasp_points, object_thickness):
    """
    Adjust Z-coordinate of grasp points to be at the center of the object.
    
    Args:
        grasp_points: List of grasp point dictionaries (in cm)
        object_thickness: Thickness of the object in meters
    
    Returns:
        List of adjusted grasp points (in cm)
    """
    adjusted_points = []
    # Convert thickness from meters to centimeters
    thickness_cm = object_thickness * 100.0
    
    for gp in grasp_points:
        adjusted = gp.copy()
        adjusted['position'] = gp['position'].copy()
        # Set Z to half the thickness in cm (center of object in marker frame)
        adjusted['position']['z'] = thickness_cm / 2.0
        adjusted_points.append(adjusted)
    
    return adjusted_points


def annotate_grasp_points_to_all_markers(
    object_name, 
    source_marker_id, 
    grasp_points_json_path,
    object_thickness=None, 
    data_dir=None
):
    """
    Main function to annotate grasp points relative to all markers.
    
    Args:
        object_name: Name of the object (e.g., "fork_orange_scaled70")
        source_marker_id: Marker ID used to generate the grasp points
        grasp_points_json_path: Path to the grasp points JSON from step 1
        object_thickness: Thickness of the object (for Z adjustment). If None, auto-detect from ArUco annotations.
        data_dir: Path to data directory
        
    Returns:
        Path to the output file
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_path = Path(data_dir)
    
    # Load grasp points from source marker
    grasp_data = load_grasp_points(grasp_points_json_path)
    
    # Load ArUco annotations
    aruco_json = data_path / "aruco" / f"{object_name}_aruco.json"
    aruco_data = load_aruco_annotations(aruco_json)
    
    # Load wireframe
    wireframe_json = data_path / "wireframe" / f"{object_name}_wireframe.json"
    wireframe_data = load_wireframe(wireframe_json)
    
    # Find source marker
    source_marker = None
    for marker in aruco_data['markers']:
        if marker['aruco_id'] == source_marker_id:
            source_marker = marker
            break
    
    if source_marker is None:
        raise ValueError(f"Source marker {source_marker_id} not found")
    
    # Auto-detect object thickness if not provided
    if object_thickness is None:
        cad_object_info = aruco_data.get('cad_object_info', {})
        if 'dimensions' in cad_object_info:
            # Use height as thickness (assumes object is oriented with thickness in Z)
            object_thickness = cad_object_info['dimensions'].get('height')
            if object_thickness is None:
                raise ValueError("Object thickness not provided and height not found in dimensions")
        else:
            raise ValueError("Object thickness not provided and could not be auto-detected from ArUco annotations")
    
    # Adjust Z-coordinates to center of object
    adjusted_grasp_points = adjust_z_to_center(
        grasp_data['grasp_points'], 
        object_thickness
    )
    
    # Convert to numpy array for easier manipulation
    source_points_3d = []
    for gp in adjusted_grasp_points:
        point = np.array([
            gp['position']['x'],
            gp['position']['y'],
            gp['position']['z']
        ])
        source_points_3d.append(point)
    
    # Transform grasp points from source marker to CAD center
    # IMPORTANT: The coordinate transformations in Step 1 already account for the marker offset.
    # So we just need to convert from cm to meters - no offset subtraction needed.
    
    grasp_points_in_cad = []
    for i, point in enumerate(source_points_3d):
        # Point is in cm relative to marker's detection frame (where marker was facing up during detection)
        # The detection frame has identity rotation - no rotation needed
        # To get grasp point in CAD center frame:
        # 1. Convert from cm to meters
        # 2. Z is set to 0.0 (CAD center plane) regardless of marker Z offset
        point_m = point / 100.0
        
        # Convert to CAD center frame (offset already accounted for in Step 1)
        point_in_cad_center = point_m
        
        grasp_points_in_cad.append({
            "id": i + 1,
            "position": {
                "x": float(point_in_cad_center[0]),
                "y": float(point_in_cad_center[1]),
                "z": 0.0  # CAD center plane (origin)
            },
            "type": "center_point",
            "approach_vector": {
                "x": 0.0,
                "y": 0.0,
                "z": 1.0
            }
        })
    
    # Create output data structure
    all_markers_grasp_data = {
        "object_name": object_name,
        "display_name": object_name.replace('_', ' ').title(),
        "source_marker_id": source_marker_id,
        "object_thickness": object_thickness,
        "total_grasp_points": len(grasp_points_in_cad),
        "coordinate_frame": "cad_center",
        "grasp_points": grasp_points_in_cad,
        "wireframe": {
            "vertices": wireframe_data['vertices'],
            "edges": wireframe_data['edges'],
            "mesh_info": wireframe_data['mesh_info']
        },
        "markers": []
    }
    
    # Add marker information (without per-marker grasp points)
    for marker in aruco_data['markers']:
        marker_id = marker['aruco_id']
        
        # Extract position and rotation from T_object_to_marker
        T_object_to_marker = marker.get('T_object_to_marker', {})
        pos_data = T_object_to_marker.get('position', {})
        rot_data = T_object_to_marker.get('rotation', {})
        
        marker_grasp_data = {
            "aruco_id": marker_id,
            "size": aruco_data.get('size', 0.02),  # Default size if not specified
            "pose_absolute": {
                "position": {
                    "x": pos_data.get('x', 0.0),
                    "y": pos_data.get('y', 0.0),
                    "z": pos_data.get('z', 0.0)
                },
                "rotation": {
                    "roll": rot_data.get('roll', 0.0),
                    "pitch": rot_data.get('pitch', 0.0),
                    "yaw": rot_data.get('yaw', 0.0)
                }
            }
        }
        
        all_markers_grasp_data["markers"].append(marker_grasp_data)
    
    # Save to grasp directory
    output_dir = data_path / "grasp"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{object_name}_grasp_points_all_markers.json"
    with open(output_file, 'w') as f:
        json.dump(all_markers_grasp_data, f, indent=2)
    
    return output_file

