#!/usr/bin/env python3
"""
Generate grasp candidates with all possible approach vector orientations.

This script reads grasp points from JSON files in the data/grasp/ folder
and generates all possible approach vector orientations for each grasp point,
saving them as separate JSON files in the data/grasp_candidates/ folder.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation

# Define all possible approach vectors (normalized directions)
# 6 cardinal directions + 12 intermediate directions = 18 total

# Normalization factor for intermediate directions (45 degrees between axes)
sqrt2_inv = 1.0 / math.sqrt(2.0)

APPROACH_VECTORS = [
    # 6 Cardinal directions
    {"x": 0.0, "y": 0.0, "z": 1.0, "name": "top"},           # 1: From above (+Z)
    {"x": 0.0, "y": 0.0, "z": -1.0, "name": "bottom"},      # 2: From below (-Z)
    {"x": 1.0, "y": 0.0, "z": 0.0, "name": "right"},        # 3: From right (+X)
    {"x": -1.0, "y": 0.0, "z": 0.0, "name": "left"},        # 4: From left (-X)
    {"x": 0.0, "y": 1.0, "z": 0.0, "name": "front"},        # 5: From front (+Y)
    {"x": 0.0, "y": -1.0, "z": 0.0, "name": "back"},        # 6: From back (-Y)
    
    # 12 Intermediate directions (45 degrees between axes)
    # Between +X and +Y
    {"x": sqrt2_inv, "y": sqrt2_inv, "z": 0.0, "name": "right_front"},      # 7
    # Between +X and -Y
    {"x": sqrt2_inv, "y": -sqrt2_inv, "z": 0.0, "name": "right_back"},       # 8
    # Between -X and +Y
    {"x": -sqrt2_inv, "y": sqrt2_inv, "z": 0.0, "name": "left_front"},      # 9
    # Between -X and -Y
    {"x": -sqrt2_inv, "y": -sqrt2_inv, "z": 0.0, "name": "left_back"},       # 10
    
    # Between +X and +Z
    {"x": sqrt2_inv, "y": 0.0, "z": sqrt2_inv, "name": "right_top"},         # 11
    # Between +X and -Z
    {"x": sqrt2_inv, "y": 0.0, "z": -sqrt2_inv, "name": "right_bottom"},      # 12
    # Between -X and +Z
    {"x": -sqrt2_inv, "y": 0.0, "z": sqrt2_inv, "name": "left_top"},         # 13
    # Between -X and -Z
    {"x": -sqrt2_inv, "y": 0.0, "z": -sqrt2_inv, "name": "left_bottom"},     # 14
    
    # Between +Y and +Z
    {"x": 0.0, "y": sqrt2_inv, "z": sqrt2_inv, "name": "front_top"},         # 15
    # Between +Y and -Z
    {"x": 0.0, "y": sqrt2_inv, "z": -sqrt2_inv, "name": "front_bottom"},     # 16
    # Between -Y and +Z
    {"x": 0.0, "y": -sqrt2_inv, "z": sqrt2_inv, "name": "back_top"},         # 17
    # Between -Y and -Z
    {"x": 0.0, "y": -sqrt2_inv, "z": -sqrt2_inv, "name": "back_bottom"},      # 18
]

def calculate_grasp_candidate_position(grasp_point_pos: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the grasp candidate position from the grasp point position.
    The candidate position is the same for all approach directions of the same grasp point.
    
    For now, the candidate position is the same as the grasp point position.
    The offset of 0.115m is applied at runtime when moving the end-effector.
    
    Args:
        grasp_point_pos: Dictionary with 'x', 'y', 'z' keys for grasp point position
    
    Returns:
        Dictionary with 'x', 'y', 'z' keys for grasp candidate position
    """
    # The candidate position is the same as the grasp point position
    # The 0.115m offset is applied at runtime along the approach direction
    return {
        "x": float(grasp_point_pos['x']),
        "y": float(grasp_point_pos['y']),
        "z": float(grasp_point_pos['z'])
    }


def approach_vector_to_quaternion(approach_vec: Dict[str, float]) -> Dict[str, float]:
    """
    Convert an approach vector (direction) to a quaternion.
    
    The approach vector represents the direction from which we approach the grasp point.
    This function creates a rotation where the Z-axis (in the local frame) aligns with
    the approach vector direction.
    
    Args:
        approach_vec: Dictionary with 'x', 'y', 'z' keys representing the approach direction
    
    Returns:
        Dictionary with 'x', 'y', 'z', 'w' keys representing the quaternion
    """
    # Get approach vector as numpy array
    approach = np.array([approach_vec['x'], approach_vec['y'], approach_vec['z']])
    approach = approach / np.linalg.norm(approach)  # Normalize
    
    # Reference direction (Z-axis in local frame)
    z_axis_ref = np.array([0.0, 0.0, 1.0])
    
    # Construct rotation matrix where Z-axis aligns with approach vector
    # We need to find two perpendicular vectors to complete the frame
    if abs(approach[2]) < 0.9:
        # Not too close to Z-axis, use cross product with Z-axis
        x_axis_local = np.cross(z_axis_ref, approach)
        if np.linalg.norm(x_axis_local) < 1e-6:
            # Vectors are parallel, use X-axis as reference
            x_axis_local = np.array([1.0, 0.0, 0.0])
        else:
            x_axis_local = x_axis_local / np.linalg.norm(x_axis_local)
        y_axis_local = np.cross(approach, x_axis_local)
        y_axis_local = y_axis_local / np.linalg.norm(y_axis_local)
        x_axis_local = np.cross(y_axis_local, approach)
        x_axis_local = x_axis_local / np.linalg.norm(x_axis_local)
    else:
        # Close to Z-axis, use X-axis as reference
        x_axis_local = np.array([1.0, 0.0, 0.0])
        y_axis_local = np.cross(approach, x_axis_local)
        y_axis_local = y_axis_local / np.linalg.norm(y_axis_local)
        x_axis_local = np.cross(y_axis_local, approach)
        x_axis_local = x_axis_local / np.linalg.norm(x_axis_local)
    
    # Construct rotation matrix: columns are x, y, z axes
    # The Z-axis column is the approach vector
    rotation_matrix = np.column_stack([x_axis_local, y_axis_local, approach])
    
    # Convert to quaternion
    rotation = Rotation.from_matrix(rotation_matrix)
    quat = rotation.as_quat()  # Returns [x, y, z, w]
    
    return {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quat[2]),
        "w": float(quat[3])
    }


def quaternion_to_rpy(quat: Dict[str, float]) -> Dict[str, float]:
    """
    Convert quaternion to roll, pitch, yaw (RPY) in degrees.
    
    Uses the same convention as grasp_points_publisher: 'xyz' Euler angles in degrees.
    Normalizes angles to [-180, 180] range for consistency.
    
    Args:
        quat: Dictionary with 'x', 'y', 'z', 'w' keys representing the quaternion
    
    Returns:
        Dictionary with 'roll', 'pitch', 'yaw' keys in degrees (normalized to [-180, 180])
    """
    quat_array = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
    rotation = Rotation.from_quat(quat_array)
    rpy = rotation.as_euler('xyz', degrees=True)
    
    # Normalize angles to [-180, 180] range
    rpy_normalized = np.array([
        ((rpy[0] + 180) % 360) - 180,
        ((rpy[1] + 180) % 360) - 180,
        ((rpy[2] + 180) % 360) - 180
    ])
    
    return {
        "roll": float(rpy_normalized[0]),
        "pitch": float(rpy_normalized[1]),
        "yaw": float(rpy_normalized[2])
    }


def generate_grasp_candidates_for_file(grasp_file: Path, output_dir: Path):
    """
    Generate grasp candidates for all grasp points in a single JSON file.
    Creates one output file per object containing all grasp candidates.
    
    Args:
        grasp_file: Path to the input grasp points JSON file
        output_dir: Directory to save the output JSON files
    
    Returns:
        Number of grasp candidates generated
    """
    print(f"\n📂 Processing: {grasp_file.name}")
    
    # Read the grasp points file
    with open(grasp_file, 'r') as f:
        grasp_data = json.load(f)
    
    object_name = grasp_data.get('object_name', 'unknown')
    grasp_points = grasp_data.get('grasp_points', [])
    
    print(f"  Found {len(grasp_points)} grasp points for object: {object_name}")
    
    # Create output structure with all grasp candidates
    output_data = {
        "object_name": object_name,
        "total_grasp_candidates": 0,
        "grasp_candidates": []
    }
    
    # Generate candidates for each grasp point
    for grasp_point in grasp_points:
        grasp_point_id = grasp_point.get('id', 0)
        grasp_point_pos = grasp_point.get('position', {})
        
        # Calculate grasp candidate position once for this grasp point
        # (same for all directions)
        candidate_pos = calculate_grasp_candidate_position(grasp_point_pos)
        
        # Generate a candidate for each approach vector
        for direction_id, approach_vec in enumerate(APPROACH_VECTORS, start=1):
            # Convert approach vector to quaternion
            approach_quaternion = approach_vector_to_quaternion(approach_vec)
            
            # Convert quaternion to RPY for verification
            approach_rpy = quaternion_to_rpy(approach_quaternion)
            
            # Debug output for first grasp point and first few directions
            if grasp_point_id == 1 and direction_id <= 6:
                print(f"    Direction {direction_id} ({approach_vec['name']}): "
                      f"approach_vec=[{approach_vec['x']:.3f}, {approach_vec['y']:.3f}, {approach_vec['z']:.3f}], "
                      f"quat=[{approach_quaternion['x']:.3f}, {approach_quaternion['y']:.3f}, "
                      f"{approach_quaternion['z']:.3f}, {approach_quaternion['w']:.3f}], "
                      f"RPY=[{approach_rpy['roll']:.2f}°, {approach_rpy['pitch']:.2f}°, {approach_rpy['yaw']:.2f}°]")
            
            # Create the candidate JSON structure
            candidate_data = {
                "grasp_point_id": grasp_point_id,
                "direction_id": direction_id,
                "grasp_candidate_position": candidate_pos,
                "approach_quaternion": approach_quaternion,
                "approach_rpy": approach_rpy  # Add RPY for verification
            }
            
            output_data["grasp_candidates"].append(candidate_data)
            output_data["total_grasp_candidates"] += 1
    
    # Generate output filename: {object_name}_grasp_candidates.json
    output_filename = f"{object_name}_grasp_candidates.json"
    output_path = output_dir / output_filename
    
    # Save the candidate JSON file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    total_candidates = output_data["total_grasp_candidates"]
    print(f"  ✅ Generated {total_candidates} grasp candidates ({len(grasp_points)} points × {len(APPROACH_VECTORS)} directions)")
    print(f"  💾 Saved to: {output_filename}")
    return total_candidates


def main():
    """Main entry point."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    grasp_dir = project_root / "data" / "grasp"
    output_dir = project_root / "data" / "grasp_candidates"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Generating grasp candidates with all possible approach vectors")
    print(f"📁 Input directory: {grasp_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Note: Candidate position is the same for all directions of the same grasp point")
    print(f"🧭 Approach directions: {len(APPROACH_VECTORS)}")
    print("   " + ", ".join([f"{i+1}. {v['name']}" for i, v in enumerate(APPROACH_VECTORS)]))
    
    # Find all grasp points JSON files
    grasp_files = list(grasp_dir.glob("*_grasp_points_all_markers.json"))
    
    if not grasp_files:
        print(f"\n❌ No grasp points files found in {grasp_dir}")
        print("   Expected files matching pattern: *_grasp_points_all_markers.json")
        return
    
    print(f"\n📋 Found {len(grasp_files)} grasp points file(s)")
    
    # Process each file
    total_candidates = 0
    for grasp_file in sorted(grasp_files):
        try:
            candidates = generate_grasp_candidates_for_file(grasp_file, output_dir)
            total_candidates += candidates
        except Exception as e:
            print(f"  ❌ Error processing {grasp_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Complete! Generated {total_candidates} total grasp candidates")
    print(f"📁 Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()

