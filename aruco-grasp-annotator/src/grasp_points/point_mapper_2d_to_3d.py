#!/usr/bin/env python3
"""
2D to 3D Point Mapper
Maps 2D grasp points from rendered images back to 3D coordinates relative to ArUco marker.
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R


class Point2Dto3DMapper:
    """Maps 2D image points to 3D coordinates relative to an ArUco marker"""
    
    def __init__(self):
        """Initialize the mapper"""
        pass
    
    def image_to_camera_coords(self, pixel_x, pixel_y, depth, intrinsics):
        """
        Convert 2D image pixel coordinates to 3D camera coordinates.
        
        Args:
            pixel_x: X coordinate in image (pixels)
            pixel_y: Y coordinate in image (pixels)
            depth: Depth value at that pixel
            intrinsics: Camera intrinsic matrix (Open3D PinholeCameraIntrinsic)
            
        Returns:
            3D point in camera coordinates [x, y, z]
        """
        # Get intrinsic parameters
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]
        
        # Convert pixel to normalized image coordinates
        x_norm = (pixel_x - cx) / fx
        y_norm = (pixel_y - cy) / fy
        
        # Scale by depth to get 3D point in camera frame
        x_cam = x_norm * depth
        y_cam = y_norm * depth
        z_cam = depth
        
        return np.array([x_cam, y_cam, z_cam])
    
    def camera_to_world_coords(self, point_camera, extrinsics):
        """
        Convert 3D camera coordinates to world coordinates.
        
        Args:
            point_camera: 3D point in camera frame [x, y, z]
            extrinsics: 4x4 camera extrinsic matrix (world to camera)
            
        Returns:
            3D point in world coordinates [x, y, z]
        """
        # Extrinsic matrix is world-to-camera, we need camera-to-world
        camera_to_world = np.linalg.inv(extrinsics)
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_camera, 1.0)
        
        # Transform to world coordinates
        point_world_homogeneous = camera_to_world @ point_homogeneous
        
        return point_world_homogeneous[:3]
    
    def world_to_marker_coords(self, point_world, transform, marker_data, cad_center):
        """
        Convert world coordinates to coordinates relative to the ArUco marker.
        
        Args:
            point_world: 3D point in world coordinates [x, y, z]
            transform: 4x4 transformation matrix that was applied to align marker
            marker_data: Dictionary containing marker pose information
            cad_center: Center of the CAD object
            
        Returns:
            3D point relative to ArUco marker frame [x, y, z]
        """
        # The point is currently in the transformed world space
        # We need to:
        # 1. Undo the transformation to get back to original CAD space
        # 2. Transform to marker-relative coordinates
        
        # Undo the transformation
        transform_inv = np.linalg.inv(transform)
        point_homogeneous = np.append(point_world, 1.0)
        point_cad_space = (transform_inv @ point_homogeneous)[:3]
        
        # Get marker position and orientation in CAD space
        marker_pos = np.array([
            marker_data['pose_relative_to_cad_center']['position']['x'],
            marker_data['pose_relative_to_cad_center']['position']['y'],
            marker_data['pose_relative_to_cad_center']['position']['z']
        ])
        
        # Adjust for CAD center (marker position is already relative to CAD center)
        # So we just need to subtract marker position
        point_relative_to_marker_pos = point_cad_space - marker_pos
        
        # Get marker rotation
        roll = marker_data['pose_relative_to_cad_center']['rotation']['roll']
        pitch = marker_data['pose_relative_to_cad_center']['rotation']['pitch']
        yaw = marker_data['pose_relative_to_cad_center']['rotation']['yaw']
        
        marker_rotation = R.from_euler('xyz', [roll, pitch, yaw])
        
        # Rotate point to marker frame
        # We need to apply inverse rotation to transform from world to marker frame
        point_in_marker_frame = marker_rotation.inv().apply(point_relative_to_marker_pos)
        
        return point_in_marker_frame
    
    def map_2d_points_to_3d_marker_relative(self, points_2d, depth_map, 
                                           intrinsics, extrinsics, 
                                           transform, marker_data, cad_center):
        """
        Map multiple 2D points to 3D coordinates relative to ArUco marker.
        
        Args:
            points_2d: List of (x, y) tuples in image coordinates
            depth_map: 2D array of depth values
            intrinsics: Camera intrinsic parameters
            extrinsics: Camera extrinsic matrix
            transform: Transformation matrix applied to align marker
            marker_data: Dictionary containing marker pose information
            cad_center: Center of the CAD object
            
        Returns:
            List of 3D points relative to ArUco marker frame
        """
        points_3d_marker_relative = []
        
        for pixel_x, pixel_y in points_2d:
            # Get depth at this pixel
            # Note: pixel coordinates might be floats, so we round them
            px = int(round(pixel_x))
            py = int(round(pixel_y))
            
            # Check bounds
            if py < 0 or py >= depth_map.shape[0] or px < 0 or px >= depth_map.shape[1]:
                print(f"  Warning: Point ({px}, {py}) is out of bounds, skipping")
                continue
            
            depth = depth_map[py, px]
            
            # Skip if depth is invalid
            if depth <= 0 or np.isnan(depth) or np.isinf(depth):
                print(f"  Warning: Invalid depth at ({px}, {py}): {depth}, skipping")
                continue
            
            # Convert image pixel to camera coordinates
            point_camera = self.image_to_camera_coords(pixel_x, pixel_y, depth, intrinsics)
            
            # Convert camera coordinates to world coordinates
            point_world = self.camera_to_world_coords(point_camera, extrinsics)
            
            # Convert world coordinates to marker-relative coordinates
            point_marker_relative = self.world_to_marker_coords(
                point_world, transform, marker_data, cad_center
            )
            
            points_3d_marker_relative.append(point_marker_relative)
        
        return points_3d_marker_relative
    
    def save_grasp_points_json(self, points_3d, marker_id, object_name, output_path):
        """
        Save 3D grasp points to JSON file in a format compatible with robotics applications.
        
        Args:
            points_3d: List of 3D points relative to ArUco marker
            marker_id: The ArUco marker ID
            object_name: Name of the object
            output_path: Path to save the JSON file
        """
        grasp_data = {
            "object_name": object_name,
            "aruco_marker_id": marker_id,
            "coordinate_frame": "aruco_marker",
            "description": "Grasp points relative to ArUco marker coordinate frame",
            "total_points": len(points_3d),
            "grasp_points": []
        }
        
        for i, point in enumerate(points_3d):
            grasp_point = {
                "id": i + 1,
                "position": {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2])
                },
                "type": "center_point",
                "approach_vector": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0
                }
            }
            grasp_data["grasp_points"].append(grasp_point)
        
        with open(output_path, 'w') as f:
            json.dump(grasp_data, f, indent=2)
        
        print(f"  Saved grasp points: {output_path}")
        return grasp_data


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python point_mapper_2d_to_3d.py <center_points_json>")
        print("\nExample:")
        print("  python point_mapper_2d_to_3d.py outputs/base_marker24_topdown_center_points.json")
        print("\nNote: This script requires rendering output data to be available")
        sys.exit(1)
    
    print("2D to 3D Point Mapper")
    print("=" * 50)
    print("\nThis module provides functions to map 2D image points to 3D coordinates")
    print("relative to an ArUco marker coordinate frame.")
    print("\nUse the full pipeline script (cad_to_grasp_pipeline.py) to run the complete workflow.")


if __name__ == "__main__":
    main()

