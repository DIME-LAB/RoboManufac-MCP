"""2D to 3D Point Mapper - Maps 2D grasp points to 3D coordinates relative to ArUco marker"""

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
        # Note: The image projection flips Y (world_to_image uses: img_y = (1.0 - (py / max_dimension + 0.5)) * image_size[1])
        # So we need to account for this when converting back
        x_norm = (pixel_x - cx) / fx
        y_norm = (pixel_y - cy) / fy
        
        # Scale by depth to get 3D point in camera frame
        x_cam = x_norm * depth
        y_cam = -y_norm * depth  # Flip Y to account for image projection Y-flip
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
    
    def world_to_marker_coords(self, point_world, transform, marker_data, mesh_center):
        """
        Convert world coordinates to coordinates relative to the ArUco marker.
        
        Args:
            point_world: 3D point in world coordinates [x, y, z] (already in transformed space after camera_to_world)
            transform: 4x4 transformation matrix that was applied to align marker
            marker_data: Dictionary containing marker pose information with T_object_to_marker
            mesh_center: Center of the transformed mesh (used during rendering, for reference)
            
        Returns:
            3D point relative to ArUco marker frame [x, y, z]
        """
        # The point_world is already in transformed space (not centered)
        # because camera_to_world_coords inverts the extrinsic matrix which had translation -mesh_center,
        # so it adds mesh_center back, putting us in transformed space.
        # We need to:
        # 1. Undo the transformation to get back to original CAD space
        # 2. Transform to marker-relative coordinates
        
        # Undo the transformation to get back to original CAD space
        transform_inv = np.linalg.inv(transform)
        point_homogeneous = np.append(point_world, 1.0)
        point_cad_space = (transform_inv @ point_homogeneous)[:3]
        
        # Get marker position and orientation from T_object_to_marker
        # T_object_to_marker.position is the position of marker relative to object center (in object frame)
        T_object_to_marker = marker_data.get('T_object_to_marker', {})
        pos_data = T_object_to_marker.get('position', {})
        
        marker_pos = np.array([
            pos_data.get('x', 0.0),
            pos_data.get('y', 0.0),
            pos_data.get('z', 0.0)
        ])
        
        # Adjust for CAD center (marker position is already relative to CAD center in object frame)
        # So we just need to subtract marker position
        point_relative_to_marker_pos = point_cad_space - marker_pos
        
        # IMPORTANT: We do NOT apply the marker's rotation because the grasp detection was done
        # with the marker facing up (identity orientation), not in its actual CAD orientation.
        # The points should be in the marker's local frame with identity orientation.
        # This matches the original Python code behavior in annotation_transformer.py.
        # The annotation_transformer will later convert these to CAD center by simply adding
        # the marker position back, without applying any rotation.
        
        # However, we need to account for coordinate system differences.
        # Apply 180° rotation about Y axis (flips X and Z, keeps Y unchanged)
        # This matches the coordinate system convention used in the annotation transformer.
        # Rotation matrix for 180° about Y: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        point_relative_to_marker_pos = np.array([
            -point_relative_to_marker_pos[0],  # Flip X (180° rotation about Y)
            point_relative_to_marker_pos[1],   # Keep Y unchanged
            -point_relative_to_marker_pos[2]   # Flip Z (180° rotation about Y)
        ])
        
        return point_relative_to_marker_pos
    
    def map_2d_points_to_3d_marker_relative(self, points_2d, depth_map, 
                                           intrinsics, extrinsics, 
                                           transform, marker_data, mesh_center):
        """
        Map multiple 2D points to 3D coordinates relative to ArUco marker.
        
        Args:
            points_2d: List of (x, y) tuples in image coordinates
            depth_map: 2D array of depth values
            intrinsics: Camera intrinsic parameters
            extrinsics: Camera extrinsic matrix
            transform: Transformation matrix applied to align marker
            marker_data: Dictionary containing marker pose information
            mesh_center: Center of the transformed mesh (used during rendering to center the object)
            
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
                continue
            
            depth = depth_map[py, px]
            
            # Skip if depth is invalid
            if depth <= 0 or np.isnan(depth) or np.isinf(depth):
                continue
            
            # Convert image pixel to camera coordinates
            point_camera = self.image_to_camera_coords(pixel_x, pixel_y, depth, intrinsics)
            
            # Convert camera coordinates to world coordinates
            point_world = self.camera_to_world_coords(point_camera, extrinsics)
            
            # Convert world coordinates to marker-relative coordinates
            point_marker_relative = self.world_to_marker_coords(
                point_world, transform, marker_data, mesh_center
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
        
        return grasp_data

