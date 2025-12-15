"""CAD to Image Renderer - Renders CAD models with specific ArUco marker orientations"""

import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R


class CADRenderer:
    """Renders CAD models with specific ArUco marker orientations"""
    
    def __init__(self, image_size=(1024, 1024)):
        """
        Initialize the CAD renderer.
        
        Args:
            image_size: Tuple of (width, height) for the rendered image
        """
        self.image_size = image_size
        self.vis = None
        
    def load_aruco_annotation(self, aruco_json_path):
        """
        Load ArUco marker annotation from JSON file.
        
        Args:
            aruco_json_path: Path to the ArUco annotation JSON file
            
        Returns:
            Dictionary containing marker information
        """
        with open(aruco_json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def find_marker_by_id(self, aruco_data, marker_id):
        """
        Find a specific marker by its ID.
        
        Args:
            aruco_data: ArUco annotation data
            marker_id: The ArUco marker ID to find
            
        Returns:
            Marker data dictionary or None if not found
        """
        for marker in aruco_data['markers']:
            if marker['aruco_id'] == marker_id:
                return marker
        return None
    
    def compute_transform_to_align_marker(self, marker_data):
        """
        Compute transformation matrix to align the marker to face upward (top-down view).
        
        The goal is to rotate the object so that:
        - The marker's Z-axis points DOWN (so camera looking down sees it)
        - The marker's surface normal becomes [0, 0, -1] after transformation
        
        Args:
            marker_data: Dictionary containing marker pose information with T_object_to_marker
            
        Returns:
            4x4 transformation matrix
        """
        # Get marker's rotation from T_object_to_marker (rotation from object frame to marker frame)
        T_object_to_marker = marker_data.get('T_object_to_marker', {})
        rot_data = T_object_to_marker.get('rotation', {})
        
        roll = rot_data.get('roll', 0.0)
        pitch = rot_data.get('pitch', 0.0)
        yaw = rot_data.get('yaw', 0.0)
        
        # Create rotation from marker's orientation
        # T_object_to_marker.rotation is the rotation from object frame to marker frame
        # So R_object_to_marker is what we need
        marker_rotation = R.from_euler('xyz', [roll, pitch, yaw])
        
        # We want the marker to face UP (normal pointing in +Z direction)
        # Current surface normal is pointing down [0, 0, -1]
        # We need to rotate so that the current normal becomes [0, 0, 1]
        
        # Desired orientation: marker facing up (normal = [0, 0, 1])
        # This means we need to flip the object 180 degrees around X axis
        desired_rotation = R.from_euler('x', np.pi)
        
        # Combine: first apply marker rotation inverse, then desired rotation
        transform_rotation = desired_rotation * marker_rotation.inv()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = transform_rotation.as_matrix()
        
        return transform
    
    def render_top_down_view(self, mesh, transform, camera_distance=0.5):
        """
        Render a true orthographic top-down view using direct vertex projection.
        
        Args:
            mesh: Open3D TriangleMesh object
            transform: 4x4 transformation matrix
            camera_distance: Distance of camera from the object center (not used in ortho)
            
        Returns:
            Rendered image as numpy array (RGB), depth image, and camera intrinsics
        """
        # Apply transformation to mesh
        mesh_transformed = o3d.geometry.TriangleMesh(mesh)
        mesh_transformed.transform(transform)
        
        # Get vertices and triangles
        vertices = np.asarray(mesh_transformed.vertices)
        triangles = np.asarray(mesh_transformed.triangles)
        
        # Compute bounding box for orthographic projection
        bbox = mesh_transformed.get_axis_aligned_bounding_box()
        mesh_center = bbox.get_center()
        mesh_extent = bbox.get_extent()
        
        # Add padding
        scale_factor = 1.2
        ortho_width = mesh_extent[0] * scale_factor
        ortho_height = mesh_extent[1] * scale_factor
        
        # Preserve aspect ratio: use the larger dimension to determine scale
        # This ensures we don't squish non-square objects
        max_dimension = max(ortho_width, ortho_height)
        
        # Create orthographic projection (simple 2D projection ignoring Z)
        # Map world coordinates to image coordinates while preserving aspect ratio
        def world_to_image(point):
            # Center the object
            px = point[0] - mesh_center[0]
            py = point[1] - mesh_center[1]
            
            # Scale to image size using max dimension to preserve aspect ratio
            # This ensures objects maintain their true proportions
            img_x = (px / max_dimension + 0.5) * self.image_size[0]
            img_y = (1.0 - (py / max_dimension + 0.5)) * self.image_size[1]  # Flip Y
            
            return int(img_x), int(img_y)
        
        # Create blank image and depth buffer
        image_np = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 45  # Dark gray background
        
        # For orthographic projection, depth is just the Z coordinate
        # We'll store the max Z (top surface) for each pixel
        depth_np = np.full((self.image_size[1], self.image_size[0]), -np.inf, dtype=np.float32)
        
        # Render each triangle
        for tri_idx in triangles:
            # Get triangle vertices
            v0, v1, v2 = vertices[tri_idx]
            
            # Get 2D projections
            p0 = world_to_image(v0)
            p1 = world_to_image(v1)
            p2 = world_to_image(v2)
            
            # Get depths (Z values) - in orthographic, we want the highest Z (top surface)
            z0, z1, z2 = v0[2], v1[2], v2[2]
            
            # Draw filled triangle
            pts = np.array([p0, p1, p2], dtype=np.int32)
            
            # Fill triangle with color (light blue/gray)
            cv2.fillPoly(image_np, [pts], color=(180, 180, 180))
            
            # Simple depth: use average Z of triangle
            avg_depth = (z0 + z1 + z2) / 3.0
            
            # Create mask for this triangle
            mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Update depth buffer (for orthographic top-down, we want maximum Z = top surface)
            depth_np = np.where(mask > 0, np.maximum(depth_np, avg_depth), depth_np)
        
        # Draw triangle edges for better visualization
        for tri_idx in triangles:
            v0, v1, v2 = vertices[tri_idx]
            p0 = world_to_image(v0)
            p1 = world_to_image(v1)
            p2 = world_to_image(v2)
            
            cv2.line(image_np, p0, p1, (120, 120, 120), 1)
            cv2.line(image_np, p1, p2, (120, 120, 120), 1)
            cv2.line(image_np, p2, p0, (120, 120, 120), 1)
        
        # Create intrinsic parameters for orthographic projection
        # For orthographic projection with our direct vertex mapping:
        # focal_length should represent the scaling from world units to pixels
        # Since we map world coordinates directly: pixel = (world / max_dimension + 0.5) * image_size
        # This means: focal_length = image_size / max_dimension
        focal_length = self.image_size[0] / max_dimension
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.image_size[0],
            height=self.image_size[1],
            fx=focal_length,
            fy=focal_length,
            cx=self.image_size[0] / 2,
            cy=self.image_size[1] / 2
        )
        
        # Create extrinsic matrix for top-down view
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = -mesh_center  # Translate to center
        
        return image_np, depth_np, intrinsic, extrinsic, mesh_center
    
    def render_object_with_marker_up(self, obj_path, aruco_json_path, marker_id, 
                                     output_image_path=None, camera_distance=0.5):
        """
        Main function to render an object with a specific marker facing up.
        
        Args:
            obj_path: Path to the OBJ/STL/PLY CAD file
            aruco_json_path: Path to the ArUco annotation JSON
            marker_id: The ArUco marker ID that should face up
            output_image_path: Optional path to save the rendered image
            camera_distance: Distance of camera from object center
            
        Returns:
            Dictionary containing:
                - image: Rendered RGB image
                - depth: Depth map
                - transform: Transformation matrix applied
                - marker_data: Information about the marker
                - intrinsics: Camera intrinsic matrix
                - extrinsics: Camera extrinsic matrix
        """
        # Load CAD model
        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Load ArUco annotation
        aruco_data = self.load_aruco_annotation(aruco_json_path)
        
        # Find the specified marker
        marker_data = self.find_marker_by_id(aruco_data, marker_id)
        
        if marker_data is None:
            raise ValueError(f"Marker ID {marker_id} not found in annotation file")
        
        # Compute transformation to align marker facing up
        transform = self.compute_transform_to_align_marker(marker_data)
        
        # Render top-down view
        image, depth, intrinsics, extrinsics, mesh_center = self.render_top_down_view(
            mesh, transform, camera_distance
        )
        
        # Save image if requested
        if output_image_path:
            cv2.imwrite(str(output_image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
        return {
            'image': image,
            'depth': depth,
            'transform': transform,
            'marker_data': marker_data,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'mesh_center': mesh_center,  # The actual center used during rendering
            'cad_center': aruco_data.get('cad_object_info', {}).get('center', [0.0, 0.0, 0.0])
        }

