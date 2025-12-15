"""Main pipeline orchestrator for CAD to grasp points conversion"""

import sys
from pathlib import Path
import cv2

from ..utils.renderer import CADRenderer
from ..utils.mapper import Point2Dto3DMapper
from ..utils.region_detector import visualize_center_points_only


class CADToGraspPipeline:
    """Complete pipeline from CAD to grasp points"""
    
    def __init__(self, data_dir=None, outputs_dir=None):
        """
        Initialize the pipeline.
        
        Args:
            data_dir: Path to the data directory containing models, aruco, wireframe folders
            outputs_dir: Path to output directory (default: outputs in app directory)
        """
        if data_dir is None:
            # Default to project data directory
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.aruco_dir = self.data_dir / "aruco"
        
        # Create output directories
        if outputs_dir is None:
            outputs_dir = Path(__file__).parent.parent / "outputs"
        self.outputs_dir = Path(outputs_dir)
        self.masks_dir = self.outputs_dir / "masks"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.renderer = CADRenderer(image_size=(1024, 1024))
        self.mapper = Point2Dto3DMapper()
    
    def find_object_files(self, object_name):
        """
        Find CAD and ArUco annotation files for an object.
        
        Args:
            object_name: Name of the object (e.g., "base_scaled70")
            
        Returns:
            Tuple of (cad_path, aruco_path)
        """
        # Try different extensions for CAD file
        cad_extensions = ['.obj', '.stl', '.ply']
        cad_path = None
        
        for ext in cad_extensions:
            potential_path = self.models_dir / f"{object_name}{ext}"
            if potential_path.exists():
                cad_path = potential_path
                break
        
        if cad_path is None:
            raise FileNotFoundError(f"CAD file not found for object: {object_name}")
        
        # Find ArUco annotation file
        aruco_path = self.aruco_dir / f"{object_name}_aruco.json"
        if not aruco_path.exists():
            raise FileNotFoundError(f"ArUco annotation not found: {aruco_path}")
        
        return cad_path, aruco_path
    
    def render_top_down_image(self, object_name, marker_id, camera_distance=0.5):
        """
        Render a top-down view of the object with specified marker facing up.
        
        Args:
            object_name: Name of the object
            marker_id: ArUco marker ID to face up
            camera_distance: Camera distance from object
            
        Returns:
            Dictionary with rendering data
        """
        # Find files
        cad_path, aruco_path = self.find_object_files(object_name)
        
        # Render
        output_image = self.outputs_dir / f"{object_name}_marker{marker_id}_topdown.png"
        render_data = self.renderer.render_object_with_marker_up(
            obj_path=cad_path,
            aruco_json_path=aruco_path,
            marker_id=marker_id,
            output_image_path=output_image,
            camera_distance=camera_distance
        )
        
        render_data['output_image_path'] = output_image
        render_data['object_name'] = object_name
        render_data['marker_id'] = marker_id
        
        return render_data
    
    def detect_grasp_points_2d(self, image_path, object_name, marker_id, min_area_threshold=1000):
        """
        Detect 2D grasp points from the rendered image using adaptive region splitting.
        
        Args:
            image_path: Path to the rendered image
            object_name: Name of the object
            marker_id: Marker ID (for naming outputs)
            min_area_threshold: Minimum area threshold for filtering regions
            
        Returns:
            Tuple of (center_points_2d, filtered_regions, mask)
        """
        # Detect grasp points
        from ..utils.region_detector import detect_grasp_points_2d as detect_2d
        center_points_2d, filtered_regions, mask = detect_2d(
            str(image_path),
            min_area_threshold=min_area_threshold
        )
        
        # Save mask
        mask_output = self.masks_dir / f"{object_name}_marker{marker_id}_mask.png"
        cv2.imwrite(str(mask_output), mask)
        
        # Visualize center points
        viz_output = self.outputs_dir / f"{object_name}_marker{marker_id}_grasp_points_2d.png"
        visualize_center_points_only(str(image_path), mask, filtered_regions, str(viz_output))
        
        return center_points_2d, filtered_regions, mask
    
    def map_to_3d_marker_relative(self, points_2d, render_data):
        """
        Map 2D grasp points to 3D coordinates relative to ArUco marker.
        
        Args:
            points_2d: List of (x, y) tuples in image coordinates
            render_data: Dictionary containing rendering information
            
        Returns:
            List of 3D points relative to ArUco marker
        """
        # Map points
        points_3d = self.mapper.map_2d_points_to_3d_marker_relative(
            points_2d=points_2d,
            depth_map=render_data['depth'],
            intrinsics=render_data['intrinsics'],
            extrinsics=render_data['extrinsics'],
            transform=render_data['transform'],
            marker_data=render_data['marker_data'],
            mesh_center=render_data['mesh_center']
        )
        
        return points_3d
    
    def export_grasp_points(self, points_3d, object_name, marker_id):
        """
        Export grasp points to JSON file.
        
        Args:
            points_3d: List of 3D points relative to ArUco marker
            object_name: Name of the object
            marker_id: ArUco marker ID
            
        Returns:
            Path to the exported JSON file
        """
        output_json = self.outputs_dir / f"{object_name}_marker{marker_id}_grasp_points_3d.json"
        
        self.mapper.save_grasp_points_json(
            points_3d=points_3d,
            marker_id=marker_id,
            object_name=object_name,
            output_path=str(output_json)
        )
        
        return output_json
    
    def run(self, object_name, marker_id, camera_distance=0.5, min_area_threshold=1000):
        """
        Run the complete pipeline.
        
        Args:
            object_name: Name of the object (e.g., "base_scaled70")
            marker_id: ArUco marker ID to use as reference
            camera_distance: Camera distance for rendering
            min_area_threshold: Minimum area threshold for filtering regions
            
        Returns:
            Dictionary containing all pipeline outputs
        """
        try:
            # Step 1: Render top-down view
            render_data = self.render_top_down_image(object_name, marker_id, camera_distance)
            
            # Step 2: Detect 2D grasp points
            points_2d, regions, mask = self.detect_grasp_points_2d(
                render_data['output_image_path'],
                object_name,
                marker_id,
                min_area_threshold=min_area_threshold
            )
            
            # Step 3: Map to 3D marker-relative coordinates
            points_3d = self.map_to_3d_marker_relative(points_2d, render_data)
            
            # Step 4: Export grasp points
            output_json = self.export_grasp_points(points_3d, object_name, marker_id)
            
            return {
                'render_data': render_data,
                'points_2d': points_2d,
                'points_3d': points_3d,
                'regions': regions,
                'mask': mask,
                'output_json': output_json
            }
            
        except Exception as e:
            raise RuntimeError(f"Pipeline failed: {str(e)}") from e

