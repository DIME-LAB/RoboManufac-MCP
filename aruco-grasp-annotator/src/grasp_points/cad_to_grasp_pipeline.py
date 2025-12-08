#!/usr/bin/env python3
"""
CAD to Grasp Points Pipeline
Complete pipeline that takes a CAD model with ArUco annotations and generates
3D grasp points relative to a specified ArUco marker.

Workflow:
1. Load CAD model and ArUco annotations
2. Render top-down view with specified marker facing up
3. Apply grasp point detection algorithm to rendered image
4. Map 2D grasp points back to 3D coordinates relative to ArUco marker
5. Export grasp points in robotics-compatible format

Usage:
    python cad_to_grasp_pipeline.py --object base_scaled70 --marker-id 24
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Import our custom modules
from cad_to_image_renderer import CADRenderer
from point_mapper_2d_to_3d import Point2Dto3DMapper
from adaptive_region_center_points import (
    generate_mask_from_image,
    clean_mask,
    find_contours,
    extract_boundary_corners,
    find_mask_boundary,
    create_boundary_aware_lines,
    find_line_intersections,
    create_regions_from_boundary_aware_lines,
    filter_regions_by_area,
    visualize_center_points_only,
    save_center_points_data
)


class CADToGraspPipeline:
    """Complete pipeline from CAD to grasp points"""
    
    def __init__(self, data_dir="../../data"):
        """
        Initialize the pipeline.
        
        Args:
            data_dir: Path to the data directory containing models, aruco, wireframe folders
        """
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.aruco_dir = self.data_dir / "aruco"
        
        # Create output directories
        self.outputs_dir = Path("outputs")
        self.masks_dir = Path("masks")
        self.outputs_dir.mkdir(exist_ok=True)
        self.masks_dir.mkdir(exist_ok=True)
        
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
        print(f"\n{'='*60}")
        print("STEP 1: RENDERING TOP-DOWN VIEW")
        print(f"{'='*60}")
        
        # Find files
        cad_path, aruco_path = self.find_object_files(object_name)
        
        print(f"  CAD file: {cad_path}")
        print(f"  ArUco annotation: {aruco_path}")
        print(f"  Marker ID: {marker_id}")
        
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
    
    def detect_grasp_points_2d(self, image_path, object_name, marker_id):
        """
        Detect 2D grasp points from the rendered image using adaptive region splitting.
        
        Args:
            image_path: Path to the rendered image
            object_name: Name of the object
            marker_id: Marker ID (for naming outputs)
            
        Returns:
            List of (x, y) tuples representing 2D grasp point centers
        """
        print(f"\n{'='*60}")
        print("STEP 2: DETECTING 2D GRASP POINTS")
        print(f"{'='*60}")
        
        # Generate mask from image
        print("  Generating mask from rendered image...")
        mask = generate_mask_from_image(image_path)
        print(f"    Mask shape: {mask.shape}")
        
        # Clean mask
        print("  Cleaning mask...")
        mask = clean_mask(mask)
        
        # Save mask
        mask_output = self.masks_dir / f"{object_name}_marker{marker_id}_mask.png"
        cv2.imwrite(str(mask_output), mask)
        print(f"    Saved mask: {mask_output}")
        
        # Find contours
        print("  Finding contours...")
        contour = find_contours(mask)
        if contour is None:
            raise ValueError("No contours found in mask")
        
        # Extract boundary corners
        print("  Extracting boundary corners...")
        corner_points = extract_boundary_corners(contour, mask)
        
        # Find mask boundary
        print("  Finding mask boundary...")
        boundary_lines = find_mask_boundary(mask)
        
        # Create boundary-aware lines
        print("  Creating boundary-aware lines...")
        lines = create_boundary_aware_lines(corner_points, mask, boundary_lines)
        
        # Find line intersections
        print("  Finding line intersections...")
        intersections = find_line_intersections(lines)
        
        # Create regions
        print("  Creating regions...")
        regions = create_regions_from_boundary_aware_lines(lines, intersections, mask)
        
        # Filter regions
        print("  Filtering regions...")
        filtered_regions = filter_regions_by_area(regions, min_area_threshold=1000)
        
        # Visualize center points
        print("  Creating visualization...")
        viz_output = self.outputs_dir / f"{object_name}_marker{marker_id}_grasp_points_2d.png"
        visualize_center_points_only(str(image_path), mask, filtered_regions, str(viz_output))
        
        # Save center points data
        print("  Saving 2D center points data...")
        save_center_points_data(filtered_regions, str(viz_output))
        
        # Extract center points as list of (x, y) tuples
        center_points_2d = [(region['center_x'], region['center_y']) 
                           for region in filtered_regions]
        
        print(f"\n  ✓ Detected {len(center_points_2d)} grasp points")
        
        return center_points_2d, filtered_regions
    
    def map_to_3d_marker_relative(self, points_2d, render_data):
        """
        Map 2D grasp points to 3D coordinates relative to ArUco marker.
        
        Args:
            points_2d: List of (x, y) tuples in image coordinates
            render_data: Dictionary containing rendering information
            
        Returns:
            List of 3D points relative to ArUco marker
        """
        print(f"\n{'='*60}")
        print("STEP 3: MAPPING TO 3D MARKER-RELATIVE COORDINATES")
        print(f"{'='*60}")
        
        print(f"  Input: {len(points_2d)} 2D points")
        
        # Map points
        points_3d = self.mapper.map_2d_points_to_3d_marker_relative(
            points_2d=points_2d,
            depth_map=render_data['depth'],
            intrinsics=render_data['intrinsics'],
            extrinsics=render_data['extrinsics'],
            transform=render_data['transform'],
            marker_data=render_data['marker_data'],
            cad_center=render_data['cad_center']
        )
        
        print(f"  ✓ Mapped to {len(points_3d)} 3D points")
        
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
        print(f"\n{'='*60}")
        print("STEP 4: EXPORTING GRASP POINTS")
        print(f"{'='*60}")
        
        output_json = self.outputs_dir / f"{object_name}_marker{marker_id}_grasp_points_3d.json"
        
        self.mapper.save_grasp_points_json(
            points_3d=points_3d,
            marker_id=marker_id,
            object_name=object_name,
            output_path=str(output_json)
        )
        
        print(f"  ✓ Exported to: {output_json}")
        
        return output_json
    
    def run(self, object_name, marker_id, camera_distance=0.5):
        """
        Run the complete pipeline.
        
        Args:
            object_name: Name of the object (e.g., "base_scaled70")
            marker_id: ArUco marker ID to use as reference
            camera_distance: Camera distance for rendering
            
        Returns:
            Dictionary containing all pipeline outputs
        """
        print(f"\n{'#'*60}")
        print(f"CAD TO GRASP POINTS PIPELINE")
        print(f"{'#'*60}")
        print(f"  Object: {object_name}")
        print(f"  Reference Marker: {marker_id}")
        print(f"  Camera Distance: {camera_distance}")
        
        try:
            # Step 1: Render top-down view
            render_data = self.render_top_down_image(object_name, marker_id, camera_distance)
            
            # Step 2: Detect 2D grasp points
            points_2d, regions = self.detect_grasp_points_2d(
                render_data['output_image_path'],
                object_name,
                marker_id
            )
            
            # Step 3: Map to 3D marker-relative coordinates
            points_3d = self.map_to_3d_marker_relative(points_2d, render_data)
            
            # Step 4: Export grasp points
            output_json = self.export_grasp_points(points_3d, object_name, marker_id)
            
            # Summary
            print(f"\n{'='*60}")
            print("PIPELINE COMPLETE!")
            print(f"{'='*60}")
            print(f"\nGenerated Files:")
            print(f"  1. Rendered Image: {render_data['output_image_path']}")
            print(f"  2. 2D Grasp Points Visualization: {self.outputs_dir}/{object_name}_marker{marker_id}_grasp_points_2d.png")
            print(f"  3. 3D Grasp Points JSON: {output_json}")
            print(f"\nSummary:")
            print(f"  - Detected {len(points_2d)} 2D grasp regions")
            print(f"  - Mapped {len(points_3d)} 3D grasp points")
            print(f"  - All coordinates are relative to ArUco marker {marker_id}")
            
            # Print first few 3D points as example
            print(f"\nSample 3D Grasp Points (relative to marker {marker_id}):")
            for i, point in enumerate(points_3d[:min(5, len(points_3d))]):
                print(f"  Point {i+1}: x={point[0]:.4f}, y={point[1]:.4f}, z={point[2]:.4f}")
            if len(points_3d) > 5:
                print(f"  ... and {len(points_3d) - 5} more points")
            
            return {
                'render_data': render_data,
                'points_2d': points_2d,
                'points_3d': points_3d,
                'regions': regions,
                'output_json': output_json
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(
        description="CAD to Grasp Points Pipeline - Generate 3D grasp points from CAD models"
    )
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help='Object name (e.g., base_scaled70, fork_orange_scaled70)'
    )
    parser.add_argument(
        '--marker-id',
        type=int,
        required=True,
        help='ArUco marker ID to use as reference frame'
    )
    parser.add_argument(
        '--camera-distance',
        type=float,
        default=0.5,
        help='Camera distance from object (default: 0.5)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../../data',
        help='Path to data directory (default: ../../data)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = CADToGraspPipeline(data_dir=args.data_dir)
    result = pipeline.run(
        object_name=args.object,
        marker_id=args.marker_id,
        camera_distance=args.camera_distance
    )
    
    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()

