"""CAD file loader for various 3D formats."""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import open3d as o3d


class CADLoader:
    """Loader for various CAD file formats."""
    
    def __init__(self):
        self.supported_formats = {'.stl', '.obj', '.ply', '.off'}
        self.unit_conversion = 1.0  # Default: no conversion (assume meters)
        
    def load_file(self, file_path: Path, input_units: str = "auto") -> o3d.geometry.TriangleMesh:
        """Load a CAD file and return an Open3D mesh.
        
        Args:
            file_path: Path to the CAD file
            input_units: Input units ("mm", "cm", "m", or "auto" for detection)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        try:
            # Load mesh using Open3D
            mesh = o3d.io.read_triangle_mesh(str(file_path))
            
            if len(mesh.vertices) == 0:
                raise ValueError("Loaded mesh has no vertices")
            
            # Detect or set unit conversion
            if input_units == "auto":
                self.unit_conversion = self._detect_units(mesh)
            else:
                self.unit_conversion = self._get_conversion_factor(input_units)
            
            # Apply unit conversion if needed
            if self.unit_conversion != 1.0:
                mesh.scale(self.unit_conversion, center=mesh.get_center())
                
            # Process mesh
            mesh = self.process_mesh(mesh)
            
            return mesh
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh: {str(e)}")
            
    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Process and clean the loaded mesh."""
        # Remove duplicated vertices
        mesh.remove_duplicated_vertices()
        
        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        # Compute normals if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            
        # Center the mesh at origin (but preserve original scale)
        mesh = self.center_mesh(mesh)
        
        # Note: We no longer normalize scale to preserve original CAD dimensions
        # This is critical for robotics applications where accurate measurements matter
        
        return mesh
        
    def center_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Center the mesh at the origin."""
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        mesh.translate(-center)
        return mesh
        
    def normalize_scale(self, mesh: o3d.geometry.TriangleMesh, target_size: float = 1.0) -> o3d.geometry.TriangleMesh:
        """Normalize mesh scale so max dimension equals target_size."""
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_extent = np.max(extent)
        
        if max_extent > 0:
            scale_factor = target_size / max_extent
            mesh.scale(scale_factor, center=mesh.get_center())
            
        return mesh
        
    def get_mesh_info(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """Get information about the mesh."""
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        
        # Try to get volume, but handle non-watertight meshes gracefully
        try:
            volume = mesh.get_volume()
        except:
            volume = 0.0  # Volume cannot be computed for non-watertight meshes
        
        # Try to get surface area
        try:
            surface_area = mesh.get_surface_area()
        except:
            surface_area = 0.0
        
        return {
            'vertices': len(mesh.vertices),
            'triangles': len(mesh.triangles),
            'bbox_min': bbox.min_bound.tolist(),
            'bbox_max': bbox.max_bound.tolist(),
            'extent': extent.tolist(),
            'dimensions': {
                'length': float(extent[0]),  # X dimension
                'width': float(extent[1]),   # Y dimension  
                'height': float(extent[2])   # Z dimension
            },
            'max_dimension': float(np.max(extent)),
            'volume': volume,
            'surface_area': surface_area,
            'has_normals': mesh.has_vertex_normals(),
            'has_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable()
        }
        
    def _detect_units(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """Detect the most likely input units based on mesh dimensions."""
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dim = np.max(extent)
        
        # Heuristics for unit detection: (need to fix in future)
        # - If max dimension > 10, likely in mm (convert to m: /1000)
        # - If max dimension between 0.1 and 10, likely in cm (convert to m: /100)  
        # - If max dimension < 0.1, likely already in m (no conversion)
        
        if max_dim > 10:
            return 0.01  
        elif max_dim > 0.1:
            return 0.1   
        else:
            return 1.0   
    
    def _get_conversion_factor(self, input_units: str) -> float:
        """Get conversion factor from input units to meters."""
        conversion_factors = {
            "mm": 0.001,    # millimeters to meters
            "cm": 0.01,     # centimeters to meters
            "m": 1.0,       # meters (no conversion)
            "in": 0.0254,   # inches to meters
            "ft": 0.3048    # feet to meters
        }
        return conversion_factors.get(input_units.lower(), 1.0)
    
    def get_input_units(self) -> str:
        """Get the detected input units."""
        if self.unit_conversion == 0.001:
            return "mm"
        elif self.unit_conversion == 0.01:
            return "cm"
        elif self.unit_conversion == 1.0:
            return "m"
        else:
            return "unknown"
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.supported_formats
        
    def get_supported_formats(self) -> list:
        """Get list of supported file formats."""
        return list(self.supported_formats)
