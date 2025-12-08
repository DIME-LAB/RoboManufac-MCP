#!/usr/bin/env python3
"""
Wireframe Export Script for 3D Meshes

This script exports wireframe information from 3D mesh files into vector relation formats
suitable for plotting in various 3D viewers.

Usage:
    python export_wireframe.py input_mesh.stl --format json --output wireframe_data.json
    python export_wireframe.py input_mesh.obj --format csv --output wireframe_data.csv
    python export_wireframe.py input_mesh.ply --format numpy --output wireframe_data.npy
"""

import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Tuple, Any
import csv


class WireframeExporter:
    """Export wireframe data from 3D meshes in various formats."""
    
    def __init__(self, unit_conversion=1.0):
        self.vertices = None
        self.edges = None
        self.mesh_info = None
        self.unit_conversion = unit_conversion  # Factor to convert to meters (e.g., 0.01 for cm->m)
    
    def load_mesh(self, file_path: str) -> bool:
        """Load a 3D mesh from file."""
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) == 0:
                print(f"ERROR: No vertices found in {file_path}")
                return False
            
            # Apply unit conversion to vertices (convert to meters)
            original_vertices = np.asarray(mesh.vertices)
            self.vertices = original_vertices * self.unit_conversion
            self.edges = self._extract_wireframe_edges(mesh)
            self.mesh_info = self._compute_mesh_info(mesh)
            
            print(f"✅ Loaded mesh: {len(self.vertices)} vertices, {len(self.edges)} edges")
            if self.unit_conversion != 1.0:
                print(f"📏 Applied unit conversion: {self.unit_conversion} (converting to meters)")
                # Show example of conversion
                if len(original_vertices) > 0:
                    orig_example = original_vertices[0]
                    conv_example = self.vertices[0]
                    print(f"   Example: {orig_example} → {conv_example}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load mesh {file_path}: {e}")
            return False
    
    def _extract_wireframe_edges(self, mesh: o3d.geometry.TriangleMesh) -> List[Tuple[int, int]]:
        """Extract wireframe edges from mesh triangles."""
        triangles = np.asarray(mesh.triangles)
        edges = []
        
        # Extract all edges from triangles
        for triangle in triangles:
            for i in range(3):
                v1, v2 = triangle[i], triangle[(i + 1) % 3]
                # Ensure consistent ordering (smaller index first)
                edge = tuple(sorted([v1, v2]))
                edges.append(edge)
        
        # Remove duplicates while preserving order
        unique_edges = []
        seen = set()
        for edge in edges:
            if edge not in seen:
                unique_edges.append(edge)
                seen.add(edge)
        
        return unique_edges
    
    def _compute_mesh_info(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """Compute mesh statistics and information."""
        vertices = np.asarray(mesh.vertices)
        
        return {
            'num_vertices': len(vertices),
            'num_edges': len(self.edges),
            'num_triangles': len(mesh.triangles),
            'bounding_box': {
                'min': vertices.min(axis=0).tolist(),
                'max': vertices.max(axis=0).tolist(),
                'center': vertices.mean(axis=0).tolist(),
                'size': (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
            },
            'has_normals': mesh.has_vertex_normals(),
            'has_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable()
        }
    
    def export_json(self, output_path: str) -> bool:
        """Export wireframe data to JSON format."""
        try:
            data = {
                'mesh_info': self.mesh_info,
                'vertices': self.vertices.tolist(),
                'edges': [[int(edge[0]), int(edge[1])] for edge in self.edges],  # Convert to Python int
                'format': 'vector_relation',
                'description': 'Wireframe data with vertices and edge connections'
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Exported JSON wireframe data to {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to export JSON: {e}")
            return False
    
    def export_csv(self, output_path: str) -> bool:
        """Export wireframe data to CSV format."""
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['# Wireframe Export - Vector Relation Format'])
                writer.writerow(['# Vertices:', len(self.vertices)])
                writer.writerow(['# Edges:', len(self.edges)])
                writer.writerow([])
                
                # Write vertices
                writer.writerow(['# VERTICES'])
                writer.writerow(['vertex_id', 'x', 'y', 'z'])
                for i, vertex in enumerate(self.vertices):
                    writer.writerow([i, vertex[0], vertex[1], vertex[2]])
                
                writer.writerow([])
                
                # Write edges
                writer.writerow(['# EDGES'])
                writer.writerow(['edge_id', 'vertex1_id', 'vertex2_id', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
                for i, (v1_idx, v2_idx) in enumerate(self.edges):
                    v1 = self.vertices[v1_idx]
                    v2 = self.vertices[v2_idx]
                    writer.writerow([i, v1_idx, v2_idx, v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]])
            
            print(f"✅ Exported CSV wireframe data to {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to export CSV: {e}")
            return False
    
    def export_numpy(self, output_path: str) -> bool:
        """Export wireframe data to NumPy format."""
        try:
            data = {
                'vertices': self.vertices,
                'edges': np.array(self.edges),
                'mesh_info': self.mesh_info
            }
            
            np.savez(output_path, **data)
            
            print(f"✅ Exported NumPy wireframe data to {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to export NumPy: {e}")
            return False
    
    def export_ply(self, output_path: str) -> bool:
        """Export wireframe as PLY file with line segments."""
        try:
            # Create line set
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(self.vertices)
            line_set.lines = o3d.utility.Vector2iVector(self.edges)
            
            # Set edge colors (white for visibility)
            colors = [[1.0, 1.0, 1.0] for _ in range(len(self.edges))]
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.io.write_line_set(output_path, line_set)
            
            print(f"✅ Exported PLY wireframe data to {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to export PLY: {e}")
            return False
    
    def export_obj(self, output_path: str) -> bool:
        """Export wireframe as OBJ file with line segments."""
        try:
            with open(output_path, 'w') as f:
                f.write("# Wireframe Export - Vector Relation Format\n")
                f.write(f"# Vertices: {len(self.vertices)}\n")
                f.write(f"# Edges: {len(self.edges)}\n\n")
                
                # Write vertices
                for vertex in self.vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                # Write edges as line segments
                for v1_idx, v2_idx in self.edges:
                    f.write(f"l {v1_idx + 1} {v2_idx + 1}\n")  # OBJ uses 1-based indexing
            
            print(f"✅ Exported OBJ wireframe data to {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to export OBJ: {e}")
            return False


def main():
    """Main function to handle command line arguments and export wireframe data."""
    parser = argparse.ArgumentParser(
        description="Export wireframe information from 3D meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_wireframe.py model.stl --format json --output wireframe.json
  python export_wireframe.py model.obj --format csv --output wireframe.csv
  python export_wireframe.py model.ply --format numpy --output wireframe.npy
  python export_wireframe.py model.stl --format ply --output wireframe.ply
  python export_wireframe.py model.obj --format obj --output wireframe.obj
  python export_wireframe.py model.stl --unit-conversion 0.01  # Convert cm to m
  python export_wireframe.py model.obj --unit-conversion 0.001 # Convert mm to m
        """
    )
    
    parser.add_argument('input_file', help='Input 3D mesh file (STL, OBJ, PLY, etc.)')
    parser.add_argument('--format', '-f', 
                       choices=['json', 'csv', 'numpy', 'ply', 'obj'],
                       default='json',
                       help='Export format (default: json)')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--info', '-i', action='store_true',
                       help='Show mesh information only')
    parser.add_argument('--unit-conversion', '-u', type=float, default=0.01,
                       help='Unit conversion factor to convert to meters (default: 0.01 for cm->m)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: Input file {args.input_file} does not exist")
        return 1
    
    # Create exporter and load mesh
    exporter = WireframeExporter(unit_conversion=args.unit_conversion)
    if not exporter.load_mesh(args.input_file):
        return 1
    
    # Show mesh info if requested
    if args.info:
        print("\n📊 Mesh Information:")
        for key, value in exporter.mesh_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        return 0
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_stem = input_path.stem
        output_path = f"{input_stem}_wireframe.{args.format}"
    
    # Export in requested format
    success = False
    if args.format == 'json':
        success = exporter.export_json(output_path)
    elif args.format == 'csv':
        success = exporter.export_csv(output_path)
    elif args.format == 'numpy':
        success = exporter.export_numpy(output_path)
    elif args.format == 'ply':
        success = exporter.export_ply(output_path)
    elif args.format == 'obj':
        success = exporter.export_obj(output_path)
    
    if success:
        print(f"\n🎯 Wireframe export completed successfully!")
        print(f"   Input: {args.input_file}")
        print(f"   Output: {output_path}")
        print(f"   Format: {args.format}")
        print(f"   Vertices: {len(exporter.vertices)}")
        print(f"   Edges: {len(exporter.edges)}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
