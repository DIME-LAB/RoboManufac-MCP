#!/usr/bin/env python3
"""
Example script showing how to plot wireframe data exported by export_wireframe.py

This demonstrates how to load and visualize the exported wireframe data in various formats.
"""

import json
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


def plot_wireframe_json(json_file: str):
    """Plot wireframe data from JSON export using Open3D."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    vertices = np.array(data['vertices'])
    edges = data['edges']
    
    print(f"Loaded wireframe: {len(vertices)} vertices, {len(edges)} edges")
    
    # Create LineSet for wireframe visualization
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edges)
    )
    
    # Set line color (bright yellow for visibility)
    colors = [[1.0, 1.0, 0.0] for _ in range(len(edges))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # Visualize with Open3D
    o3d.visualization.draw_geometries(
        [line_set, coord_frame],
        window_name="Wireframe Visualization (JSON)",
        width=1200, height=800
    )


def plot_wireframe_numpy(npy_file: str):
    """Plot wireframe data from NumPy export using Open3D."""
    data = np.load(npy_file, allow_pickle=True)
    
    vertices = data['vertices']
    edges = data['edges']
    
    print(f"Loaded wireframe: {len(vertices)} vertices, {len(edges)} edges")
    
    # Create LineSet for wireframe visualization
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edges)
    )
    
    # Set line color (bright yellow for visibility)
    colors = [[1.0, 1.0, 0.0] for _ in range(len(edges))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # Visualize with Open3D
    o3d.visualization.draw_geometries(
        [line_set, coord_frame],
        window_name="Wireframe Visualization (NumPy)",
        width=1200, height=800
    )


def plot_wireframe_csv(csv_file: str):
    """Plot wireframe data from CSV export using Open3D."""
    # Read CSV file
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    # Find vertices section
    vertices_start = None
    edges_start = None
    
    for i, line in enumerate(lines):
        if line.strip() == '# VERTICES':
            vertices_start = i + 2  # Skip header line
        elif line.strip() == '# EDGES':
            edges_start = i + 2  # Skip header line
            break
    
    if vertices_start is None or edges_start is None:
        print("ERROR: Could not find vertices or edges section in CSV")
        return
    
    # Parse vertices
    vertices = []
    for i in range(vertices_start, edges_start - 2):  # -2 for empty line and header
        if lines[i].strip() and not lines[i].startswith('#'):
            parts = lines[i].strip().split(',')
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    
    # Parse edges
    edges = []
    for i in range(edges_start, len(lines)):
        if lines[i].strip() and not lines[i].startswith('#'):
            parts = lines[i].strip().split(',')
            edges.append([int(parts[1]), int(parts[2])])
    
    print(f"Loaded wireframe: {len(vertices)} vertices, {len(edges)} edges")
    
    # Create LineSet for wireframe visualization
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edges)
    )
    
    # Set line color (bright yellow for visibility)
    colors = [[1.0, 1.0, 0.0] for _ in range(len(edges))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # Visualize with Open3D
    o3d.visualization.draw_geometries(
        [line_set, coord_frame],
        window_name="Wireframe Visualization (CSV)",
        width=1200, height=800
    )


def main():
    """Main function to handle command line arguments and plot wireframe data."""
    parser = argparse.ArgumentParser(
        description="Plot wireframe data exported by export_wireframe.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_wireframe_example.py wireframe.json
  python plot_wireframe_example.py wireframe.npy
  python plot_wireframe_example.py wireframe.csv
        """
    )
    
    parser.add_argument('wireframe_file', help='Exported wireframe file (JSON, NumPy, or CSV)')
    
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.wireframe_file)
    if not file_path.exists():
        print(f"ERROR: File {args.wireframe_file} does not exist")
        return 1
    
    # Determine file type and plot accordingly
    file_ext = file_path.suffix.lower()
    
    try:
        if file_ext == '.json':
            plot_wireframe_json(args.wireframe_file)
        elif file_ext == '.npy':
            plot_wireframe_numpy(args.wireframe_file)
        elif file_ext == '.csv':
            plot_wireframe_csv(args.wireframe_file)
        else:
            print(f"ERROR: Unsupported file format: {file_ext}")
            print("Supported formats: .json, .npy, .csv")
            return 1
        
        print("✅ Wireframe visualization completed successfully!")
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed to plot wireframe: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
