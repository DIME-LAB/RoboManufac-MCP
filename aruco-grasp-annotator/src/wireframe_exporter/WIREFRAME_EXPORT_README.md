# Wireframe Export Tools

This directory contains tools for exporting wireframe information from 3D meshes into vector relation formats suitable for plotting in various 3D viewers.

## Files

- `export_wireframe.py` - Main script to export wireframe data from 3D meshes
- `plot_wireframe_example.py` - Example script showing how to plot exported wireframe data
- `WIREFRAME_EXPORT_README.md` - This documentation file

## Quick Start

### 1. Export Wireframe Data

```bash
# Export to JSON format (default)
uv run python export_wireframe.py model.stl

# Export to CSV format
uv run python export_wireframe.py model.obj --format csv --output wireframe.csv

# Export to NumPy format
uv run python export_wireframe.py model.ply --format numpy --output wireframe.npy

# Export to PLY format (for Open3D viewers)
uv run python export_wireframe.py model.stl --format ply --output wireframe.ply

# Export to OBJ format (for Blender, etc.)
uv run python export_wireframe.py model.obj --format obj --output wireframe.obj
```

### 2. Plot Wireframe Data

```bash
# Plot JSON wireframe data
uv run python plot_wireframe_example.py wireframe.json

# Plot NumPy wireframe data
uv run python plot_wireframe_example.py wireframe.npy

# Plot CSV wireframe data
uv run python plot_wireframe_example.py wireframe.csv
```

## Export Formats

### JSON Format
- **Best for**: General use, web applications, data exchange
- **Structure**: 
  ```json
  {
    "mesh_info": {...},
    "vertices": [[x, y, z], ...],
    "edges": [[v1_idx, v2_idx], ...],
    "format": "vector_relation"
  }
  ```

### CSV Format
- **Best for**: Excel, data analysis, human-readable
- **Structure**: 
  - Vertices section with vertex_id, x, y, z
  - Edges section with edge_id, vertex1_id, vertex2_id, coordinates

### NumPy Format
- **Best for**: Python applications, machine learning
- **Structure**: NPZ file with 'vertices', 'edges', 'mesh_info' arrays

### PLY Format
- **Best for**: Open3D, MeshLab, other 3D viewers
- **Structure**: Standard PLY line set format

### OBJ Format
- **Best for**: Blender, Maya, other 3D software
- **Structure**: Standard OBJ format with line segments

## Vector Relation Format

The exported data follows a **vector relation format** where:

- **Vertices**: Array of 3D coordinates `[x, y, z]`
- **Edges**: Array of vertex index pairs `[v1_idx, v2_idx]` that define connections
- **Relations**: Each edge connects two vertices by their indices

This format is ideal for:
- Overlaying ArUco marker information
- Plotting in any 3D viewer
- Data analysis and processing
- Machine learning applications

## Usage Examples

### Export and Plot Workflow

```bash
# 1. Export wireframe from your 3D model
uv run python export_wireframe.py my_model.stl --format json --output my_wireframe.json

# 2. Plot the wireframe data
uv run python plot_wireframe_example.py my_wireframe.json

# 3. Use the data in your own applications
python -c "
import json
with open('my_wireframe.json', 'r') as f:
    data = json.load(f)
vertices = data['vertices']
edges = data['edges']
print(f'Loaded {len(vertices)} vertices and {len(edges)} edges')
"
```

### Integration with ArUco Markers

The vector relation format is perfect for overlaying ArUco marker information:

```python
import json
import numpy as np

# Load wireframe data
with open('wireframe.json', 'r') as f:
    wireframe = json.load(f)

vertices = np.array(wireframe['vertices'])
edges = wireframe['edges']

# Add ArUco marker at specific vertex
marker_position = vertices[42]  # Example: marker at vertex 42
marker_orientation = [0, 0, 0]  # Example: no rotation

# Create marker overlay data
marker_data = {
    'position': marker_position.tolist(),
    'orientation': marker_orientation,
    'size': 0.05,
    'vertex_id': 42
}

# Save combined data
combined_data = {
    'wireframe': wireframe,
    'markers': [marker_data]
}

with open('wireframe_with_markers.json', 'w') as f:
    json.dump(combined_data, f, indent=2)
```

## Command Line Options

### export_wireframe.py

```
positional arguments:
  input_file            Input 3D mesh file (STL, OBJ, PLY, etc.)

options:
  -h, --help            show this help message and exit
  --format {json,csv,numpy,ply,obj}, -f
                        Export format (default: json)
  --output OUTPUT, -o   Output file path (default: auto-generated)
  --info, -i            Show mesh information only
```

### plot_wireframe_example.py

```
positional arguments:
  wireframe_file        Exported wireframe file (JSON, NumPy, or CSV)
```

## Requirements

- Python 3.9+
- Open3D
- NumPy
- Matplotlib (for plotting example)
- Pandas (for CSV plotting)

All dependencies are included in the project's `pyproject.toml`.

## Notes

- The wireframe export extracts all edges from the mesh triangles
- Duplicate edges are automatically removed
- The coordinate system is preserved from the original mesh
- All formats include mesh metadata (bounding box, statistics, etc.)
- The vector relation format is optimized for 3D visualization and analysis
