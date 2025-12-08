# Grasp Candidates Visualizer

A web-based 3D visualization tool for viewing objects with grasp points overlayed. Built with FastAPI and Three.js for interactive 3D exploration.

## Features

- **Object Selection**: Choose from available objects with grasp point data
- **Grasp Point Visualization**: See all grasp points as colored spheres overlayed on the object
- **Grasp Point Selection**: Select individual grasp points to highlight and view details
- **Interactive 3D Viewer**: Full 3D interaction with rotate, zoom, and pan controls
- **Object Pose Support**: Automatically applies object poses from assembly JSON (similar to orientation_visualizer)
- **Real-time Updates**: Grasp point colors update when selected/deselected

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- FastAPI (>=0.104.0)
- uvicorn[standard] (>=0.24.0)

## Usage

### Prerequisites

**Important**: Start the grasp points publisher before running the application:

```bash
# Start the grasp points publisher (in a separate terminal)
# Make sure ROS2 is sourced first
source /opt/ros/humble/setup.bash
python3.10 src/grasp_candidates/grasp_points_publisher.py
```

The publisher will:
- Subscribe to `/objects_poses_sim` topic for object poses
- Publish grasp points to `/grasp_points` topic
- Publish grasp candidates to `/grasp_candidates` topic

### Generating Grasp Candidates

**Note**: Before generating grasp candidates, ensure you have already generated grasp points files (see `src/grasp_points/README.md` for details).

To generate grasp candidates (18 approach directions per grasp point):

```bash
cd src/grasp_candidates
python generate_grasp_candidates.py
```

This script will:
- Read all `*_grasp_points_all_markers.json` files from `data/grasp/`
- Generate 18 approach directions for each grasp point
- Save output to `data/grasp_candidates/{object_name}_grasp_candidates.json`

**Input**: `data/grasp/*_grasp_points_all_markers.json`  
**Output**: `data/grasp_candidates/*_grasp_candidates.json`

### Running the Application

```bash
python src/grasp_candidates/app.py
```

Or from the project root:

```bash
python -m grasp_candidates.app
```

Then open your browser to: `http://localhost:8002`

### Workflow

1. **Select Object**: Choose an object from the dropdown menu (e.g., fork_orange_scaled70, line_red_scaled70, etc.)
   - The object wireframe will be loaded and displayed
   - Grasp points will be automatically overlayed as colored spheres

2. **Select Grasp Point**: Choose a specific grasp point from the dropdown
   - The selected grasp point will change color (red) to distinguish it from others
   - Grasp point details will be displayed in the info panel
   - Camera will smoothly focus on the selected grasp point

3. **Interact with 3D Scene**:
   - **Rotate**: Right-click and drag
   - **Zoom**: Mouse wheel
   - **Pan**: Middle-click and drag
   - **Orbit**: Left-click and drag (OrbitControls)

## Data Sources

The application loads data from:

- **Grasp Points**: `data/grasp/*_grasp_points_all_markers.json`
  - Contains grasp point positions relative to CAD center
  - Includes approach vectors and point IDs

- **Wireframe Data**: `data/wireframe/*_wireframe.json`
  - Contains object wireframe vertices and edges for visualization

- **Object Poses**: `data/fmb_assembly.json`
  - Contains object positions and rotations (Euler angles)
  - Used to transform objects and grasp points to their assembly positions

## API Endpoints

- `GET /` - Main HTML interface
- `GET /api/objects` - List available objects
- `GET /api/grasp-data/{object_name}` - Get grasp points data for an object
- `GET /api/object-pose/{object_name}` - Get object pose from assembly JSON
- `GET /api/wireframe/{object_name}` - Get wireframe data for visualization

## Visualization Details

### Grasp Points

- **Default Color**: Green (`0x00ff00`) for unselected points
- **Selected Color**: Red (`0xff0000`) for the selected point
- **Size**: 3mm radius spheres (0.003m)
- **Labels**: Each grasp point shows its ID as a text label

### Object Wireframe

- **Color**: Gray (`0x888888`)
- **Rendering**: Line segments connecting wireframe vertices

### Coordinate System

- Uses Z-up coordinate system (matching assembly JSON)
- Grasp points are positioned relative to CAD center
- Object poses are applied as transformations

## Future Extensibility

The web UI approach allows for easy addition of:

- **Gripper Arm Visualizations**: Can add "tiny arms" (gripper visualizations) annotated on top of objects
- **Interactive Editing**: Edit grasp point positions interactively
- **Real-time ROS Updates**: Add WebSocket support for live pose updates
- **Multiple Object Views**: Compare multiple objects side-by-side
- **Grasp Quality Metrics**: Display quality scores or metrics for each grasp point

## Troubleshooting

### No Objects Appearing

1. Check that grasp JSON files exist in `data/grasp/`
2. Verify file naming: `{object_name}_grasp_points_all_markers.json`
3. Check browser console for errors

### Object Not Loading

1. Verify wireframe file exists: `data/wireframe/{object_name}_wireframe.json`
2. Check that object name matches between grasp and wireframe files
3. Verify assembly JSON contains the object pose

### Grasp Points Not Visible

1. Check that grasp points data is valid JSON
2. Verify grasp point positions are in meters (not centimeters)
3. Check camera position - try resetting view or zooming out

## Architecture

- **Backend**: FastAPI application serving HTML and JSON APIs
- **Frontend**: Three.js for 3D rendering with embedded HTML/JavaScript
- **Data Loading**: Loads JSON files from data directory
- **3D Rendering**: Uses Three.js LineSegments for wireframes and Spheres for grasp points

## Notes

- The application runs on port 8002 by default (to avoid conflicts with assembly_app on 8001)
- Object poses are optional - if not found in assembly JSON, objects are displayed at origin
- Grasp points are always displayed relative to CAD center coordinate frame
- The UI is responsive and works best with modern browsers supporting WebGL

