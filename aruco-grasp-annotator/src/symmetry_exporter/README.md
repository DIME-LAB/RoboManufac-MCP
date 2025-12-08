# Symmetry Exporter

An interactive web-based 3D visualization application for displaying objects with grasp points in a 3D environment.

## Features

- **Load Components**: Load all available components (base, fork_orange, fork_yellow, line_brown, line_red) with their wireframe and ArUco marker data
- **Interactive 3D Scene**: View and manipulate objects in a 3D environment
- **Grasp Points Visualization**: Load and visualize grasp points along with objects
- **Real-time Manipulation**: Click and select objects, use precision controls for positioning
- **ArUco Marker Visualization**: See ArUco markers positioned on components
- **Modern UI**: Clean, responsive interface with real-time status updates

## Quick Start

1. **Run the Application**:
   ```bash
   uv run python src/symmetry_exporter/app.py
   ```
   Or use the runner script:
   ```bash
   uv run python src/symmetry_exporter/run.py
   ```

2. **Open in Browser**:
   Navigate to `http://localhost:8002`

## Usage

1. **Load Components**: Click "Load All Components" to load all available parts
2. **Add to Scene**: Click on component names in the sidebar to add them to the 3D scene
3. **Load Grasp Points**: Use the "Load Grasp Points" button to load a grasp points JSON file
4. **Manipulate Objects**:
   - **Select**: Click objects in the scene or object list
   - **Move**: Use precision controls or arrow keys
   - **Rotate**: Use rotation controls or Q/E keys
   - **Zoom**: Mouse wheel to zoom in/out
   - **Pan**: Middle-click and drag to pan the camera
   - **Orbit**: Right-click and drag to orbit around the scene

## Controls

- **Left Click**: Select object
- **Arrow Keys**: Move selected object (X/Y axes)
- **Page Up/Down**: Move selected object (Z axis)
- **Q/E**: Rotate selected object around Y axis
- **Mouse Wheel**: Zoom camera
- **Middle Click + Drag**: Pan camera
- **Right Click + Drag**: Orbit camera
- **Delete**: Remove selected object

## Data Structure

The app loads data from:
- `data/wireframe/` - Wireframe JSON files with vertices and edges
- `data/aruco/` - ArUco marker JSON files with positions and orientations
- Grasp points JSON files (loaded via file input)

## API Endpoints

- `GET /` - Main application interface
- `GET /api/components` - Get all available components
- `GET /api/components/{name}` - Get specific component data

## Architecture

- **Backend**: FastAPI server serving component data
- **Frontend**: Three.js-based 3D viewer with interactive manipulation
- **Data**: JSON-based component definitions with wireframe and ArUco data

## Components Available

- `base_scaled70` - Base component (Green)
- `fork_orange_scaled70` - Orange fork (Orange)  
- `fork_yellow_scaled70` - Yellow fork (Yellow)
- `line_brown_scaled70` - Brown line component (Brown)
- `line_red_scaled70` - Red line component (Red)

Each component includes:
- Wireframe geometry (vertices and edges)
- ArUco marker positions and orientations
- Component metadata and display information

