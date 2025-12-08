# 3D Assembly App

An interactive web-based 3D assembly application for loading and assembling ArUco-annotated components.

## Features

- **Load Components**: Load all available components (base, fork_orange, fork_yellow, line_brown, line_red) with their wireframe and ArUco marker data
- **Interactive 3D Scene**: Drag and drop components, rotate objects, and manipulate the assembly
- **Real-time Manipulation**: Click and drag to move objects, right-click and drag to rotate
- **Grid Snapping**: Hold Shift while moving for precise grid-based positioning
- **ArUco Marker Visualization**: See ArUco markers positioned on components
- **Export Assembly**: Save your assembly configuration as JSON
- **Modern UI**: Clean, responsive interface with real-time status updates

## Quick Start

1. **Run the Application**:
   ```bash
   uv run python src/assembly_app/app.py
   ```

2. **Open in Browser**:
   Navigate to `http://localhost:8001`

## Usage

1. **Load Components**: Click "Load All Components" to load all available parts
2. **Add to Scene**: Click on component names in the sidebar to add them to the 3D scene
3. **Manipulate Objects**:
   - **Move**: Left-click and drag objects to move them
   - **Rotate**: Right-click and drag to rotate objects
   - **Zoom**: Mouse wheel to zoom in/out
   - **Pan**: Middle-click and drag to pan the camera
4. **Grid Snapping**: Hold Shift while moving for precise positioning
5. **Export**: Click "Export Assembly" to save your configuration

## Controls

- **Left Click + Drag**: Move selected object
- **Right Click + Drag**: Rotate selected object  
- **Mouse Wheel**: Zoom camera
- **Middle Click + Drag**: Pan camera
- **Shift + Move**: Grid snapping
- **Toggle Grid**: Show/hide reference grid
- **Reset Camera**: Return to default view

## Data Structure

The app loads data from:
- `data/wireframe/` - Wireframe JSON files with vertices and edges
- `data/aruco/` - ArUco marker JSON files with positions and orientations

## API Endpoints

- `GET /` - Main application interface
- `GET /api/components` - Get all available components
- `GET /api/components/{name}` - Get specific component data
- `POST /api/assembly` - Save assembly configuration
- `GET /api/assemblies` - Get saved assemblies

## Architecture

- **Backend**: FastAPI server serving component data and handling assembly operations
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
