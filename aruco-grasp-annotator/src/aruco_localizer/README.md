# ArUco Localizer

A comprehensive OpenCV-based ArUco marker detection and 3D mesh overlay system for robotics applications.

## Features

### 🎥 Camera Management
- **Automatic Camera Detection**: Scans for all available cameras (0-20)
- **Interactive Camera Selection**: Preview each camera, skip with any key, select with ESC
- **Max Localizer Style**: Exact same interface as the original max localizer
- **Real-time Streaming**: High-quality camera feed with customizable settings

### 🎯 ArUco Marker Detection
- **Multi-Marker Support**: Detects all ArUco markers from JSON annotations
- **6D Pose Estimation**: Accurate position and orientation calculation
- **Dynamic Marker Sizing**: Automatic marker size calculation including border width
- **Real-time Display**: Live pose information with distance and rotation data

### 🎨 3D Mesh Overlay
- **Wireframe Rendering**: Real-time 3D mesh overlay on detected markers
- **Coordinate System Correction**: Automatic transformation from 3D graphics to OpenCV
- **Multiple Marker Support**: Works with all marker IDs simultaneously
- **Toggle Functionality**: Enable/disable mesh overlay with 's' key

### 📊 Data Integration
- **JSON Annotation Support**: Reads ArUco marker data from annotation files
- **Wireframe Mesh Loading**: Loads 3D mesh data from JSON wireframe files
- **Dynamic Configuration**: All settings configurable via command line arguments

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Camera Streaming
```bash
# Interactive camera selection (recommended)
python camera_streamer.py

# Direct camera selection
python camera_streamer.py --camera-id 8
```

### ArUco Pose Detection
```bash
# Detect specific marker (ID 5)
python aruco_pose_detector.py

# Custom marker ID
python aruco_pose_detector.py --marker-id 3
```

### 3D Mesh Overlay
```bash
# Full mesh overlay with all markers
python aruco_mesh_overlay.py

# Custom camera and marker
python aruco_mesh_overlay.py --camera-id 8 --marker-id 5
```

## Controls

### Camera Selection
- **Any key** (except ESC) → Skip to next camera
- **ESC** → Select current camera and start streaming

### During Streaming/Detection
- **'q'** → Quit application
- **'s'** → Toggle mesh overlay (mesh overlay mode only)

## Files

- `camera_streamer.py` - Interactive camera detection and streaming
- `aruco_pose_detector.py` - 6D pose estimation for ArUco markers
- `aruco_mesh_overlay.py` - 3D wireframe mesh overlay system
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Data Files

- `fork_orange_scaled70_ v2_aruco_annotations.json` - ArUco marker annotations
- `fork_orange_scaled70_ v2_wireframe.json` - 3D wireframe mesh data

## Technical Details

### Coordinate System Transformation
The system automatically handles coordinate system differences between:
- **3D Graphics Convention**: X-right, Y-up, Z-forward (used by ArUco annotator)
- **OpenCV Convention**: X-left, Y-up, Z-backward (used by camera detection)

### Marker Detection
- **Dictionary**: DICT_4X4_50 (configurable)
- **Size Calculation**: `marker_size = base_size * (1 + border_percentage)`
- **Pose Estimation**: Uses OpenCV's `solvePnP` for accurate 6D pose

### Mesh Overlay
- **Real-time Rendering**: 3D wireframe projected onto 2D camera feed
- **Coordinate Transformation**: Automatic conversion between coordinate systems
- **Performance Optimized**: Efficient vertex projection and edge drawing

## Command Line Arguments

### camera_streamer.py
```bash
--camera-id ID     # Start with specific camera ID
--width WIDTH      # Camera width (default: 1280)
--height HEIGHT    # Camera height (default: 720)
```

### aruco_pose_detector.py
```bash
--camera-id ID     # Camera device ID (default: 8)
--marker-id ID     # ArUco marker ID to detect (default: 5)
--marker-size SIZE # Marker size in meters (default: 0.05)
```

### aruco_mesh_overlay.py
```bash
--camera-id ID           # Camera device ID (default: 8)
--marker-id ID           # ArUco marker ID (default: 5)
--marker-size SIZE       # Marker size in meters (default: 0.05)
--mesh-json PATH         # Path to wireframe JSON file
```

## Perfect for Robotics Applications

This system is designed for robotics applications where you need:
- **Real-time ArUco marker detection**
- **Accurate 6D pose estimation**
- **3D object visualization**
- **Multiple marker support**
- **Easy camera management**

## Troubleshooting

### Camera Not Detected
- Run `python camera_streamer.py` to see all available cameras
- Check camera permissions and USB connections
- Try different camera IDs (0-20)

### Mesh Not Appearing
- Ensure ArUco marker is detected (check console output)
- Verify JSON files are in the correct location
- Press 's' to toggle mesh overlay
- Check marker size and camera calibration parameters

### Coordinate System Issues
- The system automatically handles coordinate transformations
- If mesh appears in wrong position, check marker annotations in JSON file
- Verify marker size and border width settings

## Dependencies

- `opencv-python` - Computer vision and ArUco detection
- `numpy` - Numerical operations and array handling
- `scipy` - Rotation matrix conversions
- `json` - Data file parsing (built-in)

## License

This project is part of the ArUco Grasp Annotator system.