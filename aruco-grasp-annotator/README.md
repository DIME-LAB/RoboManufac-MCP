# ArUco Grasp Annotator

A professional 3D CAD annotation tool for defining grasp poses relative to ArUco markers on objects. Perfect for robotics applications requiring precise object manipulation.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

## 🎯 Features

- **3D CAD Support**: Load STL, OBJ, PLY, and OFF file formats
- **ArUco Marker Placement**: Place multiple markers on 3D objects with precise positioning
- **Grasp Pose Annotation**: Define 6DOF grasp poses with interactive 3D visualization
- **Real-time 3D Viewer**: Open3D-powered 3D visualization with mouse controls
- **JSON Export**: Export annotations for robotics pipelines
- **Coordinate Transformations**: Grasp poses relative to ArUco marker coordinate frames
- **Multi-marker Support**: Handle objects with multiple ArUco markers
- **Cross-platform**: Works on macOS, Linux, and Windows

## 🚀 Quick Start

### Prerequisites

1. **Install uv** (fast Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/aruco-grasp-annotator.git
cd aruco-grasp-annotator
```

3. **Setup environment**:
```bash
uv sync
```

This automatically creates a virtual environment and installs all dependencies.

### Running the Application

```bash
uv run aruco-annotator
```

## 🎮 Usage

1. **Load CAD Model**: Use File → Open CAD File to load your STL/OBJ/PLY file
2. **Launch 3D Viewer**: Click "Launch 3D Viewer" to open the 3D visualization
3. **Place ArUco Markers**: Use the left panel to add markers at specific positions
4. **Define Grasp Poses**: Use the right panel to create grasp annotations relative to markers
5. **Export Annotations**: Save your work as JSON for use in robotics pipelines

### 3D Viewer Controls

- **Left mouse drag**: Rotate view
- **Right mouse drag**: Pan view  
- **Mouse scroll**: Zoom in/out
- **Reset View**: Return to default camera position
- **Fit to View**: Center the model in view

## 📁 Project Structure

```
aruco-grasp-annotator/
├── src/aruco_annotator/
│   ├── gui/              # PyQt6 user interface
│   │   ├── main_window.py
│   │   ├── working_viewer_3d.py
│   │   ├── marker_panel.py
│   │   └── grasp_panel.py
│   ├── core/             # Core functionality
│   │   ├── cad_loader.py
│   │   └── annotation_manager.py
│   ├── utils/            # Utility functions
│   └── data/             # Data models
├── pyproject.toml        # Project configuration
├── README.md
└── sample_*.stl          # Example CAD files
```

## 📊 Output Format

The tool exports annotations in JSON format suitable for robotics pipelines:

```json
{
  "object_file": "example.stl",
  "created_at": "2024-01-01T12:00:00",
  "markers": [
    {
      "id": 0,
      "position": [0.1, 0.2, 0.3],
      "orientation": [1.0, 0.0, 0.0, 0.0],
      "size": 0.05,
      "grasp_poses": [
        {
          "name": "top_grasp",
          "position": [0.0, 0.0, 0.05],
          "orientation": [0.707, 0.0, 0.0, 0.707],
          "approach_vector": [0.0, 0.0, 1.0]
        }
      ]
    }
  ]
}
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

### Dependencies

- **Open3D**: 3D visualization and CAD file processing
- **PyQt6**: Modern GUI framework
- **OpenCV**: ArUco marker detection
- **NumPy/SciPy**: Mathematical operations
- **Trimesh**: Additional 3D mesh processing

## 🎯 Use Cases

- **Robotics Research**: Annotate grasp poses for manipulation tasks
- **Computer Vision**: Prepare training data for object detection
- **Industrial Automation**: Define pick-and-place operations
- **Simulation**: Create realistic grasping scenarios
- **Education**: Teach robotics and computer vision concepts

## 📋 Requirements

- **Python**: 3.9 or higher
- **Operating System**: macOS 10.15+, Linux, or Windows 10+
- **Graphics**: OpenGL support for 3D visualization
- **Memory**: 4GB RAM minimum (8GB recommended for large models)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Open3D](https://www.open3d.org/) for 3D visualization
- GUI powered by [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
- Package management with [uv](https://github.com/astral-sh/uv)

---

**Made with ❤️ for the robotics community**
