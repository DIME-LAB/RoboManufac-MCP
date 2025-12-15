# ArUco Grasp Annotator

Suite of ArUco-enabled tools for grasp annotation, object localization, assembly planning, and symmetry export for robotics workflows.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

## Features

- **3D CAD Support**: Load STL, OBJ, PLY, and OFF file formats
- **ArUco Marker Placement**: Place multiple markers on 3D objects with precise positioning
- **Grasp Pose Annotation**: Define 6DOF grasp poses with interactive 3D visualization
- **Real-time 3D Viewer**: Open3D-powered 3D visualization with mouse controls
- **JSON Export**: Export annotations for robotics pipelines
- **Coordinate Transformations**: Grasp poses relative to ArUco marker coordinate frames
- **Multi-marker Support**: Handle objects with multiple ArUco markers
- **Cross-platform**: Works on macOS, Linux, and Windows

## Quick Start

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

### Running the Applications

Use uv to launch any of the apps:

```bash
uv run aruco-annotator          # Annotate ArUco markers and grasps
uv run assembly-app             # Plan assemblies in the browser
uv run grasp-points-annotator   # Create grasp point annotations
uv run symmetry-exporter        # Visualize/export symmetry data
```

## Project Structure

```
aruco-grasp-annotator/
├── src/
│   ├── aruco_annotator/        # ArUco marker + grasp annotator
│   ├── aruco_localizer/        # Localization utilities
│   ├── assembly_app/           # 3D assembly web app
│   ├── grasp_points_annotator/ # Grasp points annotator
│   └── symmetry_exporter/      # Symmetry visualization/exporter
├── utils/                      # Shared Isaac Sim/automation scripts
├── data/                       # ArUco, grasp, symmetry, wireframe, model assets
├── pyproject.toml
└── README.md
```
