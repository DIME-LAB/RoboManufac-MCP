# Wireframe Exporter

A standalone application for exporting 3D mesh wireframe data to various formats and visualizing the exported data.

## Overview

The Wireframe Exporter is a specialized tool that extracts wireframe (skeleton) information from 3D mesh files and exports it in multiple formats suitable for further processing, analysis, or visualization.

## Features

- **Multiple Export Formats**: JSON, CSV, NumPy, PLY, and OBJ
- **3D Visualization**: Interactive wireframe visualization using Open3D
- **Scale Preservation**: Maintains original mesh dimensions and proportions
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Standalone Application**: Independent of the ArUco Grasp Annotator

## Installation

The application uses the same dependencies as the main ArUco Grasp Annotator project:

```bash
# From the main project directory
uv sync
```

## Usage

### GUI Application (Recommended)

The easiest way to use the Wireframe Exporter is through the graphical interface:

```bash
# Launch the GUI application
uv run python src/wireframe_exporter/main.py gui
```

The GUI provides:
- **Model Loading**: Browse and load 3D model files
- **Export Options**: Choose format and output location
- **Auto-export**: Automatically save to examples folder
- **3D Viewer**: View exported wireframes in a separate tab
- **Progress Tracking**: Real-time export progress
- **Export Log**: Detailed logging of operations

### Command Line Interface

The application also provides a unified CLI with three main commands:

#### Launch GUI Application

```bash
# Launch the GUI application
python src/wireframe_exporter/main.py gui
```

#### Export Wireframe Data

```bash
# Basic export to JSON
python src/wireframe_exporter/main.py export model.obj

# Export to specific format
python src/wireframe_exporter/main.py export model.obj --format csv --output model_wireframe.csv

# Show mesh information only
python src/wireframe_exporter/main.py export model.obj --info
```

#### Plot Wireframe Data

```bash
# Plot exported wireframe data
python src/wireframe_exporter/main.py plot model_wireframe.json
```

### Direct Script Usage

You can also use the individual scripts directly:

```bash
# Export wireframe
python src/wireframe_exporter/scripts/export_wireframe.py model.obj --format json

# Plot wireframe
python src/wireframe_exporter/scripts/plot_wireframe_example.py model_wireframe.json
```

## Supported Formats

### Export Formats

- **JSON**: Human-readable format with metadata
- **CSV**: Spreadsheet-compatible format
- **NumPy**: Binary format for Python processing
- **PLY**: Standard 3D format for line sets
- **OBJ**: Wavefront OBJ format for lines

### Input Formats

- **STL**: Stereolithography files
- **OBJ**: Wavefront OBJ files
- **PLY**: Polygon File Format
- **Any format supported by Open3D**

## File Structure

```
src/wireframe_exporter/
├── __init__.py                 # Package initialization
├── main.py                     # Main entry point
├── pyproject.toml             # Project configuration
├── README.md                  # This file
├── WIREFRAME_EXPORT_README.md # Detailed documentation
└── scripts/
    ├── __init__.py
    ├── export_wireframe.py    # Export functionality
    └── plot_wireframe_example.py # Visualization functionality
```

## Examples

### Export a Fork Model

```bash
# Export fork wireframe to JSON
python src/wireframe_exporter/main.py export "/path/to/fork.obj" --format json --output fork_wireframe.json

# Plot the exported wireframe
python src/wireframe_exporter/main.py plot fork_wireframe.json
```

### Batch Processing

```bash
# Export multiple models
for model in *.obj; do
    python src/wireframe_exporter/main.py export "$model" --format json
done
```

## Integration

This application is designed to work alongside the ArUco Grasp Annotator but can be used independently for any 3D mesh wireframe extraction needs.

## Dependencies

- Open3D >= 0.17.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- scikit-learn >= 1.3.0
- PyQt6 >= 6.4.0 (for GUI)

## License

Same license as the ArUco Grasp Annotator project.
