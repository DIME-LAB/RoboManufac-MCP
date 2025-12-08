#!/usr/bin/env python3
"""
Main window for the Wireframe Exporter GUI application.
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QTextEdit, QComboBox, QGroupBox, QGridLayout, QLineEdit,
    QCheckBox, QSplitter, QFrame, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon

import open3d as o3d
import numpy as np
import cv2

# Add the scripts directory to the path
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Add the aruco_annotator directory to the path
aruco_annotator_dir = current_dir.parent.parent / "aruco_annotator"
sys.path.insert(0, str(aruco_annotator_dir))

from export_wireframe import WireframeExporter
from plot_wireframe_example import plot_wireframe_json
from utils.aruco_utils import ArUcoGenerator, ArUcoMarkerInfo


class BatchExportThread(QThread):
    """Thread for batch wireframe export operations."""
    
    progress = pyqtSignal(int)
    file_progress = pyqtSignal(str, int, int)  # filename, current, total
    finished = pyqtSignal(str, bool)  # message, success
    error = pyqtSignal(str)
    
    def __init__(self, source_folder: str, output_folder: str, output_format: str = "json"):
        super().__init__()
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.output_format = output_format
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Get all OBJ files from source folder
            source_path = Path(self.source_folder)
            obj_files = list(source_path.glob("*.obj"))
            
            if not obj_files:
                self.error.emit("No OBJ files found in source folder")
                return
            
            # Create output folder if it doesn't exist
            output_path = Path(self.output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.progress.emit(20)
            
            # Process each file
            success_count = 0
            total_files = len(obj_files)
            
            for i, obj_file in enumerate(obj_files):
                try:
                    # Create output filename: {model_name}_wireframe.{format}
                    output_filename = f"{obj_file.stem}_wireframe.{self.output_format}"
                    output_file = output_path / output_filename
                    
                    self.file_progress.emit(obj_file.name, i + 1, total_files)
                    
                    # Create exporter with unit conversion (cm to m)
                    exporter = WireframeExporter(unit_conversion=0.01)
                    
                    # Load mesh
                    exporter.load_mesh(str(obj_file))
                    
                    # Export wireframe based on format
                    if self.output_format == 'json':
                        success = exporter.export_json(str(output_file))
                    elif self.output_format == 'csv':
                        success = exporter.export_csv(str(output_file))
                    elif self.output_format == 'numpy':
                        success = exporter.export_numpy(str(output_file))
                    elif self.output_format == 'ply':
                        success = exporter.export_ply(str(output_file))
                    elif self.output_format == 'obj':
                        success = exporter.export_obj(str(output_file))
                    else:
                        raise ValueError(f"Unsupported format: {self.output_format}")
                    
                    if success:
                        success_count += 1
                    
                    # Update progress
                    progress_value = 20 + int((i + 1) / total_files * 70)
                    self.progress.emit(progress_value)
                    
                except Exception as e:
                    print(f"Error processing {obj_file.name}: {str(e)}")
                    continue
            
            self.progress.emit(100)
            
            if success_count == total_files:
                self.finished.emit(f"Successfully exported {success_count}/{total_files} wireframe files to {self.output_folder}", True)
            else:
                self.finished.emit(f"Exported {success_count}/{total_files} wireframe files to {self.output_folder} (some files failed)", True)
                
        except Exception as e:
            self.error.emit(f"Batch export error: {str(e)}")


class WireframeExportThread(QThread):
    """Thread for wireframe export operations."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, bool)  # message, success
    error = pyqtSignal(str)
    
    def __init__(self, input_file: str, output_format: str, output_path: str, auto_export: bool = False):
        super().__init__()
        self.input_file = input_file
        self.output_format = output_format
        self.output_path = output_path
        self.auto_export = auto_export
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Create exporter with unit conversion (cm to m)
            exporter = WireframeExporter(unit_conversion=0.01)
            
            self.progress.emit(30)
            
            # Load mesh
            exporter.load_mesh(self.input_file)
            
            self.progress.emit(50)
            
            # Export wireframe based on format
            if self.output_format == 'json':
                success = exporter.export_json(self.output_path)
            elif self.output_format == 'csv':
                success = exporter.export_csv(self.output_path)
            elif self.output_format == 'numpy':
                success = exporter.export_numpy(self.output_path)
            elif self.output_format == 'ply':
                success = exporter.export_ply(self.output_path)
            elif self.output_format == 'obj':
                success = exporter.export_obj(self.output_path)
            else:
                raise ValueError(f"Unsupported format: {self.output_format}")
            
            self.progress.emit(80)
            
            if success:
                if self.auto_export:
                    # Also export to examples folder
                    examples_dir = Path(__file__).parent.parent / "examples"
                    examples_dir.mkdir(exist_ok=True)
                    examples_path = examples_dir / f"{Path(self.input_file).stem}_wireframe.{self.output_format}"
                    
                    # Export to examples folder using the same format
                    if self.output_format == 'json':
                        exporter.export_json(str(examples_path))
                    elif self.output_format == 'csv':
                        exporter.export_csv(str(examples_path))
                    elif self.output_format == 'numpy':
                        exporter.export_numpy(str(examples_path))
                    elif self.output_format == 'ply':
                        exporter.export_ply(str(examples_path))
                    elif self.output_format == 'obj':
                        exporter.export_obj(str(examples_path))
                
                self.progress.emit(100)
                self.finished.emit(f"Wireframe exported successfully to {self.output_path}", True)
            else:
                self.error.emit("Failed to export wireframe")
                
        except Exception as e:
            self.error.emit(f"Export error: {str(e)}")


class WireframeViewerWidget(QWidget):
    """Widget for displaying wireframe visualization."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_wireframe_file = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = QGroupBox("Wireframe Viewer Controls")
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Wireframe File")
        self.load_btn.clicked.connect(self.load_wireframe_file)
        
        self.view_btn = QPushButton("View Wireframe")
        self.view_btn.clicked.connect(self.view_wireframe)
        self.view_btn.setEnabled(False)
        
        self.export_viewer_btn = QPushButton("Export Current View")
        self.export_viewer_btn.clicked.connect(self.export_current_view)
        self.export_viewer_btn.setEnabled(False)
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.view_btn)
        control_layout.addWidget(self.export_viewer_btn)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)
        
        # Status display
        self.status_label = QLabel("No wireframe file loaded")
        layout.addWidget(self.status_label)
        
        # Info display
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        self.setLayout(layout)
    
    def load_wireframe_file(self):
        """Load a wireframe file for viewing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Wireframe File",
            str(Path.home()),
            "Wireframe Files (*.json *.csv *.npy);;JSON Files (*.json);;CSV Files (*.csv);;NumPy Files (*.npy)"
        )
        
        if file_path:
            self.current_wireframe_file = file_path
            self.view_btn.setEnabled(True)
            self.export_viewer_btn.setEnabled(True)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            
            # Display file info
            self.display_file_info(file_path)
    
    def display_file_info(self, file_path: str):
        """Display information about the loaded wireframe file."""
        try:
            if file_path.endswith('.json'):
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                info = f"""File: {Path(file_path).name}
Format: JSON
Vertices: {data.get('mesh_info', {}).get('num_vertices', 'Unknown')}
Edges: {data.get('mesh_info', {}).get('num_edges', 'Unknown')}
Bounding Box: {data.get('mesh_info', {}).get('bounding_box', {})}
"""
            else:
                info = f"File: {Path(file_path).name}\nFormat: {Path(file_path).suffix.upper()}\n(Detailed info not available for this format)"
            
            self.info_text.setText(info)
            
        except Exception as e:
            self.info_text.setText(f"Error reading file info: {str(e)}")
    
    def view_wireframe(self):
        """Open wireframe visualization."""
        if self.current_wireframe_file:
            try:
                if self.current_wireframe_file.endswith('.json'):
                    plot_wireframe_json(self.current_wireframe_file)
                else:
                    QMessageBox.warning(self, "Warning", "Only JSON wireframe files are supported for viewing in this version.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to view wireframe: {str(e)}")
    
    def export_current_view(self):
        """Export current wireframe view to examples folder."""
        if self.current_wireframe_file:
            try:
                examples_dir = Path(__file__).parent.parent / "examples"
                examples_dir.mkdir(exist_ok=True)
                
                # Copy current file to examples with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{Path(self.current_wireframe_file).stem}_view_{timestamp}{Path(self.current_wireframe_file).suffix}"
                new_path = examples_dir / new_name
                
                import shutil
                shutil.copy2(self.current_wireframe_file, new_path)
                
                QMessageBox.information(self, "Success", f"Wireframe view exported to:\n{new_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export view: {str(e)}")


class WireframeExporterMainWindow(QMainWindow):
    """Main window for the Wireframe Exporter GUI application."""
    
    def __init__(self):
        super().__init__()
        self.current_mesh_file = None
        
        # Initialize ArUco generator
        self.aruco_generator = ArUcoGenerator()
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Wireframe Exporter - 3D Mesh Wireframe Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Export tab
        self.export_tab = self.create_export_tab()
        self.tab_widget.addTab(self.export_tab, "Export Wireframe")
        
        # Viewer tab
        self.viewer_tab = WireframeViewerWidget()
        self.tab_widget.addTab(self.viewer_tab, "View Wireframe")
        
        # ArUco Only Viewer tab
        self.aruco_only_tab = self.create_aruco_only_viewer_tab()
        self.tab_widget.addTab(self.aruco_only_tab, "View ArUco")
        
        # Combined Viewer tab
        self.combined_tab = self.create_combined_viewer_tab()
        self.tab_widget.addTab(self.combined_tab, "Combined Viewer")
        
        # Batch Processing tab
        self.batch_tab = self.create_batch_processing_tab()
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    
    def create_aruco_marker_mesh(self, marker_data: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        """Create a 3D mesh representation of an ArUco marker."""
        # Get position and rotation
        pos_data = marker_data["pose_absolute"]["position"]
        rot_data = marker_data["pose_absolute"]["rotation"]
        
        position = np.array([pos_data["x"], pos_data["y"], pos_data["z"]])
        rotation = np.array([rot_data["roll"], rot_data["pitch"], rot_data["yaw"]])
        size = marker_data["size"]
        
        # Create marker base (flat rectangle)
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/20)
        
        # Apply rotation
        if not np.allclose(rotation, [0, 0, 0]):
            # Create rotation matrix from RPY angles
            R = self.euler_to_rotation_matrix(rotation)
            marker_mesh.rotate(R, center=(0, 0, 0))
        
        # Translate to position
        marker_mesh.translate(position)
        
        # Paint with different colors based on face type
        face_type = marker_data["face_type"]
        colors = {
            "top": [0.2, 0.8, 0.2],      # Green
            "bottom": [0.8, 0.2, 0.2],   # Red
            "front": [0.2, 0.2, 0.8],    # Blue
            "back": [0.8, 0.8, 0.2],     # Yellow
            "left": [0.8, 0.2, 0.8],     # Magenta
            "right": [0.2, 0.8, 0.8],    # Cyan
            "custom": [0.5, 0.5, 0.5]    # Gray
        }
        color = colors.get(face_type, [0.5, 0.5, 0.5])
        marker_mesh.paint_uniform_color(color)
        
        # Return marker mesh without coordinate frame
        return marker_mesh
    
    def euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix."""
        roll, pitch, yaw = euler_angles
        
        # Create rotation matrices
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # Combined rotation matrix (ZYX order)
        R = R_z @ R_y @ R_x
        return R
    
    
    def create_wireframe_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
        """Create wireframe representation from mesh."""
        # Extract edges
        edges = []
        for triangle in mesh.triangles:
            # Add three edges per triangle
            edges.append([triangle[0], triangle[1]])
            edges.append([triangle[1], triangle[2]])
            edges.append([triangle[2], triangle[0]])
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = mesh.vertices
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color([0.0, 1.0, 0.0])  # Green wireframe
        
        return line_set
    
    def create_detailed_aruco_marker(self, marker_data: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        """Create a detailed 3D representation of an ArUco marker using ArUcoMarkerInfo."""
        try:
            print(f"🎯 Creating detailed ArUco marker for ID {marker_data['aruco_id']}")
            
            # Convert JSON data to ArUcoMarkerInfo object
            pos_data = marker_data["pose_absolute"]["position"]
            rot_data = marker_data["pose_absolute"]["rotation"]
            
            print(f"🎯 Position: ({pos_data['x']:.3f}, {pos_data['y']:.3f}, {pos_data['z']:.3f})")
            print(f"🎯 Size: {marker_data['size']:.3f}m")
            
            # Create ArUcoMarkerInfo object
            aruco_info = ArUcoMarkerInfo(
                dictionary=marker_data["aruco_dictionary"],
                marker_id=marker_data["aruco_id"],
                position=(pos_data["x"], pos_data["y"], pos_data["z"]),
                size=marker_data["size"],
                rotation=(rot_data["roll"], rot_data["pitch"], rot_data["yaw"]),
                border_width=marker_data["border_width"],
                cad_object_pose={
                    "cad_center": marker_data["cad_object_info"]["center"],
                    "cad_dimensions": marker_data["cad_object_info"]["dimensions"],
                    "relative_position": [
                        marker_data["pose_relative_to_cad_center"]["position"]["x"],
                        marker_data["pose_relative_to_cad_center"]["position"]["y"],
                        marker_data["pose_relative_to_cad_center"]["position"]["z"]
                    ],
                    "relative_rotation": [
                        marker_data["pose_relative_to_cad_center"]["rotation"]["roll"],
                        marker_data["pose_relative_to_cad_center"]["rotation"]["pitch"],
                        marker_data["pose_relative_to_cad_center"]["rotation"]["yaw"]
                    ],
                    "surface_normal": marker_data["surface_normal"],
                    "face_type": marker_data["face_type"]
                }
            )
            
            print(f"🎯 Created ArUcoMarkerInfo object for ID {aruco_info.marker_id}")
            
            # Use the same method as ArUco annotator to create the marker
            marker_mesh = self.create_real_aruco_marker(aruco_info)
            print(f"✅ Successfully created 3D mesh for ArUco marker ID {aruco_info.marker_id}")
            return marker_mesh
            
        except Exception as e:
            print(f"❌ Error creating detailed ArUco marker for ID {marker_data.get('aruco_id', 'unknown')}: {str(e)}")
            raise
    
    def create_real_aruco_marker(self, aruco_info: ArUcoMarkerInfo) -> o3d.geometry.TriangleMesh:
        """Create a real ArUco marker with actual pixel-perfect pattern display."""
        x, y, z = aruco_info.position
        
        # Use the size from ArUcoMarkerInfo, but apply reasonable limits
        size = aruco_info.size
        
        # Debug: Show size information
        print(f"🎯 ArUco marker size: User input = {aruco_info.size:.3f}m")
        print(f"🎯 Using exact user-specified size: {size:.3f}m")
        
        # Only apply minimum size limit to prevent extremely small markers
        original_size = size
        size = max(0.0001, size)  # Minimum 0.1mm, no maximum limit
        if size != original_size:
            print(f"⚠️  Size adjusted from {original_size:.3f}m to {size:.3f}m (minimum size limit only)")
        
        print(f"🎯 Final marker size: {size:.3f}m")
        
        # Generate the actual ArUco marker image
        marker_image = self.aruco_generator.generate_marker(
            aruco_info.dictionary, 
            aruco_info.marker_id, 
            512  # High resolution
        )
        
        # Create pixel-based pattern - this approach works reliably and shows exact ArUco pattern
        # Use moderate resolution for clear pattern while keeping performance good
        display_resolution = 12  # 12x12 grid - good balance of detail vs performance
        
        # Calculate border width in actual units
        border_width_units = size * aruco_info.border_width
        
        # Calculate the actual pattern area (reduced by border)
        pattern_size = size - 2 * border_width_units
        
        # Sample the marker image to get the pattern
        pixel_size = pattern_size / display_resolution
        combined_mesh = o3d.geometry.TriangleMesh()
        
        print(f"🎯 Creating 3D ArUco marker from 2D image: {marker_image.shape} -> {display_resolution}x{display_resolution} blocks")
        print(f"🎯 Border width: {aruco_info.border_width*100:.1f}% ({border_width_units*1000:.1f}mm)")
        print(f"🎯 Pattern area: {pattern_size*1000:.1f}mm (reduced from {size*1000:.1f}mm total)")
        
        for i in range(display_resolution):
            for j in range(display_resolution):
                # Sample the corresponding pixel from the marker image
                # Map from 3D grid coordinates to 2D image coordinates
                img_x = int((i / display_resolution) * marker_image.shape[1])
                img_y = int((j / display_resolution) * marker_image.shape[0])
                
                # Ensure we don't go out of bounds
                img_x = min(img_x, marker_image.shape[1] - 1)
                img_y = min(img_y, marker_image.shape[0] - 1)
                
                # Get pixel intensity (0=black, 255=white)
                pixel_intensity = marker_image[img_y, img_x]
                
                # Create a small flat cube for this pixel
                # Position pixels within the pattern area (excluding border)
                pixel_x = x + pattern_size/2 - (i + 0.5) * pixel_size
                pixel_y = y + pattern_size/2 - (j + 0.5) * pixel_size 
                pixel_z = z - size/200  # Very thin, just above the base
                
                pixel_cube = o3d.geometry.TriangleMesh.create_box(
                    pixel_size,  # Full pixel size - no gaps
                    pixel_size, 
                    size/200  # Very thin
                )
                pixel_cube.translate([
                    pixel_x - pixel_size * 0.5,
                    pixel_y - pixel_size * 0.5,
                    pixel_z
                ])
                
                # Color based on pixel intensity
                if pixel_intensity < 128:  # Black pixel
                    pixel_cube.paint_uniform_color([0.0, 0.0, 0.0])
                else:  # White pixel
                    pixel_cube.paint_uniform_color([1.0, 1.0, 1.0])
                
                combined_mesh += pixel_cube
        
        print(f"✅ Created 3D ArUco marker with {display_resolution*display_resolution} blocks")
        
        # Add white border around the pattern if border width > 0
        if aruco_info.border_width > 0:
            # Create white border by adding a larger white rectangle behind the pattern
            border_plate = o3d.geometry.TriangleMesh.create_box(size, size, size/300)
            border_plate.translate([x - size/2, y - size/2, z - size/600])
            border_plate.paint_uniform_color([1.0, 1.0, 1.0])  # White border
            combined_mesh += border_plate
            
            # Add a slightly smaller black rectangle to create the border effect
            inner_plate = o3d.geometry.TriangleMesh.create_box(pattern_size, pattern_size, size/250)
            inner_plate.translate([x - pattern_size/2, y - pattern_size/2, z - size/500])
            inner_plate.paint_uniform_color([0.0, 0.0, 0.0])  # Black inner area
            combined_mesh += inner_plate
        
        # Add a thin base plate for better visualization
        base_plate = o3d.geometry.TriangleMesh.create_box(size, size, size/200)
        base_plate.translate([x - size/2, y - size/2, z - size/400])
        base_plate.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray base
        combined_mesh += base_plate
        
        # Apply rotation if specified
        if hasattr(aruco_info, 'rotation') and aruco_info.rotation != (0, 0, 0):
            roll, pitch, yaw = aruco_info.rotation
            R = combined_mesh.get_rotation_matrix_from_xyz([roll, pitch, yaw])
            combined_mesh.rotate(R, center=[x, y, z])
            print(f"📐 Applied rotation to ArUco marker: ({roll:.3f}, {pitch:.3f}, {yaw:.3f}) rad")
        
        # Compute normals for proper lighting
        combined_mesh.compute_vertex_normals()
        
        return combined_mesh
    
    def browse_aruco_only_file(self):
        """Browse for ArUco markers JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ArUco Markers JSON File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.aruco_only_path_edit.setText(file_path)
    
    def load_aruco_only_file(self):
        """Load ArUco markers from JSON file."""
        file_path = self.aruco_only_path_edit.text().strip()
        
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select an ArUco markers JSON file.")
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate the data structure
            if "markers" not in data:
                QMessageBox.critical(self, "Error", "Invalid ArUco markers JSON file format.")
                return
            
            self.aruco_only_data = data
            self.update_aruco_only_summary()
            self.aruco_only_launch_btn.setEnabled(True)
            
            QMessageBox.information(
                self,
                "Success",
                f"Successfully loaded {len(data['markers'])} ArUco markers!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load ArUco markers file: {str(e)}"
            )
    
    def load_aruco_only_example(self):
        """Load the fork orange example ArUco markers."""
        try:
            aruco_path = Path("/home/aaugus11/Projects/aruco-grasp-annotator/fork_orange_annotations.json")
            
            if aruco_path.exists():
                self.aruco_only_path_edit.setText(str(aruco_path))
                self.load_aruco_only_file()
                
                QMessageBox.information(
                    self,
                    "Example Loaded",
                    "Successfully loaded Fork Orange ArUco markers example!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"Example file not found: {aruco_path}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load example file: {str(e)}"
            )
    
    def update_aruco_only_summary(self):
        """Update the ArUco markers summary display."""
        if not self.aruco_only_data:
            return
        
        markers = self.aruco_only_data["markers"]
        summary_text = f"""ARUCO MARKERS SUMMARY
========================
Total Markers: {len(markers)}
Export Type: {self.aruco_only_data.get('export_type', 'Unknown')}
Version: {self.aruco_only_data.get('version', 'Unknown')}
Model File: {self.aruco_only_data.get('model_file', 'Unknown')}

MARKER DETAILS:
"""
        
        for i, marker in enumerate(markers):
            summary_text += f"""
Marker {i+1}:
  • Dictionary: {marker['aruco_dictionary']}
  • ID: {marker['aruco_id']}
  • Size: {marker['size']:.3f}m
  • Border Width: {marker['border_width']*100:.1f}%
  • Face Type: {marker['face_type']}
  • Position: ({marker['pose_absolute']['position']['x']:.3f}, {marker['pose_absolute']['position']['y']:.3f}, {marker['pose_absolute']['position']['z']:.3f})
  • Rotation: ({marker['pose_absolute']['rotation']['roll']:.3f}, {marker['pose_absolute']['rotation']['pitch']:.3f}, {marker['pose_absolute']['rotation']['yaw']:.3f}) rad
"""
        
        self.aruco_only_summary_text.setPlainText(summary_text)
    
    def launch_aruco_only_viewer(self):
        """Launch the ArUco-only 3D viewer."""
        if not self.aruco_only_data:
            QMessageBox.warning(self, "Warning", "Please load ArUco markers first.")
            return
        
        try:
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window("ArUco Markers Only", width=1400, height=900)
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coord_frame)
            
            # Add ArUco markers
            for marker_data in self.aruco_only_data["markers"]:
                marker_mesh = self.create_detailed_aruco_marker(marker_data)
                vis.add_geometry(marker_mesh)
                print(f"✅ Added ArUco marker {marker_data['aruco_id']} to viewer")
            
            # Setup view
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0.1, 0.1, 0.1])
            render_option.show_coordinate_frame = True
            render_option.mesh_show_wireframe = True
            render_option.mesh_show_back_face = True
            
            print(f"🎯 Launched ArUco-only viewer with {len(self.aruco_only_data['markers'])} markers")
            
            # Run visualization
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Viewer Error",
                f"Failed to launch ArUco-only viewer: {str(e)}"
            )
            print(f"❌ Error launching ArUco-only viewer: {e}")
    
    
    def browse_combined_wireframe_file(self):
        """Browse for wireframe JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Wireframe JSON File",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.combined_wireframe_path_edit.setText(file_path)
            self.check_combined_files_ready()
    
    def browse_combined_aruco_file(self):
        """Browse for ArUco markers JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ArUco Markers JSON File",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.combined_aruco_path_edit.setText(file_path)
            self.check_combined_files_ready()
    
    def check_combined_files_ready(self):
        """Check if both files are selected and enable load button."""
        wireframe_ready = bool(self.combined_wireframe_path_edit.text())
        aruco_ready = bool(self.combined_aruco_path_edit.text())
        
        self.load_combined_btn.setEnabled(wireframe_ready and aruco_ready)
    
    def load_combined_files(self):
        """Load both wireframe and ArUco marker files."""
        try:
            # Load wireframe data
            with open(self.combined_wireframe_path_edit.text(), 'r') as f:
                self.combined_wireframe_data = json.load(f)
            
            # Load ArUco data
            with open(self.combined_aruco_path_edit.text(), 'r') as f:
                self.combined_aruco_data = json.load(f)
            
            # Update summary
            self.update_combined_summary()
            
            # Enable viewer button
            self.launch_combined_viewer_btn.setEnabled(True)
            
            QMessageBox.information(
                self,
                "Files Loaded",
                "Successfully loaded both wireframe and ArUco marker files!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load files: {str(e)}"
            )
    
    def update_combined_summary(self):
        """Update the data summary display."""
        summary_text = "DATA SUMMARY\n"
        summary_text += "=" * 50 + "\n\n"
        
        if self.combined_wireframe_data:
            mesh_info = self.combined_wireframe_data.get("mesh_info", {})
            summary_text += "WIREFRAME DATA:\n"
            summary_text += f"  • Vertices: {mesh_info.get('num_vertices', 'Unknown')}\n"
            summary_text += f"  • Edges: {mesh_info.get('num_edges', 'Unknown')}\n"
            summary_text += f"  • Triangles: {mesh_info.get('num_triangles', 'Unknown')}\n"
            
            bbox = mesh_info.get('bounding_box', {})
            if bbox:
                size = bbox.get('size', [0, 0, 0])
                summary_text += f"  • Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} units\n"
            summary_text += "\n"
        
        if self.combined_aruco_data:
            markers = self.combined_aruco_data.get("markers", [])
            summary_text += "ARUCO MARKERS:\n"
            summary_text += f"  • Total markers: {len(markers)}\n"
            summary_text += f"  • Export version: {self.combined_aruco_data.get('version', 'Unknown')}\n"
            summary_text += f"  • Model file: {Path(self.combined_aruco_data.get('model_file', 'Unknown')).name}\n\n"
            
            summary_text += "MARKER DETAILS:\n"
            for i, marker in enumerate(markers):
                summary_text += f"  {i+1}. {marker['aruco_dictionary']} ID:{marker['aruco_id']} - {marker['face_type']} face\n"
                pos = marker['pose_absolute']['position']
                summary_text += f"     Position: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})\n"
        
        self.combined_summary_text.setPlainText(summary_text)
    
    def load_fork_example(self):
        """Load the fork orange example files."""
        try:
            # Set paths to updated example files
            examples_dir = Path(__file__).parent.parent / "examples"
            wireframe_path = examples_dir / "fork_orange_scaled70_ v2_wireframe.json"
            aruco_path = examples_dir / "fork_orange_scaled70_ v2_aruco_annotations.json"
            
            if wireframe_path.exists() and aruco_path.exists():
                self.combined_wireframe_path_edit.setText(str(wireframe_path))
                self.combined_aruco_path_edit.setText(str(aruco_path))
                self.check_combined_files_ready()
                
                # Auto-load the files
                self.load_combined_files()
                
                QMessageBox.information(
                    self,
                    "Example Loaded",
                    "Successfully loaded Fork Orange scaled70 v2 example files!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Files Not Found",
                    f"Example files not found:\nWireframe: {wireframe_path}\nArUco: {aruco_path}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load example files: {str(e)}"
            )
    
    def launch_combined_viewer(self):
        """Launch the combined 3D viewer with wireframe and ArUco markers."""
        if not self.combined_wireframe_data or not self.combined_aruco_data:
            QMessageBox.warning(self, "Warning", "Please load both files first.")
            return
        
        
        try:
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window("Combined Wireframe + ArUco Markers", width=1400, height=900)
            
            # No coordinate frame added - clean view
            
            # Add wireframe
            wireframe_mesh = self.create_wireframe_from_json(self.combined_wireframe_data)
            vis.add_geometry(wireframe_mesh)
            print("✅ Added wireframe to combined viewer")
            
            # Add ArUco markers
            markers_added = 0
            for i, marker_data in enumerate(self.combined_aruco_data["markers"]):
                try:
                    print(f"🎯 Processing ArUco marker {i+1}/{len(self.combined_aruco_data['markers'])}: ID {marker_data['aruco_id']}")
                    
                    marker_mesh = self.create_detailed_aruco_marker(marker_data)
                    vis.add_geometry(marker_mesh)
                    markers_added += 1
                    print(f"✅ Added ArUco marker {marker_data['aruco_id']} to combined viewer")
                except Exception as marker_error:
                    print(f"❌ Failed to add ArUco marker {marker_data.get('aruco_id', 'unknown')}: {str(marker_error)}")
                    # Continue with other markers even if one fails
                    continue
            
            print(f"🎯 Successfully added {markers_added}/{len(self.combined_aruco_data['markers'])} ArUco markers to combined viewer")
            
            # Setup view
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0.1, 0.1, 0.1])
            render_option.show_coordinate_frame = False
            render_option.mesh_show_wireframe = True
            render_option.mesh_show_back_face = True
            
            print(f"🎯 Launched combined viewer with wireframe + {markers_added} ArUco markers")
            
            # Run visualization
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            error_msg = f"Failed to launch combined viewer: {str(e)}"
            print(f"❌ {error_msg}")
            QMessageBox.critical(
                self,
                "Viewer Error",
                error_msg
            )
    
    def create_wireframe_from_json(self, wireframe_data: Dict[str, Any]) -> o3d.geometry.LineSet:
        """Create wireframe LineSet from JSON data."""
        try:
            print(f"🎯 Creating wireframe from JSON data")
            vertices = np.array(wireframe_data["vertices"])
            edges = np.array(wireframe_data["edges"])
            
            print(f"🎯 Wireframe vertices: {len(vertices)} points")
            print(f"🎯 Wireframe edges: {len(edges)} lines")
            
            # Create line set
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            line_set.paint_uniform_color([0.0, 1.0, 0.0])  # Green wireframe
            
            print(f"✅ Successfully created wireframe LineSet")
            return line_set
            
        except Exception as e:
            print(f"❌ Error creating wireframe from JSON: {str(e)}")
            raise
    
    
    def create_export_tab(self):
        """Create the export tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Model loading section
        model_group = QGroupBox("3D Model Loading")
        model_layout = QGridLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select a 3D model file...")
        self.model_path_edit.setReadOnly(True)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_model_file)
        
        self.model_info_label = QLabel("No model loaded")
        
        model_layout.addWidget(QLabel("Model File:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(self.browse_btn, 0, 2)
        model_layout.addWidget(self.model_info_label, 1, 0, 1, 3)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Export options section
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout()
        
        # Format selection
        export_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["json", "csv", "numpy", "ply", "obj"])
        self.format_combo.setCurrentText("json")
        export_layout.addWidget(self.format_combo, 0, 1)
        
        # Output path
        export_layout.addWidget(QLabel("Output Path:"), 1, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Auto-generated if empty")
        export_layout.addWidget(self.output_path_edit, 1, 1)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_file)
        export_layout.addWidget(self.output_browse_btn, 1, 2)
        
        # Auto-export to examples
        self.auto_export_check = QCheckBox("Auto-export to examples folder")
        self.auto_export_check.setChecked(True)
        export_layout.addWidget(self.auto_export_check, 2, 0, 1, 3)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Export controls
        control_group = QGroupBox("Export Controls")
        control_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Wireframe")
        self.export_btn.clicked.connect(self.export_wireframe)
        self.export_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addStretch()
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Log display
        log_group = QGroupBox("Export Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    
    def create_aruco_only_viewer_tab(self):
        """Create a tab for viewing only ArUco markers without wireframe."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Load ArUco JSON section
        load_group = QGroupBox("Load ArUco Markers JSON")
        load_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.aruco_only_path_edit = QLineEdit()
        self.aruco_only_path_edit.setPlaceholderText("Select ArUco markers JSON file...")
        file_layout.addWidget(self.aruco_only_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_aruco_only_file)
        file_layout.addWidget(browse_btn)
        load_layout.addLayout(file_layout)
        
        # Load button
        load_btn = QPushButton("Load ArUco Markers")
        load_btn.clicked.connect(self.load_aruco_only_file)
        load_layout.addWidget(load_btn)
        
        # Quick load example
        example_btn = QPushButton("Load Fork Orange Example")
        example_btn.clicked.connect(self.load_aruco_only_example)
        load_layout.addWidget(example_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Data summary
        summary_group = QGroupBox("ArUco Markers Summary")
        summary_layout = QVBoxLayout()
        
        self.aruco_only_summary_text = QTextEdit()
        self.aruco_only_summary_text.setMaximumHeight(150)
        self.aruco_only_summary_text.setReadOnly(True)
        self.aruco_only_summary_text.setPlaceholderText("Load an ArUco markers JSON file to see summary...")
        summary_layout.addWidget(self.aruco_only_summary_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Viewer controls
        viewer_group = QGroupBox("3D Viewer")
        viewer_layout = QVBoxLayout()
        
        launch_btn = QPushButton("Launch ArUco Markers 3D Viewer")
        launch_btn.clicked.connect(self.launch_aruco_only_viewer)
        launch_btn.setEnabled(False)
        self.aruco_only_launch_btn = launch_btn
        viewer_layout.addWidget(launch_btn)
        
        viewer_group.setLayout(viewer_layout)
        layout.addWidget(viewer_group)
        
        # Store data
        self.aruco_only_data = None
        
        tab.setLayout(layout)
        return tab
    
    def create_combined_viewer_tab(self):
        """Create the combined viewer tab for wireframe + ArUco markers."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # File loading section
        files_group = QGroupBox("Load Files")
        files_layout = QGridLayout()
        
        # Wireframe JSON file
        self.combined_wireframe_path_edit = QLineEdit()
        self.combined_wireframe_path_edit.setPlaceholderText("Select wireframe JSON file...")
        self.combined_wireframe_path_edit.setReadOnly(True)
        
        self.combined_wireframe_browse_btn = QPushButton("Browse Wireframe...")
        self.combined_wireframe_browse_btn.clicked.connect(self.browse_combined_wireframe_file)
        
        # ArUco JSON file
        self.combined_aruco_path_edit = QLineEdit()
        self.combined_aruco_path_edit.setPlaceholderText("Select ArUco markers JSON file...")
        self.combined_aruco_path_edit.setReadOnly(True)
        
        self.combined_aruco_browse_btn = QPushButton("Browse ArUco...")
        self.combined_aruco_browse_btn.clicked.connect(self.browse_combined_aruco_file)
        
        files_layout.addWidget(QLabel("Wireframe JSON:"), 0, 0)
        files_layout.addWidget(self.combined_wireframe_path_edit, 0, 1)
        files_layout.addWidget(self.combined_wireframe_browse_btn, 0, 2)
        
        files_layout.addWidget(QLabel("ArUco JSON:"), 1, 0)
        files_layout.addWidget(self.combined_aruco_path_edit, 1, 1)
        files_layout.addWidget(self.combined_aruco_browse_btn, 1, 2)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        # Load and visualize section
        load_group = QGroupBox("Load and Visualize")
        load_layout = QVBoxLayout()
        
        self.load_combined_btn = QPushButton("Load Both Files")
        self.load_combined_btn.clicked.connect(self.load_combined_files)
        self.load_combined_btn.setEnabled(False)
        
        
        self.launch_combined_viewer_btn = QPushButton("Launch Combined 3D Viewer")
        self.launch_combined_viewer_btn.clicked.connect(self.launch_combined_viewer)
        self.launch_combined_viewer_btn.setEnabled(False)
        
        load_layout.addWidget(self.load_combined_btn)
        load_layout.addWidget(self.launch_combined_viewer_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Data summary section
        summary_group = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout()
        
        self.combined_summary_text = QTextEdit()
        self.combined_summary_text.setMaximumHeight(200)
        self.combined_summary_text.setReadOnly(True)
        self.combined_summary_text.setPlaceholderText("Load files to see data summary...")
        
        summary_layout.addWidget(self.combined_summary_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Quick load examples section
        examples_group = QGroupBox("Quick Load Examples")
        examples_layout = QVBoxLayout()
        
        self.load_fork_example_btn = QPushButton("Load Fork Orange Example")
        self.load_fork_example_btn.clicked.connect(self.load_fork_example)
        
        examples_layout.addWidget(self.load_fork_example_btn)
        examples_group.setLayout(examples_layout)
        layout.addWidget(examples_group)
        
        # Initialize data storage
        self.combined_wireframe_data = None
        self.combined_aruco_data = None
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_batch_processing_tab(self):
        """Create the batch processing tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Source folder selection
        source_group = QGroupBox("Source Folder")
        source_layout = QGridLayout()
        
        self.batch_source_edit = QLineEdit()
        self.batch_source_edit.setPlaceholderText("Select folder containing OBJ files...")
        self.batch_source_edit.setReadOnly(True)
        
        self.batch_source_browse_btn = QPushButton("Browse...")
        self.batch_source_browse_btn.clicked.connect(self.browse_batch_source_folder)
        
        source_layout.addWidget(QLabel("Source Folder:"), 0, 0)
        source_layout.addWidget(self.batch_source_edit, 0, 1)
        source_layout.addWidget(self.batch_source_browse_btn, 0, 2)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Output folder selection
        output_group = QGroupBox("Output Folder")
        output_layout = QGridLayout()
        
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("Select output folder for wireframe files...")
        self.batch_output_edit.setReadOnly(True)
        
        self.batch_output_browse_btn = QPushButton("Browse...")
        self.batch_output_browse_btn.clicked.connect(self.browse_batch_output_folder)
        
        output_layout.addWidget(QLabel("Output Folder:"), 0, 0)
        output_layout.addWidget(self.batch_output_edit, 0, 1)
        output_layout.addWidget(self.batch_output_browse_btn, 0, 2)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QGridLayout()
        
        options_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.batch_format_combo = QComboBox()
        self.batch_format_combo.addItems(["json", "csv", "numpy", "ply", "obj"])
        self.batch_format_combo.setCurrentText("json")
        options_layout.addWidget(self.batch_format_combo, 0, 1)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # File preview
        preview_group = QGroupBox("Files to Process")
        preview_layout = QVBoxLayout()
        
        self.batch_file_list = QListWidget()
        self.batch_file_list.setMaximumHeight(150)
        preview_layout.addWidget(self.batch_file_list)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Batch processing controls
        control_group = QGroupBox("Batch Processing")
        control_layout = QVBoxLayout()
        
        self.batch_export_btn = QPushButton("Start Batch Export")
        self.batch_export_btn.clicked.connect(self.start_batch_export)
        self.batch_export_btn.setEnabled(False)
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        
        self.batch_file_progress_label = QLabel("")
        self.batch_file_progress_label.setVisible(False)
        
        control_layout.addWidget(self.batch_export_btn)
        control_layout.addWidget(self.batch_progress_bar)
        control_layout.addWidget(self.batch_file_progress_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Log display
        log_group = QGroupBox("Batch Export Log")
        log_layout = QVBoxLayout()
        
        self.batch_log_text = QTextEdit()
        self.batch_log_text.setMaximumHeight(200)
        self.batch_log_text.setReadOnly(True)
        
        log_layout.addWidget(self.batch_log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def browse_batch_source_folder(self):
        """Browse for source folder containing OBJ files."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Source Folder",
            str(Path.home())
        )
        
        if folder_path:
            self.batch_source_edit.setText(folder_path)
            self.update_batch_file_list()
            self.check_batch_ready()
    
    def browse_batch_output_folder(self):
        """Browse for output folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(Path.home())
        )
        
        if folder_path:
            self.batch_output_edit.setText(folder_path)
            self.check_batch_ready()
    
    def update_batch_file_list(self):
        """Update the list of files to be processed."""
        self.batch_file_list.clear()
        
        source_path = self.batch_source_edit.text().strip()
        if not source_path:
            return
        
        try:
            source_dir = Path(source_path)
            obj_files = list(source_dir.glob("*.obj"))
            
            if obj_files:
                for obj_file in obj_files:
                    item = QListWidgetItem(obj_file.name)
                    self.batch_file_list.addItem(item)
                
                self.batch_log_text.append(f"Found {len(obj_files)} OBJ files in source folder")
            else:
                self.batch_log_text.append("No OBJ files found in source folder")
                
        except Exception as e:
            self.batch_log_text.append(f"Error scanning source folder: {str(e)}")
    
    def check_batch_ready(self):
        """Check if batch processing is ready to start."""
        source_ready = bool(self.batch_source_edit.text().strip())
        output_ready = bool(self.batch_output_edit.text().strip())
        
        self.batch_export_btn.setEnabled(source_ready and output_ready)
    
    def start_batch_export(self):
        """Start the batch export process."""
        source_folder = self.batch_source_edit.text().strip()
        output_folder = self.batch_output_edit.text().strip()
        output_format = self.batch_format_combo.currentText()
        
        if not source_folder or not output_folder:
            QMessageBox.warning(self, "Warning", "Please select both source and output folders.")
            return
        
        # Start batch export in thread
        self.batch_export_thread = BatchExportThread(source_folder, output_folder, output_format)
        
        self.batch_export_thread.progress.connect(self.batch_progress_bar.setValue)
        self.batch_export_thread.file_progress.connect(self.update_batch_file_progress)
        self.batch_export_thread.finished.connect(self.batch_export_finished)
        self.batch_export_thread.error.connect(self.batch_export_error)
        
        self.batch_export_btn.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_file_progress_label.setVisible(True)
        self.batch_progress_bar.setValue(0)
        
        self.batch_log_text.append(f"Starting batch export from {source_folder} to {output_folder}...")
        self.batch_export_thread.start()
    
    def update_batch_file_progress(self, filename: str, current: int, total: int):
        """Update the file progress display."""
        self.batch_file_progress_label.setText(f"Processing: {filename} ({current}/{total})")
        self.batch_log_text.append(f"Processing {filename} ({current}/{total})...")
    
    def batch_export_finished(self, message: str, success: bool):
        """Handle batch export completion."""
        self.batch_export_btn.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        self.batch_file_progress_label.setVisible(False)
        
        self.batch_log_text.append(message)
        self.statusBar().showMessage(message)
        
        if success:
            QMessageBox.information(self, "Batch Export Complete", message)
        else:
            QMessageBox.warning(self, "Batch Export Warning", message)
    
    def batch_export_error(self, error_message: str):
        """Handle batch export error."""
        self.batch_export_btn.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        self.batch_file_progress_label.setVisible(False)
        
        self.batch_log_text.append(f"ERROR: {error_message}")
        self.statusBar().showMessage(f"Batch export failed: {error_message}")
        
        QMessageBox.critical(self, "Batch Export Error", error_message)
    
    def browse_model_file(self):
        """Browse for a 3D model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D Model File",
            str(Path.home()),
            "3D Model Files (*.obj *.stl *.ply *.off);;OBJ Files (*.obj);;STL Files (*.stl);;PLY Files (*.ply);;OFF Files (*.off)"
        )
        
        if file_path:
            self.current_mesh_file = file_path
            self.model_path_edit.setText(file_path)
            self.export_btn.setEnabled(True)
            
            # Display model info
            self.display_model_info(file_path)
            
            # Auto-generate output path
            if not self.output_path_edit.text():
                output_path = Path(file_path).with_suffix(f".wireframe.{self.format_combo.currentText()}")
                self.output_path_edit.setText(str(output_path))
    
    def browse_output_file(self):
        """Browse for output file location."""
        if not self.current_mesh_file:
            QMessageBox.warning(self, "Warning", "Please select a model file first.")
            return
        
        format_ext = self.format_combo.currentText()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Wireframe As",
            str(Path.home()),
            f"{format_ext.upper()} Files (*.{format_ext})"
        )
        
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def display_model_info(self, file_path: str):
        """Display information about the loaded model."""
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            if len(mesh.vertices) == 0:
                self.model_info_label.setText("Error: Could not load mesh")
                self.export_btn.setEnabled(False)
                return
            
            bbox = mesh.get_axis_aligned_bounding_box()
            size = bbox.max_bound - bbox.min_bound
            
            info = f"""Vertices: {len(mesh.vertices)} | Triangles: {len(mesh.triangles)} | 
            Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} units"""
            
            self.model_info_label.setText(info)
            self.log_text.append(f"Loaded model: {Path(file_path).name}")
            self.log_text.append(f"  - Vertices: {len(mesh.vertices)}")
            self.log_text.append(f"  - Triangles: {len(mesh.triangles)}")
            self.log_text.append(f"  - Bounding box: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
            
        except Exception as e:
            self.model_info_label.setText(f"Error loading model: {str(e)}")
            self.export_btn.setEnabled(False)
            self.log_text.append(f"Error loading model: {str(e)}")
    
    def export_wireframe(self):
        """Export wireframe data."""
        if not self.current_mesh_file:
            QMessageBox.warning(self, "Warning", "Please select a model file first.")
            return
        
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            output_path = str(Path(self.current_mesh_file).with_suffix(f".wireframe.{self.format_combo.currentText()}"))
            self.output_path_edit.setText(output_path)
        
        # Start export in thread
        self.export_thread = WireframeExportThread(
            self.current_mesh_file,
            self.format_combo.currentText(),
            output_path,
            self.auto_export_check.isChecked()
        )
        
        self.export_thread.progress.connect(self.progress_bar.setValue)
        self.export_thread.finished.connect(self.export_finished)
        self.export_thread.error.connect(self.export_error)
        
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_text.append(f"Starting export to {output_path}...")
        self.export_thread.start()
    
    def export_finished(self, message: str, success: bool):
        """Handle export completion."""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(message)
        self.statusBar().showMessage(message)
        
        if success:
            QMessageBox.information(self, "Success", message)
            
            # Switch to viewer tab and load the exported file
            self.tab_widget.setCurrentIndex(1)  # Switch to viewer tab
            self.viewer_tab.current_wireframe_file = self.output_path_edit.text()
            self.viewer_tab.view_btn.setEnabled(True)
            self.viewer_tab.export_viewer_btn.setEnabled(True)
            self.viewer_tab.status_label.setText(f"Loaded: {Path(self.output_path_edit.text()).name}")
            self.viewer_tab.display_file_info(self.output_path_edit.text())
    
    def export_error(self, error_message: str):
        """Handle export error."""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"ERROR: {error_message}")
        self.statusBar().showMessage(f"Export failed: {error_message}")
        
        QMessageBox.critical(self, "Export Error", error_message)


def main():
    """Main function to run the GUI application."""
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setApplicationName("Wireframe Exporter")
    app.setApplicationVersion("1.0.0")
    
    window = WireframeExporterMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
