"""Main window for the ArUco Grasp Annotator application."""

import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QKeySequence, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QMenuBar, QStatusBar, QToolBar,
    QFileDialog, QMessageBox, QPushButton, QLabel,
    QGroupBox, QListWidget, QSlider, QSpinBox,
    QCheckBox, QComboBox
)

from .working_viewer_3d import WorkingViewer3D
from .marker_panel import MarkerPanel
from ..core.cad_loader import CADLoader
from ..core.annotation_manager import AnnotationManager


class MainWindow(QMainWindow):
    """Main application window with 3D viewer and control panels."""
    
    def __init__(self) -> None:
        super().__init__()
        self.cad_loader = CADLoader()
        self.annotation_manager = AnnotationManager()
        self.current_file: Optional[Path] = None
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("ArUco Marker Annotator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        
        # Left panel for controls
        self.left_panel = self.create_left_panel()
        main_splitter.addWidget(self.left_panel)
        
        # 3D Viewer in the center
        self.viewer_3d = WorkingViewer3D()
        main_splitter.addWidget(self.viewer_3d)
        
        # Right panel for export/import
        self.right_panel = self.create_right_panel()
        main_splitter.addWidget(self.right_panel)
        
        # Set splitter proportions (left: 300px, center: expand, right: 300px)
        main_splitter.setSizes([300, 800, 300])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.create_status_bar()
        
    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        from PyQt6.QtWidgets import QScrollArea
        
        # Create a scroll area for the left panel to handle overflow
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMaximumWidth(350)
        scroll_area.setMinimumWidth(250)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)  # Reduce spacing between sections
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # File section
        file_group = QGroupBox("CAD Model")
        file_layout = QVBoxLayout(file_group)
        
        # Unit selection
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Input Units:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Auto-detect", "mm", "cm", "m", "in", "ft"])
        self.unit_combo.setCurrentText("Auto-detect")
        unit_layout.addWidget(self.unit_combo)
        file_layout.addLayout(unit_layout)
        
        self.load_button = QPushButton("Load CAD File...")
        self.load_button.clicked.connect(self.load_cad_file)
        file_layout.addWidget(self.load_button)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.file_label)
        
        layout.addWidget(file_group)
        
        
        # ArUco Markers section
        self.marker_panel = MarkerPanel()
        layout.addWidget(self.marker_panel)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # Set the panel as the scroll area's widget
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def create_right_panel(self) -> QWidget:
        """Create the right panel for export/import."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(250)
        
        layout = QVBoxLayout(panel)
        
        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        self.export_button = QPushButton("Export Annotations...")
        self.export_button.clicked.connect(self.export_annotations)
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_button)
        
        self.import_button = QPushButton("Import Annotations...")
        self.import_button.clicked.connect(self.import_annotations)
        export_layout.addWidget(self.import_button)
        
        layout.addWidget(export_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Open action
        open_action = QAction("Open CAD File...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.load_cad_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Save annotations
        save_action = QAction("Save Annotations...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.export_annotations)
        file_menu.addAction(save_action)
        
        # Load annotations
        load_action = QAction("Load Annotations...", self)
        load_action.triggered.connect(self.import_annotations)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Reset camera
        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.setShortcut("R")
        reset_camera_action.triggered.connect(self.viewer_3d.reset_camera)
        view_menu.addAction(reset_camera_action)
        
        # Fit to view
        fit_view_action = QAction("Fit to View", self)
        fit_view_action.setShortcut("F")
        fit_view_action.triggered.connect(self.viewer_3d.fit_to_view)
        view_menu.addAction(fit_view_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self) -> None:
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Load file
        load_action = QAction("Load", self)
        load_action.setToolTip("Load CAD File")
        load_action.triggered.connect(self.load_cad_file)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        # View controls
        reset_action = QAction("Reset View", self)
        reset_action.setToolTip("Reset Camera View")
        reset_action.triggered.connect(self.viewer_3d.reset_camera)
        toolbar.addAction(reset_action)
        
        fit_action = QAction("Fit View", self)
        fit_action.setToolTip("Fit Model to View")
        fit_action.triggered.connect(self.viewer_3d.fit_to_view)
        toolbar.addAction(fit_action)
        
    def create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a CAD file to begin annotation")
        
    def setup_connections(self) -> None:
        """Setup signal-slot connections between components."""
        
        # Marker panel connections
        self.marker_panel.marker_added.connect(self.viewer_3d.add_aruco_marker)
        self.marker_panel.marker_removed.connect(self.viewer_3d.remove_marker)
        self.marker_panel.marker_selected.connect(self.viewer_3d.select_marker)
        self.marker_panel.marker_position_changed.connect(self.viewer_3d.move_marker)
        self.marker_panel.marker_moved.connect(self.viewer_3d.move_marker)
        self.marker_panel.marker_orientation_changed.connect(self.viewer_3d.rotate_marker)
        self.marker_panel.placement_mode_requested.connect(self.handle_placement_mode)
        
        # Viewer to marker panel connections
        self.viewer_3d.point_picked.connect(self.marker_panel.place_marker_at_clicked_position)
        
        
    def handle_placement_mode(self, enable: bool) -> None:
        """Handle placement mode requests from marker panel."""
        if enable:
            self.viewer_3d.enable_placement_mode()
            self.status_bar.showMessage("Click on the 3D model to place a marker")
        else:
            self.viewer_3d.disable_placement_mode()
            self.status_bar.showMessage("Placement mode cancelled")
        
    def load_cad_file(self) -> None:
        """Load a CAD file for annotation."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Load CAD File",
            "",
            "CAD Files (*.stl *.obj *.ply);;STL Files (*.stl);;OBJ Files (*.obj);;PLY Files (*.ply);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_file = Path(file_path)
                
                # Clear all existing markers and reset counter for new CAD model
                self.marker_panel.clear_all_markers_automatically()
                
                # Get selected units
                selected_units = self.unit_combo.currentText()
                if selected_units == "Auto-detect":
                    input_units = "auto"
                else:
                    input_units = selected_units.lower()
                
                mesh = self.cad_loader.load_file(self.current_file, input_units)
                
                # Get mesh information including dimensions
                mesh_info = self.cad_loader.get_mesh_info(mesh)
                dimensions = mesh_info['dimensions']
                detected_units = self.cad_loader.get_input_units()
                
                # Update UI with dimension information
                dim_text = f"Loaded: {self.current_file.name}\n"
                dim_text += f"Dimensions: {dimensions['length']:.4f} × {dimensions['width']:.4f} × {dimensions['height']:.4f} m\n"
                dim_text += f"Vertices: {mesh_info['vertices']:,} | Faces: {mesh_info['triangles']:,}\n"
                if selected_units == "Auto-detect":
                    dim_text += f"Detected input units: {detected_units}"
                else:
                    dim_text += f"Input units: {selected_units}"
                
                self.file_label.setText(dim_text)
                self.file_label.setStyleSheet("color: green; font-size: 10px;")
                
                # Load mesh into viewer with dimension info
                self.viewer_3d.load_mesh(mesh, mesh_info)
                
                # Pass CAD object information to marker panel for pose calculations
                cad_info = self.viewer_3d.get_cad_object_info()
                if cad_info:
                    self.marker_panel.set_cad_object_info(cad_info)
                    print(f"📐 CAD object info passed to marker panel: {cad_info}")
                
                # Enable export button
                self.export_button.setEnabled(True)
                
                # Update status with dimensions
                self.status_bar.showMessage(
                    f"Loaded: {self.current_file.name} - "
                    f"Size: {dimensions['length']:.2f}×{dimensions['width']:.2f}×{dimensions['height']:.2f}m"
                )
                
                # Reset annotation manager
                self.annotation_manager.set_model_file(self.current_file)
                
                # Model information is now displayed inline in the working viewer
                # No popup needed - dimensions are shown in the status area
                
            except Exception as e:
                error_msg = f"Could not load file: {str(e)}"
                
                # Print error to terminal
                print("=" * 60)
                print("ERROR LOADING CAD FILE:")
                print(error_msg)
                print("Exception type:", type(e).__name__)
                import traceback
                traceback.print_exc()
                print("=" * 60)
                
                QMessageBox.critical(
                    self, 
                    "Error Loading File", 
                    error_msg
                )
                
    def show_dimension_info(self, mesh_info: dict) -> None:
        """Show dimension information dialog to confirm CAD size."""
        dimensions = mesh_info['dimensions']
        detected_units = self.cad_loader.get_input_units()
        
        # Create detailed dimension info
        info_text = f"""
<b>CAD Model Dimensions (Converted to Meters):</b><br><br>
<font size="+1">
Length (X): {dimensions['length']:.4f} m<br>
Width (Y):  {dimensions['width']:.4f} m<br>
Height (Z): {dimensions['height']:.4f} m<br>
</font><br>
<b>Model Statistics:</b><br>
• Vertices: {mesh_info['vertices']:,}<br>
• Faces: {mesh_info['triangles']:,}<br>
• Max Dimension: {mesh_info['max_dimension']:.4f} m<br>
• Volume: {mesh_info['volume']:.6f} m³<br>
• Surface Area: {mesh_info['surface_area']:.6f} m²<br><br>
<font color="blue">
<i>Input units detected: {detected_units}<br>
All dimensions converted to meters for robotics applications.</i>
</font>
        """
        
        # Show information dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("CAD Model Information")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(info_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
                
    def export_annotations(self) -> None:
        """Export annotations to JSON file."""
        if not self.current_file:
            QMessageBox.warning(self, "No Model", "Please load a CAD model first.")
            return
            
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export Annotations",
            f"{self.current_file.stem}_annotations.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Collect annotations from UI in robotics format
                robotics_markers = self.marker_panel.get_markers_for_robotics_export()
                
                # Get common marker properties from first marker (all markers have same values)
                aruco_dictionary = None
                size = None
                border_width = None
                if robotics_markers and len(robotics_markers) > 0:
                    # Get from first marker - all markers have same values in batch processing
                    first_marker_info = list(self.marker_panel.markers.values())[0].aruco_info
                    aruco_dictionary = first_marker_info.dictionary
                    size = first_marker_info.size
                    border_width = first_marker_info.border_width
                
                # Get CAD object info from viewer (common to all markers)
                cad_object_info = None
                if robotics_markers and len(robotics_markers) > 0:
                    # Get CAD info from viewer (more reliable than from marker)
                    cad_info = self.viewer_3d.get_cad_object_info()
                    if cad_info:
                        cad_object_info = {
                            "center": list(cad_info["center"]),
                            "dimensions": cad_info["dimensions"]
                        }
                
                # Create export data with simplified format
                export_data = {
                    "exported_at": datetime.now().isoformat(),
                    "model_file": str(self.current_file) if self.current_file else None,
                    "total_markers": len(robotics_markers),
                    "aruco_dictionary": aruco_dictionary,
                    "size": size,
                    "border_width": border_width,
                    "cad_object_info": cad_object_info,
                    "markers": robotics_markers
                }
                
                # Write to file
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.status_bar.showMessage(f"Annotations exported to: {Path(file_path).name}")
                QMessageBox.information(
                    self, 
                    "Export Complete", 
                    f"Annotations exported successfully to:\n{file_path}"
                )
                
            except Exception as e:
                error_msg = f"Could not export annotations: {str(e)}"
                
                # Print error to terminal
                print("=" * 60)
                print("ERROR EXPORTING ANNOTATIONS:")
                print(error_msg)
                print("Exception type:", type(e).__name__)
                import traceback
                traceback.print_exc()
                print("=" * 60)
                
                QMessageBox.critical(
                    self, 
                    "Export Error", 
                    error_msg
                )
                
    def import_annotations(self) -> None:
        """Import annotations from JSON file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Import Annotations",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Import using annotation manager
                data = self.annotation_manager.import_annotations(Path(file_path))
                
                # Load into UI panels
                self.marker_panel.load_markers(data.get("markers", []))
                
                # Update viewer
                self.viewer_3d.load_annotations(data)
                
                self.status_bar.showMessage(f"Annotations imported from: {Path(file_path).name}")
                
            except Exception as e:
                error_msg = f"Could not import annotations: {str(e)}"
                
                # Print error to terminal
                print("=" * 60)
                print("ERROR IMPORTING ANNOTATIONS:")
                print(error_msg)
                print("Exception type:", type(e).__name__)
                import traceback
                traceback.print_exc()
                print("=" * 60)
                
                QMessageBox.critical(
                    self, 
                    "Import Error", 
                    error_msg
                )
                
    def show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ArUco Marker Annotator",
            """<h3>ArUco Marker Annotator v0.1.0</h3>
            <p>A 3D CAD annotation tool for placing ArUco markers on objects.</p>
            <p><b>Features:</b></p>
            <ul>
            <li>Load STL, OBJ, and PLY files</li>
            <li>Place ArUco markers on 3D objects</li>
            <li>Visual marker editing and positioning</li>
            <li>Export annotations for robotics pipelines</li>
            </ul>
            <p><b>Controls:</b></p>
            <ul>
            <li>Mouse: Rotate, pan, zoom in 3D view</li>
            <li>R: Reset camera view</li>
            <li>F: Fit model to view</li>
            </ul>"""
        )
        
    def closeEvent(self, event) -> None:
        """Handle application close event."""
        # Check if there are unsaved changes
        if self.annotation_manager.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self.export_annotations()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
                
        event.accept()
