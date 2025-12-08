"""Main application for orientation visualizer."""

import sys
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

import open3d as o3d
from scipy.spatial.transform import Rotation

from ros_subscriber import ROS2Thread, ROS2_AVAILABLE


class Open3DWidget(QWidget):
    """Widget for displaying Open3D visualizations."""
    
    def __init__(self, title: str = "3D View"):
        """Initialize the Open3D widget.
        
        Args:
            title: Title for this view
        """
        super().__init__()
        self.title = title
        self.vis = None
        self.geometry = None
        self.coordinate_frame = None
        
        self.init_ui()
    
    def init_ui(self) -> None:
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Title and zoom controls in horizontal layout
        title_layout = QHBoxLayout()
        
        # Zoom out button
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setMaximumWidth(30)
        self.zoom_out_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_layout.addWidget(self.zoom_out_btn)
        
        # Zoom in button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setMaximumWidth(30)
        self.zoom_in_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_layout.addWidget(self.zoom_in_btn)
        
        # Title label
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title_label)
        
        layout.addLayout(title_layout)
        
        # Info label for displaying current orientation
        self.info_label = QLabel("No data")
        self.info_label.setStyleSheet("font-size: 10px; padding: 5px; background-color: #333; color: #fff;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Placeholder for 3D view
        self.view_label = QLabel("3D visualization will appear here")
        self.view_label.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 12px;")
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_label.setMinimumSize(400, 400)
        self.view_label.setScaledContents(False)  # Don't distort aspect ratio
        layout.addWidget(self.view_label)
    
    def update_orientation(self, quaternion: np.ndarray, euler_deg: np.ndarray) -> None:
        """Update the displayed orientation information.
        
        Args:
            quaternion: Rotation as quaternion [x, y, z, w]
            euler_deg: Rotation as Euler angles in degrees [x, y, z]
        """
        info_text = (
            f"Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, "
            f"{quaternion[2]:.3f}, {quaternion[3]:.3f}]\n"
            f"Euler (deg): [Roll: {euler_deg[0]:.1f}°, Pitch: {euler_deg[1]:.1f}°, "
            f"Yaw: {euler_deg[2]:.1f}°]"
        )
        self.info_label.setText(info_text)
    
    def create_coordinate_frame(self, quaternion: np.ndarray, size: float = 0.1) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame mesh with the given orientation.
        
        Args:
            quaternion: Rotation as quaternion [x, y, z, w]
            size: Size of the coordinate frame
            
        Returns:
            Open3D mesh representing the coordinate frame
        """
        # Create coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(quaternion)
        R = rotation.as_matrix()
        
        # Apply rotation
        frame.rotate(R, center=(0, 0, 0))
        
        return frame
    
    def render_to_image(self, quaternion: np.ndarray, geometry = None, zoom_level: float = 0.8) -> None:
        """Render the coordinate frame and optional geometry to an image and display it.
        
        Args:
            quaternion: Rotation as quaternion [x, y, z, w]
            geometry: Optional mesh or line set to render with the orientation
            zoom_level: Zoom level for the camera (default 0.4)
        """
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=800)
        
        # Create and add coordinate frame
        frame = self.create_coordinate_frame(quaternion, size=0.15)
        vis.add_geometry(frame)
        
        # Add geometry if provided
        if geometry is not None:
            # Convert quaternion to rotation matrix
            rotation = Rotation.from_quat(quaternion)
            R = rotation.as_matrix()
            
            # Handle different geometry types
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                # Clone the mesh so we don't modify the original
                geom_copy = o3d.geometry.TriangleMesh(geometry)
                geom_copy.rotate(R, center=(0, 0, 0))
                
                # Add the solid mesh
                vis.add_geometry(geom_copy)
                
                # Create wireframe edges for better visualization
                # Method 1: Standard wireframe
                wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(geom_copy)
                wireframe.paint_uniform_color([0.0, 0.0, 0.0])  # Black edges
                
                # Method 2: Create thicker lines by duplicating and offsetting
                wireframe_thick = o3d.geometry.LineSet(wireframe)
                wireframe_thick.scale(1.002, center=(0, 0, 0))
                wireframe_thick.paint_uniform_color([0.0, 0.0, 0.0])
                
                vis.add_geometry(wireframe)
                vis.add_geometry(wireframe_thick)
                
            elif isinstance(geometry, o3d.geometry.LineSet):
                # Clone the line set
                geom_copy = o3d.geometry.LineSet(geometry)
                geom_copy.rotate(R, center=(0, 0, 0))
                vis.add_geometry(geom_copy)
        
        # Set rendering options for better line visibility
        render_option = vis.get_render_option()
        render_option.line_width = 2.0  # Make lines thicker
        render_option.background_color = np.array([0.9, 0.9, 0.9])  # Light gray background
        
        # Set view control
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_front([0.5, 0.5, 0.5])
        ctr.set_up([0, 0, 1])
        
        # Use the provided zoom level
        ctr.set_zoom(zoom_level)
        
        # Update renderer
        vis.poll_events()
        vis.update_renderer()
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        
        # Capture and save image
        image_path = f"/tmp/orientation_{self.title.replace(' ', '_')}.png"
        vis.capture_screen_image(image_path, do_render=True)
        vis.destroy_window()
        
        # Load and display image
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap(image_path)
        
        # Scale to fit the widget while maintaining aspect ratio
        label_size = self.view_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.view_label.setPixmap(scaled_pixmap)


class OrientationVisualizerApp(QMainWindow):
    """Main application window for the orientation visualizer."""
    
    update_signal = pyqtSignal(dict)
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        self.assembly_data: Optional[Dict] = None
        self.ros_thread: Optional[ROS2Thread] = None
        self.current_object: Optional[str] = None
        self.object_components: Dict[str, Dict] = {}
        self.current_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.zoom_level: float = 0.8  # Default zoom level
        
        self.init_ui()
        self.setup_ros()
        
        # Timer for updating visualization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_current_pose)
        self.update_timer.start(100)  # Update every 100ms
        
        # Connect signal
        self.update_signal.connect(self.on_pose_update)
        
        # Auto-load the default assembly JSON file
        self.auto_load_assembly()
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Object Orientation Visualizer")
        self.setGeometry(100, 100, 1200, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Split view for left (final) and right (current) orientations
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left view - Final Assembly Orientation
        self.final_view = Open3DWidget("Final Assembly Orientation")
        
        # Connect zoom buttons to final view
        self.final_view.zoom_out_btn.clicked.connect(self.zoom_out)
        self.final_view.zoom_in_btn.clicked.connect(self.zoom_in)
        
        splitter.addWidget(self.final_view)
        
        # Right view - Current Live Orientation
        self.current_view = Open3DWidget("Current Live Orientation")
        
        # Hide zoom buttons on right panel (only show on left panel)
        self.current_view.zoom_out_btn.hide()
        self.current_view.zoom_in_btn.hide()
        
        splitter.addWidget(self.current_view)
        
        # Equal sizing
        splitter.setSizes([600, 600])
        
        # TODO: Fix the UI to make sure the final and current assembly panels 
        # start at the same vertical line. Currently there might be slight 
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready. Load assembly JSON and select an object.")
    
    def create_control_panel(self) -> QWidget:
        """Create the control panel widget.
        
        Returns:
            Control panel widget
        """
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Load assembly button
        load_btn = QPushButton("Load Assembly JSON")
        load_btn.clicked.connect(self.load_assembly)
        layout.addWidget(load_btn)
        
        # Object selection
        layout.addWidget(QLabel("Select Object:"))
        self.object_combo = QComboBox()
        self.object_combo.currentTextChanged.connect(self.on_object_changed)
        layout.addWidget(self.object_combo)
        
        # ROS topic input
        layout.addWidget(QLabel("ROS Topic:"))
        self.topic_combo = QComboBox()
        self.topic_combo.setEditable(True)
        self.topic_combo.addItems(['/objects_poses_sim', '/objects_poses'])
        self.topic_combo.currentTextChanged.connect(self.on_topic_changed)
        layout.addWidget(self.topic_combo)
        
        # Reconnect button
        reconnect_btn = QPushButton("Reconnect ROS")
        reconnect_btn.clicked.connect(self.setup_ros)
        layout.addWidget(reconnect_btn)
        
        layout.addStretch()
        
        return panel
    
    def auto_load_assembly(self) -> None:
        """Automatically load the default assembly JSON file."""
        # Try to find the assembly file in common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "data" / "fmb_assembly.json",
            Path(__file__).parent.parent.parent.parent / "data" / "fmb_assembly.json",
            Path("/home/aaugus11/Projects/aruco-grasp-annotator/data/fmb_assembly.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        self.assembly_data = json.load(f)
                    
                    # Extract component objects (not markers)
                    self.object_components.clear()
                    for component in self.assembly_data.get('components', []):
                        if component.get('type') == 'component':
                            name = component.get('name', '')
                            self.object_components[name] = component
                    
                    # Update combo box
                    self.object_combo.clear()
                    self.object_combo.addItems(sorted(self.object_components.keys()))
                    
                    self.statusBar().showMessage(f"Auto-loaded assembly from {path.name}")
                    
                    # Automatically select first object
                    if self.object_combo.count() > 0:
                        self.object_combo.setCurrentIndex(0)
                    
                    return
                except Exception as e:
                    print(f"Failed to auto-load assembly from {path}: {e}")
                    continue
        
        # If no assembly file found, show message
        self.statusBar().showMessage("No assembly file found. Click 'Load Assembly JSON' to load manually.")
    
    def load_assembly(self) -> None:
        """Load assembly JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Assembly JSON",
            str(Path.home()),
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                self.assembly_data = json.load(f)
            
            # Extract component objects (not markers)
            self.object_components.clear()
            for component in self.assembly_data.get('components', []):
                if component.get('type') == 'component':
                    name = component.get('name', '')
                    self.object_components[name] = component
            
            # Update combo box
            self.object_combo.clear()
            self.object_combo.addItems(sorted(self.object_components.keys()))
            
            self.statusBar().showMessage(f"Loaded assembly from {Path(file_path).name}")
            
            # Automatically select first object
            if self.object_combo.count() > 0:
                self.object_combo.setCurrentIndex(0)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load assembly: {str(e)}")
    
    def setup_ros(self) -> None:
        """Setup ROS2 subscriber."""
        if not ROS2_AVAILABLE:
            self.statusBar().showMessage("ROS2 not available. Install rclpy and tf2_msgs.")
            return
        
        # Stop existing thread if any
        if self.ros_thread:
            self.ros_thread.stop()
            self.ros_thread.join(timeout=2)
        
        # Start new thread
        topic_name = self.topic_combo.currentText()
        self.ros_thread = ROS2Thread(topic_name, self.ros_callback)
        self.ros_thread.start()
        
        self.statusBar().showMessage(f"Connected to ROS topic: {topic_name}")
    
    def on_topic_changed(self, topic: str) -> None:
        """Handle topic change.
        
        Args:
            topic: New topic name
        """
        self.setup_ros()
    
    def ros_callback(self, poses: Dict) -> None:
        """Callback for ROS2 pose updates.
        
        Args:
            poses: Dictionary of object poses
        """
        # Emit signal to update UI in main thread
        self.update_signal.emit(poses)
    
    def on_pose_update(self, poses: Dict) -> None:
        """Handle pose update from ROS2.
        
        Args:
            poses: Dictionary of object poses
        """
        # This runs in the main thread
        pass  # The update happens in update_current_pose via timer
    
    def load_object_model(self, object_name: str) -> None:
        """Load the 3D OBJ model for the specified object.
        
        Args:
            object_name: Name of the object to load
        """
        # Try to find the OBJ model file
        possible_paths = [
            Path(__file__).parent.parent.parent / "data" / "models" / f"{object_name}.obj",
            Path(__file__).parent.parent.parent.parent / "data" / "models" / f"{object_name}.obj",
            Path(f"/home/aaugus11/Projects/aruco-grasp-annotator/data/models/{object_name}.obj")
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    # Load OBJ file using Open3D
                    self.current_mesh = o3d.io.read_triangle_mesh(str(path))
                    if not self.current_mesh.has_vertex_normals():
                        self.current_mesh.compute_vertex_normals()
                    # Paint it with a nice color
                    self.current_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                    self.statusBar().showMessage(f"Loaded model: {path.name}")
                    return
                except Exception as e:
                    print(f"Failed to load model from {path}: {e}")
                    continue
        
        # If no model found, clear current mesh
        self.current_mesh = None
        print(f"No OBJ model found for {object_name}")
    
    def zoom_in(self) -> None:
        """Zoom in by decreasing the zoom level."""
        self.zoom_level -= 0.1
        
        # Update both views
        self.update_final_pose()
        self.update_current_pose()
    
    def zoom_out(self) -> None:
        """Zoom out by increasing the zoom level."""
        self.zoom_level += 0.1
        
        # Update both views
        self.update_final_pose()
        self.update_current_pose()
    
    def on_object_changed(self, object_name: str) -> None:
        """Handle object selection change.
        
        Args:
            object_name: Name of the selected object
        """
        if not object_name or object_name not in self.object_components:
            return
        
        self.current_object = object_name
        
        # Load the 3D model
        self.load_object_model(object_name)
        
        self.update_final_pose()
        self.update_current_pose()
        
        self.statusBar().showMessage(f"Monitoring: {object_name}")
    
    def update_final_pose(self) -> None:
        """Update the final (assembly) pose visualization."""
        if not self.current_object or not self.object_components:
            return
        
        component = self.object_components[self.current_object]
        rotation_euler = component.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        
        # Convert Euler angles (XYZ) to quaternion
        euler_rad = np.array([
            rotation_euler.get('x', 0),
            rotation_euler.get('y', 0),
            rotation_euler.get('z', 0)
        ])
        
        rotation = Rotation.from_euler('xyz', euler_rad, degrees=False)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        euler_deg = np.degrees(euler_rad)
        
        # Update display
        self.final_view.update_orientation(quaternion, euler_deg)
        self.final_view.render_to_image(quaternion, self.current_mesh, self.zoom_level)
    
    def map_object_name_to_ros(self, object_name: str) -> Optional[str]:
        """Map assembly object name to ROS topic name.
        
        TODO: Update the names of all wireframe, models, and assembly JSON files 
        to match the ROS2 topic names (remove '_scaled70' suffix). This will eliminate 
        the need for this name mapping function.
        Files to rename:
        - data/models/*.obj (e.g., base_scaled70.obj -> base.obj)
        - data/wireframe/*.json (e.g., base_scaled70_wireframe.json -> base_wireframe.json)
        - Update assembly JSON component names to match ROS topic names
        
        Args:
            object_name: Object name from assembly (e.g., 'base_scaled70')
            
        Returns:
            Mapped ROS object name (e.g., 'base') or None if not found
        """
        if not self.ros_thread:
            return None
        
        available_objects = self.ros_thread.get_all_objects()
        
        # Try direct match first
        if object_name in available_objects:
            return object_name
        
        # Try removing '_scaled70' suffix
        if object_name.endswith('_scaled70'):
            base_name = object_name.replace('_scaled70', '')
            if base_name in available_objects:
                return base_name
        
        # Try removing any '_scaledXX' pattern
        import re
        base_name = re.sub(r'_scaled\d+$', '', object_name)
        if base_name in available_objects:
            return base_name
        
        return None
    
    def update_current_pose(self) -> None:
        """Update the current (live) pose visualization."""
        if not self.current_object:
            self.statusBar().showMessage("No object selected")
            return
            
        if not self.ros_thread:
            self.statusBar().showMessage("ROS not connected")
            return
        
        # Get available objects from ROS
        available_objects = self.ros_thread.get_all_objects()
        
        # Map object name to ROS name
        ros_object_name = self.map_object_name_to_ros(self.current_object)
        
        if not ros_object_name:
            # Try to show helpful message
            if available_objects:
                self.statusBar().showMessage(
                    f"Object '{self.current_object}' not found in ROS. "
                    f"Available: {', '.join(available_objects)}"
                )
            else:
                self.statusBar().showMessage(f"No data from ROS topic yet...")
            return
        
        # Get pose using mapped name
        pose = self.ros_thread.get_pose(ros_object_name)
        if not pose:
            return
        
        quaternion = pose['quaternion']  # [x, y, z, w]
        
        # Convert to Euler for display
        rotation = Rotation.from_quat(quaternion)
        euler_rad = rotation.as_euler('xyz', degrees=False)
        euler_deg = np.degrees(euler_rad)
        
        # Update display
        self.current_view.update_orientation(quaternion, euler_deg)
        self.current_view.render_to_image(quaternion, self.current_mesh, self.zoom_level)
        
        # Update status to show we're receiving data
        self.statusBar().showMessage(f"Monitoring: {self.current_object} → {ros_object_name} (receiving data)")
    
    def closeEvent(self, event) -> None:
        """Handle window close event.
        
        Args:
            event: Close event
        """
        if self.ros_thread:
            self.ros_thread.stop()
            self.ros_thread.join(timeout=2)
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    window = OrientationVisualizerApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

