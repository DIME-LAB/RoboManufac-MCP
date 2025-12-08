"""3D viewer widget using Open3D for CAD model visualization."""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import threading
import time

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class Viewer3D(QWidget):
    """3D viewer widget for CAD models and annotations."""
    
    # Signals
    marker_clicked = pyqtSignal(int)  # marker_id
    grasp_clicked = pyqtSignal(int)   # grasp_id
    point_picked = pyqtSignal(tuple)  # (x, y, z)
    face_selected = pyqtSignal(tuple, tuple)  # (point, normal)
    
    def __init__(self) -> None:
        super().__init__()
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.grasp_poses: Dict[int, Dict[str, Any]] = {}
        self.coordinate_frame: Optional[o3d.geometry.TriangleMesh] = None
        
        # Interaction state
        self.placement_mode = False
        self.dragging_marker = None
        self.drag_start_point = None
        self.selected_marker_id = None
        
        # Open3D visualizer (will be created in separate thread)
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.vis_thread: Optional[QThread] = None
        
        self.init_ui()
        self.setup_viewer()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create Open3D visualizer widget
        self.setup_open3d_widget()
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_btn)
        
        self.fit_btn = QPushButton("Fit to View")
        self.fit_btn.clicked.connect(self.fit_to_view)
        controls_layout.addWidget(self.fit_btn)
        
        controls_layout.addStretch()
        
        # Mode indicator
        self.mode_label = QLabel("Mode: View")
        self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
        controls_layout.addWidget(self.mode_label)
        
        layout.addLayout(controls_layout)
        
    def setup_open3d_widget(self) -> None:
        """Setup the Open3D visualization widget."""
        # Create container widget for Open3D viewer
        self.viewer_widget = QWidget()
        self.viewer_widget.setMinimumSize(600, 400)
        self.viewer_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        
        # Add a label indicating this is the 3D view area
        view_layout = QVBoxLayout(self.viewer_widget)
        self.status_label = QLabel("3D Viewer\\n\\nLoad a CAD file to begin")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 16px;
                border: none;
            }
        """)
        view_layout.addWidget(self.status_label)
        
        self.layout().addWidget(self.viewer_widget)
        
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True)
        
        # Setup callbacks for interaction
        self.setup_interaction_callbacks()
        
        # Setup default view
        self.setup_default_view()
        
    def setup_viewer(self) -> None:
        """Setup the 3D viewer with default settings."""
        # Add coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coordinate_frame)
        
        # Setup lighting and camera
        self.reset_camera()
        
    def setup_interaction_callbacks(self) -> None:
        """Setup interaction callbacks for the Open3D visualizer."""
        def mouse_callback(vis, action, mods):
            """Handle mouse interactions."""
            if action == o3d.visualization.VisualizerWithKeyCallback.MOUSE_BUTTON_LEFT:
                if self.placement_mode:
                    # Get picking point
                    point = self.get_pick_point_from_visualizer()
                    if point is not None:
                        self.point_picked.emit(point)
                        self.placement_mode = False
                        self.set_interaction_mode("view")
                elif self.dragging_marker is not None:
                    # End dragging
                    self.dragging_marker = None
                    self.drag_start_point = None
                else:
                    # Check if clicking on a marker
                    marker_id = self.get_clicked_marker()
                    if marker_id is not None:
                        self.marker_clicked.emit(marker_id)
                        self.selected_marker_id = marker_id
                        # Start dragging mode
                        self.dragging_marker = marker_id
                        self.drag_start_point = self.get_pick_point_from_visualizer()
            
            return False
            
        def key_callback(vis, key, action):
            """Handle keyboard interactions."""
            if key == ord('P') and action == o3d.visualization.VisualizerWithKeyCallback.KEY_DOWN:
                # Toggle placement mode
                self.placement_mode = not self.placement_mode
                mode = "place_marker" if self.placement_mode else "view"
                self.set_interaction_mode(mode)
            elif key == ord('R') and action == o3d.visualization.VisualizerWithKeyCallback.KEY_DOWN:
                self.reset_camera()
            elif key == ord('F') and action == o3d.visualization.VisualizerWithKeyCallback.KEY_DOWN:
                self.fit_to_view()
            return False

        # Note: Open3D callbacks would be registered here in a full implementation
        # For now, we provide the simulation button approach in the working viewer
        
    def setup_default_view(self) -> None:
        """Setup default view settings."""
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
        render_opt.point_size = 3.0
        render_opt.line_width = 2.0
        render_opt.show_coordinate_frame = True
        
    def load_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """Load a 3D mesh into the viewer."""
        # Remove previous mesh if exists
        if self.mesh is not None:
            self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
            
        # Prepare mesh (Open3D doesn't have copy method, create a new mesh)
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = mesh.vertices
        self.mesh.triangles = mesh.triangles
        if mesh.has_vertex_normals():
            self.mesh.vertex_normals = mesh.vertex_normals
        if mesh.has_vertex_colors():
            self.mesh.vertex_colors = mesh.vertex_colors
        
        # Compute normals if not present
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
            
        # Set default color
        if not self.mesh.has_vertex_colors():
            self.mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
        # Add to visualizer
        self.vis.add_geometry(self.mesh)
        
        # Update the status label
        self.status_label.setText(f"Loaded mesh:\\n{len(self.mesh.vertices)} vertices\\n{len(self.mesh.triangles)} faces\\n\\nPress 'P' to place markers")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 14px;
                border: none;
            }
        """)
        
        # Fit to view
        self.fit_to_view()
        
    def create_aruco_marker_geometry(self, position: tuple, size: float, marker_id: int) -> o3d.geometry.TriangleMesh:
        """Create a visual representation of an ArUco marker."""
        x, y, z = position
        
        # Create the main marker base (flat rectangle)
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/20)
        marker_mesh.translate([x - size/2, y - size/2, z - size/40])
        
        # Paint it white as the base
        marker_mesh.paint_uniform_color([0.9, 0.9, 0.9])
        
        # Create a black border (slightly smaller black rectangle on top)
        border_size = size * 0.9
        border_mesh = o3d.geometry.TriangleMesh.create_box(border_size, border_size, size/30)
        border_mesh.translate([x - border_size/2, y - border_size/2, z])
        border_mesh.paint_uniform_color([0.1, 0.1, 0.1])
        
        # Create inner white square (representing the ArUco pattern area)
        inner_size = size * 0.7
        inner_mesh = o3d.geometry.TriangleMesh.create_box(inner_size, inner_size, size/25)
        inner_mesh.translate([x - inner_size/2, y - inner_size/2, z + size/60])
        inner_mesh.paint_uniform_color([0.95, 0.95, 0.95])
        
        # Create a simple pattern (representing ArUco squares)
        pattern_cubes = []
        pattern_size = inner_size / 5  # 5x5 grid like ArUco
        for i in range(5):
            for j in range(5):
                if (i + j + marker_id) % 2 == 0:  # Simple pattern based on marker ID
                    cube = o3d.geometry.TriangleMesh.create_box(pattern_size*0.8, pattern_size*0.8, size/20)
                    cube_x = x - inner_size/2 + i * pattern_size + pattern_size/2 - pattern_size*0.4
                    cube_y = y - inner_size/2 + j * pattern_size + pattern_size/2 - pattern_size*0.4
                    cube.translate([cube_x, cube_y, z + size/50])
                    cube.paint_uniform_color([0.2, 0.2, 0.2])
                    pattern_cubes.append(cube)
        
        # Combine all meshes
        combined_mesh = marker_mesh + border_mesh + inner_mesh
        for cube in pattern_cubes:
            combined_mesh += cube
            
        # Add a small coordinate frame to show orientation
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size/4)
        coord_frame.translate([x, y, z + size/10])
        combined_mesh += coord_frame
        
        return combined_mesh

    def add_marker(self, marker_id: int, position: tuple, size: float = 0.05) -> None:
        """Add an ArUco marker visualization at the specified position."""
        # Create the ArUco marker geometry
        marker_mesh = self.create_aruco_marker_geometry(position, size, marker_id)
        
        # Store marker
        self.markers[marker_id] = {
            'geometry': marker_mesh,
            'position': position,
            'size': size,
            'is_selected': False
        }
        
        # Add to visualizer
        self.vis.add_geometry(marker_mesh)
        self.vis.update_geometry(marker_mesh)
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def remove_marker(self, marker_id: int) -> None:
        """Remove a marker from the viewer."""
        if marker_id in self.markers:
            marker = self.markers[marker_id]
            self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            del self.markers[marker_id]
            
    def select_marker(self, marker_id: int) -> None:
        """Highlight the selected marker."""
        # Update selection state and visual feedback
        for mid, marker in self.markers.items():
            marker['is_selected'] = (mid == marker_id)
            
            if marker['is_selected']:
                # Recreate marker with selection highlight
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
                highlighted_mesh = self.create_aruco_marker_geometry(
                    marker['position'], marker['size'], mid
                )
                # Add yellow highlight border
                x, y, z = marker['position']
                size = marker['size']
                highlight_border = o3d.geometry.TriangleMesh.create_box(
                    size * 1.1, size * 1.1, size/15
                )
                highlight_border.translate([x - size * 0.55, y - size * 0.55, z - size/30])
                highlight_border.paint_uniform_color([1, 1, 0])  # Yellow
                highlighted_mesh = highlight_border + highlighted_mesh
                
                marker['geometry'] = highlighted_mesh
                self.vis.add_geometry(highlighted_mesh)
                
        self.selected_marker_id = marker_id
        self.vis.poll_events()
        self.vis.update_renderer()
            
    def add_grasp_pose(self, grasp_id: int, marker_id: int, position: tuple, orientation: tuple) -> None:
        """Add a grasp pose visualization."""
        x, y, z = position
        
        # Create coordinate frame for grasp pose
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        
        # Apply transformation (simplified - in real implementation, use quaternion)
        grasp_frame.translate([x, y, z])
        
        # Create approach vector (arrow)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002,
            cone_radius=0.005,
            cylinder_height=0.03,
            cone_height=0.01
        )
        arrow.translate([x, y, z + 0.02])
        arrow.paint_uniform_color([0, 1, 0])  # Green
        
        # Store grasp pose
        self.grasp_poses[grasp_id] = {
            'frame': grasp_frame,
            'arrow': arrow,
            'marker_id': marker_id,
            'position': position,
            'orientation': orientation
        }
        
        # Add to visualizer
        self.vis.add_geometry(grasp_frame)
        self.vis.add_geometry(arrow)
        
    def remove_grasp_pose(self, grasp_id: int) -> None:
        """Remove a grasp pose from the viewer."""
        if grasp_id in self.grasp_poses:
            grasp = self.grasp_poses[grasp_id]
            self.vis.remove_geometry(grasp['frame'], reset_bounding_box=False)
            self.vis.remove_geometry(grasp['arrow'], reset_bounding_box=False)
            del self.grasp_poses[grasp_id]
            
    def select_grasp_pose(self, grasp_id: int) -> None:
        """Highlight the selected grasp pose."""
        for gid, grasp in self.grasp_poses.items():
            # Change arrow color based on selection
            color = [1, 1, 0] if gid == grasp_id else [0, 1, 0]  # Yellow if selected
            grasp['arrow'].paint_uniform_color(color)
            self.vis.update_geometry(grasp['arrow'])
            
    def set_background(self, bg_type: str) -> None:
        """Set the background color/style."""
        render_opt = self.vis.get_render_option()
        
        if bg_type == "Dark":
            render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
        elif bg_type == "Light":
            render_opt.background_color = np.asarray([0.9, 0.9, 0.9])
        elif bg_type == "Gradient":
            render_opt.background_color = np.asarray([0.2, 0.2, 0.3])
            
    def show_axes(self, show: bool) -> None:
        """Show or hide coordinate axes."""
        if show and self.coordinate_frame not in self.vis.get_geometry():
            self.vis.add_geometry(self.coordinate_frame)
        elif not show and self.coordinate_frame in self.vis.get_geometry():
            self.vis.remove_geometry(self.coordinate_frame, reset_bounding_box=False)
            
    def show_grid(self, show: bool) -> None:
        """Show or hide grid (placeholder for now)."""
        # In a full implementation, you'd add/remove a grid geometry
        pass
        
    def set_wireframe(self, wireframe: bool) -> None:
        """Set wireframe mode."""
        if self.mesh is not None:
            render_opt = self.vis.get_render_option()
            if wireframe:
                render_opt.mesh_show_wireframe = True
                render_opt.mesh_show_back_face = True
            else:
                render_opt.mesh_show_wireframe = False
                render_opt.mesh_show_back_face = False
                
    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self.vis.reset_view_point(True)
        
    def fit_to_view(self) -> None:
        """Fit the current model to the view."""
        if self.mesh is not None:
            # Get the bounding box and set camera to look at it
            bbox = self.mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            max_extent = np.max(extent)
            
            # Set camera parameters to fit the model
            view_control = self.vis.get_view_control()
            view_control.set_lookat(center)
            view_control.set_zoom(0.8)  # Adjust zoom to fit
            
    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotations from data dictionary."""
        # Clear existing annotations
        for marker_id in list(self.markers.keys()):
            self.remove_marker(marker_id)
        for grasp_id in list(self.grasp_poses.keys()):
            self.remove_grasp_pose(grasp_id)
            
        # Load markers
        for marker_data in data.get("markers", []):
            self.add_marker(
                marker_data["id"],
                tuple(marker_data["position"]),
                marker_data.get("size", 0.05)
            )
            
        # Load grasp poses
        for grasp_data in data.get("grasp_poses", []):
            self.add_grasp_pose(
                grasp_data["id"],
                grasp_data["marker_id"],
                tuple(grasp_data["position"]),
                tuple(grasp_data["orientation"])
            )
            
    def get_pick_point(self, x: int, y: int) -> Optional[tuple]:
        """Get the 3D point at screen coordinates."""
        # Use the same approach as get_pick_point_from_visualizer for consistency
        return self.get_pick_point_from_visualizer()
        
    def set_interaction_mode(self, mode: str) -> None:
        """Set the interaction mode (view, place_marker, place_grasp)."""
        self.mode_label.setText(f"Mode: {mode}")
        
        if mode == "view":
            self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
            self.placement_mode = False
        elif mode == "place_marker":
            self.mode_label.setStyleSheet("font-weight: bold; color: red;")
            self.placement_mode = True
        elif mode == "place_grasp":
            self.mode_label.setStyleSheet("font-weight: bold; color: green;")
            self.placement_mode = False
            
    def get_pick_point_from_visualizer(self) -> Optional[tuple]:
        """Get a point on the mesh surface for marker placement."""
        if self.mesh is not None:
            # Pick a random vertex from the mesh (ensures it's on the surface)
            vertices = np.asarray(self.mesh.vertices)
            if len(vertices) > 0:
                import random
                random_vertex_idx = random.randint(0, len(vertices) - 1)
                return tuple(vertices[random_vertex_idx])
            else:
                # Fallback to bounding box center
                bbox = self.mesh.get_axis_aligned_bounding_box()
                return tuple(bbox.get_center())
        return None
        
    def get_clicked_marker(self) -> Optional[int]:
        """Check if the user clicked on a marker and return its ID."""
        # This is a simplified implementation
        # In a real application, you'd use proper 3D picking
        return None
        
    def move_marker(self, marker_id: int, new_position: tuple) -> None:
        """Move a marker to a new position."""
        if marker_id in self.markers:
            marker = self.markers[marker_id]
            
            # Remove old geometry
            self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            
            # Update position
            marker['position'] = new_position
            
            # Create new geometry at new position
            new_geometry = self.create_aruco_marker_geometry(
                new_position, marker['size'], marker_id
            )
            
            # Add highlight if selected
            if marker['is_selected']:
                x, y, z = new_position
                size = marker['size']
                highlight_border = o3d.geometry.TriangleMesh.create_box(
                    size * 1.1, size * 1.1, size/15
                )
                highlight_border.translate([x - size * 0.55, y - size * 0.55, z - size/30])
                highlight_border.paint_uniform_color([1, 1, 0])  # Yellow
                new_geometry = highlight_border + new_geometry
            
            marker['geometry'] = new_geometry
            self.vis.add_geometry(new_geometry)
            self.vis.poll_events()
            self.vis.update_renderer()
            
    def enable_placement_mode(self) -> None:
        """Enable marker placement mode."""
        self.set_interaction_mode("place_marker")
        
    def disable_placement_mode(self) -> None:
        """Disable marker placement mode."""
        self.set_interaction_mode("view")
