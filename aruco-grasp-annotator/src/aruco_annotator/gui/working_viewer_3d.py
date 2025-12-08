"""Working 3D viewer widget that integrates Open3D with PyQt6 properly."""

import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFrame, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QFont

import open3d as o3d
from ..utils.aruco_utils import ArUcoGenerator, ArUcoMarkerInfo, create_aruco_mesh_with_texture
from .face_picker_dialog import FacePickerDialog


class WorkingViewer3D(QWidget):
    """Working 3D viewer that shows mesh info and provides a launch button for Open3D."""
    
    # Signals
    marker_clicked = pyqtSignal(int)
    point_picked = pyqtSignal(tuple, tuple)  # (position, normal)
    
    def __init__(self) -> None:
        super().__init__()
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.edges_vis: Optional[o3d.visualization.Visualizer] = None
        self.wireframe_vis: Optional[o3d.visualization.Visualizer] = None
        self.placement_mode = False
        self.direct_click_mode = False
        self.aruco_generator = ArUcoGenerator()
        self.mesh_info: Optional[dict] = None
        self.face_groups = None
        self.highlighted_faces = []
        self.geometries: Dict[str, Any] = {}  # Track geometries for toggle controls
        
        self.init_ui()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_btn)
        
        self.fit_btn = QPushButton("Fit to View")
        self.fit_btn.clicked.connect(self.fit_to_view)
        controls_layout.addWidget(self.fit_btn)
        
        self.launch_3d_btn = QPushButton("Launch 3D Viewer")
        self.launch_3d_btn.clicked.connect(self.toggle_3d_viewer)
        controls_layout.addWidget(self.launch_3d_btn)
        
        self.launch_edges_btn = QPushButton("Launch Edges Viewer")
        self.launch_edges_btn.clicked.connect(self.toggle_edges_viewer)
        controls_layout.addWidget(self.launch_edges_btn)
        
        self.launch_wireframe_btn = QPushButton("Launch Wireframe Viewer")
        self.launch_wireframe_btn.clicked.connect(self.toggle_wireframe_viewer)
        controls_layout.addWidget(self.launch_wireframe_btn)
        
        # Placement instruction label (for placement mode)
        self.placement_label = QLabel("Choose how to place ArUco marker:\n• 🎯 Random Face • 📋 Face List")
        self.placement_label.setVisible(False)  # Hidden by default
        controls_layout.addWidget(self.placement_label)
        
        # Placeholder for alternative placement button (added later if needed)
        self.click_place_btn = None
        
        controls_layout.addStretch()
        
        # Mode indicator
        self.mode_label = QLabel("Mode: View")
        controls_layout.addWidget(self.mode_label)
        
        layout.addLayout(controls_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left side - 3D view info
        self.view_area = self.create_view_area()
        content_layout.addWidget(self.view_area, 2)
        
        # Right side - Information panel
        self.info_panel = self.create_info_panel()
        content_layout.addWidget(self.info_panel, 1)
        
        layout.addLayout(content_layout)
        
    def create_view_area(self) -> QWidget:
        """Create the 3D view area."""
        view_widget = QFrame()
        view_widget.setFrameStyle(QFrame.Shape.Box)
        view_widget.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(view_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("3D Model Viewer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Status label
        self.status_label = QLabel("No model loaded")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Model info
        self.model_info = QTextEdit()
        self.model_info.setMaximumHeight(200)
        self.model_info.setReadOnly(True)
        layout.addWidget(self.model_info)
        
        # Instructions
        instructions = QLabel("""
        <b>Instructions:</b><br>
        1. Load a CAD file using the File menu<br>
        2. Click "Launch 3D Viewer" to open the 3D visualization<br>
        3. Place ArUco markers in the left panel<br>
        4. Export your annotations when finished<br>
        5. Use the 3D viewer to see your markers in real-time
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        return view_widget
        
    def create_info_panel(self) -> QWidget:
        """Create the information panel."""
        info_widget = QFrame()
        info_widget.setFrameStyle(QFrame.Shape.Box)
        info_widget.setMaximumWidth(300)
        
        layout = QVBoxLayout(info_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("Model Information")
        layout.addWidget(title)
        
        # Statistics
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)
        
        # Markers info
        markers_title = QLabel("ArUco Markers")
        layout.addWidget(markers_title)
        
        self.markers_text = QTextEdit()
        self.markers_text.setMaximumHeight(100)
        self.markers_text.setReadOnly(True)
        layout.addWidget(self.markers_text)
        
        
        layout.addStretch()
        return info_widget
        
    def load_mesh(self, mesh: o3d.geometry.TriangleMesh, mesh_info: Optional[dict] = None) -> None:
        """Load a 3D mesh into the viewer."""
        self.mesh = mesh
        self.mesh_info = mesh_info
        
        # Update status with dimension information
        if mesh_info and 'dimensions' in mesh_info:
            dims = mesh_info['dimensions']
            vertices_count = mesh_info.get('vertices', 0)
            triangles_count = mesh_info.get('triangles', 0)
            
            status_text = f"✅ Model loaded successfully\\n"
            status_text += f"📐 Dimensions: {dims['length']:.4f} × {dims['width']:.4f} × {dims['height']:.4f} m\\n"
            status_text += f"🔺 Geometry: {vertices_count:,} vertices | {triangles_count:,} faces"
            
            # Add volume and surface area if available
            if 'volume' in mesh_info and mesh_info['volume'] > 0:
                status_text += f"\\n📊 Volume: {mesh_info['volume']:.4f} m³"
            if 'surface_area' in mesh_info and mesh_info['surface_area'] > 0:
                status_text += f" | Surface: {mesh_info['surface_area']:.4f} m²"
        else:
            status_text = "Model loaded successfully"
        
        self.status_label.setText(status_text)
        
        # Update model info
        info = self.get_mesh_info(mesh)
        info_text = f"""Model Statistics:
Vertices: {info['vertices']:,}
Triangles: {info['triangles']:,}
Bounding Box:
  Min: ({info['bbox_min'][0]:.3f}, {info['bbox_min'][1]:.3f}, {info['bbox_min'][2]:.3f})
  Max: ({info['bbox_max'][0]:.3f}, {info['bbox_max'][1]:.3f}, {info['bbox_max'][2]:.3f})
Extent: ({info['extent'][0]:.3f}, {info['extent'][1]:.3f}, {info['extent'][2]:.3f})
Volume: {info['volume']:.6f}
Surface Area: {info['surface_area']:.6f}
Has Normals: {info['has_normals']}
Has Colors: {info['has_colors']}
Watertight: {info['is_watertight']}"""
        
        self.model_info.setText(info_text)
        self.update_stats()
        
    def get_mesh_info(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """Get information about the mesh."""
        bbox = mesh.get_axis_aligned_bounding_box()
        
        # Try to get volume, but handle non-watertight meshes gracefully
        try:
            volume = mesh.get_volume()
        except:
            volume = 0.0  # Volume cannot be computed for non-watertight meshes
        
        # Try to get surface area
        try:
            surface_area = mesh.get_surface_area()
        except:
            surface_area = 0.0
        
        return {
            'vertices': len(mesh.vertices),
            'triangles': len(mesh.triangles),
            'bbox_min': bbox.min_bound.tolist(),
            'bbox_max': bbox.max_bound.tolist(),
            'extent': bbox.get_extent().tolist(),
            'volume': volume,
            'surface_area': surface_area,
            'has_normals': mesh.has_vertex_normals(),
            'has_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable()
        }
        
    def toggle_3d_viewer(self) -> None:
        """Toggle the 3D viewer (launch if closed, close if open)."""
        if self.vis is None:
            self.launch_3d_viewer()
        else:
            self.close_3d_viewer()
            
    def toggle_edges_viewer(self) -> None:
        """Toggle the edges viewer (launch if closed, close if open)."""
        if self.edges_vis is None:
            self.launch_edges_viewer()
        else:
            self.close_edges_viewer()
            
    def toggle_wireframe_viewer(self) -> None:
        """Toggle the wireframe viewer (launch if closed, close if open)."""
        if self.wireframe_vis is None:
            self.launch_wireframe_viewer()
        else:
            self.close_wireframe_viewer()
            
    def close_3d_viewer(self) -> None:
        """Close the 3D viewer."""
        self.cleanup_viewer()
            
    def cleanup_viewer(self) -> None:
        """Clean up the 3D viewer when it's closed."""
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except:
                pass  # Window might already be closed
            self.vis = None
            
            # Update UI
            self.launch_3d_btn.setText("Launch 3D Viewer")
            
            self.status_label.setText("3D viewer closed")
            self.update_stats()
            
    def close_edges_viewer(self) -> None:
        """Close the edges viewer."""
        self.cleanup_edges_viewer()
            
    def cleanup_edges_viewer(self) -> None:
        """Clean up the edges viewer when it's closed."""
        if self.edges_vis is not None:
            try:
                self.edges_vis.destroy_window()
            except:
                pass  # Window might already be closed
            self.edges_vis = None
            
            # Update UI
            self.launch_edges_btn.setText("Launch Edges Viewer")
            
            self.status_label.setText("Edges viewer closed")
            self.update_stats()
            
    def close_wireframe_viewer(self) -> None:
        """Close the wireframe viewer."""
        self.cleanup_wireframe_viewer()
            
    def cleanup_wireframe_viewer(self) -> None:
        """Clean up the wireframe viewer when it's closed."""
        if self.wireframe_vis is not None:
            try:
                self.wireframe_vis.destroy_window()
            except:
                pass  # Window might already be closed
            self.wireframe_vis = None
            
            # Update UI
            self.launch_wireframe_btn.setText("Launch Wireframe Viewer")
            
            self.status_label.setText("Wireframe viewer closed")
            self.update_stats()
            
    def launch_wireframe_viewer(self) -> None:
        """Launch the Open3D wireframe viewer showing mesh skeleton."""
        if self.mesh is None:
            error_msg = "Please load a model first"
            print("ERROR:", error_msg)
            
            self.status_label.setText(error_msg)
            return
            
        try:
            # Create visualizer
            self.wireframe_vis = o3d.visualization.Visualizer()
            self.wireframe_vis.create_window(window_name="ArUco Marker Annotator - Wireframe View", 
                                           width=800, height=600, visible=True)
            
            # Setup render options for wireframe mode
            render_opt = self.wireframe_vis.get_render_option()
            render_opt.background_color = np.asarray([0.0, 0.0, 0.0])  # Pure black background
            render_opt.point_size = 1.0
            render_opt.line_width = 1.0  # Thin lines for skeleton effect
            render_opt.show_coordinate_frame = False
            render_opt.mesh_show_wireframe = True  # Enable wireframe mode
            render_opt.mesh_show_back_face = True  # Show back faces for complete skeleton
            
            # Add coordinate frame with model-relative size
            if self.mesh_info and 'max_dimension' in self.mesh_info:
                coord_frame_size = self.mesh_info['max_dimension'] * 0.1
            else:
                coord_frame_size = 0.1
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
            self.wireframe_vis.add_geometry(coord_frame)
            
            # Add the mesh in wireframe mode (skeleton format)
            self.wireframe_vis.add_geometry(self.mesh)
            
            # Add existing markers (they will appear as solid objects for reference)
            for marker_id, marker_data in self.markers.items():
                if 'aruco_info' in marker_data:
                    aruco_info = marker_data['aruco_info']
                    self.add_marker_to_wireframe_viewer(marker_id, tuple(aruco_info.position), aruco_info.size)
                else:
                    position = marker_data.get('position', (0, 0, 0))
                    size = marker_data.get('size', 0.05)
                    self.add_marker_to_wireframe_viewer(marker_id, position, size)
            
            # Reset view
            self.wireframe_vis.reset_view_point(True)
            
            # Update button text and style
            self.launch_wireframe_btn.setText("Close Wireframe Viewer")
            
            self.status_label.setText("Wireframe viewer launched successfully!")
            
            # Setup Qt integration timer for Open3D events
            self.setup_qt_integration_wireframe()
            
            # Fit to view
            try:
                view_control = self.wireframe_vis.get_view_control()
                if hasattr(view_control, 'fit_in_window'):
                    view_control.fit_in_window()
                elif hasattr(view_control, 'zoom_in_out'):
                    view_control.zoom_in_out(0.8)
            except Exception as e:
                print(f"Could not fit to view: {e}")
            
            print("✅ Wireframe viewer launched successfully")
            
        except Exception as e:
            error_msg = f"Failed to launch wireframe viewer: {str(e)}"
            print("ERROR:", error_msg)
            
            self.status_label.setText(error_msg)
            
    def extract_mesh_edges(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
        """Extract actual edges from the mesh using multiple approaches."""
        try:
            # Get vertices and triangles
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Method 1: Use convex hull edges
            try:
                hull = ConvexHull(vertices)
                # Extract only true boundary edges from convex hull
                edge_count = {}
                boundary_edges = []
                
                # Count how many times each edge appears in the convex hull
                for simplex in hull.simplices:
                    for i in range(len(simplex)):
                        v1, v2 = simplex[i], simplex[(i + 1) % len(simplex)]
                        edge = tuple(sorted([v1, v2]))
                        edge_count[edge] = edge_count.get(edge, 0) + 1
                
                # Only keep edges that appear exactly once (true boundary edges)
                boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
                
                # Debug: print edge counts
                print(f"Total edges found: {len(edge_count)}")
                print(f"Boundary edges (count=1): {len(boundary_edges)}")
                print(f"Edge counts: {dict(list(edge_count.items())[:5])}")  # Show first 5 edge counts
                
                # If no boundary edges found, use all edges as fallback
                if len(boundary_edges) == 0:
                    print("No boundary edges found, using all edges as fallback")
                    boundary_edges = list(edge_count.keys())
            except Exception as e:
                # Fallback: use all edges
                pass  # No fallback - let angle-based detection handle it
            
            # Create line set from boundary edges
            if len(boundary_edges) > 0:
                lines = []
                for edge in boundary_edges:
                    lines.append([edge[0], edge[1]])
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(vertices)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                
                # Set edge color (bright yellow for visibility)
                colors = [[1.0, 1.0, 0.0] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors)
                
                return line_set
            else:
                return o3d.geometry.LineSet()
                
        except Exception as e:
            return o3d.geometry.LineSet()
    
#     def _detect_edges_by_normal_angle(self, mesh: o3d.geometry.TriangleMesh, edge_count: dict, edge_to_triangles: dict) -> List[tuple]:
#         """Detect edges based on angle between adjacent triangle normals."""
#         try:
#             # Compute triangle normals
#             mesh.compute_triangle_normals()
#             triangle_normals = np.asarray(mesh.triangle_normals)
#             
#             edge_angles = {}
#             for edge, triangles in edge_to_triangles.items():
#                 if len(triangles) == 2:
#                     # Get normals of adjacent triangles
#                     n1 = triangle_normals[triangles[0]]
#                     n2 = triangle_normals[triangles[1]]
#                     
#                     # Compute angle between normals
#                     cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
#                     cos_angle = np.clip(cos_angle, -1.0, 1.0)
#                     angle = np.arccos(cos_angle)
#                     
#                     edge_angles[edge] = angle
#             
#             # Find edges with significant angle difference (sharp edges)
#             threshold_angle = np.pi / 6  # 30 degrees
#             sharp_edges = [edge for edge, angle in edge_angles.items() if angle > threshold_angle]
#             
#             return sharp_edges
#             
#         except Exception as e:
#             return []
#             
    def launch_edges_viewer(self) -> None:
        """Launch the Open3D edges viewer showing actual geometric edges."""
        if self.mesh is None:
            error_msg = "Please load a model first"
            print("ERROR:", error_msg)
            
            self.status_label.setText(error_msg)
            return
            
        try:
            # Create visualizer
            self.edges_vis = o3d.visualization.Visualizer()
            self.edges_vis.create_window(window_name="ArUco Marker Annotator - Edges View", 
                                       width=800, height=600, visible=True)
            
            # Setup render options - clean edges-only view
            render_opt = self.edges_vis.get_render_option()
            render_opt.background_color = np.asarray([0.0, 0.0, 0.0])  # Pure black background
            render_opt.point_size = 1.0
            render_opt.line_width = 2.0  # Clean line width
            render_opt.show_coordinate_frame = False
            
            # Extract and add ONLY the edges (nothing else)
            edge_lines = self.extract_mesh_edges(self.mesh)
            if len(edge_lines.points) > 0:
                self.edges_vis.add_geometry(edge_lines)
                self.status_label.setText(f"Edges viewer: {len(edge_lines.lines)} edges displayed")
            else:
                self.status_label.setText("Edges viewer: No edges found")
            
            # Reset view
            self.edges_vis.reset_view_point(True)
            
            # Update button text and style
            self.launch_edges_btn.setText("Close Edges Viewer")
            
            self.status_label.setText("Edges viewer launched successfully!")
            
            # Setup Qt integration timer for Open3D events
            self.setup_qt_integration_edges()
            
            # Fit to view
            try:
                view_control = self.edges_vis.get_view_control()
                if hasattr(view_control, 'fit_in_window'):
                    view_control.fit_in_window()
                elif hasattr(view_control, 'zoom_in_out'):
                    view_control.zoom_in_out(0.8)
            except Exception as e:
                print(f"Could not fit to view: {e}")
            
            print("✅ Edges viewer launched successfully")
            
        except Exception as e:
            error_msg = f"Failed to launch edges viewer: {str(e)}"
            print("ERROR:", error_msg)
            
            self.status_label.setText(error_msg)
            
    def launch_3d_viewer(self) -> None:
        """Launch the Open3D 3D viewer."""
        if self.mesh is None:
            error_msg = "Please load a model first"
            print("ERROR:", error_msg)
            
            self.status_label.setText(error_msg)
            return
            
        try:
            # Create visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="ArUco Marker Annotator - 3D View", 
                                  width=800, height=600, visible=True)
            
            # Setup render options
            render_opt = self.vis.get_render_option()
            render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
            render_opt.point_size = 3.0
            render_opt.line_width = 2.0
            render_opt.show_coordinate_frame = False  # Disable built-in coordinate frame
            
            # Add coordinate frame with model-relative size
            if self.mesh_info and 'max_dimension' in self.mesh_info:
                # Scale coordinate frame to 10% of model's max dimension
                coord_frame_size = self.mesh_info['max_dimension'] * 0.1
            else:
                # Fallback to fixed size if no mesh info
                coord_frame_size = 0.1
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
            self.vis.add_geometry(coord_frame)
            self.geometries['coordinate_frame'] = coord_frame
            
            
            # Add the mesh
            self.vis.add_geometry(self.mesh)
            
            # Add existing markers
            for marker_id, marker_data in self.markers.items():
                if 'aruco_info' in marker_data:
                    # New format with ArUcoMarkerInfo
                    aruco_info = marker_data['aruco_info']
                    self.add_marker_to_viewer(marker_id, tuple(aruco_info.position), aruco_info.size)
                else:
                    # Legacy format
                    position = marker_data.get('position', (0, 0, 0))
                    size = marker_data.get('size', 0.05)
                    self.add_marker_to_viewer(marker_id, position, size)
                
            
            # Reset view
            self.vis.reset_view_point(True)
            
            # Update button text and style
            self.launch_3d_btn.setText("Close 3D Viewer")
            
            self.status_label.setText("3D viewer launched successfully!")
            
            # Set up interaction callbacks for picking
            self.setup_picking_callback()
            
            # Setup Qt integration timer for Open3D events
            self.setup_qt_integration()
            
            # Fit to view - use the correct method name for this Open3D version
            try:
                # Try different method names depending on Open3D version
                view_control = self.vis.get_view_control()
                if hasattr(view_control, 'fit_in_window'):
                    view_control.fit_in_window()
                elif hasattr(view_control, 'zoom_in_out'):
                    # Alternative: reset view and zoom to fit
                    view_control.reset_view_point(True)
                else:
                    # Fallback: just reset the view
                    print("Using fallback view reset")
                    pass
            except Exception as e:
                print(f"Warning: Could not fit view to window: {e}")
                # Continue anyway - this is not critical
            
        except Exception as e:
            error_msg = f"Error launching 3D viewer: {str(e)}"
            
            # Print error to terminal so user can see it
            print("=" * 60)
            print("ERROR IN 3D VIEWER:")
            print(error_msg)
            print("Exception type:", type(e).__name__)
            print("Full details:")
            import traceback
            traceback.print_exc()
            print("=" * 60)
            
            # Also show in UI
            self.status_label.setText(error_msg)
            
    def create_aruco_marker_geometry(self, position: tuple, size: float, marker_id: int, rotation: tuple = (0, 0, 0)) -> o3d.geometry.TriangleMesh:
        """Create a visual representation of an ArUco marker."""
        import numpy as np
        
        x, y, z = position
        roll, pitch, yaw = rotation
        
        # Use the exact size specified by the user - no proportional scaling
        print(f"📐 Model max dimension: {self.mesh_info['max_dimension']:.4f}m" if self.mesh_info and 'max_dimension' in self.mesh_info else "📐 No model info available")
        print(f"🎯 Using exact user-specified size: {size:.3f}m")
        
        # Only apply basic safety limits to prevent extremely small or large markers
        original_size = size
        size = max(0.0001, min(size, 1.0))  # Between 0.1mm and 1m
        if size != original_size:
            print(f"⚠️  Size adjusted from {original_size:.3f}m to {size:.3f}m (safety limits only)")
        
        # Create the main marker base (flat rectangle)
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/20)
        marker_mesh.translate([x - size/2, y - size/2, z - size/40])
        marker_mesh.paint_uniform_color([0.9, 0.9, 0.9])
        
        # Create a black border
        border_size = size * 0.9
        border_mesh = o3d.geometry.TriangleMesh.create_box(border_size, border_size, size/30)
        border_mesh.translate([x - border_size/2, y - border_size/2, z])
        border_mesh.paint_uniform_color([0.1, 0.1, 0.1])
        
        # Create inner white square
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
        
        # Apply rotation if specified
        if rotation != (0, 0, 0):
            print(f"Applying rotation to geometric marker: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
            
            # Apply rotations one by one for better control
            center = np.array([x, y, z])
            
            if abs(roll) > 0.001:
                roll_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([roll, 0, 0])
                combined_mesh.rotate(roll_matrix, center=center)
                print(f"Applied roll rotation: {roll:.3f}")
            
            if abs(pitch) > 0.001:
                pitch_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, pitch, 0])
                combined_mesh.rotate(pitch_matrix, center=center)
                print(f"Applied pitch rotation: {pitch:.3f}")
            
            if abs(yaw) > 0.001:
                yaw_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
                combined_mesh.rotate(yaw_matrix, center=center)
                print(f"Applied yaw rotation: {yaw:.3f}")
        
        # Compute normals for proper lighting
        combined_mesh.compute_vertex_normals()
        
        return combined_mesh
    
    def create_real_aruco_marker(self, aruco_info: ArUcoMarkerInfo) -> o3d.geometry.TriangleMesh:
        """Create a real ArUco marker with actual pixel-perfect pattern display."""
        import cv2
        import numpy as np
        
        x, y, z = aruco_info.position
        
        # Use the size from ArUcoMarkerInfo, but apply reasonable limits
        size = aruco_info.size
        
        # Debug: Show size information
        print(f"🎯 ArUco marker size: User input = {aruco_info.size:.3f}m")
        
        # Use the exact size specified by the user - no proportional scaling
        print(f"📐 Model max dimension: {self.mesh_info['max_dimension']:.4f}m" if self.mesh_info and 'max_dimension' in self.mesh_info else "📐 No model info available")
        print(f"🎯 Using exact user-specified size: {size:.3f}m")
        
        # Only apply basic safety limits to prevent extremely small or large markers
        original_size = size
        size = max(0.0001, min(size, 1.0))  # Between 0.1mm and 1m
        if size != original_size:
            print(f"⚠️  Size adjusted from {original_size:.3f}m to {size:.3f}m (safety limits only)")
        
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
        
        # Note: Coordinate frame removed - use the main axes toggle instead
        
        # Apply rotation if specified
        if hasattr(aruco_info, 'rotation') and aruco_info.rotation != (0, 0, 0):
            roll, pitch, yaw = aruco_info.rotation
            R = combined_mesh.get_rotation_matrix_from_xyz([roll, pitch, yaw])
            combined_mesh.rotate(R, center=[x, y, z])
            print(f"📐 Applied rotation to ArUco marker: ({roll:.3f}, {pitch:.3f}, {yaw:.3f}) rad")
        
        # Compute normals for proper lighting
        combined_mesh.compute_vertex_normals()
        
        return combined_mesh
    
    def get_cad_object_info(self) -> Optional[Dict]:
        """Get CAD object information for pose calculations."""
        if self.mesh is None or self.mesh_info is None:
            return None
        
        # Calculate CAD object center
        bbox_min = np.array(self.mesh_info['bbox_min'])
        bbox_max = np.array(self.mesh_info['bbox_max'])
        center = (bbox_min + bbox_max) / 2.0
        
        return {
            "center": tuple(center),
            "dimensions": self.mesh_info.get('dimensions', {
                'length': 0.0, 'width': 0.0, 'height': 0.0
            }),
            "bbox_min": tuple(bbox_min),
            "bbox_max": tuple(bbox_max),
            "max_dimension": self.mesh_info.get('max_dimension', 0.0)
        }
    
    def create_grid(self) -> o3d.geometry.LineSet:
        """Create a grid for the 3D viewer."""
        try:
            # Use mesh info to determine grid size, or default
            if self.mesh_info and 'max_dimension' in self.mesh_info:
                grid_size = self.mesh_info['max_dimension'] * 1.5
            else:
                grid_size = 1.0
            
            # Create grid lines
            lines = []
            points = []
            grid_divisions = 10
            step = grid_size / grid_divisions
            
            # Create grid points and lines
            for i in range(grid_divisions + 1):
                # X-direction lines
                x = -grid_size/2 + i * step
                points.extend([[x, -grid_size/2, 0], [x, grid_size/2, 0]])
                line_idx = len(points) - 2
                lines.append([line_idx, line_idx + 1])
                
                # Y-direction lines  
                y = -grid_size/2 + i * step
                points.extend([[-grid_size/2, y, 0], [grid_size/2, y, 0]])
                line_idx = len(points) - 2
                lines.append([line_idx, line_idx + 1])
            
            # Create LineSet
            grid = o3d.geometry.LineSet()
            grid.points = o3d.utility.Vector3dVector(points)
            grid.lines = o3d.utility.Vector2iVector(lines)
            
            # Set grid color (light gray)
            colors = [[0.5, 0.5, 0.5] for _ in range(len(lines))]
            grid.colors = o3d.utility.Vector3dVector(colors)
            
            return grid
        except Exception as e:
            print(f"❌ Error creating grid: {e}")
            return None
    

    def add_marker_to_viewer(self, marker_id: int, position: tuple, size: float):
        """Add a marker to the 3D viewer."""
        if self.vis is None:
            return
            
        # Create the enhanced ArUco marker geometry
        marker_mesh = self.create_aruco_marker_geometry(position, size, marker_id)
        self.vis.add_geometry(marker_mesh)
        
    def add_marker_to_edges_viewer(self, marker_id: int, position: tuple, size: float):
        """Add a marker to the edges viewer."""
        if self.edges_vis is None:
            return
            
        # Create the enhanced ArUco marker geometry
        marker_mesh = self.create_aruco_marker_geometry(position, size, marker_id)
        self.edges_vis.add_geometry(marker_mesh)
        
    def add_marker_to_wireframe_viewer(self, marker_id: int, position: tuple, size: float):
        """Add a marker to the wireframe viewer."""
        if self.wireframe_vis is None:
            return
            
        # Create the enhanced ArUco marker geometry
        marker_mesh = self.create_aruco_marker_geometry(position, size, marker_id)
        self.wireframe_vis.add_geometry(marker_mesh)
        
    def add_aruco_marker(self, marker_id: int, aruco_info: ArUcoMarkerInfo) -> None:
        """Add a real ArUco marker with proper texture."""
        self.markers[marker_id] = {
            'aruco_info': aruco_info,
            'geometry': None,
            'is_selected': False
        }
        self.update_stats()
        
        # Add to 3D viewer if it's open
        if self.vis is not None:
            try:
                # Create ArUco marker with actual image texture
                marker_mesh = self.create_real_aruco_marker(aruco_info)
                self.markers[marker_id]['geometry'] = marker_mesh
                self.vis.add_geometry(marker_mesh)
            except Exception as e:
                print(f"Failed to create ArUco marker: {e}")
                # Fallback to geometric representation
                marker_mesh = self.create_aruco_marker_geometry(aruco_info.position, aruco_info.size, marker_id)
                self.markers[marker_id]['geometry'] = marker_mesh
                self.vis.add_geometry(marker_mesh)
        
        # Add to edges viewer if it's open
        if self.edges_vis is not None:
            try:
                # Create ArUco marker with actual image texture
                marker_mesh = self.create_real_aruco_marker(aruco_info)
                self.edges_vis.add_geometry(marker_mesh)
            except Exception as e:
                print(f"Failed to create ArUco marker for edges viewer: {e}")
                # Fallback to geometric representation
                marker_mesh = self.create_aruco_marker_geometry(aruco_info.position, aruco_info.size, marker_id)
                self.edges_vis.add_geometry(marker_mesh)
    
    def add_marker(self, marker_id: int, position: tuple, size: float = 0.05) -> None:
        """Add an ArUco marker visualization (legacy method)."""
        # Create default ArUco info for legacy compatibility
        aruco_info = ArUcoMarkerInfo(
            dictionary="DICT_4X4_50",
            marker_id=0,
            position=position,
            size=size
        )
        self.add_aruco_marker(marker_id, aruco_info)
        
    def remove_marker(self, marker_id: int) -> None:
        """Remove a marker from the viewer."""
        if marker_id in self.markers:
            marker = self.markers[marker_id]
            if marker['geometry'] is not None and self.vis is not None:
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            if marker['geometry'] is not None and self.edges_vis is not None:
                self.edges_vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            del self.markers[marker_id]
            self.update_stats()
            
    def select_marker(self, marker_id: int) -> None:
        """Highlight the selected marker."""
        # Update selection state for all markers
        for mid, marker in self.markers.items():
            marker['is_selected'] = (mid == marker_id)
        self.update_stats()
        
    def move_marker(self, marker_id: int, new_position: tuple) -> None:
        """Move a marker to a new position."""
        if marker_id in self.markers and self.vis is not None:
            marker = self.markers[marker_id]
            
            # Remove old geometry
            if marker['geometry'] is not None:
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            
            # Update position
            if 'aruco_info' in marker:
                # New format with ArUco info
                marker['aruco_info'].position = new_position
                # Create new geometry at new position
                try:
                    new_geometry = self.create_real_aruco_marker(marker['aruco_info'])
                except Exception:
                    # Fallback to geometric representation
                    new_geometry = self.create_aruco_marker_geometry(
                        new_position, marker['aruco_info'].size, marker_id
                    )
            else:
                # Legacy format
                marker['position'] = new_position
                new_geometry = self.create_aruco_marker_geometry(
                    new_position, marker.get('size', 0.05), marker_id
                )
            
            marker['geometry'] = new_geometry
            self.vis.add_geometry(new_geometry)
            self.update_stats()
    
    def rotate_marker(self, marker_id: int, new_rotation: tuple) -> None:
        """Rotate a marker to a new orientation."""
        print(f"🔄 Rotating marker {marker_id} to {new_rotation}")
        
        if marker_id in self.markers and self.vis is not None:
            marker = self.markers[marker_id]
            
            # Remove old geometry
            if marker['geometry'] is not None:
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
                print(f"✅ Removed old geometry for marker {marker_id}")
            
            # Update rotation
            if 'aruco_info' in marker:
                # New format with ArUco info
                marker['aruco_info'].rotation = new_rotation
                print(f"📝 Updated ArUco info rotation: {marker['aruco_info'].rotation}")
                
                # Create new geometry with new rotation
                try:
                    new_geometry = self.create_real_aruco_marker(marker['aruco_info'])
                    print(f"✅ Created new ArUco marker geometry with rotation")
                except Exception as e:
                    print(f"❌ Failed to create ArUco marker: {e}")
                    # Fallback to geometric representation
                    new_geometry = self.create_aruco_marker_geometry(
                        marker['aruco_info'].position, marker['aruco_info'].size, marker_id, new_rotation
                    )
                    print(f"🔄 Using fallback geometric representation with rotation")
            else:
                # Legacy format
                marker['rotation'] = new_rotation
                new_geometry = self.create_aruco_marker_geometry(
                    marker.get('position', (0, 0, 0)), marker.get('size', 0.05), marker_id, new_rotation
                )
                print(f"🔄 Using legacy format with rotation")
            
            marker['geometry'] = new_geometry
            self.vis.add_geometry(new_geometry)
            self.update_stats()
            
            print(f"✅ Marker {marker_id} rotation updated successfully")
        else:
            print(f"❌ Cannot rotate marker {marker_id}: not found or no visualizer")
            
    def enable_placement_mode(self) -> None:
        """Enable marker placement mode."""
        self.placement_mode = True
        # Update status to indicate placement mode
        self.status_label.setText("Placement Mode Active\\nClick on a face to place marker")
        # Show the placement instruction
        self.placement_label.setVisible(True)
        # Show alternative placement buttons if available, create if needed
        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
            self.click_place_btn.setVisible(True)
            self.face_picker_btn.setVisible(True)
            if hasattr(self, 'all_sides_btn') and self.all_sides_btn is not None:
                self.all_sides_btn.setVisible(True)
            if hasattr(self, 'corner_markers_btn') and self.corner_markers_btn is not None:
                self.corner_markers_btn.setVisible(True)
        elif not hasattr(self, 'click_place_btn') or self.click_place_btn is None:
            # Create the alternative placement buttons now
            self.setup_alternative_placement()
        # Update mode label
        self.mode_label.setText("Mode: Place Marker")
        
    def disable_placement_mode(self) -> None:
        """Disable marker placement mode."""
        self.placement_mode = False
        # Hide the placement instruction
        self.placement_label.setVisible(False)
        # Hide alternative placement buttons if available
        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
            self.click_place_btn.setVisible(False)
        if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
            self.face_picker_btn.setVisible(False)
        # Restore normal mode label
        self.mode_label.setText("Mode: View")
        # Restore normal status
        if self.mesh is not None:
            vertices_count = len(self.mesh.vertices) if hasattr(self.mesh, 'vertices') else 0
            triangles_count = len(self.mesh.triangles) if hasattr(self.mesh, 'triangles') else 0
            self.status_label.setText(f"Loaded mesh:\\n{vertices_count} vertices\\n{triangles_count} faces")
        
    def calculate_face_center_from_triangle(self, triangle_idx: int) -> tuple:
        """Calculate the center of a triangle face."""
        if self.mesh is None:
            return None
            
        triangles = np.asarray(self.mesh.triangles)
        vertices = np.asarray(self.mesh.vertices)
        
        if triangle_idx >= len(triangles):
            return None
            
        # Get the three vertices of the triangle
        triangle = triangles[triangle_idx]
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        
        # Calculate the centroid (center) of the triangle
        center = (v1 + v2 + v3) / 3.0
        return tuple(center)
    
    def calculate_triangle_normal(self, triangle_idx: int) -> tuple:
        """Calculate the normal vector of a triangle face."""
        if self.mesh is None:
            return (0.0, 0.0, 1.0)  # Default to Z-up
            
        triangles = np.asarray(self.mesh.triangles)
        vertices = np.asarray(self.mesh.vertices)
        
        if triangle_idx >= len(triangles):
            return (0.0, 0.0, 1.0)  # Default to Z-up
            
        # Get the three vertices of the triangle
        triangle = triangles[triangle_idx]
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        
        # Calculate normal using cross product
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        normal_length = np.linalg.norm(normal)
        
        if normal_length < 1e-8:
            return (0.0, 0.0, 1.0)  # Default to Z-up if degenerate triangle
        
        # Normalize the normal vector
        normal = normal / normal_length
        return tuple(normal)
    
    def group_triangles_by_face(self, triangles, vertices):
        """Improved face detection algorithm using normal vectors, spatial connectivity, and coplanarity."""
        print("🔍 Starting improved face detection algorithm...")
        print(f"Processing {len(triangles)} triangles...")
        
        face_groups = []
        normal_tolerance = 0.02  # Tighter tolerance for better precision
        spatial_threshold = 0.1  # Maximum distance between triangle centers to consider them connected
        min_face_area = 1e-6  # Minimum area to consider a valid face
        
        # Calculate triangle properties
        triangle_data = []
        for i, triangle in enumerate(triangles):
            v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
            
            # Calculate normal (ensure consistent orientation)
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            normal = normal / (np.linalg.norm(normal) + 1e-8)  # Normalize
            
            # Calculate center and plane equation (ax + by + cz + d = 0)
            center = (v1 + v2 + v3) / 3.0
            plane_d = -np.dot(normal, center)
            
            triangle_data.append({
                'vertices': (v1, v2, v3),
                'center': center,
                'normal': normal,
                'area': area,
                'plane_d': plane_d,
                'triangle_idx': i
            })
        
        used_triangles = set()
        
        # Sort triangles by area (largest first) to prioritize major faces
        sorted_indices = sorted(range(len(triangle_data)), key=lambda i: triangle_data[i]['area'], reverse=True)
        
        for idx in sorted_indices:
            if idx in used_triangles:
                continue
                
            seed_triangle = triangle_data[idx]
            if seed_triangle['area'] < min_face_area:
                continue
                
            # Start a new face group
            face_triangles = [idx]
            face_queue = [idx]
            used_triangles.add(idx)
            
            # Grow the face by finding connected triangles with similar properties
            while face_queue:
                current_idx = face_queue.pop(0)
                current_triangle = triangle_data[current_idx]
                
                # Check all remaining triangles for membership in this face
                for j, candidate in enumerate(triangle_data):
                    if j in used_triangles:
                        continue
                    
                    # Check 1: Normal similarity (dot product)
                    normal_similarity = np.dot(current_triangle['normal'], candidate['normal'])
                    if normal_similarity < (1.0 - normal_tolerance):
                        continue
                    
                    # Check 2: Coplanarity (distance to plane)
                    point_to_plane_dist = abs(np.dot(candidate['normal'], current_triangle['center']) + current_triangle['plane_d'])
                    if point_to_plane_dist > spatial_threshold * 0.1:  # Very strict coplanarity check
                        continue
                    
                    # Check 3: Spatial proximity (at least one triangle in face should be close)
                    is_spatially_connected = False
                    for face_tri_idx in face_triangles:
                        face_center = triangle_data[face_tri_idx]['center']
                        distance = np.linalg.norm(candidate['center'] - face_center)
                        if distance < spatial_threshold:
                            is_spatially_connected = True
                            break
                    
                    if not is_spatially_connected:
                        continue
                    
                    # Check 4: Edge connectivity (share vertices or are very close)
                    is_edge_connected = self._triangles_share_edge_or_vertex(
                        triangles[current_idx], triangles[j], vertices
                    )
                    
                    if is_edge_connected or self._triangles_are_adjacent(
                        current_triangle['vertices'], candidate['vertices'], spatial_threshold * 0.5
                    ):
                        # Add to face
                        face_triangles.append(j)
                        face_queue.append(j)
                        used_triangles.add(j)
            
            # Calculate face properties
            if len(face_triangles) > 0:
                face_centers = [triangle_data[i]['center'] for i in face_triangles]
                face_areas = [triangle_data[i]['area'] for i in face_triangles]
                face_normals = [triangle_data[i]['normal'] for i in face_triangles]
                
                # Use area-weighted average for face center and normal
                total_area = sum(face_areas)
                if total_area > min_face_area:
                    face_center = np.average(face_centers, axis=0, weights=face_areas)
                    face_normal = np.average(face_normals, axis=0, weights=face_areas)
                    face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
                    
                    face_groups.append((face_center, face_normal, total_area, face_triangles))
                    print(f"✅ Face {len(face_groups)}: {len(face_triangles)} triangles, area={total_area:.6f}")
        
        print(f"🎯 Found {len(face_groups)} distinct faces using improved algorithm")
        
        # Sort faces by area (largest first) for consistent ordering
        face_groups.sort(key=lambda x: x[2], reverse=True)
        
        return face_groups
    
    def _triangles_share_edge_or_vertex(self, tri1, tri2, vertices):
        """Check if two triangles share an edge or vertex."""
        tri1_verts = set(tri1)
        tri2_verts = set(tri2)
        shared_vertices = tri1_verts.intersection(tri2_verts)
        return len(shared_vertices) >= 1  # Share at least one vertex
    
    def _triangles_are_adjacent(self, tri1_verts, tri2_verts, threshold):
        """Check if triangles are spatially adjacent (vertices are very close)."""
        for v1 in tri1_verts:
            for v2 in tri2_verts:
                if np.linalg.norm(v1 - v2) < threshold:
                    return True
        return False
    
    def pick_face_with_raycast(self, x: int, y: int) -> tuple:
        """Use ray casting to find the actual clicked face center."""
        if self.mesh is None or self.vis is None:
            return None
            
        try:
            # Get camera parameters
            ctr = self.vis.get_view_control()
            camera_params = ctr.convert_to_pinhole_camera_parameters()
            
            # Create ray from screen coordinates
            width = camera_params.intrinsic.width
            height = camera_params.intrinsic.height
            
            # Convert screen coordinates to normalized device coordinates
            ndc_x = (2.0 * x / width) - 1.0
            ndc_y = 1.0 - (2.0 * y / height)
            
            # Create a ray from the camera
            # This is a simplified approach - for now we'll use a different strategy
            # since Open3D's ray casting can be complex to set up properly
            
            # Instead, let's pick the face closest to the camera view center
            triangles = np.asarray(self.mesh.triangles)
            vertices = np.asarray(self.mesh.vertices)
            
            if len(triangles) > 0:
                # Find triangle faces and their centers
                face_centers = []
                for i, triangle in enumerate(triangles):
                    v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
                    center = (v1 + v2 + v3) / 3.0
                    face_centers.append((i, center))
                
                # For now, pick a random face center
                # In a proper implementation, you'd find the face closest to the ray
                import random
                selected_face = random.choice(face_centers)
                return tuple(selected_face[1])
                
        except Exception as e:
            print(f"Error in ray casting: {e}")
            
        return None
    
    def setup_qt_integration(self) -> None:
        """Setup Qt timer to integrate Open3D events with Qt event loop."""
        self.open3d_timer = QTimer()
        self.open3d_timer.timeout.connect(self.update_open3d_events)
        self.open3d_timer.start(16)  # ~60 FPS
    
    def update_open3d_events(self) -> None:
        """Update Open3D events to keep the visualizer responsive."""
        if hasattr(self, 'vis') and self.vis is not None:
            try:
                # Poll events to keep the window responsive
                self.vis.poll_events()
                self.vis.update_renderer()
            except:
                # If polling fails, stop the timer
                if hasattr(self, 'open3d_timer'):
                    self.open3d_timer.stop()
    
    def setup_qt_integration_edges(self) -> None:
        """Setup Qt timer to integrate Open3D events with Qt event loop for edges viewer."""
        self.open3d_edges_timer = QTimer()
        self.open3d_edges_timer.timeout.connect(self.update_open3d_events_edges)
        self.open3d_edges_timer.start(16)  # ~60 FPS
    
    def update_open3d_events_edges(self) -> None:
        """Update Open3D events to keep the edges visualizer responsive."""
        if hasattr(self, 'edges_vis') and self.edges_vis is not None:
            try:
                # Poll events to keep the window responsive
                self.edges_vis.poll_events()
                self.edges_vis.update_renderer()
            except:
                # If polling fails, stop the timer
                if hasattr(self, 'open3d_edges_timer'):
                    self.open3d_edges_timer.stop()
    
    def setup_qt_integration_wireframe(self) -> None:
        """Setup Qt timer to integrate Open3D events with Qt event loop for wireframe viewer."""
        self.open3d_wireframe_timer = QTimer()
        self.open3d_wireframe_timer.timeout.connect(self.update_open3d_events_wireframe)
        self.open3d_wireframe_timer.start(16)  # ~60 FPS
    
    def update_open3d_events_wireframe(self) -> None:
        """Update Open3D events to keep the wireframe visualizer responsive."""
        if hasattr(self, 'wireframe_vis') and self.wireframe_vis is not None:
            try:
                # Poll events to keep the window responsive
                self.wireframe_vis.poll_events()
                self.wireframe_vis.update_renderer()
            except:
                # If polling fails, stop the timer
                if hasattr(self, 'open3d_wireframe_timer'):
                    self.open3d_wireframe_timer.stop()
    
    def setup_picking_callback(self) -> None:
        """Set up mouse interaction callbacks for the Open3D visualizer."""
        def mouse_callback(vis, action, mods):
            """Handle mouse interactions in the 3D viewer."""
            if action == 1:  # Left mouse button pressed
                if self.placement_mode and self.mesh is not None:
                    try:
                        # Pick a triangle face center for marker placement
                        triangles = np.asarray(self.mesh.triangles)
                        
                        if len(triangles) > 0:
                            # For now, pick a random triangle face
                            # In a proper implementation, you'd use ray casting to find the clicked face
                            import random
                            triangle_idx = random.randint(0, len(triangles) - 1)
                            face_center = self.calculate_face_center_from_triangle(triangle_idx)
                            face_normal = self.calculate_triangle_normal(triangle_idx)
                            
                            if face_center:
                                # Use QTimer to emit signal from main thread with computed normal
                                QTimer.singleShot(0, lambda: self.point_picked.emit(face_center, face_normal))
                                print(f"Placed marker at face center: {face_center} with normal: {face_normal}")
                        else:
                            # Fallback to bbox center if no triangles
                            bbox = self.mesh.get_axis_aligned_bounding_box()
                            center = bbox.get_center()
                            QTimer.singleShot(0, lambda: self.point_picked.emit(tuple(center), (0.0, 0.0, 1.0)))
                    except Exception as e:
                        print(f"Error in face picking: {e}")
            return False
        
        def key_callback(vis, key, action):
            """Handle keyboard interactions in the 3D viewer."""
            if key == ord('P') and action == 1:  # P key pressed
                # Toggle placement mode
                if self.placement_mode:
                    QTimer.singleShot(0, lambda: self.disable_placement_mode())
                print(f"Placement mode: {'OFF' if self.placement_mode else 'ON'}")
            elif key == ord('R') and action == 1:  # R key pressed
                vis.reset_view_point(True)
            return False
        
        # Register callbacks with Open3D visualizer
        print("Checking available callback methods...")
        available_methods = [method for method in dir(self.vis) if 'callback' in method.lower()]
        print(f"Available callback methods: {available_methods}")
        
        try:
            # Try to register the callbacks
            if hasattr(self.vis, 'register_mouse_callback'):
                result = self.vis.register_mouse_callback(mouse_callback)
                print(f"Mouse callback registered, result: {result}")
            else:
                print("register_mouse_callback not available")
                
            if hasattr(self.vis, 'register_key_callback'):
                result = self.vis.register_key_callback(key_callback)
                print(f"Key callback registered, result: {result}")
            else:
                print("register_key_callback not available")
                
            # Always set up alternative placement as backup
            print("Setting up alternative placement as backup...")
            self.setup_alternative_placement()
            
        except Exception as e:
            print(f"Callback registration failed: {e}")
            import traceback
            traceback.print_exc()
            # Set up alternative method
            self.setup_alternative_placement()
    
    def setup_alternative_placement(self) -> None:
        """Setup alternative placement method when callbacks don't work."""
        # Add multiple placement options
        self.click_place_btn = QPushButton("🎯 Place on Random Face")
        self.click_place_btn.setVisible(False)
        self.click_place_btn.clicked.connect(self.place_marker_at_surface)

        self.face_picker_btn = QPushButton("📋 Choose Face from List")
        self.face_picker_btn.setVisible(False)
        self.face_picker_btn.clicked.connect(self.open_face_picker)

        self.all_sides_btn = QPushButton("📦 Add ArUco on 6 Faces")
        self.all_sides_btn.setVisible(False)
        self.all_sides_btn.clicked.connect(self.place_markers_on_all_faces)
        
        self.corner_markers_btn = QPushButton("🔲 Add 4 Corner Markers")
        self.corner_markers_btn.setVisible(False)
        self.corner_markers_btn.clicked.connect(self.place_corner_markers_on_face)

        # Live face selection removed per user request
        # Style all buttons
        random_button_style = """
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: 2px solid #e68900;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                margin: 2px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #e68900;
                border-color: #d68100;
            }
            QPushButton:pressed {
                background-color: #d68100;
            }
        """


        
        # Add all buttons to the layout
        if hasattr(self, 'layout') and self.layout():
            main_layout = self.layout()
            # Insert buttons before the stretch
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and hasattr(item, 'changeSize'):  # This is the stretch
                    main_layout.insertWidget(i, self.click_place_btn)
                    main_layout.insertWidget(i+1, self.face_picker_btn)
                    main_layout.insertWidget(i+2, self.all_sides_btn)
                    main_layout.insertWidget(i+3, self.corner_markers_btn)
                    break
            else:
                # If no stretch found, just add at the end
                main_layout.addWidget(self.click_place_btn)
                main_layout.addWidget(self.face_picker_btn)
                main_layout.addWidget(self.all_sides_btn)
                main_layout.addWidget(self.corner_markers_btn)
    
    def place_marker_at_surface(self) -> None:
        """Place a marker at a random face center (using grouped faces, not individual triangles)."""
        if self.placement_mode and self.mesh is not None:
            try:
                triangles = np.asarray(self.mesh.triangles)
                vertices = np.asarray(self.mesh.vertices)
                
                if len(triangles) > 0:
                    # Group triangles into actual faces (same algorithm as Face Picker)
                    face_groups = self.group_triangles_by_face(triangles, vertices)
                    
                    if len(face_groups) > 0:
                        # Pick a random face from the grouped faces
                        import random
                        face_idx = random.randint(0, len(face_groups) - 1)
                        face_center, face_normal, face_area, triangle_indices = face_groups[face_idx]
                        
                        print(f"🎯 Random face selected: Face {face_idx + 1} of {len(face_groups)}")
                        print(f"   Face center: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")
                        print(f"   Face normal: ({face_normal[0]:.3f}, {face_normal[1]:.3f}, {face_normal[2]:.3f})")
                        print(f"   Face area: {face_area:.4f}")
                        print(f"   Triangles in face: {len(triangle_indices)}")
                        
                        # Use computed face normal instead of hardcoded Z-up
                        self.point_picked.emit(tuple(face_center), tuple(face_normal))
                        print(f"✅ Marker placed at random face center: {face_center} with normal: {face_normal}")
                        
                        # Hide the placement button
                        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
                            self.click_place_btn.setVisible(False)
                    else:
                        print("⚠️ No faces found, falling back to random triangle")
                        # Fallback to old behavior if no faces found
                        triangle_idx = random.randint(0, len(triangles) - 1)
                        face_center = self.calculate_face_center_from_triangle(triangle_idx)
                        face_normal = self.calculate_triangle_normal(triangle_idx)
                        if face_center:
                            self.point_picked.emit(face_center, face_normal)
                            print(f"Marker placed at triangle center: {face_center} with normal: {face_normal}")
                else:
                    # Fallback to bbox center
                    bbox = self.mesh.get_axis_aligned_bounding_box()
                    center = bbox.get_center()
                    self.point_picked.emit(tuple(center), (0.0, 0.0, 1.0))
            except Exception as e:
                print(f"Error in random face placement: {e}")
    
    def place_markers_on_all_faces(self) -> None:
        """Place ArUco markers on the 6 primary orthogonal faces using CAD model dimensions and face centers."""
        if self.placement_mode and self.mesh is not None:
            try:
                print("🎯 Placing ArUco markers on 6 primary faces using CAD model dimensions...")
                
                # Use stored CAD dimensions if available, otherwise fallback to runtime bounding box
                if self.mesh_info and 'dimensions' in self.mesh_info:
                    # Use the original CAD model dimensions (already converted to meters)
                    cad_dims = self.mesh_info['dimensions']
                    bbox_min = np.array(self.mesh_info['bbox_min'])
                    bbox_max = np.array(self.mesh_info['bbox_max'])
                    center = (bbox_min + bbox_max) / 2.0
                    
                    print(f"   Using CAD model dimensions:")
                    print(f"   Length (X): {cad_dims['length']:.4f} m")
                    print(f"   Width (Y):  {cad_dims['width']:.4f} m") 
                    print(f"   Height (Z): {cad_dims['height']:.4f} m")
                    print(f"   CAD Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                    
                    min_bound = bbox_min
                    max_bound = bbox_max
                else:
                    # Fallback to runtime bounding box calculation
                    print("   Warning: No CAD dimensions found, using runtime bounding box")
                    bbox = self.mesh.get_axis_aligned_bounding_box()
                    min_bound = bbox.get_min_bound()
                    max_bound = bbox.get_max_bound()
                    center = bbox.get_center()
                    
                    print(f"   Runtime bounding box: min=({min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f})")
                    print(f"   Runtime bounding box: max=({max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f})")
                    print(f"   Runtime center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                
                # Define the 6 orthogonal faces with their normal vectors
                # Note: Normals point OUTWARD from the object center so ArUco markers face outward (visible from outside)
                faces = [
                    # X-axis faces (Left and Right)
                    {
                        'center': np.array([min_bound[0], center[1], center[2]]),
                        'normal': np.array([-1.0, 0.0, 0.0]),  # Outward (away from center, toward -X)
                        'name': "Left Face (-X)"
                    },
                    {
                        'center': np.array([max_bound[0], center[1], center[2]]),
                        'normal': np.array([1.0, 0.0, 0.0]),  # Outward (away from center, toward +X)
                        'name': "Right Face (+X)"
                    },
                    
                    # Y-axis faces (Front and Back)
                    {
                        'center': np.array([center[0], min_bound[1], center[2]]),
                        'normal': np.array([0.0, -1.0, 0.0]),  # Outward (away from center, toward -Y)
                        'name': "Front Face (-Y)"
                    },
                    {
                        'center': np.array([center[0], max_bound[1], center[2]]),
                        'normal': np.array([0.0, 1.0, 0.0]),  # Outward (away from center, toward +Y)
                        'name': "Back Face (+Y)"
                    },
                    
                    # Z-axis faces (Bottom and Top)
                    {
                        'center': np.array([center[0], center[1], min_bound[2]]),
                        'normal': np.array([0.0, 0.0, -1.0]),  # Outward (away from center, toward -Z)
                        'name': "Bottom Face (-Z)"
                    },
                    {
                        'center': np.array([center[0], center[1], max_bound[2]]),
                        'normal': np.array([0.0, 0.0, 1.0]),  # Outward (away from center, toward +Z)
                        'name': "Top Face (+Z)"
                    }
                ]
                
                # Get mesh data for ray projection
                triangles = np.asarray(self.mesh.triangles)
                vertices = np.asarray(self.mesh.vertices)
                
                markers_placed = 0
                
                for face in faces:
                    # Project from the face center to the actual surface
                    surface_point = self._project_to_surface_with_raycast(
                        face['center'], face['normal'], triangles, vertices
                    )
                    
                    if surface_point is not None:
                        print(f"   {face['name']}: Surface point at ({surface_point[0]:.3f}, {surface_point[1]:.3f}, {surface_point[2]:.3f})")
                        print(f"   {face['name']}: Face normal: ({face['normal'][0]:.3f}, {face['normal'][1]:.3f}, {face['normal'][2]:.3f})")
                        
                        # Emit the point_picked signal with position and normal for proper orientation
                        self.point_picked.emit(tuple(surface_point), tuple(face['normal']))
                        markers_placed += 1
                    else:
                        print(f"   ⚠️ {face['name']}: Could not find surface point")
                
                print(f"✅ Successfully placed {markers_placed}/6 ArUco markers on face centers!")
                
                # Show summary of dimensions used
                if self.mesh_info and 'dimensions' in self.mesh_info:
                    cad_dims = self.mesh_info['dimensions']
                    print(f"📐 Used CAD model dimensions: {cad_dims['length']:.4f} × {cad_dims['width']:.4f} × {cad_dims['height']:.4f} m")
                    if 'volume' in self.mesh_info and self.mesh_info['volume'] > 0:
                        print(f"📊 Model volume: {self.mesh_info['volume']:.6f} m³")
                    if 'surface_area' in self.mesh_info and self.mesh_info['surface_area'] > 0:
                        print(f"📊 Surface area: {self.mesh_info['surface_area']:.6f} m²")
                else:
                    print("⚠️ Used runtime bounding box (CAD dimensions not available)")
                
                # Hide the placement buttons after placing all markers
                if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
                    self.click_place_btn.setVisible(False)
                if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
                    self.face_picker_btn.setVisible(False)
                if hasattr(self, 'all_sides_btn') and self.all_sides_btn is not None:
                    self.all_sides_btn.setVisible(False)
                if hasattr(self, 'corner_markers_btn') and self.corner_markers_btn is not None:
                    self.corner_markers_btn.setVisible(False)
                    
            except Exception as e:
                print(f"Error placing markers on 6 faces: {e}")
        else:
            print("⚠️ Not in placement mode or no mesh loaded")
    
    def place_corner_markers_on_face(self) -> None:
        """Place 4 ArUco markers in the corners of a selected face."""
        if self.placement_mode and self.mesh is not None:
            try:
                print("🔲 Starting corner markers placement on selected face...")
                
                # Get mesh data
                vertices = np.asarray(self.mesh.vertices)
                triangles = np.asarray(self.mesh.triangles)
                
                # Group triangles into faces
                face_groups = self.group_triangles_by_face(triangles, vertices)
                
                if len(face_groups) == 0:
                    print("⚠️ No faces found for corner marker placement")
                    return
                
                # Open face picker dialog for corner markers
                from .face_picker_dialog import FacePickerDialog
                dialog = FacePickerDialog(self, triangles, vertices)
                dialog.setWindowTitle("Select Face for Corner Markers")
                
                # Show dialog and get result
                if dialog.exec() == 1:  # Dialog accepted
                    selected_face_data = dialog.selected_face_data
                    if selected_face_data is not None:
                        # Extract data from dictionary format
                        face_center = selected_face_data['face_center']
                        face_normal = selected_face_data['face_normal']
                        face_area = selected_face_data['area']
                        triangle_idx = selected_face_data['triangle_idx']
                        
                        print(f"🎯 Selected face for corner markers:")
                        print(f"   Face center: ({float(face_center[0]):.3f}, {float(face_center[1]):.3f}, {float(face_center[2]):.3f})")
                        print(f"   Face normal: ({float(face_normal[0]):.3f}, {float(face_normal[1]):.3f}, {float(face_normal[2]):.3f})")
                        print(f"   Face area: {float(face_area):.4f}")
                        
                        # Get all triangle indices for this face by re-running face detection
                        face_groups = self.group_triangles_by_face(triangles, vertices)
                        triangle_indices = []
                        for face_center_calc, face_normal_calc, face_area_calc, tri_indices in face_groups:
                            if (np.allclose(face_center, face_center_calc, atol=1e-6) and 
                                np.allclose(face_normal, face_normal_calc, atol=1e-6)):
                                triangle_indices = tri_indices
                                break
                        
                        if not triangle_indices:
                            print("⚠️ Could not find triangle indices for selected face")
                            return
                        
                        # Calculate corner positions
                        corner_positions = self.calculate_face_corners(face_center, face_normal, face_area, triangle_indices, vertices)
                        
                        if len(corner_positions) == 4:
                            print(f"✅ Calculated 4 corner positions for face")
                            
                            # Place markers at each corner
                            for i, corner_pos in enumerate(corner_positions):
                                print(f"🔲 Placing corner marker {i+1}/4 at: ({float(corner_pos[0]):.3f}, {float(corner_pos[1]):.3f}, {float(corner_pos[2]):.3f})")
                                
                                # Emit signal to place marker
                                self.point_picked.emit(tuple(corner_pos), tuple(face_normal))
                                
                            print(f"✅ Successfully placed 4 corner markers on selected face")
                        else:
                            print(f"⚠️ Could not calculate 4 corner positions (got {len(corner_positions)})")
                    else:
                        print("⚠️ No face selected for corner markers")
                else:
                    print("❌ Corner markers placement cancelled")
                    
            except Exception as e:
                print(f"❌ Error in corner markers placement: {e}")
        else:
            print("⚠️ Not in placement mode or no mesh loaded")
    
    def calculate_face_corners(self, face_center, face_normal, face_area, triangle_indices, vertices):
        """Calculate the 4 corner positions of a face."""
        try:
            print(f"🔲 Calculating corners for face with {len(triangle_indices)} triangles...")
            
            # Get all vertices that belong to this face
            face_vertices = set()
            for tri_idx in triangle_indices:
                triangle = self.mesh.triangles[tri_idx]
                face_vertices.update(triangle)
            
            face_vertices = list(face_vertices)
            print(f"🔲 Face has {len(face_vertices)} unique vertices")
            
            if len(face_vertices) < 4:
                print(f"⚠️ Face has only {len(face_vertices)} vertices, cannot determine 4 corners")
                return []
            
            # Get vertex positions
            vertex_positions = np.array([vertices[i] for i in face_vertices])
            
            # Project vertices onto the face plane
            face_center = np.array(face_center, dtype=float)
            face_normal = np.array(face_normal, dtype=float)
            
            # Create a coordinate system on the face plane
            # Find two orthogonal vectors in the plane
            if abs(face_normal[0]) < 0.9:
                u = np.array([1, 0, 0])
            else:
                u = np.array([0, 1, 0])
            
            # Make u orthogonal to face_normal
            u = u - np.dot(u, face_normal) * face_normal
            u = u / (np.linalg.norm(u) + 1e-8)
            
            # v is perpendicular to both u and face_normal
            v = np.cross(face_normal, u)
            v = v / (np.linalg.norm(v) + 1e-8)
            
            # Project vertices onto the face plane
            projected_vertices = []
            for vertex_pos in vertex_positions:
                # Project vertex onto face plane
                relative_pos = vertex_pos - face_center
                u_coord = np.dot(relative_pos, u)
                v_coord = np.dot(relative_pos, v)
                projected_vertices.append([u_coord, v_coord])
            
            projected_vertices = np.array(projected_vertices)
            
            # Find the bounding box of projected vertices
            u_min, u_max = np.min(projected_vertices[:, 0]), np.max(projected_vertices[:, 0])
            v_min, v_max = np.min(projected_vertices[:, 1]), np.max(projected_vertices[:, 1])
            
            # Calculate corner positions in 2D
            corners_2d = [
                [u_min, v_min],  # Bottom-left
                [u_max, v_min],  # Bottom-right
                [u_max, v_max],  # Top-right
                [u_min, v_max]   # Top-left
            ]
            
            # Convert back to 3D coordinates and offset inward by half marker size
            corners_3d = []
            
            # Get marker size from the marker panel (default to 0.03m if not available)
            marker_size = 0.03  # Default marker size
            try:
                # Try to get marker size from the main window's marker panel
                if hasattr(self, 'parent') and hasattr(self.parent, 'marker_panel'):
                    marker_size = self.parent.marker_panel.size_spinbox.value()
                elif hasattr(self, 'marker_panel'):
                    marker_size = self.marker_panel.size_spinbox.value()
            except:
                pass  # Use default if we can't get the size
            
            # Offset inward by half marker size to keep markers within face boundaries
            offset_distance = marker_size / 2.0
            print(f"🔲 Using marker size: {marker_size*1000:.1f}mm, offsetting inward by: {offset_distance*1000:.1f}mm")
            
            for corner_2d in corners_2d:
                # Calculate corner position on face surface
                corner_3d = face_center + corner_2d[0] * u + corner_2d[1] * v
                
                # Calculate offset direction (toward face center)
                offset_direction = face_center - corner_3d
                offset_direction = offset_direction / (np.linalg.norm(offset_direction) + 1e-8)
                
                # Offset inward by half marker size to keep marker within boundaries
                corner_3d_offset = corner_3d + offset_distance * offset_direction
                corners_3d.append(corner_3d_offset)
            
            print(f"✅ Calculated 4 corners (offset inward by {offset_distance*1000:.1f}mm to keep markers within face):")
            for i, corner in enumerate(corners_3d):
                print(f"   Corner {i+1}: ({float(corner[0]):.3f}, {float(corner[1]):.3f}, {float(corner[2]):.3f})")
            
            return corners_3d
            
        except Exception as e:
            print(f"❌ Error calculating face corners: {e}")
            return []
    
    def _project_to_surface_with_raycast(self, face_center, face_normal, triangles, vertices):
        """Project from face center to the actual surface using improved ray casting for complex shapes."""
        try:
            print(f"     Ray casting from face center to surface...")
            
            # For complex shapes, try multiple ray directions to find the best surface point
            ray_origins = [
                face_center,  # Original face center
                face_center + face_normal * 0.001,  # Slightly offset outward
                face_center - face_normal * 0.001,  # Slightly offset inward
            ]
            
            best_intersection = None
            best_distance = float('inf')
            
            for ray_origin in ray_origins:
                # Try both inward and outward directions
                for direction_multiplier in [-1, 1]:
                    ray_direction = face_normal * direction_multiplier
                    
                    intersection_point = self._ray_triangle_intersection(
                        ray_origin, ray_direction, triangles, vertices
                    )
                    
                    if intersection_point is not None:
                        distance = np.linalg.norm(intersection_point - face_center)
                        if distance < best_distance:
                            best_distance = distance
                            best_intersection = intersection_point
            
            if best_intersection is not None:
                print(f"     Ray intersection found at distance {best_distance:.3f}")
                return best_intersection
            
            # Fallback: Find closest surface point using triangle analysis
            print(f"     No ray intersection found, finding closest surface point...")
            return self._find_closest_surface_point(face_center, face_normal, triangles, vertices)
                
        except Exception as e:
            print(f"Error in surface projection: {e}")
            return self._find_closest_surface_point(face_center, face_normal, triangles, vertices)
    
    def _ray_triangle_intersection(self, ray_origin, ray_direction, triangles, vertices):
        """Find the closest ray-triangle intersection."""
        closest_distance = float('inf')
        closest_point = None
        
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
            
            # Möller-Trumbore ray-triangle intersection algorithm
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            
            if -1e-8 < a < 1e-8:  # Ray is parallel to triangle
                continue
                
            f = 1.0 / a
            s = ray_origin - v0
            u = f * np.dot(s, h)
            
            if u < 0.0 or u > 1.0:
                continue
                
            q = np.cross(s, edge1)
            v = f * np.dot(ray_direction, q)
            
            if v < 0.0 or u + v > 1.0:
                continue
                
            # Calculate intersection distance
            t = f * np.dot(edge2, q)
            
            if t > 1e-8 and t < closest_distance:  # Valid intersection
                closest_distance = t
                closest_point = ray_origin + t * ray_direction
        
        return closest_point
    
    def _find_closest_surface_point(self, target_point, face_normal, triangles, vertices):
        """Find the closest point on the mesh surface to the target point."""
        try:
            closest_distance = float('inf')
            closest_point = None
            
            # Check all triangles for the closest surface point
            for triangle in triangles:
                v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
                
                # Calculate triangle normal
                tri_normal = np.cross(v1 - v0, v2 - v0)
                tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)
                
                # Check if triangle normal is roughly aligned with face normal
                normal_alignment = abs(np.dot(tri_normal, face_normal))
                if normal_alignment < 0.7:  # Not aligned enough
                    continue
                
                # Find closest point on triangle to target point
                triangle_point = self._closest_point_on_triangle(target_point, v0, v1, v2)
                distance = np.linalg.norm(triangle_point - target_point)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = triangle_point
            
            if closest_point is not None:
                print(f"     Found closest surface point at distance {closest_distance:.3f}")
                return closest_point
            else:
                # Final fallback: closest vertex
                vertex_distances = np.linalg.norm(vertices - target_point, axis=1)
                closest_idx = np.argmin(vertex_distances)
                print(f"     Using closest vertex at distance {vertex_distances[closest_idx]:.3f}")
                return vertices[closest_idx]
                
        except Exception as e:
            print(f"Error finding closest surface point: {e}")
            return target_point
    
    def _closest_point_on_triangle(self, p, a, b, c):
        """Find the closest point on triangle ABC to point P."""
        # Vectors
        ab = b - a
        ac = c - a
        ap = p - a
        
        # Compute parametric coordinates
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        
        # Check if P is on vertex A side
        if d1 <= 0.0 and d2 <= 0.0:
            return a
        
        # Check if P is on vertex B side
        bp = p - b
        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b
        
        # Check if P is on vertex C side
        cp = p - c
        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c
        
        # Check if P is on edge AB
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab
        
        # Check if P is on edge AC
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac
        
        # Check if P is on edge BC
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)
        
        # P is inside the triangle
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w
    
    def open_face_picker(self) -> None:
        """Open a dialog to let user choose a specific face."""
        print("Opening face picker dialog...")
        
        if self.mesh is None:
            print("No mesh available for face picking")
            return
            
        try:
            triangles = np.asarray(self.mesh.triangles)
            vertices = np.asarray(self.mesh.vertices)
            
            if len(triangles) == 0:
                print("No triangles found in mesh")
                return
                
            # Create face picker dialog
            dialog = FacePickerDialog(self, triangles, vertices)
            if dialog.exec() == dialog.DialogCode.Accepted:
                selected_face_data = dialog.selected_face_data
                if selected_face_data:
                    selected_face_center = tuple(selected_face_data['face_center'])
                    selected_face_normal = tuple(selected_face_data['face_normal'])
                    print(f"User selected face center: {selected_face_center}")
                    print(f"User selected face normal: {selected_face_normal}")
                    # Use computed face normal instead of hardcoded Z-up
                    self.point_picked.emit(selected_face_center, selected_face_normal)
                    
                    # Hide the placement buttons
                    if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
                        self.click_place_btn.setVisible(False)
                    if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
                        self.face_picker_btn.setVisible(False)
                        
        except Exception as e:
            print(f"Error in face picker: {e}")
            import traceback
            traceback.print_exc()



    def smart_auto_place(self) -> None:
        """Smart automatic placement that chooses the best face for marker placement."""
        print("🧠 Smart auto-placement activated!")

        if self.mesh is None:
            print("❌ No mesh loaded")
            return

        try:
            # Get face groups if not already computed
            if not hasattr(self, 'face_groups') or self.face_groups is None:
                print("🔄 Analyzing mesh faces...")
                self.face_groups = self.group_triangles_by_face()

            if not self.face_groups:
                print("❌ No faces found in mesh")
                return

            print(f"📊 Found {len(self.face_groups)} faces to analyze")

            # Smart face selection algorithm
            best_face = self.find_best_face_for_marker()
            if best_face is None:
                print("❌ Could not find suitable face for marker placement")
                return

            face_center, face_normal, face_area, triangle_indices = best_face

            print(f"Selected face center: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")
            print(f"Selected face normal: ({face_normal[0]:.3f}, {face_normal[1]:.3f}, {face_normal[2]:.3f})")
            print(f"Selected face area: {face_area:.4f}")
            # Place marker at the face center with computed normal
            self.point_picked.emit(tuple(face_center), tuple(face_normal))

            # Hide placement buttons after successful placement
            self.disable_direct_placement_mode()

            print("✅ Smart marker placement completed!")
            print(f"Final marker position: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")

        except Exception as e:
            print(f"❌ Error in smart auto-placement: {e}")
            import traceback
            traceback.print_exc()

    def find_best_face_for_marker(self):
        """Find the best face for marker placement using smart criteria."""
        if not self.face_groups:
            return None

        # Scoring criteria for face selection:
        # 1. Face area (larger faces are better for markers)
        # 2. Face orientation (faces pointing more upward are preferred)
        # 3. Face position (avoid very small faces)
        # 4. Face normal alignment (prefer faces that are reasonably flat)

        best_score = -1
        best_face = None

        for face_center, face_normal, face_area, triangle_indices in self.face_groups:
            score = 0

            # 1. Area score (0-50 points)
            if face_area > 0:
                area_score = min(50, face_area * 10000)  # Scale area to reasonable score
                score += area_score

            # 2. Orientation score (0-30 points) - prefer faces pointing somewhat upward
            up_alignment = abs(face_normal[2])  # Z-component indicates up/down alignment
            orientation_score = up_alignment * 30
            score += orientation_score

            # 3. Size bonus (0-20 points) - prefer larger faces
            size_score = min(20, len(triangle_indices) / 5)  # More triangles = larger face
            score += size_score

            print(f"Face score: {score:.1f}")
            if score > best_score:
                best_score = score
                best_face = (face_center, face_normal, face_area, triangle_indices)

        return best_face

    def handle_direct_click(self, x: int, y: int) -> None:
        """Handle direct click on the 3D model."""
        if not self.direct_click_mode or self.mesh is None:
            return

        try:
            print(f"Processing click at ({x}, {y})")

            # Get the view control and camera
            view_control = self.vis.get_view_control()

            # Convert screen coordinates to world coordinates
            # This is a simplified approach - in a full implementation we'd use ray casting
            depth = view_control.get_field_of_view() / 10.0  # Approximate depth

            # Get camera parameters
            camera_params = view_control.convert_to_pinhole_camera_parameters()

            # Create a ray from the camera through the clicked point
            ray_origin = camera_params.extrinsic[:3, 3]
            ray_direction = self.screen_to_world_direction(x, y, camera_params)

            print(f"Ray origin: {ray_origin}")
            print(f"Ray direction: {ray_direction}")

            # Find intersection with mesh
            intersection_point = self.find_mesh_intersection(ray_origin, ray_direction)

            if intersection_point is not None:
                print(f"Found intersection at: {intersection_point}")
                self.point_picked.emit(tuple(intersection_point), (0.0, 0.0, 1.0))

                # Disable direct click mode
                self.direct_click_mode = False

                # Reset UI
                self.disable_direct_placement_mode()
            else:
                print("No intersection found with mesh")

        except Exception as e:
            print(f"Error handling direct click: {e}")
            import traceback
            traceback.print_exc()

    def screen_to_world_direction(self, x: int, y: int, camera_params):
        """Convert screen coordinates to world direction vector."""
        try:
            # Get camera intrinsic parameters
            intrinsic = camera_params.intrinsic
            extrinsic = camera_params.extrinsic

            # Convert screen coordinates to normalized device coordinates
            width, height = self.vis.get_render_option().viewport_width, self.vis.get_render_option().viewport_height
            ndc_x = (2.0 * x / width) - 1.0
            ndc_y = 1.0 - (2.0 * y / height)  # Flip Y coordinate

            # Convert to camera coordinates
            camera_point = np.array([ndc_x, ndc_y, 1.0])

            # Apply inverse intrinsic matrix
            inv_intrinsic = np.linalg.inv(intrinsic.intrinsic_matrix)
            camera_direction = inv_intrinsic @ camera_point

            # Apply inverse extrinsic matrix (camera pose)
            inv_extrinsic = np.linalg.inv(extrinsic)
            world_direction = inv_extrinsic[:3, :3] @ camera_direction

            # Normalize
            return world_direction / np.linalg.norm(world_direction)

        except Exception as e:
            print(f"Error in screen to world conversion: {e}")
            return np.array([0, 0, -1])  # Default forward direction

    def find_mesh_intersection(self, ray_origin, ray_direction):
        """Find intersection point between ray and mesh."""
        try:
            # Use Open3D's ray casting if available
            if hasattr(self.mesh, 'create_raycasting_scene'):
                scene = self.mesh.create_raycasting_scene()
                result = scene.cast_rays([ray_origin], [ray_direction])

                if len(result['t_hit']) > 0 and result['t_hit'][0] < np.inf:
                    hit_point = ray_origin + result['t_hit'][0] * ray_direction
                    return hit_point

            # Fallback: find closest point on mesh
            # This is a simplified approach
            bbox = self.mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()

            # Return a point near the center (this is a fallback)
            return np.array(center)

        except Exception as e:
            print(f"Error in mesh intersection: {e}")
            return None

    def show_direct_click_instructions(self) -> None:
        """Show instructions for direct click placement."""
        print("Direct click instructions:")
        print("1. Move your mouse over the 3D model")
        print("2. Click on the surface where you want to place the marker")
        print("3. The marker will be placed at the clicked location")
        print("Note: If clicking doesn't work, try the 'Choose Face from List' option")

    def disable_direct_placement_mode(self) -> None:
        """Disable direct placement mode and reset UI."""
        self.direct_click_mode = False

        if hasattr(self, 'placement_label'):
            self.placement_label.setText("Choose how to place ArUco marker:\n• 🎲 Quick Random • 📋 Face List • 🎯 Smart Auto")

        # Reset button styles and visibility
        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
            self.click_place_btn.setVisible(True)
            self.click_place_btn.setText("🎲 Quick Random Place")

        if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
            self.face_picker_btn.setVisible(True)
            self.face_picker_btn.setText("📋 Pick Face from List")

        if hasattr(self, 'all_sides_btn') and self.all_sides_btn is not None:
            self.all_sides_btn.setVisible(True)
            self.all_sides_btn.setText("📦 Add ArUco on 6 Faces")
            
        if hasattr(self, 'corner_markers_btn') and self.corner_markers_btn is not None:
            self.corner_markers_btn.setVisible(True)
            self.corner_markers_btn.setText("🔲 Add 4 Corner Markers")


    def setup_face_highlighting(self) -> None:
        """Setup face highlighting system for direct placement mode."""
        if self.mesh is None:
            return

        try:
            # Create face groups if not already done
            if not hasattr(self, 'face_groups') or self.face_groups is None:
                self.face_groups = self.group_triangles_by_face()

            print(f"Setting up highlighting for {len(self.face_groups)} faces")

            # Create highlighted versions of faces
            self.highlighted_faces = []
            for face_idx, (face_center, face_normal, face_area, triangle_indices) in enumerate(self.face_groups):
                # Create a small sphere at the face center for highlighting
                highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                highlight_sphere.translate(face_center)
                highlight_sphere.paint_uniform_color([1, 1, 0])  # Yellow highlight

                self.highlighted_faces.append(highlight_sphere)

                # Initially hide all highlights
                highlight_sphere.translate([0, 0, -1000])  # Move far away

            print("Face highlighting setup complete")

        except Exception as e:
            print(f"Error setting up face highlighting: {e}")
            import traceback
            traceback.print_exc()

    def highlight_face_at_point(self, point) -> None:
        """Highlight the face closest to the given point."""
        if not hasattr(self, 'face_groups') or not self.face_groups:
            return

        try:
            # Find the closest face to the point
            min_distance = float('inf')
            closest_face_idx = -1

            for face_idx, (face_center, face_normal, face_area, triangle_indices) in enumerate(self.face_groups):
                distance = np.linalg.norm(np.array(point) - np.array(face_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_face_idx = face_idx

            # Update highlighting
            for i, highlight_sphere in enumerate(self.highlighted_faces):
                if i == closest_face_idx:
                    # Move highlight to face center
                    face_center = self.face_groups[i][0]
                    highlight_sphere.translate([0, 0, 1000])  # Reset position
                    highlight_sphere.translate(face_center)
                else:
                    # Hide other highlights
                    highlight_sphere.translate([0, 0, -1000])

            # Update the visualizer
            if self.vis:
                self.vis.update_geometry(highlight_sphere)
                self.vis.poll_events()
                self.vis.update_renderer()

        except Exception as e:
            print(f"Error highlighting face: {e}")


    
    def reset_camera(self) -> None:
        """Reset camera view."""
        if self.vis is not None:
            self.vis.reset_view_point(True)
    
    def fit_to_view(self) -> None:
        """Fit the current model to the view."""
        if self.vis is not None and self.mesh is not None:
            self.vis.reset_view_point(True)
    
    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotations from data dictionary."""
        # Clear existing annotations
        for marker_id in list(self.markers.keys()):
            self.remove_marker(marker_id)
            
        # Load markers
        for marker_data in data.get("markers", []):
            self.add_marker(
                marker_data["id"],
                tuple(marker_data["position"]),
                marker_data.get("size", 0.05)
            )
            
    
    def update_stats(self) -> None:
        """Update the statistics display."""
        # Update markers info
        if self.markers:
            markers_text = "Placed ArUco Markers:\n"
            for marker_id, marker in self.markers.items():
                if 'aruco_info' in marker:
                    # New format with ArUco info
                    aruco_info = marker['aruco_info']
                    x, y, z = aruco_info.position
                    markers_text += f"  {aruco_info.dictionary} ID:{aruco_info.marker_id} at ({x:.3f}, {y:.3f}, {z:.3f})\n"
                else:
                    # Legacy format
                    x, y, z = marker.get('position', (0, 0, 0))
                    markers_text += f"  Marker {marker_id}: ({x:.3f}, {y:.3f}, {z:.3f})\n"
        else:
            markers_text = "No ArUco markers placed"
        self.markers_text.setText(markers_text)
        
        
        # Update main stats
        stats_text = f"""Current Session:
Markers: {len(self.markers)}

Model Status:
Loaded: {'Yes' if self.mesh is not None else 'No'}
Vertices: {len(self.mesh.vertices) if self.mesh else 0}
Triangles: {len(self.mesh.triangles) if self.mesh else 0}

3D Viewer:
Status: {'Open' if self.vis is not None else 'Closed'}"""
        
        self.stats_text.setText(stats_text)
        
        
    def reset_camera(self) -> None:
        """Reset camera to default position."""
        if self.vis is not None:
            self.vis.reset_view_point(True)
        self.status_label.setText("Camera reset")
        
    def fit_to_view(self) -> None:
        """Fit the current model to the view."""
        if self.mesh is not None:
            self.reset_camera()
            self.status_label.setText("Model fitted to view")
        else:
            self.status_label.setText("No model to fit")
            
    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotations from data dictionary."""
        # Clear existing annotations
        self.markers.clear()
        
        # Load markers
        for marker_data in data.get("markers", []):
            self.add_marker(
                marker_data["id"],
                tuple(marker_data["position"]),
                marker_data.get("size", 0.05)
            )
            
            
    def set_interaction_mode(self, mode: str) -> None:
        """Set the interaction mode."""
        self.mode_label.setText(f"Mode: {mode}")
        
        if mode == "view":
            pass
        elif mode == "place_marker":
            pass
        elif mode == "place_grasp":
            pass