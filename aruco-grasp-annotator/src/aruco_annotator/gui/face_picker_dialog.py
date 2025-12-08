"""Face picker dialog for selecting specific faces from a mesh."""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QApplication
)


class FacePickerDialog(QWidget):
    """Dialog for picking a specific face from the mesh."""
    
    def __init__(self, parent, triangles, vertices):
        super().__init__()
        self.parent = parent
        self.triangles = triangles
        self.vertices = vertices
        self.selected_face_data = None
        self.dialog_result = 0
        
        self.setWindowTitle("Select Face for ArUco Marker Placement")
        # Make dialog smaller and more compact for Linux displays
        self.setGeometry(200, 200, 550, 400)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)  # Reduce spacing between elements
        layout.setContentsMargins(10, 10, 10, 10)  # Reduce margins
        
        # Title
        title = QLabel("Choose Face for ArUco Marker")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        layout.addWidget(title)
        
        # Instructions - more compact
        instructions = QLabel("Select a face and click 'Place Marker Here'")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("margin: 2px; color: #666; font-size: 11px;")
        layout.addWidget(instructions)
        
        # Face list - more compact
        self.face_list = QListWidget()
        self.face_list.setMaximumHeight(250)  # Limit height for smaller screens
        self.face_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 3px;
                font-family: monospace;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
        """)
        
        # Populate face list with face information
        self.populate_face_list()
        layout.addWidget(self.face_list)
        
        # Buttons - more compact
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 5, 0, 0)  # Reduce top margin
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setFixedHeight(32)  # Fixed smaller height
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #616161; }
        """)
        
        self.select_btn = QPushButton("🎯 Place Marker Here")
        self.select_btn.clicked.connect(self.accept)
        self.select_btn.setEnabled(False)
        self.select_btn.setFixedHeight(32)  # Fixed smaller height
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.select_btn)
        layout.addLayout(button_layout)
        
        # Connect selection change
        self.face_list.itemSelectionChanged.connect(self.on_selection_changed)
        
    def populate_face_list(self):
        """Populate the face list with actual faces (not individual triangles)."""
        print(f"Analyzing {len(self.triangles)} triangles to find actual faces...")
        print("NOTE: Algorithm finds faces DYNAMICALLY - it will find as many faces as exist in your model!")
        print(f"Mesh type: {type(self.mesh).__name__ if hasattr(self, 'mesh') else 'Unknown'}")

        # Group triangles by their normal vectors to find actual faces
        face_groups = self.group_triangles_by_face()

        print(f"Algorithm found {len(face_groups)} actual faces (not hardcoded!)")
        print("Each face is made up of triangles pointing in the same direction.")
        
        # Additional debugging for complex shapes
        if len(face_groups) > 10:
            print(f"⚠️  Complex shape detected with {len(face_groups)} faces!")
            print("This might be a detailed model with many small faces.")
        elif len(face_groups) < 4:
            print(f"⚠️  Simple shape detected with only {len(face_groups)} faces!")
            print("This might be a basic geometric shape.")
        
        for face_idx, (face_center, face_normal, face_area, triangle_indices) in enumerate(face_groups):
            # Determine face orientation
            face_name = self.get_face_name(face_normal)
            
            item_text = f"{face_name}: Center ({face_center[0]:6.3f}, {face_center[1]:6.3f}, {face_center[2]:6.3f}) | Area: {face_area:.4f}"
            
            item = QListWidgetItem(item_text)
            # Store the face center and representative triangle index
            item.setData(Qt.ItemDataRole.UserRole, {
                'face_center': face_center,
                'triangle_idx': triangle_indices[0],  # Use first triangle as representative
                'face_normal': face_normal,
                'area': face_area
            })
            self.face_list.addItem(item)
            
        print(f"Added {self.face_list.count()} faces to the list")
        
    def group_triangles_by_face(self):
        """Improved face detection algorithm using normal vectors, spatial connectivity, and coplanarity."""
        print("🔍 Starting improved face detection algorithm...")
        print(f"Processing {len(self.triangles)} triangles...")
        
        face_groups = []
        normal_tolerance = 0.02  # Tighter tolerance for better precision
        spatial_threshold = 0.1  # Maximum distance between triangle centers to consider them connected
        min_face_area = 1e-6  # Minimum area to consider a valid face
        
        # Calculate triangle properties
        triangle_data = []
        for i, triangle in enumerate(self.triangles):
            v1, v2, v3 = self.vertices[triangle[0]], self.vertices[triangle[1]], self.vertices[triangle[2]]
            
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
                        self.triangles[current_idx], self.triangles[j]
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
    
    def _triangles_share_edge_or_vertex(self, tri1, tri2):
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
    
    def get_face_name(self, normal):
        """Get a descriptive name for the face based on its normal vector."""
        # Normalize the normal vector
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Define primary directions with more tolerance
        directions = {
            'Top': np.array([0, 0, 1]),
            'Bottom': np.array([0, 0, -1]),
            'Front': np.array([0, 1, 0]),
            'Back': np.array([0, -1, 0]),
            'Right': np.array([1, 0, 0]),
            'Left': np.array([-1, 0, 0])
        }
        
        # Find the closest direction
        best_match = 'Unknown'
        best_similarity = -1
        
        for name, direction in directions.items():
            similarity = abs(np.dot(normal, direction))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # If similarity is low, use coordinate-based naming
        if best_similarity < 0.7:  # Not well aligned with standard directions
            # Create a more descriptive name based on normal components
            x, y, z = normal
            if abs(x) > abs(y) and abs(x) > abs(z):
                direction = "X" if x > 0 else "-X"
            elif abs(y) > abs(z):
                direction = "Y" if y > 0 else "-Y"
            else:
                direction = "Z" if z > 0 else "-Z"
            return f"Face ({direction})"
        
        return f"{best_match} Face"
        
    def on_selection_changed(self):
        """Handle face selection change."""
        current_item = self.face_list.currentItem()
        if current_item:
            self.selected_face_data = current_item.data(Qt.ItemDataRole.UserRole)
            self.select_btn.setEnabled(True)
            face_center = self.selected_face_data['face_center']
            print(f"Selected face center: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")
        else:
            self.selected_face_data = None
            self.select_btn.setEnabled(False)
            
    def get_selected_face_center(self):
        """Get the center of the selected face."""
        if self.selected_face_data is not None:
            return tuple(self.selected_face_data['face_center'])
        return None
        
    def accept(self):
        """Accept the selection and close."""
        self.dialog_result = 1
        self.close()
        
    def reject(self):
        """Reject the selection and close."""
        self.dialog_result = 0
        self.close()
        
    def exec(self):
        """Show the dialog modally."""
        self.show()
        # Process events until dialog is closed
        while self.isVisible():
            QApplication.processEvents()
        return self.dialog_result
        
    class DialogCode:
        Accepted = 1
        Rejected = 0
