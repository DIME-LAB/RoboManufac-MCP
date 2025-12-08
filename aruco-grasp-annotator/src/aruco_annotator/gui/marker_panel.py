"""Panel for managing ArUco markers on the 3D model."""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy.spatial.transform import Rotation
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem, QLabel,
    QSpinBox, QDoubleSpinBox, QInputDialog, QMessageBox,
    QComboBox, QTextEdit
)

from ..utils.aruco_utils import ArUcoGenerator, ArUcoMarkerInfo


class MarkerListItem(QListWidgetItem):
    """Custom list item for ArUco markers."""
    
    def __init__(self, marker_id: int, aruco_info: ArUcoMarkerInfo):
        super().__init__()
        self.marker_id = marker_id
        self.aruco_info = aruco_info
        self.update_text()
        
    def update_text(self) -> None:
        """Update the display text with pose information."""
        x, y, z = self.aruco_info.position
        pose_summary = self.aruco_info.get_pose_summary()
        self.setText(f"ArUco {self.marker_id}: {self.aruco_info.dictionary} ID:{self.aruco_info.marker_id} ({x:.3f}, {y:.3f}, {z:.3f}) - {pose_summary}")
        
    def get_data(self) -> Dict[str, Any]:
        """Get marker data as dictionary."""
        data = self.aruco_info.to_dict()
        data["id"] = self.marker_id  # Add our internal marker ID
        return data


class MarkerPanel(QGroupBox):
    """Panel for managing ArUco markers."""
    
    # Signals
    marker_added = pyqtSignal(int, ArUcoMarkerInfo)   # marker_id, aruco_info
    marker_removed = pyqtSignal(int)                  # marker_id
    marker_selected = pyqtSignal(int)                 # marker_id
    marker_position_changed = pyqtSignal(int, tuple)  # marker_id, new_position
    marker_moved = pyqtSignal(int, tuple)             # marker_id, new_position
    marker_orientation_changed = pyqtSignal(int, tuple)  # marker_id, new_orientation (roll, pitch, yaw)
    placement_mode_requested = pyqtSignal(bool)       # enable/disable placement mode
    
    def __init__(self):
        super().__init__("ArUco Markers")
        self.next_marker_id = 0
        self.markers: Dict[int, MarkerListItem] = {}
        self.aruco_generator = ArUcoGenerator()
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # Reduce spacing between sections
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # ArUco Configuration section - more compact
        aruco_config_group = QGroupBox("ArUco Configuration")
        aruco_config_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; }")
        aruco_layout = QVBoxLayout(aruco_config_group)
        aruco_layout.setSpacing(4)  # Reduce spacing
        aruco_layout.setContentsMargins(5, 10, 5, 5)  # Reduce margins
        
        # Dictionary selection
        dict_layout = QHBoxLayout()
        dict_layout.addWidget(QLabel("Dictionary:"))
        self.dict_combo = QComboBox()
        self.dict_combo.addItems(ArUcoGenerator.get_available_dictionaries())
        self.dict_combo.setCurrentText("DICT_4X4_50")  # Default to 4x4_50
        self.dict_combo.setToolTip("ArUco dictionary to use")
        dict_layout.addWidget(self.dict_combo)
        aruco_layout.addLayout(dict_layout)
        
        # Marker ID selection
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Marker ID:"))
        self.marker_id_spinbox = QSpinBox()
        self.marker_id_spinbox.setRange(0, 49)  # Default for 4x4_50
        self.marker_id_spinbox.setValue(0)
        self.marker_id_spinbox.setToolTip("ID of the ArUco marker (must be unique)")
        id_layout.addWidget(self.marker_id_spinbox)
        aruco_layout.addLayout(id_layout)
        
        # Marker size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setRange(0.001, 1.0)  # Allow 1mm to 1m
        self.size_spinbox.setSingleStep(0.01)
        self.size_spinbox.setDecimals(4)  # Allow 4 decimal places (0.0001m = 0.1mm precision)
        self.size_spinbox.setSuffix(" m")
        self.size_spinbox.setToolTip("Physical size of the ArUco marker")
        self.size_spinbox.setValue(0.03)
        size_layout.addWidget(self.size_spinbox)
        aruco_layout.addLayout(size_layout)
        
        # Border width control
        border_layout = QHBoxLayout()
        border_layout.addWidget(QLabel("Border Width:"))
        self.border_spinbox = QDoubleSpinBox()
        self.border_spinbox.setRange(0.0, 0.5)  # Allow 0% to 50% of marker size
        self.border_spinbox.setSingleStep(0.01)
        self.border_spinbox.setDecimals(3)  # Allow 3 decimal places
        self.border_spinbox.setSuffix(" %")
        self.border_spinbox.setToolTip("White border width as percentage of marker size (0% = no border, 10% = 10% border)")
        self.border_spinbox.setValue(0.2)  # Default 10% border
        border_layout.addWidget(self.border_spinbox)
        aruco_layout.addLayout(border_layout)
        
        layout.addWidget(aruco_config_group)
        
        # Add marker section
        add_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add ArUco Marker")
        self.add_button.setToolTip("Click to place a new ArUco marker on the 3D model")
        add_layout.addWidget(self.add_button)
        
        layout.addLayout(add_layout)
        
        # Marker list - more compact
        list_label = QLabel("Placed Markers:")
        list_label.setStyleSheet("font-size: 11px; font-weight: bold; margin: 2px;")
        layout.addWidget(list_label)
        
        self.marker_list = QListWidget()
        self.marker_list.setMaximumHeight(120)  # Reduced height
        self.marker_list.setToolTip("List of placed ArUco markers")
        self.marker_list.setStyleSheet("font-size: 10px;")  # Smaller font
        layout.addWidget(self.marker_list)
        
        # Marker controls
        controls_layout = QHBoxLayout()
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.setEnabled(False)
        self.edit_button.setToolTip("Edit selected marker properties")
        controls_layout.addWidget(self.edit_button)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip("Remove selected marker")
        controls_layout.addWidget(self.remove_button)
        
        self.delete_all_button = QPushButton("🗑️ Delete All")
        self.delete_all_button.setEnabled(False)  # Enabled when there are markers
        self.delete_all_button.setToolTip("Delete all markers and reset counter")
        self.delete_all_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        controls_layout.addWidget(self.delete_all_button)
        
        layout.addLayout(controls_layout)
        
        # Detailed pose information section
        pose_info_group = QGroupBox("Marker Pose Information")
        pose_info_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; }")
        pose_info_layout = QVBoxLayout(pose_info_group)
        pose_info_layout.setSpacing(4)
        pose_info_layout.setContentsMargins(5, 10, 5, 5)
        
        # Pose information display
        self.pose_info_text = QTextEdit()
        self.pose_info_text.setMaximumHeight(120)
        self.pose_info_text.setReadOnly(True)
        self.pose_info_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 9px;
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        self.pose_info_text.setPlaceholderText("Select a marker to view detailed pose information...")
        pose_info_layout.addWidget(self.pose_info_text)
        
        # Wireframe attachment info
        wireframe_layout = QHBoxLayout()
        wireframe_layout.addWidget(QLabel("Wireframe Attachment:"))
        self.wireframe_attachment_label = QLabel("N/A")
        self.wireframe_attachment_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 9px; color: #0066cc;")
        wireframe_layout.addWidget(self.wireframe_attachment_label)
        pose_info_layout.addLayout(wireframe_layout)
        
        layout.addWidget(pose_info_group)
        
        # Marker movement controls - more compact
        movement_group = QGroupBox("Marker Movement")
        movement_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; }")
        movement_layout = QVBoxLayout(movement_group)
        movement_layout.setSpacing(4)  # Reduce spacing
        movement_layout.setContentsMargins(5, 10, 5, 5)  # Reduce margins
        
        # Step size control
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:"))
        self.step_spinbox = QDoubleSpinBox()
        self.step_spinbox.setRange(0.001, 0.1)
        self.step_spinbox.setValue(0.01)
        self.step_spinbox.setSingleStep(0.001)
        self.step_spinbox.setDecimals(3)
        self.step_spinbox.setSuffix(" m")
        self.step_spinbox.setToolTip("Distance to move marker per step")
        step_layout.addWidget(self.step_spinbox)
        movement_layout.addLayout(step_layout)
        
        # Directional buttons
        direction_layout = QVBoxLayout()
        
        # Up button
        up_layout = QHBoxLayout()
        up_layout.addStretch()
        self.up_button = QPushButton("↑")
        self.up_button.setFixedSize(40, 30)
        self.up_button.setEnabled(False)
        self.up_button.setToolTip("Move marker up")
        up_layout.addWidget(self.up_button)
        up_layout.addStretch()
        direction_layout.addLayout(up_layout)
        
        # Left, Down, Right buttons
        ldr_layout = QHBoxLayout()
        self.left_button = QPushButton("←")
        self.left_button.setFixedSize(40, 30)
        self.left_button.setEnabled(False)
        self.left_button.setToolTip("Move marker left")
        ldr_layout.addWidget(self.left_button)
        
        self.down_button = QPushButton("↓")
        self.down_button.setFixedSize(40, 30)
        self.down_button.setEnabled(False)
        self.down_button.setToolTip("Move marker down")
        ldr_layout.addWidget(self.down_button)
        
        self.right_button = QPushButton("→")
        self.right_button.setFixedSize(40, 30)
        self.right_button.setEnabled(False)
        self.right_button.setToolTip("Move marker right")
        ldr_layout.addWidget(self.right_button)
        direction_layout.addLayout(ldr_layout)
        
        movement_layout.addLayout(direction_layout)
        layout.addWidget(movement_group)
        
        # Marker orientation controls - more compact
        orientation_group = QGroupBox("Marker Orientation")
        orientation_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; }")
        orientation_layout = QVBoxLayout(orientation_group)
        orientation_layout.setSpacing(4)  # Reduce spacing
        orientation_layout.setContentsMargins(5, 10, 5, 5)  # Reduce margins
        
        # Rotation step size control (fixed at 90 degrees)
        rot_step_layout = QHBoxLayout()
        rot_step_layout.addWidget(QLabel("Rotation Step:"))
        self.rot_step_label = QLabel("90°")
        self.rot_step_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        self.rot_step_label.setToolTip("Fixed 90-degree rotation steps")
        rot_step_layout.addWidget(self.rot_step_label)
        orientation_layout.addLayout(rot_step_layout)
        
        # Rotation buttons
        rotation_layout = QVBoxLayout()
        
        # Pitch controls (X-axis rotation)
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch (X):"))
        self.pitch_up_button = QPushButton("↗")
        self.pitch_up_button.setFixedSize(40, 30)
        self.pitch_up_button.setEnabled(False)
        self.pitch_up_button.setToolTip("Rotate marker up 90° (positive pitch)")
        pitch_layout.addWidget(self.pitch_up_button)
        
        self.pitch_down_button = QPushButton("↘")
        self.pitch_down_button.setFixedSize(40, 30)
        self.pitch_down_button.setEnabled(False)
        self.pitch_down_button.setToolTip("Rotate marker down 90° (negative pitch)")
        pitch_layout.addWidget(self.pitch_down_button)
        rotation_layout.addLayout(pitch_layout)
        
        # Yaw controls (Y-axis rotation)
        yaw_layout = QHBoxLayout()
        yaw_layout.addWidget(QLabel("Yaw (Y):"))
        self.yaw_left_button = QPushButton("↖")
        self.yaw_left_button.setFixedSize(40, 30)
        self.yaw_left_button.setEnabled(False)
        self.yaw_left_button.setToolTip("Rotate marker left 90° (negative yaw)")
        yaw_layout.addWidget(self.yaw_left_button)
        
        self.yaw_right_button = QPushButton("↗")
        self.yaw_right_button.setFixedSize(40, 30)
        self.yaw_right_button.setEnabled(False)
        self.yaw_right_button.setToolTip("Rotate marker right 90° (positive yaw)")
        yaw_layout.addWidget(self.yaw_right_button)
        rotation_layout.addLayout(yaw_layout)
        
        # Roll controls (Z-axis rotation)
        roll_layout = QHBoxLayout()
        roll_layout.addWidget(QLabel("Roll (Z):"))
        self.roll_left_button = QPushButton("↶")
        self.roll_left_button.setFixedSize(40, 30)
        self.roll_left_button.setEnabled(False)
        self.roll_left_button.setToolTip("Rotate marker counter-clockwise 90° (negative roll)")
        roll_layout.addWidget(self.roll_left_button)
        
        self.roll_right_button = QPushButton("↷")
        self.roll_right_button.setFixedSize(40, 30)
        self.roll_right_button.setEnabled(False)
        self.roll_right_button.setToolTip("Rotate marker clockwise 90° (positive roll)")
        roll_layout.addWidget(self.roll_right_button)
        rotation_layout.addLayout(roll_layout)
        
        # Reset orientation button
        reset_orientation_layout = QHBoxLayout()
        self.reset_orientation_button = QPushButton("Reset Orientation")
        self.reset_orientation_button.setEnabled(False)
        self.reset_orientation_button.setToolTip("Reset marker orientation to default")
        reset_orientation_layout.addWidget(self.reset_orientation_button)
        rotation_layout.addLayout(reset_orientation_layout)
        
        orientation_layout.addLayout(rotation_layout)
        layout.addWidget(orientation_group)
        
        # Position input section (for manual entry) - more compact
        position_group = QGroupBox("Manual Position Entry")
        position_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; }")
        position_layout = QVBoxLayout(position_group)
        position_layout.setSpacing(4)  # Reduce spacing
        position_layout.setContentsMargins(5, 10, 5, 5)  # Reduce margins
        
        coords_layout = QHBoxLayout()
        
        # X coordinate
        coords_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-10.0, 10.0)
        self.x_spinbox.setDecimals(3)
        self.x_spinbox.setSuffix(" m")
        coords_layout.addWidget(self.x_spinbox)
        
        # Y coordinate
        coords_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-10.0, 10.0)
        self.y_spinbox.setDecimals(3)
        self.y_spinbox.setSuffix(" m")
        coords_layout.addWidget(self.y_spinbox)
        
        # Z coordinate
        coords_layout.addWidget(QLabel("Z:"))
        self.z_spinbox = QDoubleSpinBox()
        self.z_spinbox.setRange(-10.0, 10.0)
        self.z_spinbox.setDecimals(3)
        self.z_spinbox.setSuffix(" m")
        coords_layout.addWidget(self.z_spinbox)
        
        position_layout.addLayout(coords_layout)
        
        self.place_at_coords_button = QPushButton("Place at Coordinates")
        self.place_at_coords_button.setToolTip("Place marker at manually entered coordinates")
        position_layout.addWidget(self.place_at_coords_button)
        
        layout.addWidget(position_group)
        
        # Statistics
        self.stats_label = QLabel("Markers: 0")
        self.stats_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.stats_label)
        
    def setup_connections(self) -> None:
        """Setup signal-slot connections."""
        self.add_button.clicked.connect(self.enter_placement_mode)
        self.place_at_coords_button.clicked.connect(self.place_marker_at_coordinates)
        self.edit_button.clicked.connect(self.edit_selected_marker)
        self.remove_button.clicked.connect(self.remove_selected_marker)
        self.delete_all_button.clicked.connect(self.delete_all_markers)
        
        # Connect movement buttons
        self.up_button.clicked.connect(lambda: self.move_marker('up'))
        self.down_button.clicked.connect(lambda: self.move_marker('down'))
        self.left_button.clicked.connect(lambda: self.move_marker('left'))
        self.right_button.clicked.connect(lambda: self.move_marker('right'))
        
        # Connect orientation buttons
        self.pitch_up_button.clicked.connect(lambda: self.rotate_marker('pitch_up'))
        self.pitch_down_button.clicked.connect(lambda: self.rotate_marker('pitch_down'))
        self.yaw_left_button.clicked.connect(lambda: self.rotate_marker('yaw_left'))
        self.yaw_right_button.clicked.connect(lambda: self.rotate_marker('yaw_right'))
        self.roll_left_button.clicked.connect(lambda: self.rotate_marker('roll_left'))
        self.roll_right_button.clicked.connect(lambda: self.rotate_marker('roll_right'))
        self.reset_orientation_button.clicked.connect(self.reset_marker_orientation)
        
        self.marker_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.marker_list.itemDoubleClicked.connect(self.edit_selected_marker)
        
        # Connect spinbox changes to update marker position
        self.x_spinbox.valueChanged.connect(self.on_coordinate_changed)
        self.y_spinbox.valueChanged.connect(self.on_coordinate_changed)
        self.z_spinbox.valueChanged.connect(self.on_coordinate_changed)
        
        # Connect ArUco configuration changes
        self.dict_combo.currentTextChanged.connect(self.on_dictionary_changed)
        
    def on_dictionary_changed(self) -> None:
        """Update marker ID range when dictionary changes."""
        current_dict = self.dict_combo.currentText()
        max_id = ArUcoGenerator.get_max_id_for_dict(current_dict)
        self.marker_id_spinbox.setRange(0, max_id)
        self.marker_id_spinbox.setToolTip(f"ID of the ArUco marker (0-{max_id} for {current_dict})")
        
        # Reset to 0 if current value is out of range
        if self.marker_id_spinbox.value() > max_id:
            self.marker_id_spinbox.setValue(0)
        
    def enter_placement_mode(self) -> None:
        """Enter marker placement mode."""
        # Emit signal to tell the viewer to enter placement mode
        self.placement_mode_requested.emit(True)
        
        # Update button text to indicate placement mode
        self.add_button.setText("Cancel Placement")
        self.add_button.clicked.disconnect()
        self.add_button.clicked.connect(self.cancel_placement_mode)
        
        # Disable other controls during placement
        self.place_at_coords_button.setEnabled(False)
        self.edit_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        
    def cancel_placement_mode(self) -> None:
        """Cancel marker placement mode."""
        # Emit signal to tell the viewer to exit placement mode
        self.placement_mode_requested.emit(False)
        
        # Restore button text and connection
        self.add_button.setText("Add Marker")
        self.add_button.clicked.disconnect()
        self.add_button.clicked.connect(self.enter_placement_mode)
        
        # Re-enable controls
        self.place_at_coords_button.setEnabled(True)
        current_item = self.marker_list.currentItem()
        has_selection = current_item is not None
        self.edit_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        
    def place_marker_at_coordinates(self) -> None:
        """Place marker at manually entered coordinates."""
        position = (
            self.x_spinbox.value(),
            self.y_spinbox.value(),
            self.z_spinbox.value()
        )
        self.place_marker_at_position(position)
        
    def place_marker_at_position(self, position: tuple, normal: tuple = (0.0, 0.0, 1.0)) -> None:
        """Place a marker at the specified position with proper orientation based on surface normal."""
        marker_id = self.next_marker_id
        print(f"🔢 Creating marker with ID: {marker_id} (next_marker_id was: {self.next_marker_id})")
        
        # Get ArUco configuration
        dictionary = self.dict_combo.currentText()
        aruco_id = self.marker_id_spinbox.value()
        size = self.size_spinbox.value()
        border_width = self.border_spinbox.value()
        
        # Check if this ArUco ID is already used
        for existing_item in self.markers.values():
            if (existing_item.aruco_info.dictionary == dictionary and 
                existing_item.aruco_info.marker_id == aruco_id):
                QMessageBox.warning(
                    self,
                    "Duplicate ArUco Marker",
                    f"ArUco marker {dictionary} ID:{aruco_id} is already placed.\n"
                    f"Each ArUco marker must have a unique dictionary-ID combination."
                )
                return
        
        # Calculate marker orientation from surface normal
        # Convert normal vector to rotation angles that align marker with surface
        rotation = self._calculate_rotation_from_normal(normal)
        
        print(f"📐 Placing marker with normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
        print(f"📐 Calculated rotation: ({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}) rad")
        
        # Note: The marker position becomes the origin (0,0,0) for the object coordinate system
        # This is important for robotics applications where object poses are relative to the marker
        
        # Create ArUco marker info
        aruco_info = ArUcoMarkerInfo(
            dictionary=dictionary,
            marker_id=aruco_id,
            position=position,
            size=size,
            rotation=rotation,  # Add rotation based on surface normal
            border_width=border_width
        )
        
        # Update CAD object pose information
        self._update_marker_cad_pose(aruco_info, position, normal)
        
        # Create list item
        item = MarkerListItem(marker_id, aruco_info)
        self.markers[marker_id] = item
        
        # Add to list
        self.marker_list.addItem(item)
        
        # Emit signal
        self.marker_added.emit(marker_id, aruco_info)
        
        # Update stats
        self.update_stats()
        
        # Increment ID for next marker
        self.next_marker_id += 1
        
        # Auto-increment ArUco ID for convenience (if not at max)
        max_id = ArUcoGenerator.get_max_id_for_dict(dictionary)
        if aruco_id < max_id:
            self.marker_id_spinbox.setValue(aruco_id + 1)
    
    def _update_marker_cad_pose(self, aruco_info: ArUcoMarkerInfo, position: tuple, normal: tuple) -> None:
        """Update marker with CAD object pose information."""
        # Get CAD object information from the 3D viewer
        cad_info = self._get_cad_object_info()
        
        if cad_info:
            # Determine face type based on surface normal
            face_type = self._determine_face_type(normal)
            
            # Update the marker's CAD object pose information
            aruco_info.update_cad_object_pose(
                cad_center=cad_info["center"],
                cad_dimensions=cad_info["dimensions"],
                surface_normal=normal,
                face_type=face_type
            )
            
            print(f"📐 Updated marker CAD pose: {aruco_info.get_pose_summary()}")
        else:
            print("⚠️  No CAD object information available for pose calculation")
    
    def _get_cad_object_info(self) -> Optional[Dict]:
        """Get CAD object information from the 3D viewer."""
        # Use stored CAD object information if available
        return self.get_cad_object_info()
    
    def set_cad_object_info(self, cad_info: Dict) -> None:
        """Set CAD object information for pose calculations."""
        self.cad_object_info = cad_info
    
    def get_cad_object_info(self) -> Optional[Dict]:
        """Get stored CAD object information."""
        return getattr(self, 'cad_object_info', None)
    
    def _update_marker_pose_info(self, aruco_info: ArUcoMarkerInfo) -> None:
        """Update the CAD object pose information for a marker after position/rotation changes."""
        # Get CAD object information
        cad_info = self.get_cad_object_info()
        
        if cad_info:
            # Get the current surface normal (we'll need to recalculate this)
            # For now, we'll use the existing surface normal from the pose info
            current_normal = aruco_info.cad_object_pose.get("surface_normal", (0.0, 0.0, 1.0))
            current_face_type = aruco_info.cad_object_pose.get("face_type", "unknown")
            
            # Update the marker's CAD object pose information
            aruco_info.update_cad_object_pose(
                cad_center=cad_info["center"],
                cad_dimensions=cad_info["dimensions"],
                surface_normal=current_normal,
                face_type=current_face_type
            )
            
            print(f"📐 Updated marker CAD pose after movement: {aruco_info.get_pose_summary()}")
        else:
            print("⚠️  No CAD object information available for pose update")
    
    def _determine_face_type(self, normal: tuple) -> str:
        """Determine the type of face based on surface normal."""
        import numpy as np
        
        n = np.array(normal)
        n = n / (np.linalg.norm(n) + 1e-8)
        
        # Check for primary axis alignments
        if np.allclose(n, [0, 0, 1], atol=1e-6):
            return "top"
        elif np.allclose(n, [0, 0, -1], atol=1e-6):
            return "bottom"
        elif np.allclose(n, [1, 0, 0], atol=1e-6):
            return "right"
        elif np.allclose(n, [-1, 0, 0], atol=1e-6):
            return "left"
        elif np.allclose(n, [0, 1, 0], atol=1e-6):
            return "front"
        elif np.allclose(n, [0, -1, 0], atol=1e-6):
            return "back"
        else:
            return "custom"
    
    def _calculate_rotation_from_normal(self, normal: tuple) -> tuple:
        """Calculate rotation angles to align marker Z-axis with surface normal."""
        import numpy as np
        
        # Convert normal to numpy array and normalize
        n = np.array(normal)
        n = n / (np.linalg.norm(n) + 1e-8)
        
        # Default marker orientation is Z-up (0, 0, 1)
        z_axis = np.array([0.0, 0.0, 1.0])
        
        # If normal is already aligned with Z-axis, no rotation needed
        if np.allclose(n, z_axis, atol=1e-6):
            return (0.0, 0.0, 0.0)
        
        # If normal is opposite to Z-axis, rotate 180 degrees around X
        if np.allclose(n, -z_axis, atol=1e-6):
            return (np.pi, 0.0, 0.0)
        
        # Calculate rotation axis (cross product of z_axis and normal)
        rotation_axis = np.cross(z_axis, n)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
        
        # Calculate rotation angle
        cos_angle = np.dot(z_axis, n)
        rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Convert axis-angle to Euler angles (roll, pitch, yaw)
        # This is a simplified approach for robotics applications
        
        # For face normals along primary axes, use simplified rotations
        if np.allclose(n, [1, 0, 0], atol=1e-6):  # +X face
            return (0.0, np.pi/2, 0.0)  # Pitch 90 degrees
        elif np.allclose(n, [-1, 0, 0], atol=1e-6):  # -X face
            return (0.0, -np.pi/2, 0.0)  # Pitch -90 degrees
        elif np.allclose(n, [0, 1, 0], atol=1e-6):  # +Y face
            return (-np.pi/2, 0.0, 0.0)  # Roll -90 degrees
        elif np.allclose(n, [0, -1, 0], atol=1e-6):  # -Y face
            return (np.pi/2, 0.0, 0.0)  # Roll 90 degrees
        elif np.allclose(n, [0, 0, -1], atol=1e-6):  # -Z face
            return (np.pi, 0.0, 0.0)  # Roll 180 degrees
        else:
            # For arbitrary normals, use a general approach
            # Convert rotation axis and angle to Euler angles
            # This is simplified - for more complex cases, quaternions would be better
            if abs(rotation_axis[2]) > 0.9:  # Rotation mostly around Z-axis
                return (0.0, 0.0, rotation_angle if rotation_axis[2] > 0 else -rotation_angle)
            elif abs(rotation_axis[1]) > 0.9:  # Rotation mostly around Y-axis  
                return (0.0, rotation_angle if rotation_axis[1] > 0 else -rotation_angle, 0.0)
            else:  # Rotation mostly around X-axis
                return (rotation_angle if rotation_axis[0] > 0 else -rotation_angle, 0.0, 0.0)
        
    def place_marker_at_clicked_position(self, position: tuple, normal: tuple = (0.0, 0.0, 1.0)) -> None:
        """Place a marker at a position clicked in the 3D viewer with proper orientation."""
        self.place_marker_at_position(position, normal)
        # Exit placement mode after placing
        self.cancel_placement_mode()
        
    def remove_selected_marker(self) -> None:
        """Remove the currently selected marker."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            reply = QMessageBox.question(
                self,
                "Remove Marker",
                f"Remove Marker {current_item.marker_id}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove from list
                row = self.marker_list.row(current_item)
                self.marker_list.takeItem(row)
                
                # Remove from dictionary
                marker_id = current_item.marker_id
                del self.markers[marker_id]
                
                # Emit signal
                self.marker_removed.emit(marker_id)
                
                # Update stats
                self.update_stats()
                
    def edit_selected_marker(self) -> None:
        """Edit the currently selected marker."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            # Get new position
            x, y, z = current_item.aruco_info.position
            
            new_x, ok1 = QInputDialog.getDouble(
                self, "Edit X Position", "X coordinate:", x, -10.0, 10.0, 3
            )
            if not ok1:
                return
                
            new_y, ok2 = QInputDialog.getDouble(
                self, "Edit Y Position", "Y coordinate:", y, -10.0, 10.0, 3
            )
            if not ok2:
                return
                
            new_z, ok3 = QInputDialog.getDouble(
                self, "Edit Z Position", "Z coordinate:", z, -10.0, 10.0, 3
            )
            if not ok3:
                return
                
            # Update position
            new_position = (new_x, new_y, new_z)
            current_item.aruco_info.position = new_position
            current_item.update_text()
            
            # Emit signal
            self.marker_position_changed.emit(current_item.marker_id, new_position)
            
            # Update the coordinate spinboxes to reflect the change
            self.x_spinbox.setValue(new_x)
            self.y_spinbox.setValue(new_y)
            self.z_spinbox.setValue(new_z)
            
    def on_selection_changed(self) -> None:
        """Handle marker selection changes."""
        current_item = self.marker_list.currentItem()
        has_selection = current_item is not None
        
        self.edit_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        self.up_button.setEnabled(has_selection)
        self.down_button.setEnabled(has_selection)
        self.left_button.setEnabled(has_selection)
        self.right_button.setEnabled(has_selection)
        
        # Enable/disable orientation buttons
        self.pitch_up_button.setEnabled(has_selection)
        self.pitch_down_button.setEnabled(has_selection)
        self.yaw_left_button.setEnabled(has_selection)
        self.yaw_right_button.setEnabled(has_selection)
        self.roll_left_button.setEnabled(has_selection)
        self.roll_right_button.setEnabled(has_selection)
        self.reset_orientation_button.setEnabled(has_selection)
        
        if has_selection and isinstance(current_item, MarkerListItem):
            # Emit selection signal
            self.marker_selected.emit(current_item.marker_id)
            
            # Update coordinate spinboxes WITHOUT triggering signals
            # Block signals to prevent unwanted position updates during selection
            self.x_spinbox.blockSignals(True)
            self.y_spinbox.blockSignals(True)
            self.z_spinbox.blockSignals(True)
            
            x, y, z = current_item.aruco_info.position
            self.x_spinbox.setValue(x)
            self.y_spinbox.setValue(y)
            self.z_spinbox.setValue(z)
            
            # Re-enable signals
            self.x_spinbox.blockSignals(False)
            self.y_spinbox.blockSignals(False)
            self.z_spinbox.blockSignals(False)
            
            # Update pose information display
            self.update_pose_information_display(current_item.aruco_info)
        else:
            # Clear pose information when no marker is selected
            self.pose_info_text.clear()
            self.wireframe_attachment_label.setText("N/A")
    
    def update_pose_information_display(self, aruco_info: ArUcoMarkerInfo) -> None:
        """Update the pose information display with detailed marker information."""
        try:
            # Get pose information
            pose_info = aruco_info.cad_object_pose
            
            # Format detailed pose information
            info_text = f"""MARKER POSE INFORMATION
========================
ArUco Marker: {aruco_info.dictionary} ID:{aruco_info.marker_id}
Size: {aruco_info.size:.3f} m
Border Width: {aruco_info.border_width*100:.1f}%

ABSOLUTE 6D POSE
================
Position (X, Y, Z): ({aruco_info.position[0]:.6f}, {aruco_info.position[1]:.6f}, {aruco_info.position[2]:.6f}) m
Rotation (RPY): ({aruco_info.rotation[0]:.6f}, {aruco_info.rotation[1]:.6f}, {aruco_info.rotation[2]:.6f}) rad
Rotation (RPY): ({aruco_info.rotation[0]*180/3.14159:.2f}°, {aruco_info.rotation[1]*180/3.14159:.2f}°, {aruco_info.rotation[2]*180/3.14159:.2f}°)

CAD OBJECT RELATIVE 6D POSE
===========================
Face Type: {pose_info['face_type']}
Surface Normal: ({pose_info['surface_normal'][0]:.3f}, {pose_info['surface_normal'][1]:.3f}, {pose_info['surface_normal'][2]:.3f})

CAD Object Center: ({pose_info['cad_center'][0]:.6f}, {pose_info['cad_center'][1]:.6f}, {pose_info['cad_center'][2]:.6f})
CAD Dimensions: L={pose_info['cad_dimensions']['length']:.3f}m, W={pose_info['cad_dimensions']['width']:.3f}m, H={pose_info['cad_dimensions']['height']:.3f}m

RELATIVE 6D POSE (to CAD center)
================================
Position (X, Y, Z): ({pose_info['relative_position'][0]:.6f}, {pose_info['relative_position'][1]:.6f}, {pose_info['relative_position'][2]:.6f}) m
Rotation (RPY): ({pose_info['relative_rotation'][0]:.6f}, {pose_info['relative_rotation'][1]:.6f}, {pose_info['relative_rotation'][2]:.6f}) rad
Rotation (RPY): ({pose_info['relative_rotation'][0]*180/3.14159:.2f}°, {pose_info['relative_rotation'][1]*180/3.14159:.2f}°, {pose_info['relative_rotation'][2]*180/3.14159:.2f}°)

WIREFRAME ATTACHMENT
===================
The wireframe mesh will attach at the marker position.
This provides the reference point for robotics applications."""
            
            self.pose_info_text.setPlainText(info_text)
            
            # Update wireframe attachment point
            attachment_point = aruco_info.get_wireframe_attachment_point()
            self.wireframe_attachment_label.setText(f"({attachment_point[0]:.3f}, {attachment_point[1]:.3f}, {attachment_point[2]:.3f})")
            
        except Exception as e:
            self.pose_info_text.setPlainText(f"Error displaying pose information: {str(e)}")
            self.wireframe_attachment_label.setText("Error")
    
    def move_marker(self, direction: str) -> None:
        """Move the selected marker in the specified direction."""
        current_item = self.marker_list.currentItem()
        if not current_item or not isinstance(current_item, MarkerListItem):
            return
        
        # Get current position and step size
        x, y, z = current_item.aruco_info.position
        step = self.step_spinbox.value()
        
        # Calculate new position based on direction
        if direction == 'up':
            new_position = (x, y + step, z)
        elif direction == 'down':
            new_position = (x, y - step, z)
        elif direction == 'left':
            new_position = (x - step, y, z)
        elif direction == 'right':
            new_position = (x + step, y, z)
        else:
            return
        
        # Update the marker's position
        current_item.aruco_info.position = new_position
        current_item.update_text()
        
        # Update CAD object pose information
        self._update_marker_pose_info(current_item.aruco_info)
        
        # Update coordinate spinboxes
        self.x_spinbox.setValue(new_position[0])
        self.y_spinbox.setValue(new_position[1])
        self.z_spinbox.setValue(new_position[2])
        
        # Update pose information display
        self.update_pose_information_display(current_item.aruco_info)
        
        # Emit movement signal
        self.marker_moved.emit(current_item.marker_id, new_position)
            
    def on_coordinate_changed(self) -> None:
        """Handle coordinate spinbox changes for real-time marker movement."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            # Get new position from spinboxes
            new_position = (
                self.x_spinbox.value(),
                self.y_spinbox.value(),
                self.z_spinbox.value()
            )
            
            # Update the marker item
            current_item.aruco_info.position = new_position
            current_item.update_text()
            
            # Update CAD object pose information
            self._update_marker_pose_info(current_item.aruco_info)
            
            # Update pose information display
            self.update_pose_information_display(current_item.aruco_info)
            
            # Emit signal to update 3D viewer
            self.marker_position_changed.emit(current_item.marker_id, new_position)
            
    def update_stats(self) -> None:
        """Update statistics display."""
        count = len(self.markers)
        self.stats_label.setText(f"Markers: {count}")
        
        # Enable/disable the "Delete All" button based on whether there are markers
        self.delete_all_button.setEnabled(count > 0)
        
    def get_all_markers(self) -> List[Dict[str, Any]]:
        """Get all markers as a list of dictionaries."""
        return [item.get_data() for item in self.markers.values()]
    
    def get_markers_for_robotics_export(self) -> List[Dict[str, Any]]:
        """Get markers formatted for robotics export with 6D pose relative to CAD object center."""
        robotics_markers = []
        
        for item in self.markers.values():
            aruco_info = item.aruco_info
            pose_info = aruco_info.cad_object_pose
            
            # Convert RPY to quaternion for relative pose
            rel_roll = pose_info["relative_rotation"][0]
            rel_pitch = pose_info["relative_rotation"][1]
            rel_yaw = pose_info["relative_rotation"][2]
            rel_quat = Rotation.from_euler('xyz', [rel_roll, rel_pitch, rel_yaw], degrees=False).as_quat()
            
            # Convert RPY to quaternion for absolute pose
            abs_roll = aruco_info.rotation[0]
            abs_pitch = aruco_info.rotation[1]
            abs_yaw = aruco_info.rotation[2]
            abs_quat = Rotation.from_euler('xyz', [abs_roll, abs_pitch, abs_yaw], degrees=False).as_quat()
            
            # Create robotics export format (common fields moved to top level)
            robotics_marker = {
                "aruco_id": aruco_info.marker_id,
                "face_type": pose_info["face_type"],
                "surface_normal": pose_info["surface_normal"],
                "pose_relative_to_cad_center": {
                    "position": {
                        "x": pose_info["relative_position"][0],
                        "y": pose_info["relative_position"][1], 
                        "z": pose_info["relative_position"][2]
                    },
                    "rotation": {
                        "roll": rel_roll,
                        "pitch": rel_pitch,
                        "yaw": rel_yaw,
                        "quaternion": {
                            "x": float(rel_quat[0]),
                            "y": float(rel_quat[1]),
                            "z": float(rel_quat[2]),
                            "w": float(rel_quat[3])
                        }
                    }
                },
                "pose_absolute": {
                    "position": {
                        "x": aruco_info.position[0],
                        "y": aruco_info.position[1],
                        "z": aruco_info.position[2]
                    },
                    "rotation": {
                        "roll": abs_roll,
                        "pitch": abs_pitch,
                        "yaw": abs_yaw,
                        "quaternion": {
                            "x": float(abs_quat[0]),
                            "y": float(abs_quat[1]),
                            "z": float(abs_quat[2]),
                            "w": float(abs_quat[3])
                        }
                    }
                }
            }
            
            robotics_markers.append(robotics_marker)
        
        return robotics_markers
    
    def get_export_preview(self) -> str:
        """Get a preview of the export format."""
        robotics_markers = self.get_markers_for_robotics_export()
        
        if not robotics_markers:
            return "No markers to export"
        
        # Show first marker as example
        example_marker = robotics_markers[0]
        
        preview = f"""EXPORT FORMAT PREVIEW
====================
Total markers: {len(robotics_markers)}

Example marker export format:
{{
  "aruco_dictionary": "{example_marker['aruco_dictionary']}",
  "aruco_id": {example_marker['aruco_id']},
  "size": {example_marker['size']:.3f},
  "border_width": {example_marker['border_width']:.3f},
  "face_type": "{example_marker['face_type']}",
  "pose_relative_to_cad_center": {{
    "position": {{
      "x": {example_marker['pose_relative_to_cad_center']['position']['x']:.6f},
      "y": {example_marker['pose_relative_to_cad_center']['position']['y']:.6f},
      "z": {example_marker['pose_relative_to_cad_center']['position']['z']:.6f}
    }},
    "rotation": {{
      "roll": {example_marker['pose_relative_to_cad_center']['rotation']['roll']:.6f},
      "pitch": {example_marker['pose_relative_to_cad_center']['rotation']['pitch']:.6f},
      "yaw": {example_marker['pose_relative_to_cad_center']['rotation']['yaw']:.6f}
    }}
  }}
}}

This format provides:
- ArUco dictionary and ID for marker detection
- Size and border width used to create the ArUco marker
- 6D pose (x,y,z,roll,pitch,yaw) relative to CAD object center
- Face type and surface normal for context
- Both relative and absolute poses for flexibility"""
        
        return preview
        
    def load_markers(self, markers_data: List[Dict[str, Any]]) -> None:
        """Load markers from data."""
        # Clear existing markers
        self.marker_list.clear()
        self.markers.clear()
        
        # Load each marker
        for marker_data in markers_data:
            marker_id = marker_data["id"]
            
            # Create ArUcoMarkerInfo from data
            if "dictionary" in marker_data:
                # New format with ArUco info
                aruco_info = ArUcoMarkerInfo.from_dict(marker_data)
            else:
                # Legacy format - convert to ArUco info
                position = tuple(marker_data["position"])
                size = marker_data.get("size", 0.05)
                aruco_info = ArUcoMarkerInfo(
                    dictionary="DICT_4X4_50",  # Default
                    marker_id=0,  # Default
                    position=position,
                    size=size
                )
            
            # Create item
            item = MarkerListItem(marker_id, aruco_info)
            self.markers[marker_id] = item
            self.marker_list.addItem(item)
            
            # Update next ID
            self.next_marker_id = max(self.next_marker_id, marker_id + 1)
            
        # Update stats
        self.update_stats()
        
    def clear_all_markers(self) -> None:
        """Clear all markers."""
        reply = QMessageBox.question(
            self,
            "Clear All Markers",
            "Remove all markers?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Remove all markers
            for marker_id in list(self.markers.keys()):
                self.marker_removed.emit(marker_id)
                
            self.marker_list.clear()
            self.markers.clear()
            self.update_stats()
            
    def delete_all_markers(self) -> None:
        """Delete all markers and reset the counter to 0."""
        if len(self.markers) == 0:
            QMessageBox.information(
                self,
                "No Markers",
                "There are no markers to delete."
            )
            return
            
        reply = QMessageBox.question(
            self,
            "Delete All Markers",
            f"Delete all {len(self.markers)} markers and reset counter to 0?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No  # Default to No for safety
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            print(f"🗑️ Deleting all {len(self.markers)} markers and resetting counter...")
            
            # Remove all markers (emit signals for 3D viewer)
            for marker_id in list(self.markers.keys()):
                self.marker_removed.emit(marker_id)
                
            # Clear all data structures
            self.marker_list.clear()
            self.markers.clear()
            
            # Reset the counter to 0
            self.next_marker_id = 0
            
            # Reset ArUco marker ID spinbox to 0
            self.marker_id_spinbox.setValue(0)
            
            # Update UI
            self.update_stats()
            
            print("✅ All markers deleted and counter reset to 0")
            
    def clear_all_markers_automatically(self) -> None:
        """Clear all markers automatically without user confirmation (for new CAD model loading)."""
        if len(self.markers) == 0:
            return  # Nothing to clear
            
        print(f"🗑️ Automatically clearing all {len(self.markers)} markers for new CAD model...")
        
        # Remove all markers (emit signals for 3D viewer)
        for marker_id in list(self.markers.keys()):
            self.marker_removed.emit(marker_id)
            
        # Clear all data structures
        self.marker_list.clear()
        self.markers.clear()
        
        # Reset the counter to 0
        self.next_marker_id = 0
        print(f"🔢 Counter reset to: {self.next_marker_id}")
        
        # Reset ArUco marker ID spinbox to 0
        self.marker_id_spinbox.setValue(0)
        print(f"🔢 ArUco marker ID spinbox reset to: 0")
        
        # Update UI
        self.update_stats()
        
        print("✅ All markers cleared and counter reset to 0 for new CAD model")
            
    def get_selected_marker_id(self) -> int:
        """Get the ID of the currently selected marker."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            return current_item.marker_id
        return -1
    
    def rotate_marker(self, direction: str) -> None:
        """Rotate the selected marker in the specified direction."""
        current_item = self.marker_list.currentItem()
        if not current_item or not isinstance(current_item, MarkerListItem):
            return
            
        # Get current rotation (default to 0,0,0 if not set)
        current_rotation = getattr(current_item.aruco_info, 'rotation', (0.0, 0.0, 0.0))
        roll, pitch, yaw = current_rotation
        
        # Fixed 90-degree rotation step
        step_degrees = 90.0
        step_radians = step_degrees * 3.14159 / 180.0
        
        # Apply rotation based on direction
        if direction == 'pitch_up':
            pitch += step_radians
        elif direction == 'pitch_down':
            pitch -= step_radians
        elif direction == 'yaw_left':
            yaw -= step_radians
        elif direction == 'yaw_right':
            yaw += step_radians
        elif direction == 'roll_left':
            roll -= step_radians
        elif direction == 'roll_right':
            roll += step_radians
            
        # Normalize angles to [-π, π]
        roll = ((roll + 3.14159) % (2 * 3.14159)) - 3.14159
        pitch = ((pitch + 3.14159) % (2 * 3.14159)) - 3.14159
        yaw = ((yaw + 3.14159) % (2 * 3.14159)) - 3.14159
        
        new_rotation = (roll, pitch, yaw)
        
        # Update the marker's rotation
        current_item.aruco_info.rotation = new_rotation
        current_item.update_text()
        
        # Update CAD object pose information
        self._update_marker_pose_info(current_item.aruco_info)
        
        # Update pose information display
        self.update_pose_information_display(current_item.aruco_info)
        
        # Emit orientation change signal
        self.marker_orientation_changed.emit(current_item.marker_id, new_rotation)
        
        print(f"Rotated marker {current_item.marker_id} {direction}: {new_rotation}")
    
    def reset_marker_orientation(self) -> None:
        """Reset the selected marker's orientation to default."""
        current_item = self.marker_list.currentItem()
        if not current_item or not isinstance(current_item, MarkerListItem):
            return
            
        # Reset to default rotation
        default_rotation = (0.0, 0.0, 0.0)
        current_item.aruco_info.rotation = default_rotation
        current_item.update_text()
        
        # Update CAD object pose information
        self._update_marker_pose_info(current_item.aruco_info)
        
        # Update pose information display
        self.update_pose_information_display(current_item.aruco_info)
        
        # Emit orientation change signal
        self.marker_orientation_changed.emit(current_item.marker_id, default_rotation)
        
        print(f"Reset rotation for marker {current_item.marker_id}")
