#!/usr/bin/env python3
"""
ArUco Grasp Annotator Web Application
FastAPI-based web application for placing ArUco markers on 3D CAD objects
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2
import base64
import io
import warnings

# Import existing modules
from .core.cad_loader import CADLoader
from .core.annotation_manager import AnnotationManager
from .utils.aruco_utils import ArUcoGenerator

app = FastAPI(
    title="ArUco Grasp Annotator",
    description="3D CAD annotation tool for placing ArUco markers on objects",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
cad_loader = CADLoader()
annotation_manager = AnnotationManager()
aruco_generator = ArUcoGenerator()

# Data directory path - automatically resolve relative to project root
_app_dir = Path(__file__).parent
_project_root = _app_dir.parent.parent
DATA_DIR = _project_root / "data"

# Global session state (in production, use database)
session_state = {
    "mesh": None,
    "mesh_info": None,
    "markers": {},
    "next_marker_id": 0,
    "cad_object_info": None,
    "current_file": None
}

class MarkerData:
    """Enhanced marker data with separated geometric and rotation info"""
    def __init__(self, aruco_id, dictionary, size, border_width, 
                 position, face_normal, face_type):
        self.aruco_id = aruco_id
        self.dictionary = dictionary
        self.size = size
        self.border_width = border_width
        
        # Geometric properties (INVARIANT - don't change with in-plane rotation)
        # Position and normal are stored in OBJECT LOCAL coordinate system
        self.position = np.array(position)  # Marker position relative to object center (object local space)
        self.face_normal = np.array(face_normal) / (np.linalg.norm(face_normal) + 1e-8)  # Normal in object local space
        self.face_type = face_type
        
        # Calculate base orientation (marker Z+ aligned with face normal)
        # This rotation is relative to object's local coordinate system
        self.base_rotation_matrix = self._calculate_base_rotation()
        
        # In-plane rotation offset (rotation around face normal)
        # This is what gets adjusted when user rotates the marker
        self.in_plane_rotation_deg = 0.0  # Degrees
        
        # In-plane translation offset (translation along marker's X and Y axes)
        # This tracks cumulative translation from initial placement position
        self.in_plane_translation = np.array([0.0, 0.0])  # [x_offset, y_offset] in marker's local plane
        
        # Store initial position for translation calculations
        self.initial_position = np.array(position).copy()
        
    def _calculate_base_rotation(self):
        """Calculate rotation matrix that aligns marker Z+ with face normal"""
        z_world = np.array([0.0, 0.0, 1.0])
        normal = self.face_normal
        
        # Handle special cases
        if np.allclose(normal, z_world, atol=1e-6):
            return np.eye(3)
        if np.allclose(normal, -z_world, atol=1e-6):
            return Rotation.from_euler('x', 180, degrees=True).as_matrix()
        
        # General case: rotate Z-axis to align with normal
        rotation_axis = np.cross(z_world, normal)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
        rotation_angle = np.arccos(np.clip(np.dot(z_world, normal), -1.0, 1.0))
        
        return Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()
    
    def get_current_rotation_matrix(self):
        """Get current rotation matrix including in-plane rotation using axis-angle to avoid gimbal lock"""
        # Start with base rotation (aligns Z+ with normal)
        R_base = self.base_rotation_matrix
        
        # Apply in-plane rotation using axis-angle representation
        # Rotate around face_normal (in world space) by in_plane_rotation_deg
        # This avoids gimbal lock issues
        if abs(self.in_plane_rotation_deg) < 1e-6:
            # No rotation, return base rotation
            return R_base
        
        # Convert angle to radians
        angle_rad = np.deg2rad(self.in_plane_rotation_deg)
        
        # Use axis-angle: rotate around face_normal
        # The face_normal is in world space, so we need to apply this rotation in world space
        # But we want the final rotation to be: R_base @ R_inplane_world
        # However, R_inplane should be in the marker frame (around Z+)
        # So: R_final = R_base @ R_inplane_marker
        # Where R_inplane_marker rotates around Z+ in marker frame
        
        # Rotate around Z-axis in marker frame (this is equivalent to rotating around face_normal in world)
        R_inplane_marker = Rotation.from_euler('z', self.in_plane_rotation_deg, degrees=True).as_matrix()
        
        # Combined rotation: first base, then in-plane
        return R_base @ R_inplane_marker
    
    def set_in_plane_rotation(self, degrees):
        """Set in-plane rotation in degrees"""
        self.in_plane_rotation_deg = degrees % 360.0
    
    def _get_quaternion_from_normal(self):
        """
        Get quaternion for primary axis faces based on the actual stored normal.
        This computes the quaternion directly from the normal vector, avoiding gimbal lock.
        Returns (quaternion, is_primary_axis) tuple.
        Quaternion format: [x, y, z, w]
        """
        normal = self.face_normal
        in_plane_rad = np.deg2rad(self.in_plane_rotation_deg)
        
        # Check if this is a primary axis face
        is_primary = (np.any(np.allclose(normal, [1, 0, 0], atol=1e-6)) or
                     np.any(np.allclose(normal, [-1, 0, 0], atol=1e-6)) or
                     np.any(np.allclose(normal, [0, 1, 0], atol=1e-6)) or
                     np.any(np.allclose(normal, [0, -1, 0], atol=1e-6)) or
                     np.any(np.allclose(normal, [0, 0, 1], atol=1e-6)) or
                     np.any(np.allclose(normal, [0, 0, -1], atol=1e-6)))
        
        if not is_primary:
            return None, False
        
        # Compute base rotation matrix from the actual normal (same as _calculate_base_rotation)
        z_world = np.array([0.0, 0.0, 1.0])
        
        if np.allclose(normal, z_world, atol=1e-6):
            # Top face: no base rotation
            R_base = np.eye(3)
        elif np.allclose(normal, -z_world, atol=1e-6):
            # Bottom face: roll = π
            R_base = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        else:
            # General case: rotate Z-axis to align with normal
            rotation_axis = np.cross(z_world, normal)
            rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
            rotation_angle = np.arccos(np.clip(np.dot(z_world, normal), -1.0, 1.0))
            R_base = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()
        
        # Convert base rotation to quaternion
        rot_base = Rotation.from_matrix(R_base)
        quat_base = rot_base.as_quat()  # [x, y, z, w]
        
        # Apply in-plane rotation (rotation around face normal, which is now Z+ in marker frame)
        if abs(in_plane_rad) > 1e-6:
            # Rotate around Z-axis in marker frame
            rot_inplane = Rotation.from_euler('z', in_plane_rad, degrees=False)
            quat_inplane = rot_inplane.as_quat()
            
            # Combine: quat_final = quat_base * quat_inplane
            rot_combined = Rotation.from_quat(quat_base) * rot_inplane
            quat = rot_combined.as_quat()
        else:
            quat = quat_base
        
        return quat, True
    
    def _rotation_matrix_to_euler_avoiding_gimbal_lock(self, R):
        """
        Convert rotation matrix to Euler angles (xyz) using quaternion as intermediate.
        This avoids direct Euler extraction which can have gimbal lock issues.
        """
        # Convert to quaternion first, then to Euler
        rot_scipy = Rotation.from_matrix(R)
        quat = rot_scipy.as_quat()
        rot_from_quat = Rotation.from_quat(quat)
        
        # Extract Euler angles with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            euler = rot_from_quat.as_euler('xyz')
        
        return euler
        
        # For arbitrary normals, use axis-angle as intermediate to avoid gimbal lock
        # Convert rotation matrix to axis-angle, then to Euler
        rot_scipy = Rotation.from_matrix(R)
        
        # Get axis-angle representation
        axis_angle = rot_scipy.as_rotvec()
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-6:
            # No rotation
            return np.array([0.0, 0.0, 0.0])
        
        # Convert axis-angle to quaternion, then to Euler
        # This avoids direct Euler extraction which can have gimbal lock issues
        quat = rot_scipy.as_quat()
        rot_from_quat = Rotation.from_quat(quat)
        
        # Extract Euler angles with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            euler = rot_from_quat.as_euler('xyz')
        
        return euler
    
    def get_T_object_to_marker(self, cad_center_local):
        """
        Calculate T_object_to_marker (includes full rotation: base + in-plane)
        
        Args:
            cad_center_local: Object center in object local space (typically [0,0,0] or the geometric center)
        
        Returns:
            T_object_to_marker: Transform from object frame to marker frame (in object local space)
            - position: vector from object center to marker (in object frame)
            - rotation: FULL rotation from object frame to marker frame (base + in-plane rotation)
        """
        cad_center_local = np.array(cad_center_local)
        
        # Vector from object center to marker position in object local space
        # Both position and cad_center are in object local space
        vec_object_to_marker = self.position - cad_center_local
        
        # Use quaternion as primary representation (no gimbal lock issues)
        # For primary axis faces, compute quaternion directly from normal
        quat, is_primary = self._get_quaternion_from_normal()
        
        if is_primary:
            # For primary axis faces, compute Euler angles explicitly from normal
            # This avoids gimbal lock issues when extracting from quaternion
            normal = self.face_normal
            in_plane_rad = np.deg2rad(self.in_plane_rotation_deg)
            
            # Compute base Euler angles based on normal direction
            if np.allclose(normal, [1, 0, 0], atol=1e-6):  # Right face (+X)
                euler = np.array([0.0, np.pi/2, in_plane_rad])
            elif np.allclose(normal, [-1, 0, 0], atol=1e-6):  # Left face (-X)
                euler = np.array([0.0, -np.pi/2, in_plane_rad])
            elif np.allclose(normal, [0, 1, 0], atol=1e-6):  # Back face (+Y)
                euler = np.array([-np.pi/2, 0.0, in_plane_rad])
            elif np.allclose(normal, [0, -1, 0], atol=1e-6):  # Front face (-Y)
                euler = np.array([np.pi/2, 0.0, in_plane_rad])
            elif np.allclose(normal, [0, 0, 1], atol=1e-6):  # Top face (+Z)
                euler = np.array([0.0, 0.0, in_plane_rad])
            elif np.allclose(normal, [0, 0, -1], atol=1e-6):  # Bottom face (-Z)
                euler = np.array([np.pi, 0.0, in_plane_rad])
            else:
                # Fallback: extract from quaternion
                rot_from_quat = Rotation.from_quat(quat)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    euler = rot_from_quat.as_euler('xyz')
                euler[2] = in_plane_rad  # Override yaw with in-plane rotation
        else:
            # For arbitrary normals, get rotation matrix and extract both Euler and quaternion
            R_object_to_marker_full = self.get_current_rotation_matrix().T
            rot_scipy = Rotation.from_matrix(R_object_to_marker_full)
            quat = rot_scipy.as_quat()  # [x, y, z, w]
            
            # Get Euler angles using axis-angle to avoid gimbal lock
            euler = self._rotation_matrix_to_euler_avoiding_gimbal_lock(R_object_to_marker_full)
        
        return {
            "position": {
                "x": float(vec_object_to_marker[0]),
                "y": float(vec_object_to_marker[1]),
                "z": float(vec_object_to_marker[2])
            },
            "rotation": {
                "roll": float(euler[0]),
                "pitch": float(euler[1]),
                "yaw": float(euler[2]),
                "quaternion": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3])
                }
            }
        }

def group_triangles_by_face(triangles, vertices):
    """Face detection algorithm from working_viewer_3d.py"""
    face_groups = []
    normal_tolerance = 0.02
    spatial_threshold = 0.1
    min_face_area = 1e-6
    
    triangle_data = []
    for i, triangle in enumerate(triangles):
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(normal)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
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
    sorted_indices = sorted(range(len(triangle_data)), key=lambda i: triangle_data[i]['area'], reverse=True)
    
    for idx in sorted_indices:
        if idx in used_triangles:
            continue
            
        seed_triangle = triangle_data[idx]
        if seed_triangle['area'] < min_face_area:
            continue
            
        face_triangles = [idx]
        face_queue = [idx]
        used_triangles.add(idx)
        
        while face_queue:
            current_idx = face_queue.pop(0)
            current_triangle = triangle_data[current_idx]
            
            for j, candidate in enumerate(triangle_data):
                if j in used_triangles:
                    continue
                
                normal_similarity = np.dot(current_triangle['normal'], candidate['normal'])
                if normal_similarity < (1.0 - normal_tolerance):
                    continue
                
                point_to_plane_dist = abs(np.dot(candidate['normal'], current_triangle['center']) + current_triangle['plane_d'])
                if point_to_plane_dist > spatial_threshold * 0.1:
                    continue
                
                is_spatially_connected = False
                for face_tri_idx in face_triangles:
                    face_center = triangle_data[face_tri_idx]['center']
                    distance = np.linalg.norm(candidate['center'] - face_center)
                    if distance < spatial_threshold:
                        is_spatially_connected = True
                        break
                
                if not is_spatially_connected:
                    continue
                
                # Check edge connectivity
                tri1_verts = set(triangles[current_idx])
                tri2_verts = set(triangles[j])
                shared_vertices = tri1_verts.intersection(tri2_verts)
                is_edge_connected = len(shared_vertices) >= 1
                
                if is_edge_connected:
                    face_triangles.append(j)
                    face_queue.append(j)
                    used_triangles.add(j)
        
        if len(face_triangles) > 0:
            face_centers = [triangle_data[i]['center'] for i in face_triangles]
            face_areas = [triangle_data[i]['area'] for i in face_triangles]
            face_normals = [triangle_data[i]['normal'] for i in face_triangles]
            
            total_area = sum(face_areas)
            if total_area > min_face_area:
                face_center = np.average(face_centers, axis=0, weights=face_areas)
                face_normal = np.average(face_normals, axis=0, weights=face_areas)
                face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
                
                face_groups.append((face_center, face_normal, total_area, face_triangles))
    
    face_groups.sort(key=lambda x: x[2], reverse=True)
    return face_groups

def calculate_rotation_from_normal(normal: tuple) -> tuple:
    """Calculate RPY rotation from surface normal."""
    n = np.array(normal)
    n = n / (np.linalg.norm(n) + 1e-8)
    
    z_axis = np.array([0.0, 0.0, 1.0])
    
    if np.allclose(n, z_axis, atol=1e-6):
        return (0.0, 0.0, 0.0)
    if np.allclose(n, -z_axis, atol=1e-6):
        return (np.pi, 0.0, 0.0)
    
    rotation_axis = np.cross(z_axis, n)
    rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
    
    cos_angle = np.dot(z_axis, n)
    rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Handle primary axes
    if np.allclose(n, [1, 0, 0], atol=1e-6):
        return (0.0, np.pi/2, 0.0)
    elif np.allclose(n, [-1, 0, 0], atol=1e-6):
        return (0.0, -np.pi/2, 0.0)
    elif np.allclose(n, [0, 1, 0], atol=1e-6):
        return (-np.pi/2, 0.0, 0.0)
    elif np.allclose(n, [0, -1, 0], atol=1e-6):
        return (np.pi/2, 0.0, 0.0)
    elif np.allclose(n, [0, 0, -1], atol=1e-6):
        return (np.pi, 0.0, 0.0)
    else:
        if abs(rotation_axis[2]) > 0.9:
            return (0.0, 0.0, rotation_angle if rotation_axis[2] > 0 else -rotation_angle)
        elif abs(rotation_axis[1]) > 0.9:
            return (0.0, rotation_angle if rotation_axis[1] > 0 else -rotation_angle, 0.0)
        else:
            return (rotation_angle if rotation_axis[0] > 0 else -rotation_angle, 0.0, 0.0)

def determine_face_type(normal: tuple) -> str:
    """Determine face type from normal vector."""
    n = np.array(normal)
    n = n / (np.linalg.norm(n) + 1e-8)
    
    if np.allclose(n, [0, 0, 1], atol=0.1):
        return "top"
    elif np.allclose(n, [0, 0, -1], atol=0.1):
        return "bottom"
    elif np.allclose(n, [0, 1, 0], atol=0.1):
        return "front"
    elif np.allclose(n, [0, -1, 0], atol=0.1):
        return "back"
    elif np.allclose(n, [1, 0, 0], atol=0.1):
        return "right"
    elif np.allclose(n, [-1, 0, 0], atol=0.1):
        return "left"
    else:
        return "custom"

def _marker_to_json(internal_id, marker: MarkerData):
    """Convert MarkerData to JSON format for frontend"""
    cad_info = session_state["cad_object_info"]
    
    # Check if CAD model is loaded
    if cad_info is None:
        raise ValueError("No CAD model loaded. Please load a CAD model first.")
    
    # Get CAD object pose to transform marker to world space for display
    cad_position = np.array(cad_info.get("position", [0.0, 0.0, 0.0]))
    cad_rotation = cad_info.get("rotation", {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    })
    
    # Create rotation matrix from CAD object's rotation
    if "quaternion" in cad_rotation:
        quat = cad_rotation["quaternion"]
        R_cad = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    else:
        R_cad = Rotation.from_euler('xyz', [
            np.deg2rad(cad_rotation.get("roll", 0.0)),
            np.deg2rad(cad_rotation.get("pitch", 0.0)),
            np.deg2rad(cad_rotation.get("yaw", 0.0))
        ]).as_matrix()
    
    # Transform marker position from object local to world space
    cad_center_local = np.array(cad_info["center"])
    position_world = cad_position + R_cad @ (np.array(marker.position) - cad_center_local) + R_cad @ cad_center_local
    
    # Get current rotation (base + in-plane) in object local space, then transform to world
    R_marker_to_object = marker.get_current_rotation_matrix()
    R_marker_to_world = R_cad @ R_marker_to_object
    
    rot_scipy = Rotation.from_matrix(R_marker_to_world)
    quat = rot_scipy.as_quat()
    
    # Get Euler angles with warning suppression (for display only)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = rot_scipy.as_euler('xyz')
    
    # Get T_object_to_marker (INVARIANT) - already in object local space
    T_object_to_marker = marker.get_T_object_to_marker(cad_info["center"])
    
    return {
        "aruco_id": marker.aruco_id,
        "internal_id": internal_id,
        "dictionary": marker.dictionary,
        "face_type": marker.face_type,
        "surface_normal": marker.face_normal.tolist(),
        "size": marker.size,
        "border_width": marker.border_width,
        "in_plane_rotation_deg": marker.in_plane_rotation_deg,
        "translation_offset": {
            "x": float(marker.in_plane_translation[0]),
            "y": float(marker.in_plane_translation[1])
        },
        "T_object_to_marker": T_object_to_marker,
        "pose_absolute": {
            "position": {
                "x": float(position_world[0]),
                "y": float(position_world[1]),
                "z": float(position_world[2])
            },
            "rotation": {
                "roll": float(euler[0]),
                "pitch": float(euler[1]),
                "yaw": float(euler[2]),
                "quaternion": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3])
                }
            }
        }
    }


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface."""
    # This will be a large HTML file - I'll create it in parts
    return get_html_interface()

def get_html_interface():
    """Generate the HTML interface."""
    # Reading the assembly app HTML as reference, then adapting it
    # This is a large HTML string - I'll build it step by step
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArUco Grasp Annotator</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            height: 100vh;
            overflow: hidden;
        }
        
        .app-container {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 350px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        
        .sidebar-left {
            border-right: 1px solid #ddd;
        }
        
        .sidebar-right {
            border-left: 1px solid #ddd;
        }
        
        .main-viewer {
            flex: 1;
            position: relative;
            display: flex;
            flex-direction: column;
        }
        
        .viewer-container {
            flex: 1;
            background: #1a1a1a;
            position: relative;
        }
        
        .controls-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        
        .controls-panel h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 16px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        
        .btn {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            margin: 3px;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
        }
        
        .btn-small {
            padding: 6px 12px;
            font-size: 12px;
        }
        
        .marker-list {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            margin-top: 10px;
        }
        
        .marker-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
        }
        
        .marker-item:hover {
            background: #f8f9fa;
        }
        
        .marker-item.selected {
            background: #e3f2fd;
        }
        
        input[type="file"] {
            margin: 10px 0;
            width: 100%;
        }
        
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            background: #e7f3ff;
        }
        
        .status.error {
            background: #ffebee;
            color: #c62828;
        }
        
        .status.success {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .input-group {
            margin: 10px 0;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #34495e;
        }
        
        .input-group input, .input-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .input-group input[type="range"] {
            padding: 0;
            height: 8px;
            cursor: pointer;
        }
        
        .input-group label span {
            font-weight: bold;
            color: #3498db;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="sidebar sidebar-left">
            <div class="controls-panel">
                <h3>📁 CAD Model</h3>
                <input type="file" id="fileInput" accept=".stl,.obj,.ply,.off">
                <button class="btn" onclick="loadModel()">Load CAD File</button>
                <div id="modelStatus" class="status">No model loaded</div>
            </div>
            
            <div class="controls-panel">
                <h3>🎯 ArUco Configuration</h3>
                <div class="input-group">
                    <label>Dictionary:</label>
                    <select id="dictSelect">
                        <option value="DICT_4X4_50">DICT_4X4_50</option>
                        <option value="DICT_4X4_100">DICT_4X4_100</option>
                        <option value="DICT_5X5_50">DICT_5X5_50</option>
                        <option value="DICT_5X5_100">DICT_5X5_100</option>
                        <option value="DICT_6X6_50">DICT_6X6_50</option>
                        <option value="DICT_6X6_100">DICT_6X6_100</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Marker ID:</label>
                    <input type="number" id="markerId" value="0" min="0" max="49">
                </div>
                <div class="input-group">
                    <label>Size (m):</label>
                    <input type="number" id="markerSize" value="0.021" step="0.001" min="0.001" max="1.0">
                </div>
                <div class="input-group">
                    <label>Border Width (%):</label>
                    <input type="number" id="borderWidth" value="0.05" step="0.01" min="0" max="0.5">
                </div>
            </div>
            
            <div class="controls-panel">
                <h3>📍 Placement Modes</h3>
                <!-- <button class="btn" onclick="enterPlacementMode('click')">🖱️ Click Placement</button> -->
                <!-- <button class="btn" onclick="enterPlacementMode('random')">🎲 Random Face</button> -->
                <!-- <button class="btn" onclick="enterPlacementMode('face-picker')">📋 Face Picker</button> -->
                <!-- <button class="btn" onclick="enterPlacementMode('smart')">🎯 Smart Auto</button> -->
                <button class="btn" onclick="enterPlacementMode('all-6')">🔲 All 6 Faces</button>
                <button class="btn" onclick="enterPlacementMode('corner')">🔲 Corner Markers (4 per face)</button>
                <button class="btn" onclick="enterPlacementMode('single-face')">📍 Single Marker on Face</button>
                <!--<button class="btn" onclick="enterPlacementMode('manual')">⌨️ Manual Coordinates</button>-->
            </div>
            
            <div class="controls-panel">
                <h3>📝 Placed Markers</h3>
                <div id="markersList" class="marker-list"></div>
                <button class="btn btn-secondary btn-small" onclick="removeSelectedMarker()">Remove Selected</button>
                <button class="btn btn-secondary btn-small" onclick="clearAllMarkers()">Clear All</button>
            </div>
            
            <div class="controls-panel">
                <h3>💾 Export/Import</h3>
                <button class="btn" onclick="exportAnnotations()">Export ArUco Annotations</button>
                <button class="btn" onclick="exportWireframe()">Export Wireframe</button>
                <input type="file" id="importFile" accept=".json" style="margin: 10px 0;">
                <button class="btn" onclick="importAnnotations()">Import Annotations</button>
            </div>
        </div>
        
        <div class="main-viewer">
            <div class="viewer-container" id="viewer"></div>
        </div>
        
        <div class="sidebar sidebar-right">
            <div class="controls-panel" id="cadObjectControls" style="display: none;">
                <h3>🎛️ CAD Object Transform</h3>
                
                <h4 style="margin-top: 10px; margin-bottom: 5px; font-size: 14px; color: #555;">Position (m)</h4>
                <div class="input-group">
                    <label>X:</label>
                    <input type="number" id="cadPosX" step="0.001" onchange="updateCADPose()">
                </div>
                <div class="input-group">
                    <label>Y:</label>
                    <input type="number" id="cadPosY" step="0.001" onchange="updateCADPose()">
                </div>
                <div class="input-group">
                    <label>Z:</label>
                    <input type="number" id="cadPosZ" step="0.001" onchange="updateCADPose()">
                </div>
                
                <h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #555;">Rotation (degrees)</h4>
                <div class="input-group">
                    <label>Roll (X):</label>
                    <input type="number" id="cadRotRoll" step="1" min="-180" max="180" onchange="updateCADPose()">
                </div>
                <div class="input-group">
                    <label>Pitch (Y):</label>
                    <input type="number" id="cadRotPitch" step="1" min="-180" max="180" onchange="updateCADPose()">
                </div>
                <div class="input-group">
                    <label>Yaw (Z):</label>
                    <input type="number" id="cadRotYaw" step="1" min="-180" max="180" onchange="updateCADPose()">
                </div>
                
                <button class="btn btn-secondary" onclick="resetCADPose()" style="margin-top: 10px;">Reset to Origin</button>
                
                <h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #555;">Current Pose</h4>
                <div style="background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 12px; font-family: monospace;">
                    <div><strong>Position:</strong> <span id="cadCurrentPos">(0, 0, 0)</span></div>
                    <div style="margin-top: 5px;"><strong>RPY:</strong> <span id="cadCurrentRPY">(0°, 0°, 0°)</span></div>
                    <div style="margin-top: 5px;"><strong>Quaternion:</strong> <span id="cadCurrentQuat">(0, 0, 0, 1)</span></div>
                    <div style="margin-top: 5px;"><strong>Axis-Angle:</strong> <span id="cadCurrentAxisAngle">[0, 0, 1] @ 0°</span></div>
                </div>
            </div>
            
            <div class="controls-panel" id="rotationControls" style="display: none;">
                <h3>🔄 Rotate Selected Marker</h3>
                <div class="input-group">
                    <label>Roll (X-axis): <span id="rollValue">0°</span></label>
                    <input type="range" id="rotateRoll" value="0" step="5" min="-180" max="180" 
                           oninput="updateRotationDisplay('roll', this.value)" disabled
                           title="Roll is part of base (geometric) rotation and cannot be changed">
                </div>
                <div class="input-group">
                    <label>Pitch (Y-axis): <span id="pitchValue">0°</span></label>
                    <input type="range" id="rotatePitch" value="0" step="5" min="-180" max="180"
                           oninput="updateRotationDisplay('pitch', this.value)" disabled
                           title="Pitch is part of base (geometric) rotation and cannot be changed">
                </div>
                <div class="input-group">
                    <label>Yaw (In-plane rotation): <span id="yawValue">0°</span></label>
                    <input type="range" id="rotateYaw" value="0" step="5" min="-180" max="180"
                           oninput="updateRotationDisplay('yaw', this.value)"
                           title="Rotate marker around its face normal (in-plane rotation)">
                </div>
                <button class="btn" onclick="applyRotation()">Apply Rotation</button>
                <button class="btn btn-secondary" onclick="resetRotationControls()">Reset</button>
            </div>
            
            <div class="controls-panel" id="translationControls" style="display: none;">
                <h3>📍 Translate Selected Marker (In-Plane)</h3>
                <p style="font-size: 12px; color: #666; margin-bottom: 10px;">Move marker along X and Y axes in marker's local plane (in meters)</p>
                
                <!-- Step size control -->
                <div class="input-group">
                    <label>Step Size (m):</label>
                    <input type="number" id="inplaneStepSize" value="0.0005" step="0.0001" min="0.0001" max="0.1">
                </div>
                
                <!-- Arrow buttons for in-plane movement -->
                <div style="margin: 15px 0;">
                    <div style="text-align: center; margin-bottom: 10px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Axis 1:</label>
                        <button class="btn" onclick="moveInPlane('axis1_neg')" style="margin: 0 5px;">←</button>
                        <button class="btn" onclick="moveInPlane('axis1_pos')" style="margin: 0 5px;">→</button>
                    </div>
                    <div style="text-align: center;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Axis 2:</label>
                        <button class="btn" onclick="moveInPlane('axis2_neg')" style="margin: 0 5px;">↓</button>
                        <button class="btn" onclick="moveInPlane('axis2_pos')" style="margin: 0 5px;">↑</button>
                    </div>
                </div>
                
                <div class="input-group">
                    <label>X (in-plane, m):</label>
                    <input type="number" id="translateX" value="0" step="0.0001" min="-0.1" max="0.1"
                           onchange="updateTranslationDisplay('x', this.value)"
                           title="Move marker along X-axis in marker's local plane">
                </div>
                <div class="input-group">
                    <label>Y (in-plane, m):</label>
                    <input type="number" id="translateY" value="0" step="0.0001" min="-0.1" max="0.1"
                           onchange="updateTranslationDisplay('y', this.value)"
                           title="Move marker along Y-axis in marker's local plane">
                </div>
                <button class="btn" onclick="applyTranslation()">Apply Translation</button>
                <button class="btn btn-secondary" onclick="resetTranslationControls()">Reset</button>
            </div>
            
            <div class="controls-panel">
                <h3>🔄 Swap Marker Positions</h3>
                <p style="font-size: 12px; color: #666; margin-bottom: 10px;">Select a marker, then click "Assign to Marker 1" or "Assign to Marker 2"</p>
                <div class="input-group">
                    <label>Marker 1:</label>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="swapMarker1Display" style="flex: 1; padding: 8px; background: #f5f5f5; border-radius: 4px; min-height: 20px;">Not assigned</span>
                        <button class="btn btn-small" onclick="assignToMarker1()" style="width: auto; padding: 8px 12px;">Assign Selected</button>
                    </div>
                </div>
                <div class="input-group">
                    <label>Marker 2:</label>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="swapMarker2Display" style="flex: 1; padding: 8px; background: #f5f5f5; border-radius: 4px; min-height: 20px;">Not assigned</span>
                        <button class="btn btn-small" onclick="assignToMarker2()" style="width: auto; padding: 8px 12px;">Assign Selected</button>
                    </div>
                </div>
                <button class="btn" onclick="swapMarkerPositions()">Swap Positions</button>
                <button class="btn btn-secondary btn-small" onclick="clearSwapSelection()" style="margin-top: 5px;">Clear Selection</button>
            </div>
        </div>
    </div>
    
    <script>
        // Three.js setup
        let scene, camera, renderer, controls;
        let meshObject = null;
        let markers = [];
        let selectedMarkerId = null;
        let selectedMarkerMesh = null; // Reference to selected marker 3D object
        let placementMode = null;
        let raycaster, mouse;
        
        function init() {
            const container = document.getElementById('viewer');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);
            
            camera = new THREE.PerspectiveCamera(75, width / height, 0.01, 1000);
            camera.position.set(0.5, -0.5, 0.5);  // Position camera to look at scene
            camera.up.set(0, 0, 1);  // Set Z as up vector
            camera.lookAt(0, 0, 0);
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, -10, 10);  // Light from above in Z-up system
            scene.add(directionalLight);
            
            // Raycaster for mouse picking
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();
            
            // Mouse click handler
            renderer.domElement.addEventListener('click', onMouseClick);
            
            // Grid - rotate to make it horizontal in XY plane (Z-up)
            const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
            gridHelper.rotateX(Math.PI / 2);  // Rotate 90 degrees to make it horizontal in XY plane
            scene.add(gridHelper);
            
            // Custom coordinate frame matching Blender convention (Z-up)
            // Blender: X=Right (Red), Y=Forward (Green), Z=Up (Blue)
            const axesLength = 0.5;
            const axesGeometry = new THREE.BufferGeometry();
            const axesMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
            
            // Create axes: X (Red, Right), Y (Green, Forward), Z (Blue, Up)
            const axesVertices = new Float32Array([
                0, 0, 0,  // Origin
                axesLength, 0, 0,  // X axis end (Right)
                0, 0, 0,  // Origin
                0, axesLength, 0,  // Y axis end (Forward)
                0, 0, 0,  // Origin
                0, 0, axesLength   // Z axis end (Up)
            ]);
            
            const axesColors = new Float32Array([
                1, 0, 0,  // Red for origin
                1, 0, 0,  // Red for X
                0, 1, 0,  // Green for origin
                0, 1, 0,  // Green for Y
                0, 0, 1,  // Blue for origin
                0, 0, 1   // Blue for Z
            ]);
            
            axesGeometry.setAttribute('position', new THREE.BufferAttribute(axesVertices, 3));
            axesGeometry.setAttribute('color', new THREE.BufferAttribute(axesColors, 3));
            
            const axesHelper = new THREE.LineSegments(axesGeometry, axesMaterial);
            scene.add(axesHelper);
            
            animate();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        function onMouseClick(event) {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            
            // First check if clicking on a marker
            let markerClicked = false;
            if (markers.length > 0) {
                const markerIntersects = raycaster.intersectObjects(markers, false);
                if (markerIntersects.length > 0) {
                    const clickedMarker = markerIntersects[0].object;
                    // Ensure clickedInternalId is normalized to a number for consistent comparison
                    const clickedInternalIdRaw = clickedMarker.userData.internalId;
                    const clickedInternalId = typeof clickedInternalIdRaw === 'number' 
                        ? clickedInternalIdRaw 
                        : parseInt(clickedInternalIdRaw);
                    
                    // Normalize selectedMarkerId for comparison
                    const selectedIdNum = selectedMarkerId !== null 
                        ? (typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId))
                        : null;
                    
                    // If clicking the same marker, deselect it
                    if (selectedIdNum === clickedInternalId) {
                        deselectMarker();
                    } else {
                        selectMarkerInScene(clickedInternalId, clickedMarker);
                    }
                    markerClicked = true;
                }
            }
            
            // If not clicking on marker
            if (!markerClicked) {
                if (placementMode === 'click') {
                    // In click placement mode, place marker on mesh
                    if (meshObject) {
                        const intersects = raycaster.intersectObject(meshObject, true);
                        if (intersects.length > 0) {
                            const point = intersects[0].point;
                            const normal = intersects[0].face.normal;
                            placeMarkerAtPosition([point.x, point.y, point.z], [normal.x, normal.y, normal.z]);
                        }
                    }
                } else {
                    // Clicking on object (not in placement mode) - show object controls
                    if (meshObject) {
                        const intersects = raycaster.intersectObject(meshObject, true);
                        if (intersects.length > 0) {
                            // Deselect any marker
                            deselectMarker();
                            // Show object controls
                            document.getElementById('cadObjectControls').style.display = 'block';
                        }
                    }
                }
            }
        }
        
        function selectMarkerInScene(internalId, markerMesh) {
            // Deselect previous marker
            if (selectedMarkerMesh && selectedMarkerMesh.material) {
                // Restore original color
                if (selectedMarkerMesh.userData.originalColor !== undefined) {
                    selectedMarkerMesh.material.color.setHex(selectedMarkerMesh.userData.originalColor);
                }
            }
            
            // Select new marker - ensure internalId is always a number
            selectedMarkerId = typeof internalId === 'number' ? internalId : parseInt(internalId);
            selectedMarkerMesh = markerMesh;
            
            // Store original color if not already stored
            if (markerMesh.userData.originalColor === undefined) {
                markerMesh.userData.originalColor = markerMesh.material.color.getHex();
            }
            
            // Change color to green
            markerMesh.material.color.setHex(0x00ff00);
            
            // Update list selection
            document.querySelectorAll('.marker-item').forEach(item => {
                item.classList.remove('selected');
                const itemId = parseInt(item.dataset.internalId);
                const compareId = typeof internalId === 'string' ? parseInt(internalId) : internalId;
                if (itemId === compareId || String(itemId) === String(compareId)) {
                    item.classList.add('selected');
                }
            });
            
            // Hide object controls, show marker controls
            document.getElementById('cadObjectControls').style.display = 'none';
            
            showRotationControls(internalId);
        }
        
        function deselectMarker() {
            if (selectedMarkerMesh && selectedMarkerMesh.material) {
                // Restore original color
                if (selectedMarkerMesh.userData.originalColor !== undefined) {
                    selectedMarkerMesh.material.color.setHex(selectedMarkerMesh.userData.originalColor);
                }
            }
            
            selectedMarkerId = null;
            selectedMarkerMesh = null;
            
            // Update list selection
            document.querySelectorAll('.marker-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // Hide rotation and translation controls
            document.getElementById('rotationControls').style.display = 'none';
            document.getElementById('translationControls').style.display = 'none';
            
            // Show object controls if model is loaded
            if (meshObject) {
                document.getElementById('cadObjectControls').style.display = 'block';
            }
        }
        
        async function loadModel() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/load-model', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (response.ok) {
                    displayModel(data);
                    session_state.current_file = file.name;
                    updateStatus('modelStatus', 'Model loaded: ' + file.name, 'success');
                    // Load CAD pose from backend
                    await loadCADPose();
                } else {
                    updateStatus('modelStatus', 'Error: ' + (data.detail || data.message || 'Unknown error'), 'error');
                }
            } catch (error) {
                updateStatus('modelStatus', 'Error loading model: ' + error.message, 'error');
            }
        }
        
        function displayModel(data) {
            // Clear all existing markers when loading a new model
            markers.forEach(marker => {
                if (marker.parent) {
                    marker.parent.remove(marker);
                }
            });
            markers = [];
            selectedMarkerId = null;
            selectedMarkerMesh = null;
            
            // Remove existing mesh
            if (meshObject) {
                scene.remove(meshObject);
            }
            
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices, 3));
            geometry.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3));
            geometry.setIndex(data.faces);
            geometry.computeVertexNormals();
            
            const material = new THREE.MeshStandardMaterial({
                color: 0x888888,
                side: THREE.DoubleSide,
                wireframe: false
            });
            
            meshObject = new THREE.Mesh(geometry, material);
            scene.add(meshObject);
            
            // Refresh markers list (markers were cleared, so this will be empty)
            // Markers are cleared on backend when model loads, so refresh will show empty list
            refreshMarkers().catch(err => console.error('Error refreshing markers:', err));
            
            // Show CAD object controls
            document.getElementById('cadObjectControls').style.display = 'block';
            
            // Initialize CAD pose (reset to origin)
            resetCADPose();
            
            // Fit camera to model
            const box = new THREE.Box3().setFromObject(meshObject);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const distance = maxDim * 2;
            
            camera.position.set(center.x + distance, center.y + distance, center.z + distance);
            camera.lookAt(center);
            controls.target.copy(center);
            controls.update();
        }
        
        async function updateCADPose() {
            if (!meshObject) return;
            
            const posX = parseFloat(document.getElementById('cadPosX').value) || 0;
            const posY = parseFloat(document.getElementById('cadPosY').value) || 0;
            const posZ = parseFloat(document.getElementById('cadPosZ').value) || 0;
            
            const roll = parseFloat(document.getElementById('cadRotRoll').value) || 0;
            const pitch = parseFloat(document.getElementById('cadRotPitch').value) || 0;
            const yaw = parseFloat(document.getElementById('cadRotYaw').value) || 0;
            
            // Update mesh position
            meshObject.position.set(posX, posY, posZ);
            
            // Update mesh rotation (convert degrees to radians)
            const rollRad = THREE.MathUtils.degToRad(roll);
            const pitchRad = THREE.MathUtils.degToRad(pitch);
            const yawRad = THREE.MathUtils.degToRad(yaw);
            
            // Apply rotation using Euler angles (XYZ order)
            meshObject.rotation.set(rollRad, pitchRad, yawRad);
            
            // Update display
            updateCADPoseDisplay(posX, posY, posZ, roll, pitch, yaw);
            
            // Update backend
            try {
                await fetch('/api/cad-pose', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        position: { x: posX, y: posY, z: posZ },
                        rotation: { roll: roll, pitch: pitch, yaw: yaw }
                    })
                });
            } catch (error) {
                console.error('Error updating CAD pose:', error);
            }
        }
        
        function updateCADPoseDisplay(x, y, z, roll, pitch, yaw) {
            // Update position display
            document.getElementById('cadCurrentPos').textContent = `(${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)})`;
            
            // Update RPY display
            document.getElementById('cadCurrentRPY').textContent = `(${roll.toFixed(1)}°, ${pitch.toFixed(1)}°, ${yaw.toFixed(1)}°)`;
            
            // Convert to quaternion
            const rollRad = THREE.MathUtils.degToRad(roll);
            const pitchRad = THREE.MathUtils.degToRad(pitch);
            const yawRad = THREE.MathUtils.degToRad(yaw);
            const euler = new THREE.Euler(rollRad, pitchRad, yawRad, 'XYZ');
            const quat = new THREE.Quaternion().setFromEuler(euler);
            document.getElementById('cadCurrentQuat').textContent = `(${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}, ${quat.w.toFixed(4)})`;
            
            // Convert to axis-angle
            const angle = Math.acos(Math.max(-1, Math.min(1, Math.abs(quat.w)))) * 2;
            const s = Math.sqrt(1 - quat.w * quat.w);
            let axisX = 0, axisY = 0, axisZ = 1;
            if (s > 0.0001) {
                axisX = quat.x / s;
                axisY = quat.y / s;
                axisZ = quat.z / s;
            } else {
                // For very small angles, use default axis [0, 0, 1]
                axisX = 0;
                axisY = 0;
                axisZ = 1;
            }
            const angleDeg = THREE.MathUtils.radToDeg(angle);
            document.getElementById('cadCurrentAxisAngle').textContent = `[${axisX.toFixed(4)}, ${axisY.toFixed(4)}, ${axisZ.toFixed(4)}] @ ${angleDeg.toFixed(2)}°`;
        }
        
        async function resetCADPose() {
            if (!meshObject) return;
            
            // Reset to origin
            document.getElementById('cadPosX').value = '0';
            document.getElementById('cadPosY').value = '0';
            document.getElementById('cadPosZ').value = '0';
            document.getElementById('cadRotRoll').value = '0';
            document.getElementById('cadRotPitch').value = '0';
            document.getElementById('cadRotYaw').value = '0';
            
            await updateCADPose();
        }
        
        async function loadCADPose() {
            try {
                const response = await fetch('/api/cad-pose');
                if (response.ok) {
                    const data = await response.json();
                    const pos = data.position || { x: 0, y: 0, z: 0 };
                    const rot = data.rotation || { roll: 0, pitch: 0, yaw: 0 };
                    
                    document.getElementById('cadPosX').value = pos.x;
                    document.getElementById('cadPosY').value = pos.y;
                    document.getElementById('cadPosZ').value = pos.z;
                    document.getElementById('cadRotRoll').value = rot.roll;
                    document.getElementById('cadRotPitch').value = rot.pitch;
                    document.getElementById('cadRotYaw').value = rot.yaw;
                    
                    // Update mesh
                    if (meshObject) {
                        meshObject.position.set(pos.x, pos.y, pos.z);
                        const rollRad = THREE.MathUtils.degToRad(rot.roll);
                        const pitchRad = THREE.MathUtils.degToRad(rot.pitch);
                        const yawRad = THREE.MathUtils.degToRad(rot.yaw);
                        meshObject.rotation.set(rollRad, pitchRad, yawRad);
                        updateCADPoseDisplay(pos.x, pos.y, pos.z, rot.roll, rot.pitch, rot.yaw);
                    }
                }
            } catch (error) {
                console.error('Error loading CAD pose:', error);
            }
        }
        
        async function enterPlacementMode(mode) {
            placementMode = mode;
            
            if (mode === 'random') {
                await placeRandomMarker();
            } else if (mode === 'face-picker') {
                await showFacePicker();
            } else if (mode === 'smart') {
                await placeSmartMarker();
            } else if (mode === 'all-6') {
                await placeAll6Faces();
            } else if (mode === 'corner') {
                await placeCornerMarkers();
            } else if (mode === 'single-face') {
                await placeSingleMarkerOnFace();
            } else if (mode === 'manual') {
                await showManualPlacement();
            }
            // 'click' mode is handled by mouse click
        }
        
        async function placeRandomMarker() {
            try {
                const config = {
                    dictionary: document.getElementById('dictSelect').value,
                    aruco_id: parseInt(document.getElementById('markerId').value),
                    size: parseFloat(document.getElementById('markerSize').value),
                    border_width: parseFloat(document.getElementById('borderWidth').value)
                };
                const response = await fetch('/api/place-marker/random', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                if (response.ok) {
                    await addMarkerToScene(data);
                    await refreshMarkers();
                    // Auto-increment marker ID
                    const markerIdInput = document.getElementById('markerId');
                    const maxId = getMaxIdForDict(config.dictionary);
                    if (parseInt(markerIdInput.value) < maxId) {
                        markerIdInput.value = parseInt(markerIdInput.value) + 1;
                    }
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function getMaxIdForDict(dictName) {
            const parts = dictName.split('_');
            if (parts.length >= 3) {
                return parseInt(parts[parts.length - 1]) - 1;
            }
            return 49;
        }
        
        async function placeMarkerAtPosition(position, normal) {
            const config = {
                dictionary: document.getElementById('dictSelect').value,
                aruco_id: parseInt(document.getElementById('markerId').value),
                size: parseFloat(document.getElementById('markerSize').value),
                border_width: parseFloat(document.getElementById('borderWidth').value),
                position: { x: position[0], y: position[1], z: position[2] },
                normal: { x: normal[0], y: normal[1], z: normal[2] }
            };
            
            try {
                const response = await fetch('/api/add-marker', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                if (response.ok) {
                    await addMarkerToScene(data);
                    await refreshMarkers();
                    placementMode = null;
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function addMarkerToScene(markerData) {
            const pos = markerData.pose_absolute.position;
            const size = markerData.size || 0.021;
            const dictionary = markerData.dictionary || 'DICT_4X4_50';
            const arucoId = markerData.aruco_id;
            
            // Create geometry with thickness to prevent z-fighting
            // Use BoxGeometry instead of PlaneGeometry for better depth handling
            const thickness = size * 0.02; // 2% of marker size for thickness
            const geometry = new THREE.BoxGeometry(size, size, thickness);
            
            // Load ArUco marker texture
            const textureLoader = new THREE.TextureLoader();
            const markerImageUrl = `/api/marker-image?dictionary=${encodeURIComponent(dictionary)}&marker_id=${arucoId}&size=512`;
            
            try {
                const texture = await new Promise((resolve, reject) => {
                    textureLoader.load(
                        markerImageUrl,
                        (texture) => {
                            // Configure texture for better quality and less flickering
                            texture.wrapS = THREE.ClampToEdgeWrapping;
                            texture.wrapT = THREE.ClampToEdgeWrapping;
                            texture.minFilter = THREE.LinearMipmapLinearFilter; // Better filtering
                            texture.magFilter = THREE.LinearFilter; // Smooth but clear
                            texture.generateMipmaps = true;
                            texture.flipY = true; // Three.js default - flip Y for correct texture orientation
                            texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
                            resolve(texture);
                        },
                        undefined,
                        (error) => {
                            console.warn('Failed to load marker texture, using fallback:', error);
                            resolve(null);
                        }
                    );
                });
                
                // Create material with texture or fallback color
                // Disable transparency to prevent flickering and z-fighting issues
                const material = new THREE.MeshStandardMaterial({
                    map: texture || null,
                    color: texture ? 0xffffff : 0x00ff00,
                    side: THREE.DoubleSide,
                    transparent: false, // Disable transparency to prevent flickering
                    opacity: 1.0,
                    depthWrite: true,
                    depthTest: true
                });
                
                const marker = new THREE.Mesh(geometry, material);
                
                // Store original color for selection highlighting
                const originalColor = texture ? 0xffffff : 0x00ff00;
                marker.userData.originalColor = originalColor;
                
                // Offset marker slightly along its normal to prevent z-fighting with mesh
                // The normal is the forward direction of the marker (Z+ in local space)
                const offsetDistance = thickness * 0.6; // Push forward by 60% of thickness
                const offset = new THREE.Vector3(0, 0, offsetDistance);
                
                // Apply rotation from quaternion
                const quat = markerData.pose_absolute.rotation.quaternion;
                console.log('Setting marker quaternion:', {
                    internal_id: markerData.internal_id,
                    aruco_id: markerData.aruco_id,
                    in_plane_rotation_deg: markerData.in_plane_rotation_deg,
                    quaternion: quat
                });
                marker.quaternion.set(quat.x, quat.y, quat.z, quat.w);
                
                // pose_absolute.position is already in world space, accounting for CAD object transformation
                // Since marker is a child of meshObject, we need to convert world position to meshObject's local space
                // If meshObject is at origin with no transform, world position = local position
                // But if meshObject has been transformed, we need to account for that
                const worldPos = new THREE.Vector3(pos.x, pos.y, pos.z);
                
                // Transform offset to marker's local space and apply
                offset.applyQuaternion(marker.quaternion);
                
                // If meshObject exists and has been transformed, convert world position to local
                if (meshObject && (meshObject.position.lengthSq() > 1e-6 || meshObject.quaternion.w !== 1)) {
                    // meshObject has been transformed, need to convert world to local
                    const worldToLocal = new THREE.Matrix4().copy(meshObject.matrixWorld).invert();
                    const localPos = worldPos.clone().applyMatrix4(worldToLocal);
                    marker.position.copy(localPos).add(offset);
                } else {
                    // meshObject is at origin, world position = local position
                    marker.position.copy(worldPos).add(offset);
                }
                
                marker.userData.markerId = arucoId;
                // Always use internal_id (required for backend API calls)
                // internal_id is the dictionary key in session_state["markers"]
                if (!markerData.internal_id && markerData.internal_id !== 0) {
                    console.error('Marker data missing internal_id:', markerData);
                }
                // Ensure internalId is stored as a number for consistent comparison
                marker.userData.internalId = typeof markerData.internal_id === 'number' 
                    ? markerData.internal_id 
                    : parseInt(markerData.internal_id);
                
                // Make marker a child of meshObject so it moves/rotates with the CAD object
                if (meshObject) {
                    meshObject.add(marker);
                } else {
                    // Fallback: add to scene if meshObject not available
                    scene.add(marker);
                }
                markers.push(marker);
            } catch (error) {
                console.error('Error creating marker:', error);
                // Fallback: create marker without texture
                const material = new THREE.MeshStandardMaterial({
                    color: 0x00ff00,
                    side: THREE.DoubleSide,
                    transparent: false,
                    opacity: 1.0,
                    depthWrite: true,
                    depthTest: true
                });
                const marker = new THREE.Mesh(geometry, material);
                
                // Store original color for selection highlighting
                marker.userData.originalColor = 0x00ff00;
                
                // Apply offset for fallback marker too
                const offsetDistance = thickness * 0.6;
                const offset = new THREE.Vector3(0, 0, offsetDistance);
                const quat = markerData.pose_absolute.rotation.quaternion;
                marker.quaternion.set(quat.x, quat.y, quat.z, quat.w);
                
                // pose_absolute.position is already in world space
                const worldPos = new THREE.Vector3(pos.x, pos.y, pos.z);
                
                offset.applyQuaternion(marker.quaternion);
                
                // If meshObject exists and has been transformed, convert world position to local
                if (meshObject && (meshObject.position.lengthSq() > 1e-6 || meshObject.quaternion.w !== 1)) {
                    const worldToLocal = new THREE.Matrix4().copy(meshObject.matrixWorld).invert();
                    const localPos = worldPos.clone().applyMatrix4(worldToLocal);
                    marker.position.copy(localPos).add(offset);
                } else {
                    marker.position.copy(worldPos).add(offset);
                }
                
                marker.userData.markerId = arucoId;
                // Always use internal_id (required for backend API calls)
                // internal_id is the dictionary key in session_state["markers"]
                if (!markerData.internal_id) {
                    console.error('Marker data missing internal_id:', markerData);
                }
                marker.userData.internalId = markerData.internal_id;
                
                // Make marker a child of meshObject so it moves/rotates with the CAD object
                if (meshObject) {
                    meshObject.add(marker);
                } else {
                    scene.add(marker);
                }
                markers.push(marker);
            }
        }
        
        async function refreshMarkers(skipRotationControlsUpdate = false, skipTranslationUpdate = false) {
            try {
                // Store current selection before refresh
                const previousSelectedId = selectedMarkerId;
                
                const response = await fetch('/api/markers');
                const data = await response.json();
                
                // Clear existing markers from scene or meshObject
                markers.forEach(m => {
                    if (m.parent) {
                        m.parent.remove(m);
                    }
                });
                markers = [];
                selectedMarkerMesh = null;
                
                // Add all markers (await each one)
                for (const marker of data.markers) {
                    // Debug: log marker rotation BEFORE adding to scene
                    if (marker.internal_id === previousSelectedId || marker.aruco_id === previousSelectedId) {
                        console.log('Refreshing marker with data:', {
                            internal_id: marker.internal_id,
                            aruco_id: marker.aruco_id,
                            in_plane_rotation_deg: marker.in_plane_rotation_deg,
                            quaternion: marker.pose_absolute.rotation.quaternion,
                            position: marker.pose_absolute.position
                        });
                    }
                    await addMarkerToScene(marker);
                    // Debug: log marker rotation AFTER adding to scene
                    if (marker.internal_id === previousSelectedId || marker.aruco_id === previousSelectedId) {
                        const markerMesh = markers.find(m => m.userData.internalId === (marker.internal_id || marker.aruco_id));
                        if (markerMesh) {
                            console.log('Marker mesh after adding to scene:', {
                                internal_id: markerMesh.userData.internalId,
                                quaternion: {
                                    x: markerMesh.quaternion.x,
                                    y: markerMesh.quaternion.y,
                                    z: markerMesh.quaternion.z,
                                    w: markerMesh.quaternion.w
                                },
                                position: {
                                    x: markerMesh.position.x,
                                    y: markerMesh.position.y,
                                    z: markerMesh.position.z
                                }
                            });
                        }
                    }
                }
                
                // Restore selection if marker still exists
                if (previousSelectedId !== null) {
                    const prevIdNum = typeof previousSelectedId === 'string' ? parseInt(previousSelectedId) : previousSelectedId;
                    const markerMesh = markers.find(m => {
                        const meshId = m.userData.internalId;
                        const meshIdNum = typeof meshId === 'string' ? parseInt(meshId) : meshId;
                        return meshIdNum === prevIdNum;
                    });
                    if (markerMesh) {
                        // Just update the visual selection
                        selectedMarkerId = prevIdNum;
                        selectedMarkerMesh = markerMesh;
                        if (markerMesh.userData.originalColor !== undefined) {
                            markerMesh.material.color.setHex(0x00ff00); // Green for selected
                        }
                        // Update list selection
                        document.querySelectorAll('.marker-item').forEach(item => {
                            item.classList.remove('selected');
                            const itemId = parseInt(item.dataset.internalId);
                            if (itemId === prevIdNum || String(itemId) === String(prevIdNum)) {
                                item.classList.add('selected');
                            }
                        });
                        
                        // Only reload rotation controls if not skipping (e.g., after rotation update)
                        if (!skipRotationControlsUpdate) {
                            showRotationControls(previousSelectedId, skipTranslationUpdate);
                        }
                    } else {
                        selectedMarkerId = null;
                    }
                }
                
                // Update marker list
                updateMarkerList(data.markers);
            } catch (error) {
                console.error('Error refreshing markers:', error);
            }
        }
        
        function updateMarkerList(markers) {
            const list = document.getElementById('markersList');
            list.innerHTML = '';
            
            markers.forEach(marker => {
                const item = document.createElement('div');
                item.className = 'marker-item';
                // Always use internal_id (required for backend API calls)
                // internal_id is the dictionary key in session_state["markers"]
                // Note: internal_id can be 0, so we need to check for undefined/null explicitly
                if (marker.internal_id === undefined || marker.internal_id === null) {
                    console.error('Marker missing internal_id:', marker);
                    return; // Skip markers without internal_id
                }
                const internalId = marker.internal_id;
                item.textContent = `ArUco ${marker.aruco_id} (${marker.face_type})`;
                item.dataset.internalId = String(internalId); // Ensure it's a string for dataset
                item.onclick = (e) => {
                    selectMarker(internalId, e);
                };
                list.appendChild(item);
            });
            
            // Update swap display if markers are assigned
            if (typeof updateSwapDisplay === 'function') {
                updateSwapDisplay().catch(() => {}); // Ignore errors
            }
        }
        
        async function selectMarker(internalId, event) {
            // internalId should always be internal_id (not aruco_id)
            // Convert to number for comparison
            const internalIdNum = typeof internalId === 'string' ? parseInt(internalId) : internalId;
            
            // Find the marker mesh in the scene
            const markerMesh = markers.find(m => {
                const meshInternalId = m.userData.internalId;
                // Compare as numbers
                return (typeof meshInternalId === 'number' ? meshInternalId : parseInt(meshInternalId)) === internalIdNum;
            });
            
            if (markerMesh) {
                selectMarkerInScene(internalIdNum, markerMesh);
            } else {
                // Fallback: just update list selection if marker mesh not found
                // Try to fetch marker data to ensure we have the right ID
                try {
                    const response = await fetch('/api/markers');
                    const data = await response.json();
                    const marker = data.markers.find(m => {
                        const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                        return markerId === internalIdNum;
                    });
                    if (marker) {
                        // Ensure internal_id is stored as a number
                        const markerInternalId = typeof marker.internal_id === 'number' 
                            ? marker.internal_id 
                            : parseInt(marker.internal_id);
                        selectedMarkerId = markerInternalId;
                        document.querySelectorAll('.marker-item').forEach(item => {
                            item.classList.remove('selected');
                            const itemId = parseInt(item.dataset.internalId);
                            if (itemId === markerInternalId || String(itemId) === String(markerInternalId)) {
                                item.classList.add('selected');
                            }
                        });
                        showRotationControls(markerInternalId);
                    } else {
                        alert('Marker not found in scene or API');
                    }
                } catch (e) {
                    console.error('Error fetching markers:', e);
                    alert('Error selecting marker');
                }
            }
        }
        
        async function showRotationControls(internalId, skipTranslationUpdate = false) {
            // Show rotation and translation controls and load current values
            const rotationPanel = document.getElementById('rotationControls');
            const translationPanel = document.getElementById('translationControls');
            rotationPanel.style.display = 'block';
            translationPanel.style.display = 'block';
            
            try {
                const response = await fetch('/api/markers');
                const data = await response.json();
                const marker = data.markers.find(m => (m.internal_id || m.aruco_id) === internalId);
                
                if (marker) {
                    const rot = marker.pose_absolute.rotation;
                    // Convert radians to degrees and snap to 5-degree increments
                    const rollDeg = snapTo5Degrees(rot.roll * 180 / Math.PI);
                    const pitchDeg = snapTo5Degrees(rot.pitch * 180 / Math.PI);
                    // ALWAYS use in_plane_rotation_deg if available (preferred)
                    // Only fall back to calculating from yaw if in_plane_rotation_deg is not present
                    let yawDeg;
                    if (marker.in_plane_rotation_deg !== undefined && marker.in_plane_rotation_deg !== null) {
                        yawDeg = snapTo5Degrees(marker.in_plane_rotation_deg);
                    } else {
                        // Fallback: calculate from yaw (may have gimbal lock issues)
                        yawDeg = snapTo5Degrees(rot.yaw * 180 / Math.PI);
                    }
                    
                    document.getElementById('rotateRoll').value = rollDeg;
                    document.getElementById('rotatePitch').value = pitchDeg;
                    document.getElementById('rotateYaw').value = yawDeg;
                    
                    updateRotationDisplay('roll', rollDeg);
                    updateRotationDisplay('pitch', pitchDeg);
                    updateRotationDisplay('yaw', yawDeg);
                    
                    // Load current translation offset (only if not skipping)
                    if (!skipTranslationUpdate) {
                        let transX = 0, transY = 0;
                        if (marker.translation_offset !== undefined) {
                            transX = marker.translation_offset.x || 0;
                            transY = marker.translation_offset.y || 0;
                        }
                        // Set input fields to current translation offset
                        document.getElementById('translateX').value = transX.toFixed(4);
                        document.getElementById('translateY').value = transY.toFixed(4);
                    }
                }
            } catch (error) {
                console.error('Error loading marker rotation:', error);
            }
        }
        
        function snapTo5Degrees(value) {
            // Snap value to nearest 5-degree increment
            return Math.round(parseFloat(value) / 5) * 5;
        }
        
        function updateRotationDisplay(axis, value, skipSliderUpdate = false) {
            const valueSpan = document.getElementById(axis + 'Value');
            if (valueSpan) {
                const snapped = snapTo5Degrees(value);
                valueSpan.textContent = snapped + '°';
                // Update slider value to snapped value (but don't trigger oninput)
                // IMPORTANT: For yaw (in-plane rotation), we should be careful not to reset
                // the slider if the user is actively adjusting it
                if (!skipSliderUpdate) {
                    const sliderId = 'rotate' + axis.charAt(0).toUpperCase() + axis.slice(1);
                    const slider = document.getElementById(sliderId);
                    if (slider) {
                        const currentValue = parseFloat(slider.value) || 0;
                        const snappedCurrent = snapTo5Degrees(currentValue);
                        
                        // Only update if the snapped value is different AND
                        // we're not in the middle of a user interaction
                        // For yaw slider, be more conservative to avoid resetting user input
                        if (axis === 'yaw') {
                            // For yaw, only update if the difference is significant (more than snapping threshold)
                            // This prevents resetting the slider when user is adjusting it
                            if (Math.abs(snappedCurrent - snapped) > 2.5) {
                                const originalOninput = slider.oninput;
                                slider.oninput = null;
                                slider.value = snapped;
                                slider.oninput = originalOninput;
                            }
                        } else {
                            // For roll/pitch (disabled), always update to match display
                            if (parseFloat(slider.value) !== snapped) {
                                const originalOninput = slider.oninput;
                                slider.oninput = null;
                                slider.value = snapped;
                                slider.oninput = originalOninput;
                            }
                        }
                    }
                }
            }
        }
        
        async function resetRotationControls() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            
            // Only reset in-plane rotation (yaw/Z-axis), not base rotation (roll/pitch)
            // Base rotation aligns marker with face normal and should not be changed
            try {
                // Get marker to find internal_id
                const markersResponse = await fetch('/api/markers');
                const markersData = await markersResponse.json();
                
                // Normalize selectedMarkerId for comparison
                const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
                const marker = markersData.markers.find(m => {
                    const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                    return markerId === selectedIdNum;
                });
                
                if (!marker) {
                    console.error('Marker not found. selectedMarkerId:', selectedMarkerId, 'Available markers:', markersData.markers.map(m => ({ internal_id: m.internal_id, aruco_id: m.aruco_id })));
                    alert('Marker not found. Check console for details.');
                    return;
                }
                
                // Use internal_id for the API call (required by backend)
                if (marker.internal_id === undefined && marker.internal_id !== 0) {
                    alert('Marker internal_id not found');
                    return;
                }
                // Ensure internal_id is a number
                const markerInternalId = typeof marker.internal_id === 'number' 
                    ? marker.internal_id 
                    : parseInt(marker.internal_id);
                
                const response = await fetch(`/api/markers/${markerInternalId}/rotation`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        mode: 'absolute',
                        yaw: 0  // Reset in-plane rotation to 0
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    // Update UI
                    document.getElementById('rotateYaw').value = 0;
                    updateRotationDisplay('yaw', 0);
                    // Refresh markers to show updated rotation
                    await refreshMarkers();
                } else {
                    const errorData = await response.json();
                    alert('Error resetting rotation: ' + errorData.detail);
                }
            } catch (error) {
                alert('Error resetting rotation: ' + error.message);
            }
        }
        
        function updateTranslationDisplay(axis, value) {
            // No longer needed since we're using number inputs directly
            // But keep function for compatibility
        }
        
        async function applyTranslation() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            
            const x = parseFloat(document.getElementById('translateX').value) || 0;
            const y = parseFloat(document.getElementById('translateY').value) || 0;
            
            try {
                // Get marker to find internal_id
                const markersResponse = await fetch('/api/markers');
                const markersData = await markersResponse.json();
                
                // Normalize selectedMarkerId for comparison
                const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
                const marker = markersData.markers.find(m => {
                    const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                    return markerId === selectedIdNum;
                });
                
                if (!marker) {
                    console.error('Marker not found. selectedMarkerId:', selectedMarkerId, 'Available markers:', markersData.markers.map(m => ({ internal_id: m.internal_id, aruco_id: m.aruco_id })));
                    alert('Marker not found. Check console for details.');
                    return;
                }
                
                // Use internal_id for the API call (required by backend)
                if (marker.internal_id === undefined && marker.internal_id !== 0) {
                    alert('Marker internal_id not found');
                    return;
                }
                // Ensure internal_id is a number
                const markerInternalId = typeof marker.internal_id === 'number' 
                    ? marker.internal_id 
                    : parseInt(marker.internal_id);
                
                // Use absolute mode - the value entered is the actual translation offset
                const response = await fetch(`/api/markers/${markerInternalId}/translation`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        mode: 'absolute',
                        x: x,
                        y: y
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    // Update input fields with the actual translation offset that was set
                    document.getElementById('translateX').value = data.translation_offset.x.toFixed(4);
                    document.getElementById('translateY').value = data.translation_offset.y.toFixed(4);
                    // Refresh markers to show updated position (only the selected marker should move)
                    // Skip translation update to prevent overwriting the values we just set
                    await refreshMarkers(false, true);
                } else {
                    const errorData = await response.json();
                    alert('Error applying translation: ' + errorData.detail);
                }
            } catch (error) {
                alert('Error applying translation: ' + error.message);
            }
        }
        
        async function resetTranslationControls() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            
            try {
                // Get marker to find internal_id
                const markersResponse = await fetch('/api/markers');
                const markersData = await markersResponse.json();
                
                // Normalize selectedMarkerId for comparison
                const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
                const marker = markersData.markers.find(m => {
                    const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                    return markerId === selectedIdNum;
                });
                
                if (!marker) {
                    console.error('Marker not found. selectedMarkerId:', selectedMarkerId, 'Available markers:', markersData.markers.map(m => ({ internal_id: m.internal_id, aruco_id: m.aruco_id })));
                    alert('Marker not found. Check console for details.');
                    return;
                }
                
                // Use internal_id for the API call (required by backend)
                if (marker.internal_id === undefined && marker.internal_id !== 0) {
                    alert('Marker internal_id not found');
                    return;
                }
                // Ensure internal_id is a number
                const markerInternalId = typeof marker.internal_id === 'number' 
                    ? marker.internal_id 
                    : parseInt(marker.internal_id);
                
                // Reset translation offset to 0,0 on backend
                const response = await fetch(`/api/markers/${markerInternalId}/translation`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        mode: 'absolute',
                        x: 0,
                        y: 0
                    })
                });
                
                if (response.ok) {
                    // Reset UI sliders
                    document.getElementById('translateX').value = 0;
                    document.getElementById('translateY').value = 0;
                    updateTranslationDisplay('x', 0);
                    updateTranslationDisplay('y', 0);
                    // Refresh markers to show updated position
                    await refreshMarkers();
                } else {
                    const errorData = await response.json();
                    alert('Error resetting translation: ' + errorData.detail);
                }
            } catch (error) {
                alert('Error resetting translation: ' + error.message);
            }
        }
        
        async function moveInPlane(direction) {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            
            try {
                // Get step size
                const stepSize = parseFloat(document.getElementById('inplaneStepSize').value) || 0.0005;
                
                // Get marker to find internal_id
                const markersResponse = await fetch('/api/markers');
                const markersData = await markersResponse.json();
                
                const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
                const marker = markersData.markers.find(m => {
                    const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                    return markerId === selectedIdNum;
                });
                
                if (!marker) {
                    alert('Marker not found');
                    return;
                }
                
                const markerInternalId = typeof marker.internal_id === 'number' 
                    ? marker.internal_id 
                    : parseInt(marker.internal_id);
                
                // Determine movement direction
                let xDelta = 0;
                let yDelta = 0;
                
                if (direction === 'axis1_neg') {
                    xDelta = -stepSize;
                } else if (direction === 'axis1_pos') {
                    xDelta = stepSize;
                } else if (direction === 'axis2_neg') {
                    yDelta = -stepSize;
                } else if (direction === 'axis2_pos') {
                    yDelta = stepSize;
                }
                
                // Apply relative translation
                const response = await fetch(`/api/markers/${markerInternalId}/translation`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        mode: 'relative',
                        x: xDelta,
                        y: yDelta
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // Update input fields
                    document.getElementById('translateX').value = data.translation_offset.x.toFixed(4);
                    document.getElementById('translateY').value = data.translation_offset.y.toFixed(4);
                    
                    // Refresh markers to show updated position
                    // Skip translation update to prevent overwriting the values we just set
                    await refreshMarkers(false, true);
                } else {
                    const errorData = await response.json();
                    alert('Error moving marker: ' + errorData.detail);
                }
            } catch (error) {
                alert('Error moving marker: ' + error.message);
            }
        }
        
        async function applyRotation() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            
            // IMPORTANT: Only yaw slider controls in-plane rotation
            // Roll and pitch are part of the base (geometric) rotation and are NOT editable
            // Get yaw slider value directly (this is the in-plane rotation in degrees)
            const yawSlider = document.getElementById('rotateYaw');
            if (!yawSlider) {
                alert('Yaw slider not found');
                return;
            }
            
            // Read the slider value BEFORE any API calls (to avoid race conditions)
            const yawSliderValue = parseFloat(yawSlider.value) || 0;
            const yaw = snapTo5Degrees(yawSliderValue);
            
            console.log('Reading slider value:', {
                rawValue: yawSliderValue,
                snappedValue: yaw,
                sliderElement: yawSlider.value
            });
            
            // Get current rotation to calculate delta
            try {
                const markersResponse = await fetch('/api/markers');
                const markersData = await markersResponse.json();
                
                // selectedMarkerId should be internal_id, but handle type conversion for safety
                const selectedIdNum = typeof selectedMarkerId === 'string' ? parseInt(selectedMarkerId) : selectedMarkerId;
                const marker = markersData.markers.find(m => {
                    const markerInternalId = typeof m.internal_id === 'string' ? parseInt(m.internal_id) : m.internal_id;
                    return markerInternalId === selectedIdNum;
                });
                
                if (!marker) {
                    console.error('Marker not found. selectedMarkerId:', selectedMarkerId, 'Available markers:', markersData.markers.map(m => ({ internal_id: m.internal_id, aruco_id: m.aruco_id })));
                    alert('Marker not found. Check console for details.');
                    return;
                }
                
                // Get current in-plane rotation directly (in degrees)
                // This is the ONLY rotation value we can change (in-plane rotation)
                const currentInPlaneDeg = marker.in_plane_rotation_deg !== undefined 
                    ? marker.in_plane_rotation_deg 
                    : 0;
                
                console.log('Applying rotation:', {
                    selectedMarkerId: selectedMarkerId,
                    markerInternalId: marker.internal_id,
                    currentInPlaneDeg: currentInPlaneDeg,
                    sliderYaw: yaw,
                    delta: yaw - currentInPlaneDeg,
                    note: 'Only yaw (in-plane rotation) is editable. Roll/pitch are geometric and fixed.'
                });
                
                // Calculate delta for in-plane rotation
                // The slider shows the desired absolute rotation, so calculate delta
                const yawDelta = yaw - currentInPlaneDeg;
                
                // Use internal_id for the API call (required by backend)
                // internal_id is the dictionary key in the backend
                if (marker.internal_id === undefined && marker.internal_id !== 0) {
                    alert('Marker internal_id not found');
                    return;
                }
                // Ensure internal_id is a number
                const markerInternalId = typeof marker.internal_id === 'number' 
                    ? marker.internal_id 
                    : parseInt(marker.internal_id);
                
                const response = await fetch(`/api/markers/${markerInternalId}/rotation`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        yaw: yawDelta,  // Only send yaw for in-plane rotation
                        mode: "relative"
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('Rotation response:', data);
                    console.log('Response in_plane_rotation_deg:', data.in_plane_rotation_deg);
                    
                    // Update rotation controls FIRST with response data (before refresh)
                    const newInPlaneDeg = data.in_plane_rotation_deg !== undefined 
                        ? snapTo5Degrees(data.in_plane_rotation_deg)
                        : 0;
                    
                    console.log('Setting slider to:', newInPlaneDeg);
                    
                    // Set slider values directly, temporarily disabling oninput to prevent loops
                    const rollSlider = document.getElementById('rotateRoll');
                    const pitchSlider = document.getElementById('rotatePitch');
                    const yawSlider = document.getElementById('rotateYaw');
                    
                    // Store original handlers
                    const rollHandler = rollSlider ? rollSlider.oninput : null;
                    const pitchHandler = pitchSlider ? pitchSlider.oninput : null;
                    const yawHandler = yawSlider ? yawSlider.oninput : null;
                    
                    // Also get full rotation for display
                    const newRot = data.rotation;
                    const rollDeg = snapTo5Degrees(newRot[0] * 180 / Math.PI);
                    const pitchDeg = snapTo5Degrees(newRot[1] * 180 / Math.PI);
                    
                    // Temporarily disable handlers and set values
                    if (rollSlider) {
                        rollSlider.oninput = null;
                        rollSlider.value = rollDeg;
                        rollSlider.oninput = rollHandler;
                    }
                    if (pitchSlider) {
                        pitchSlider.oninput = null;
                        pitchSlider.value = pitchDeg;
                        pitchSlider.oninput = pitchHandler;
                    }
                    if (yawSlider) {
                        yawSlider.oninput = null;
                        yawSlider.value = newInPlaneDeg;
                        yawSlider.oninput = yawHandler;
                        console.log('Yaw slider value after setting:', yawSlider.value);
                    }
                    
                    // Update display text (but don't update slider again to avoid loops)
                    const rollValueSpan = document.getElementById('rollValue');
                    const pitchValueSpan = document.getElementById('pitchValue');
                    const yawValueSpan = document.getElementById('yawValue');
                    if (rollValueSpan) rollValueSpan.textContent = rollDeg + '°';
                    if (pitchValueSpan) pitchValueSpan.textContent = pitchDeg + '°';
                    if (yawValueSpan) yawValueSpan.textContent = newInPlaneDeg + '°';
                    
                    console.log('Before refresh - slider values:', {
                        roll: rollSlider ? rollSlider.value : 'N/A',
                        pitch: pitchSlider ? pitchSlider.value : 'N/A',
                        yaw: yawSlider ? yawSlider.value : 'N/A'
                    });
                    
                    // NOW refresh markers to update visualization (skip rotation controls update)
                    // Add a small delay to ensure backend has processed the update
                    await new Promise(resolve => setTimeout(resolve, 50));
                    await refreshMarkers(true);
                    
                    console.log('After refresh - slider values:', {
                        roll: rollSlider ? rollSlider.value : 'N/A',
                        pitch: pitchSlider ? pitchSlider.value : 'N/A',
                        yaw: yawSlider ? yawSlider.value : 'N/A'
                    });
                    
                    // Verify slider value is still correct after refresh
                    // Also verify the marker data was actually updated
                    const verifyResponse = await fetch('/api/markers');
                    const verifyData = await verifyResponse.json();
                    const updatedMarker = verifyData.markers.find(m => (m.internal_id || m.aruco_id) === selectedMarkerId);
                    
                    if (updatedMarker) {
                        console.log('Marker data after update:', {
                            in_plane_rotation_deg: updatedMarker.in_plane_rotation_deg,
                            expected: newInPlaneDeg,
                            match: Math.abs(updatedMarker.in_plane_rotation_deg - newInPlaneDeg) < 0.1,
                            quaternion: updatedMarker.pose_absolute.rotation.quaternion
                        });
                        
                        // Also check the actual marker mesh in the scene
                        const markerMesh = markers.find(m => m.userData.internalId === selectedMarkerId);
                        if (markerMesh) {
                            console.log('Marker mesh quaternion in scene:', {
                                internal_id: markerMesh.userData.internalId,
                                quaternion: {
                                    x: markerMesh.quaternion.x,
                                    y: markerMesh.quaternion.y,
                                    z: markerMesh.quaternion.z,
                                    w: markerMesh.quaternion.w
                                }
                            });
                            
                            // Check if they match
                            const quatMatch = Math.abs(markerMesh.quaternion.x - updatedMarker.pose_absolute.rotation.quaternion.x) < 0.001 &&
                                            Math.abs(markerMesh.quaternion.y - updatedMarker.pose_absolute.rotation.quaternion.y) < 0.001 &&
                                            Math.abs(markerMesh.quaternion.z - updatedMarker.pose_absolute.rotation.quaternion.z) < 0.001 &&
                                            Math.abs(markerMesh.quaternion.w - updatedMarker.pose_absolute.rotation.quaternion.w) < 0.001;
                            console.log('Quaternion match:', quatMatch);
                            if (!quatMatch) {
                                console.error('MISMATCH: Scene quaternion does not match API quaternion!');
                                console.error('API quaternion:', updatedMarker.pose_absolute.rotation.quaternion);
                                console.error('Scene quaternion:', {
                                    x: markerMesh.quaternion.x,
                                    y: markerMesh.quaternion.y,
                                    z: markerMesh.quaternion.z,
                                    w: markerMesh.quaternion.w
                                });
                            }
                        } else {
                            console.error('Marker mesh not found in scene after refresh!');
                        }
                    }
                    
                    if (yawSlider) {
                        const currentValue = parseFloat(yawSlider.value);
                        console.log('Yaw slider value after refresh:', {
                            current: currentValue,
                            expected: newInPlaneDeg,
                            difference: Math.abs(currentValue - newInPlaneDeg)
                        });
                        
                        // If it changed, set it again (something reset it)
                        if (Math.abs(currentValue - newInPlaneDeg) > 0.1) {
                            console.log('⚠️ Slider value changed after refresh! Resetting to:', newInPlaneDeg);
                            yawSlider.oninput = null;
                            yawSlider.value = newInPlaneDeg;
                            yawSlider.oninput = yawHandler;
                            // Update display directly without calling updateRotationDisplay
                            const yawValueSpan = document.getElementById('yawValue');
                            if (yawValueSpan) yawValueSpan.textContent = newInPlaneDeg + '°';
                        } else {
                            console.log('✓ Slider value is correct after refresh');
                        }
                    }
                    
                    // Re-select the marker visually (but don't reload rotation controls)
                    if (selectedMarkerId !== null) {
                        const markerMesh = markers.find(m => m.userData.internalId === selectedMarkerId);
                        if (markerMesh) {
                            // Just update visual selection, don't call showRotationControls
                            selectedMarkerMesh = markerMesh;
                            if (markerMesh.userData.originalColor !== undefined) {
                                markerMesh.material.color.setHex(0x00ff00);
                            }
                        }
                    }
                } else {
                    let errorMessage = 'Unknown error';
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.detail || JSON.stringify(errorData);
                    } catch (e) {
                        errorMessage = await response.text() || 'Unknown error';
                    }
                    alert('Error: ' + errorMessage);
                }
            } catch (error) {
                alert('Error: ' + (error.message || String(error)));
            }
        }
        
        // Store selected markers for swapping
        let swapMarker1Id = null;
        let swapMarker2Id = null;
        
        async function updateSwapDisplay() {
            const display1 = document.getElementById('swapMarker1Display');
            const display2 = document.getElementById('swapMarker2Display');
            
            if (swapMarker1Id !== null) {
                try {
                    const response = await fetch('/api/markers');
                    const data = await response.json();
                    // swapMarker1Id should be internal_id (number), normalize for comparison
                    const swap1IdNum = typeof swapMarker1Id === 'number' ? swapMarker1Id : parseInt(swapMarker1Id);
                    const m = data.markers.find(m => {
                        const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                        return markerId === swap1IdNum;
                    });
                    if (m) {
                        display1.textContent = `ArUco ${m.aruco_id}`;
                    } else {
                        display1.textContent = 'Not assigned';
                        swapMarker1Id = null; // Clear invalid assignment
                    }
                } catch (e) {
                    display1.textContent = 'Not assigned';
                }
            } else {
                display1.textContent = 'Not assigned';
            }
            
            if (swapMarker2Id !== null) {
                try {
                    const response = await fetch('/api/markers');
                    const data = await response.json();
                    // swapMarker2Id should be internal_id (number), normalize for comparison
                    const swap2IdNum = typeof swapMarker2Id === 'number' ? swapMarker2Id : parseInt(swapMarker2Id);
                    const m = data.markers.find(m => {
                        const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                        return markerId === swap2IdNum;
                    });
                    if (m) {
                        display2.textContent = `ArUco ${m.aruco_id}`;
                    } else {
                        display2.textContent = 'Not assigned';
                        swapMarker2Id = null; // Clear invalid assignment
                    }
                } catch (e) {
                    display2.textContent = 'Not assigned';
                }
            } else {
                display2.textContent = 'Not assigned';
            }
        }
        
        function assignToMarker1() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            // selectedMarkerId should already be internal_id (number), but normalize to ensure consistency
            swapMarker1Id = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
            if (isNaN(swapMarker1Id)) {
                alert('Invalid marker selection');
                return;
            }
            updateSwapDisplay();
        }
        
        function assignToMarker2() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            // selectedMarkerId should already be internal_id (number), but normalize to ensure consistency
            swapMarker2Id = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
            if (isNaN(swapMarker2Id)) {
                alert('Invalid marker selection');
                return;
            }
            updateSwapDisplay();
        }
        
        function clearSwapSelection() {
            swapMarker1Id = null;
            swapMarker2Id = null;
            updateSwapDisplay();
        }
        
        async function swapMarkerPositions() {
            if (swapMarker1Id === null || swapMarker2Id === null) {
                alert('Please assign both markers first (select a marker and click "Assign Selected" for Marker 1 and Marker 2)');
                return;
            }
            
            if (swapMarker1Id === swapMarker2Id) {
                alert('Cannot swap marker with itself');
                return;
            }
            
            // Get marker info to verify internal_ids and get ArUco IDs for display
            // swapMarker1Id and swapMarker2Id should already be internal_id (numbers)
            const swap1IdNum = typeof swapMarker1Id === 'number' ? swapMarker1Id : parseInt(swapMarker1Id);
            const swap2IdNum = typeof swapMarker2Id === 'number' ? swapMarker2Id : parseInt(swapMarker2Id);
            
            let marker1ArucoId = null;
            let marker2ArucoId = null;
            let marker1InternalId = swap1IdNum;
            let marker2InternalId = swap2IdNum;
            
            try {
                const response = await fetch('/api/markers');
                const data = await response.json();
                
                // Find markers by internal_id only (swapMarker1Id and swapMarker2Id are internal_ids)
                const m1 = data.markers.find(m => {
                    const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                    return markerId === swap1IdNum;
                });
                const m2 = data.markers.find(m => {
                    const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                    return markerId === swap2IdNum;
                });
                
                if (!m1) {
                    alert(`Marker 1 (internal_id: ${swap1IdNum}) not found`);
                    return;
                }
                if (!m2) {
                    alert(`Marker 2 (internal_id: ${swap2IdNum}) not found`);
                    return;
                }
                
                // Use internal_id for the swap (required by backend)
                marker1InternalId = typeof m1.internal_id === 'number' ? m1.internal_id : parseInt(m1.internal_id);
                marker2InternalId = typeof m2.internal_id === 'number' ? m2.internal_id : parseInt(m2.internal_id);
                marker1ArucoId = m1.aruco_id;
                marker2ArucoId = m2.aruco_id;
            } catch (e) {
                alert('Error fetching marker information: ' + e.message);
                return;
            }
            
            if (!confirm(`Swap ArUco IDs of markers ${marker1ArucoId} and ${marker2ArucoId}?`)) {
                return;
            }
            
            try {
                // marker1InternalId and marker2InternalId should already be numbers, but ensure they are
                const marker1IdInt = typeof marker1InternalId === 'number' ? marker1InternalId : parseInt(marker1InternalId);
                const marker2IdInt = typeof marker2InternalId === 'number' ? marker2InternalId : parseInt(marker2InternalId);
                
                if (isNaN(marker1IdInt) || isNaN(marker2IdInt)) {
                    alert('Invalid marker IDs');
                    return;
                }
                
                const response = await fetch('/api/markers/swap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ marker1_id: marker1IdInt, marker2_id: marker2IdInt })
                });
                
                if (response.ok) {
                    await refreshMarkers();
                    // Clear swap selection
                    clearSwapSelection();
                } else {
                    let errorMessage = 'Unknown error';
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.detail || JSON.stringify(errorData);
                    } catch (e) {
                        errorMessage = await response.text() || 'Unknown error';
                    }
                    alert('Error: ' + errorMessage);
                }
            } catch (error) {
                alert('Error: ' + (error.message || String(error)));
            }
        }
        
        async function removeSelectedMarker() {
            if (selectedMarkerId === null) {
                alert('Please select a marker first');
                return;
            }
            
            try {
                const response = await fetch(`/api/markers/${selectedMarkerId}`, { method: 'DELETE' });
                if (response.ok) {
                    await refreshMarkers();
                    selectedMarkerId = null;
                } else {
                    const data = await response.json();
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function clearAllMarkers() {
            if (!confirm('Clear all markers?')) return;
            
            try {
                const response = await fetch('/api/markers', { method: 'DELETE' });
                if (response.ok) {
                    await refreshMarkers();
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function exportAnnotations() {
            try {
                const response = await fetch('/api/export');
                if (!response.ok) {
                    const data = await response.json();
                    alert('Error: ' + data.detail);
                    return;
                }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const filename = session_state.current_file ? 
                    session_state.current_file.replace(/\\.[^/.]+$/, '') + '_aruco.json' : 
                    'aruco.json';
                a.download = filename;
                a.click();
                window.URL.revokeObjectURL(url);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function exportWireframe() {
            try {
                const response = await fetch('/api/export-wireframe');
                if (!response.ok) {
                    const data = await response.json();
                    alert('Error: ' + data.detail);
                    return;
                }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const filename = session_state.current_file ? 
                    session_state.current_file.replace(/\\.[^/.]+$/, '') + '_wireframe.json' : 
                    'wireframe.json';
                a.download = filename;
                a.click();
                window.URL.revokeObjectURL(url);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function importAnnotations() {
            const fileInput = document.getElementById('importFile');
            const file = fileInput.files[0];
            
            try {
                let response;
                let source = '';
                
                // If user uploaded a file, use it
                if (file) {
            const formData = new FormData();
            formData.append('file', file);
            
                    response = await fetch('/api/import', {
                    method: 'POST',
                    body: formData
                });
                    source = 'uploaded file';
                } else {
                    // No file uploaded, try to auto-load from data folder
                    response = await fetch('/api/import-auto', {
                        method: 'POST'
                    });
                    source = 'data folder';
                }
                
                if (response.ok) {
                    await refreshMarkers();
                    // Load CAD pose after import
                    await loadCADPose();
                    alert(`Annotations imported successfully from ${source}`);
                } else {
                    const data = await response.json();
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function updateStatus(id, message, type) {
            const status = document.getElementById(id);
            status.textContent = message;
            status.className = 'status ' + (type || '');
        }
        
        // Placeholder functions for other placement modes
        async function showFacePicker() {
            try {
                const response = await fetch('/api/faces');
                const data = await response.json();
                
                // Create a better face picker dialog
                let faceList = 'Select a face:\\n\\n';
                data.faces.forEach((face, idx) => {
                    faceList += `${idx}: ${face.face_type || 'Face ' + idx} (area: ${face.area.toFixed(4)})\\n`;
                });
                
                const faceIndex = prompt(faceList + '\\nEnter face number:');
                if (faceIndex !== null) {
                    const idx = parseInt(faceIndex);
                    if (idx >= 0 && idx < data.faces.length) {
                        const face = data.faces[idx];
                        await placeMarkerAtPosition(face.center, face.normal);
                    }
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function placeSmartMarker() {
            try {
                const config = {
                    dictionary: document.getElementById('dictSelect').value,
                    aruco_id: parseInt(document.getElementById('markerId').value),
                    size: parseFloat(document.getElementById('markerSize').value),
                    border_width: parseFloat(document.getElementById('borderWidth').value)
                };
                const response = await fetch('/api/place-marker/smart', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                if (response.ok) {
                    await addMarkerToScene(data);
                    await refreshMarkers();
                    const markerIdInput = document.getElementById('markerId');
                    const maxId = getMaxIdForDict(config.dictionary);
                    if (parseInt(markerIdInput.value) < maxId) {
                        markerIdInput.value = parseInt(markerIdInput.value) + 1;
                    }
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function placeAll6Faces() {
            try {
                const config = {
                    dictionary: document.getElementById('dictSelect').value,
                    aruco_id: parseInt(document.getElementById('markerId').value),
                    size: parseFloat(document.getElementById('markerSize').value),
                    border_width: parseFloat(document.getElementById('borderWidth').value)
                };
                const response = await fetch('/api/place-marker/all-6', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                if (response.ok) {
                    for (const marker of data.markers) {
                        await addMarkerToScene(marker);
                    }
                    await refreshMarkers();
                    const markerIdInput = document.getElementById('markerId');
                    markerIdInput.value = parseInt(markerIdInput.value) + data.markers.length;
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function placeCornerMarkers() {
            try {
                // Get the 6 primary faces
                const response = await fetch('/api/faces/primary');
                const data = await response.json();
                
                if (!response.ok) {
                    alert('Error: ' + data.detail);
                    return;
                }
                
                // Show face selection dialog with only 6 primary faces
                let faceList = 'Select a face for corner markers (4 markers will be placed on corners):\\n\\n';
                data.faces.forEach((face, idx) => {
                    faceList += `${idx}: ${face.name} (normal: [${face.normal[0].toFixed(2)}, ${face.normal[1].toFixed(2)}, ${face.normal[2].toFixed(2)}])\\n`;
                });
                
                const faceIndex = prompt(faceList + '\\nEnter face number (0-5):');
                if (faceIndex === null) {
                    return; // User cancelled
                }
                
                const idx = parseInt(faceIndex);
                if (idx < 0 || idx >= data.faces.length) {
                    alert('Invalid face number. Please select 0-5.');
                    return;
                }
                
                // Place corner markers on selected face
                const config = {
                    dictionary: document.getElementById('dictSelect').value,
                    aruco_id: parseInt(document.getElementById('markerId').value),
                    size: parseFloat(document.getElementById('markerSize').value),
                    border_width: parseFloat(document.getElementById('borderWidth').value),
                    face_index: idx
                };
                
                const cornerResponse = await fetch('/api/place-marker/corner', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                
                const cornerData = await cornerResponse.json();
                if (cornerResponse.ok) {
                    console.log(`Corner placement: ${cornerData.markers.length} markers returned from backend`);
                    if (cornerData.warnings && cornerData.warnings.length > 0) {
                        console.warn('Corner placement warnings:', cornerData.warnings);
                    }
                    for (const marker of cornerData.markers) {
                        if (!marker.internal_id && marker.internal_id !== 0) {
                            console.error('Marker missing internal_id:', marker);
                        }
                        await addMarkerToScene(marker);
                    }
                    await refreshMarkers();
                    const markerIdInput = document.getElementById('markerId');
                    markerIdInput.value = parseInt(markerIdInput.value) + cornerData.markers.length;
                } else {
                    alert('Error: ' + cornerData.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function placeSingleMarkerOnFace() {
            try {
                // Get the 6 primary faces
                const response = await fetch('/api/faces/primary');
                const data = await response.json();
                
                if (!response.ok) {
                    alert('Error: ' + data.detail);
                    return;
                }
                
                // Show face selection dialog with only 6 primary faces
                let faceList = 'Select a face for marker placement (1 marker will be placed at center):\\n\\n';
                data.faces.forEach((face, idx) => {
                    faceList += `${idx}: ${face.name} (normal: [${face.normal[0].toFixed(2)}, ${face.normal[1].toFixed(2)}, ${face.normal[2].toFixed(2)}])\\n`;
                });
                
                const faceIndex = prompt(faceList + '\\nEnter face number (0-5):');
                if (faceIndex === null) {
                    return; // User cancelled
                }
                
                const idx = parseInt(faceIndex);
                if (idx < 0 || idx >= data.faces.length) {
                    alert('Invalid face number. Please select 0-5.');
                    return;
                }
                
                const selectedFace = data.faces[idx];
                
                // Place single marker at center of selected face
                await placeMarkerAtPosition(selectedFace.center, selectedFace.normal);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function showManualPlacement() {
            const x = parseFloat(prompt('X position:') || '0');
            const y = parseFloat(prompt('Y position:') || '0');
            const z = parseFloat(prompt('Z position:') || '0');
            await placeMarkerAtPosition([x, y, z], [0, 0, 1]);
        }
        
        // Store session state
        let session_state = {
            current_file: null
        };
        
        // Initialize on load
        window.addEventListener('resize', () => {
            const container = document.getElementById('viewer');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
        
        // Update dictionary max ID when changed
        document.getElementById('dictSelect').addEventListener('change', function() {
            const dictName = this.value;
            const parts = dictName.split('_');
            const maxId = parts.length >= 3 ? parseInt(parts[parts.length - 1]) - 1 : 49;
            const markerIdInput = document.getElementById('markerId');
            markerIdInput.max = maxId;
            if (parseInt(markerIdInput.value) > maxId) {
                markerIdInput.value = 0;
            }
        });
        
        init();
    </script>
</body>
</html>"""

# API Routes

@app.post("/api/load-model")
async def load_model(file: UploadFile = File(...)):
    """Load a CAD model file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Load mesh
        mesh = cad_loader.load_file(tmp_path, input_units="auto")
        mesh_info = cad_loader.get_mesh_info(mesh)
        
        # Calculate CAD object info
        bbox_min = np.array(mesh_info['bbox_min'])
        bbox_max = np.array(mesh_info['bbox_max'])
        center = (bbox_min + bbox_max) / 2.0
        
        session_state["mesh"] = mesh
        session_state["mesh_info"] = mesh_info
        session_state["cad_object_info"] = {
            "center": center.tolist(),
            "dimensions": mesh_info['dimensions'],
            "position": [0.0, 0.0, 0.0],  # Initial position at origin
            "rotation": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            }
        }
        session_state["current_file"] = file.filename
        session_state["markers"] = {}
        session_state["next_marker_id"] = 0
        
        # Convert mesh to JSON format for Three.js
        vertices = np.asarray(mesh.vertices).flatten().tolist()
        normals = np.asarray(mesh.vertex_normals).flatten().tolist()
        faces = np.asarray(mesh.triangles).flatten().tolist()
        
        # Clean up temp file
        tmp_path.unlink()
        
        return JSONResponse({
            "success": True,
            "vertices": vertices,
            "normals": normals,
            "faces": faces,
            "mesh_info": mesh_info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-info")
async def get_model_info():
    """Get loaded model information."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return JSONResponse({
        "mesh_info": session_state["mesh_info"],
        "cad_object_info": session_state["cad_object_info"]
    })

@app.get("/api/config")
async def get_config():
    """Get current ArUco configuration."""
    return JSONResponse({
        "dictionaries": ArUcoGenerator.get_available_dictionaries(),
        "default_dictionary": "DICT_4X4_50",
        "default_size": 0.021,
        "default_border_width": 0.05
    })

@app.get("/api/cad-pose")
async def get_cad_pose():
    """Get the current CAD object pose (position and rotation)."""
    if session_state.get("cad_object_info") is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded")
    
    cad_info = session_state["cad_object_info"]
    rotation = cad_info.get("rotation", {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    })
    
    # Get position (default to center if not set)
    position = cad_info.get("position", cad_info.get("center", [0.0, 0.0, 0.0]))
    
    return JSONResponse({
        "position": {
            "x": float(position[0]) if isinstance(position, (list, tuple)) else float(position.get("x", 0.0)),
            "y": float(position[1]) if isinstance(position, (list, tuple)) else float(position.get("y", 0.0)),
            "z": float(position[2]) if isinstance(position, (list, tuple)) else float(position.get("z", 0.0))
        },
        "rotation": {
            "roll": float(rotation.get("roll", 0.0)),
            "pitch": float(rotation.get("pitch", 0.0)),
            "yaw": float(rotation.get("yaw", 0.0)),
            "quaternion": rotation.get("quaternion", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        }
    })

@app.post("/api/cad-pose")
async def update_cad_pose(pose_data: dict):
    """Update the CAD object pose (position and rotation)."""
    if session_state.get("cad_object_info") is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded")
    
    # Get position and rotation from request
    position_data = pose_data.get("position", {})
    rotation_data = pose_data.get("rotation", {})
    
    # Update position
    position = [
        float(position_data.get("x", 0.0)),
        float(position_data.get("y", 0.0)),
        float(position_data.get("z", 0.0))
    ]
    
    # Update rotation
    roll = float(rotation_data.get("roll", 0.0))
    pitch = float(rotation_data.get("pitch", 0.0))
    yaw = float(rotation_data.get("yaw", 0.0))
    
    # Convert to quaternion
    from scipy.spatial.transform import Rotation
    rot_scipy = Rotation.from_euler('xyz', [np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)])
    quat = rot_scipy.as_quat()  # [x, y, z, w]
    
    # Update session state
    session_state["cad_object_info"]["position"] = position
    session_state["cad_object_info"]["rotation"] = {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "quaternion": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3])
        }
    }
    
    return JSONResponse({"success": True})

@app.post("/api/add-marker")
async def add_marker(config: Dict[str, Any]):
    """Add an ArUco marker at specified position."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        dictionary = config.get("dictionary", "DICT_4X4_50")
        aruco_marker_id = config.get("aruco_id", session_state["next_marker_id"])
        size = config.get("size", 0.021)
        border_width = config.get("border_width", 0.05)
        
        pos_data = config.get("position", {})
        normal_data = config.get("normal", {"x": 0, "y": 0, "z": 1})
        
        # Position and normal come in as world space, need to convert to object local space
        position_world = np.array([pos_data.get("x", 0), pos_data.get("y", 0), pos_data.get("z", 0)])
        normal_world = np.array([normal_data.get("x", 0), normal_data.get("y", 0), normal_data.get("z", 1)])
        
        # Get CAD object pose to transform to object local space
        cad_info = session_state["cad_object_info"]
        cad_position = np.array(cad_info.get("position", [0.0, 0.0, 0.0]))
        cad_rotation = cad_info.get("rotation", {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        })
        
        # Create rotation matrix from CAD object's rotation
        if "quaternion" in cad_rotation:
            quat = cad_rotation["quaternion"]
            R_cad = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
        else:
            R_cad = Rotation.from_euler('xyz', [
                np.deg2rad(cad_rotation.get("roll", 0.0)),
                np.deg2rad(cad_rotation.get("pitch", 0.0)),
                np.deg2rad(cad_rotation.get("yaw", 0.0))
            ]).as_matrix()
        
        # Transform position and normal to object local space
        # A point p_local in object local space transforms to world as: p_world = cad_position + R_cad @ p_local
        # So to go from world to local: p_local = R_cad.T @ (p_world - cad_position)
        cad_center_local = np.array(cad_info["center"])
        position_local = R_cad.T @ (position_world - cad_position)
        normal_local = R_cad.T @ normal_world
        
        # Check for duplicate ArUco ID
        for existing_id, existing_marker in session_state["markers"].items():
            if existing_marker.dictionary == dictionary and existing_marker.aruco_id == aruco_marker_id:
                raise HTTPException(status_code=400, 
                    detail=f"ArUco marker {dictionary} ID:{aruco_marker_id} already exists")
        
        face_type = determine_face_type(tuple(normal_local))
        
        # Create MarkerData with object-relative position and normal
        marker = MarkerData(
            aruco_id=aruco_marker_id,
            dictionary=dictionary,
            size=size,
            border_width=border_width,
            position=tuple(position_local),
            face_normal=tuple(normal_local),
            face_type=face_type
        )
        
        # Store with internal ID
        internal_id = session_state["next_marker_id"]
        session_state["markers"][internal_id] = marker
        session_state["next_marker_id"] += 1
        
        # Return marker data for frontend
        return JSONResponse(_marker_to_json(internal_id, marker))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/markers")
async def get_markers():
    """Get all markers."""
    # Check if CAD model is loaded
    if session_state.get("cad_object_info") is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded. Please load a CAD model first.")
    
    markers_list = []
    for internal_id, marker in session_state["markers"].items():
        markers_list.append(_marker_to_json(internal_id, marker))
    
    return JSONResponse({"markers": markers_list})

@app.delete("/api/markers/{marker_id}")
async def delete_marker(marker_id: int):
    """Delete a marker."""
    if marker_id not in session_state["markers"]:
        raise HTTPException(status_code=404, detail="Marker not found")
    
    del session_state["markers"][marker_id]
    return JSONResponse({"success": True})

@app.delete("/api/markers")
async def clear_all_markers():
    """Clear all markers."""
    session_state["markers"] = {}
    session_state["next_marker_id"] = 0
    return JSONResponse({"success": True})

@app.patch("/api/markers/{marker_id}/rotation")
async def update_marker_rotation(marker_id: int, rotation: Dict[str, Any] = Body(...)):
    """
    Update marker IN-PLANE rotation only.
    This rotates the marker around its face normal, not the geometric transform.
    """
    # Ensure marker_id is an integer (FastAPI should handle this, but be explicit)
    marker_id = int(marker_id)
    
    if marker_id not in session_state["markers"]:
        available_ids = list(session_state["markers"].keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Marker not found. Requested ID: {marker_id} (type: {type(marker_id).__name__}), Available IDs: {available_ids}"
        )
    
    marker = session_state["markers"][marker_id]
    
    mode = rotation.get("mode", "relative")  # "relative" or "absolute"
    
    if mode == "absolute":
        # Set absolute in-plane rotation (in degrees)
        # Frontend sends degrees, so use directly
        yaw_deg = float(rotation.get("yaw", 0))
        marker.set_in_plane_rotation(yaw_deg)
    else:
        # Add rotation delta (in degrees)
        # Frontend sends degrees, so use directly
        yaw_delta_deg = float(rotation.get("yaw", 0))
        new_rotation_deg = marker.in_plane_rotation_deg + yaw_delta_deg
        marker.set_in_plane_rotation(new_rotation_deg)
    
    # Get updated rotation for response
    # Use axis-angle representation to avoid gimbal lock
    R_current = marker.get_current_rotation_matrix()
    rot_scipy = Rotation.from_matrix(R_current)
    quat = rot_scipy.as_quat()
    
    # Convert to axis-angle representation (avoids gimbal lock)
    axis_angle = rot_scipy.as_rotvec()  # Returns [x, y, z] where magnitude is angle in radians
    angle_rad = np.linalg.norm(axis_angle)
    if angle_rad > 1e-6:
        axis = axis_angle / angle_rad
    else:
        axis = np.array([0, 0, 1])  # Default axis
    
    # Also get Euler for backward compatibility (suppress warnings)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = rot_scipy.as_euler('xyz')
    
    return JSONResponse({
        "success": True,
        "in_plane_rotation_deg": float(marker.in_plane_rotation_deg),
        "rotation": [float(e) for e in euler],  # For display, but use quaternion/axis-angle for actual rotation
        "quaternion": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3])
        },
        "axis_angle": {
            "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            "angle_rad": float(angle_rad),
            "angle_deg": float(np.degrees(angle_rad))
        }
    })

@app.patch("/api/markers/{marker_id}/translation")
async def update_marker_translation(marker_id: int, translation: Dict[str, Any] = Body(...)):
    """
    Update marker position by translating in-plane (along marker's X and Y axes).
    This moves the marker within its plane, not perpendicular to it.
    """
    if marker_id not in session_state["markers"]:
        raise HTTPException(status_code=404, detail="Marker not found")
    
    marker = session_state["markers"][marker_id]
    
    mode = translation.get("mode", "relative")  # "relative" or "absolute"
    x_delta = float(translation.get("x", 0.0))  # Translation along marker's X-axis (in-plane)
    y_delta = float(translation.get("y", 0.0))  # Translation along marker's Y-axis (in-plane)
    
    # Get the marker's base rotation matrix (without in-plane rotation) to determine X and Y axes
    # We use base rotation so that translation axes don't change with in-plane rotation
    R_base = marker.base_rotation_matrix
    
    # Extract X and Y axes from base rotation (first two columns)
    marker_x_axis = R_base[:, 0]  # Marker's X-axis in object local space
    marker_y_axis = R_base[:, 1]  # Marker's Y-axis in object local space
    
    if mode == "absolute":
        # Set absolute translation offset
        marker.in_plane_translation = np.array([x_delta, y_delta])
    else:
        # Relative mode: add translation delta to cumulative offset
        marker.in_plane_translation = marker.in_plane_translation + np.array([x_delta, y_delta])
    
    # Apply the cumulative translation offset to the base position
    # Calculate translation vector in object local space
    translation_vector = marker.in_plane_translation[0] * marker_x_axis + marker.in_plane_translation[1] * marker_y_axis
    
    # Ensure initial_position is set (for backward compatibility with old markers)
    if not hasattr(marker, 'initial_position') or marker.initial_position is None:
        # For existing markers without initial_position, use current position as base
        # and subtract current translation to get initial position
        if np.linalg.norm(marker.in_plane_translation) > 1e-6:
            # Has translation, need to reverse it to get initial position
            prev_translation = marker.in_plane_translation[0] * marker_x_axis + marker.in_plane_translation[1] * marker_y_axis
            marker.initial_position = marker.position - prev_translation
        else:
            # No translation yet, current position is initial
            marker.initial_position = marker.position.copy()
    
    # Update position: initial_position + current_translation
    marker.position = marker.initial_position + translation_vector
    
    return JSONResponse({
        "success": True,
        "position": {
            "x": float(marker.position[0]),
            "y": float(marker.position[1]),
            "z": float(marker.position[2])
        },
        "translation_offset": {
            "x": float(marker.in_plane_translation[0]),
            "y": float(marker.in_plane_translation[1])
        }
    })

def _find_closest_surface_point_and_normal(target_point, mesh, triangles, vertices):
    """Find the closest point on mesh surface and its normal."""
    closest_distance = float('inf')
    closest_point = None
    closest_normal = None
    
    # Check all triangles for the closest surface point
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        
        # Calculate triangle normal
        tri_normal = np.cross(v1 - v0, v2 - v0)
        tri_length = np.linalg.norm(tri_normal)
        if tri_length < 1e-8:
            continue
        tri_normal = tri_normal / tri_length
        
        # Find closest point on triangle to target point
        # Project target point onto triangle plane
        edge1 = v1 - v0
        edge2 = v2 - v0
        v0_to_p = target_point - v0
        
        # Barycentric coordinates
        dot00 = np.dot(edge1, edge1)
        dot01 = np.dot(edge1, edge2)
        dot02 = np.dot(edge1, v0_to_p)
        dot11 = np.dot(edge2, edge2)
        dot12 = np.dot(edge2, v0_to_p)
        
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Clamp to triangle
        u = max(0, min(1, u))
        v = max(0, min(1, v))
        if u + v > 1:
            u = 1 - v
            v = 1 - u
        
        triangle_point = v0 + u * edge1 + v * edge2
        distance = np.linalg.norm(triangle_point - target_point)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_point = triangle_point
            closest_normal = tri_normal
    
    if closest_point is None:
        # Fallback: use closest vertex
        vertex_distances = np.linalg.norm(vertices - target_point, axis=1)
        closest_idx = np.argmin(vertex_distances)
        closest_point = vertices[closest_idx]
        # Use vertex normal if available, otherwise use default
        if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > closest_idx:
            closest_normal = np.array(mesh.vertex_normals[closest_idx])
        else:
            closest_normal = np.array([0.0, 0.0, 1.0])
    
    return closest_point, closest_normal

def _project_to_surface_with_raycast(start_point, direction, triangles, vertices, max_distance=1.0):
    """Project a point onto the mesh surface along a direction vector (like placing a marker on a face)."""
    # Normalize direction
    direction = np.array(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Cast ray from start_point along direction
    # Find intersection with mesh triangles
    closest_intersection = None
    closest_distance = float('inf')
    closest_normal = None
    
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        
        # Calculate triangle normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        tri_normal = np.cross(edge1, edge2)
        tri_length = np.linalg.norm(tri_normal)
        if tri_length < 1e-8:
            continue
        tri_normal = tri_normal / tri_length
        
        # Ray-plane intersection
        # Ray: start_point + t * direction
        # Plane: (point - v0) . tri_normal = 0
        denom = np.dot(direction, tri_normal)
        if abs(denom) < 1e-8:  # Ray parallel to plane
            continue
        
        t = np.dot(v0 - start_point, tri_normal) / denom
        if t < 0 or t > max_distance:  # Intersection behind start or too far
            continue
        
        intersection = start_point + t * direction
        
        # Check if intersection is inside triangle using barycentric coordinates
        v0_to_p = intersection - v0
        dot00 = np.dot(edge1, edge1)
        dot01 = np.dot(edge1, edge2)
        dot02 = np.dot(edge1, v0_to_p)
        dot11 = np.dot(edge2, edge2)
        dot12 = np.dot(edge2, v0_to_p)
        
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is inside triangle
        if u >= 0 and v >= 0 and u + v <= 1:
            if t < closest_distance:
                closest_distance = t
                closest_intersection = intersection
                closest_normal = tri_normal
    
    if closest_intersection is not None:
        return closest_intersection, closest_normal
    
    # Fallback: use closest surface point
    # Create a dummy mesh object for the function call
    class DummyMesh:
        def __init__(self, triangles, vertices):
            self.triangles = triangles
            self.vertices = vertices
    
    dummy_mesh = DummyMesh(triangles, vertices)
    return _find_closest_surface_point_and_normal(start_point, dummy_mesh, triangles, vertices)

@app.post("/api/markers/swap")
async def swap_marker_positions(swap_data: Dict[str, Any]):
    """Swap ArUco IDs between two markers.
    
    This swaps which ArUco ID is on which face. The faces remain the same - we just
    reassign which ID goes to which face. Each marker keeps its position, rotation, and
    face information, but gets the other marker's ArUco ID.
    """
    marker1_id = swap_data.get("marker1_id")
    marker2_id = swap_data.get("marker2_id")
    
    if marker1_id is None:
        raise HTTPException(status_code=400, detail="marker1_id is missing")
    if marker2_id is None:
        raise HTTPException(status_code=400, detail="marker2_id is missing")
    
    # Convert to int if needed
    try:
        marker1_id = int(marker1_id)
        marker2_id = int(marker2_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Marker IDs must be integers")
    
    if marker1_id not in session_state["markers"]:
        raise HTTPException(status_code=404, detail=f"Marker {marker1_id} not found")
    if marker2_id not in session_state["markers"]:
        raise HTTPException(status_code=404, detail=f"Marker {marker2_id} not found")
    
    if marker1_id == marker2_id:
        raise HTTPException(status_code=400, detail="Cannot swap marker with itself")
    
    marker1 = session_state["markers"][marker1_id]
    marker2 = session_state["markers"][marker2_id]
    
    # Store original ArUco IDs
    aruco_id1 = marker1.aruco_id
    aruco_id2 = marker2.aruco_id
    
    # Check if swapping would create duplicate IDs
    if aruco_id1 == aruco_id2:
        raise HTTPException(status_code=400, detail="Markers already have the same ArUco ID")
    
    # Check for conflicts with other markers
    for existing_id, existing_info in session_state["markers"].items():
        if existing_id != marker1_id and existing_id != marker2_id:
            if existing_info.dictionary == marker1.dictionary and existing_info.aruco_id == aruco_id2:
                raise HTTPException(status_code=400, detail=f"ArUco ID {aruco_id2} already exists in another marker")
            if existing_info.dictionary == marker2.dictionary and existing_info.aruco_id == aruco_id1:
                raise HTTPException(status_code=400, detail=f"ArUco ID {aruco_id1} already exists in another marker")
    
    # Simply swap the ArUco IDs
    # Everything else (position, rotation, face info) stays the same
    marker1.aruco_id = aruco_id2
    marker2.aruco_id = aruco_id1
    
    return JSONResponse({"success": True, "swapped": [marker1_id, marker2_id]})

@app.get("/api/faces")
async def get_faces():
    """Get detected faces from the mesh."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    triangles = np.asarray(session_state["mesh"].triangles)
    vertices = np.asarray(session_state["mesh"].vertices)
    
    face_groups = group_triangles_by_face(triangles, vertices)
    
    faces = []
    for face_center, face_normal, face_area, triangle_indices in face_groups:
        face_type = determine_face_type(tuple(face_normal))
        faces.append({
            "center": face_center.tolist(),
            "normal": face_normal.tolist(),
            "area": float(face_area),
            "triangle_count": len(triangle_indices),
            "face_type": face_type
        })
    
    return JSONResponse({"faces": faces})

@app.post("/api/place-marker/random")
async def place_random_marker(config: Dict[str, Any] = None):
    """Place marker at random face."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    triangles = np.asarray(session_state["mesh"].triangles)
    vertices = np.asarray(session_state["mesh"].vertices)
    
    face_groups = group_triangles_by_face(triangles, vertices)
    if len(face_groups) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")
    
    import random
    face_center, face_normal, face_area, triangle_indices = face_groups[random.randint(0, len(face_groups) - 1)]
    
    if config is None:
        config = {}
    
    config.update({
        "position": {"x": float(face_center[0]), "y": float(face_center[1]), "z": float(face_center[2])},
        "normal": {"x": float(face_normal[0]), "y": float(face_normal[1]), "z": float(face_normal[2])}
    })
    
    # Reuse add_marker logic
    return await add_marker(config)

@app.post("/api/place-marker/smart")
async def place_smart_marker(config: Dict[str, Any] = None):
    """Place marker using smart auto placement."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    triangles = np.asarray(session_state["mesh"].triangles)
    vertices = np.asarray(session_state["mesh"].vertices)
    
    face_groups = group_triangles_by_face(triangles, vertices)
    if len(face_groups) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")
    
    # Find best face (largest area, prefer upward-facing)
    best_score = -1
    best_face = None
    
    for face_center, face_normal, face_area, triangle_indices in face_groups:
        score = 0
        if face_area > 0:
            score += min(50, face_area * 10000)
        up_alignment = abs(face_normal[2])
        score += up_alignment * 30
        score += min(20, len(triangle_indices) / 5)
        
        if score > best_score:
            best_score = score
            best_face = (face_center, face_normal, face_area, triangle_indices)
    
    if best_face is None:
        raise HTTPException(status_code=400, detail="Could not find suitable face")
    
    face_center, face_normal, face_area, triangle_indices = best_face
    
    if config is None:
        config = {}
    
    config.update({
        "position": {"x": float(face_center[0]), "y": float(face_center[1]), "z": float(face_center[2])},
        "normal": {"x": float(face_normal[0]), "y": float(face_normal[1]), "z": float(face_normal[2])}
    })
    
    return await add_marker(config)

@app.post("/api/place-marker/all-6")
async def place_all_6_faces(config: Dict[str, Any] = None):
    """Place markers on all 6 primary faces."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if config is None:
        config = {}
    
    cad_info = session_state["cad_object_info"]
    bbox_min = np.array(session_state["mesh_info"]["bbox_min"])
    bbox_max = np.array(session_state["mesh_info"]["bbox_max"])
    center = np.array(cad_info["center"])
    
    faces = [
        {"center": [bbox_min[0], center[1], center[2]], "normal": [-1.0, 0.0, 0.0], "name": "Left"},
        {"center": [bbox_max[0], center[1], center[2]], "normal": [1.0, 0.0, 0.0], "name": "Right"},
        {"center": [center[0], bbox_min[1], center[2]], "normal": [0.0, -1.0, 0.0], "name": "Front"},
        {"center": [center[0], bbox_max[1], center[2]], "normal": [0.0, 1.0, 0.0], "name": "Back"},
        {"center": [center[0], center[1], bbox_min[2]], "normal": [0.0, 0.0, -1.0], "name": "Bottom"},
        {"center": [center[0], center[1], bbox_max[2]], "normal": [0.0, 0.0, 1.0], "name": "Top"}
    ]
    
    markers = []
    base_aruco_id = config.get("aruco_id", session_state["next_marker_id"])
    
    for i, face in enumerate(faces):
        face_config = config.copy()
        face_config.update({
            "aruco_id": base_aruco_id + i,
            "position": {"x": face["center"][0], "y": face["center"][1], "z": face["center"][2]},
            "normal": {"x": face["normal"][0], "y": face["normal"][1], "z": face["normal"][2]}
        })
        result = await add_marker(face_config)
        markers.append(json.loads(result.body))
    
    return JSONResponse({"markers": markers})

@app.get("/api/faces/primary")
async def get_primary_faces():
    """Get the 6 primary faces (top, bottom, front, back, left, right)."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    cad_info = session_state["cad_object_info"]
    bbox_min = np.array(session_state["mesh_info"]["bbox_min"])
    bbox_max = np.array(session_state["mesh_info"]["bbox_max"])
    center = np.array(cad_info["center"])
    
    faces = [
        {"name": "Left", "center": [bbox_min[0], center[1], center[2]], "normal": [-1.0, 0.0, 0.0]},
        {"name": "Right", "center": [bbox_max[0], center[1], center[2]], "normal": [1.0, 0.0, 0.0]},
        {"name": "Front", "center": [center[0], bbox_min[1], center[2]], "normal": [0.0, -1.0, 0.0]},
        {"name": "Back", "center": [center[0], bbox_max[1], center[2]], "normal": [0.0, 1.0, 0.0]},
        {"name": "Bottom", "center": [center[0], center[1], bbox_min[2]], "normal": [0.0, 0.0, -1.0]},
        {"name": "Top", "center": [center[0], center[1], bbox_max[2]], "normal": [0.0, 0.0, 1.0]}
    ]
    
    return JSONResponse({"faces": faces})

@app.post("/api/place-marker/corner")
async def place_corner_markers(config: Dict[str, Any] = None):
    """Place 4 markers on the corners of a selected primary face."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if config is None:
        config = {}
    
    cad_info = session_state["cad_object_info"]
    bbox_min = np.array(session_state["mesh_info"]["bbox_min"])
    bbox_max = np.array(session_state["mesh_info"]["bbox_max"])
    center = np.array(cad_info["center"])
    
    # Get face index (0-5 for the 6 primary faces)
    face_index = config.get("face_index")
    if face_index is None:
        raise HTTPException(status_code=400, detail="face_index is required")
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="face_index must be an integer")
    if face_index < 0 or face_index > 5:
        raise HTTPException(status_code=400, detail="Face index must be 0-5")
    
    # Define the 6 primary faces and their 4 corners
    face_definitions = [
        # Left face (X = bbox_min[0])
        {
            "name": "Left",
            "normal": [-1.0, 0.0, 0.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_min[2]],  # Bottom-Front
                [bbox_min[0], bbox_max[1], bbox_min[2]],  # Bottom-Back
                [bbox_min[0], bbox_max[1], bbox_max[2]],  # Top-Back
                [bbox_min[0], bbox_min[1], bbox_max[2]]   # Top-Front
            ]
        },
        # Right face (X = bbox_max[0])
        {
            "name": "Right",
            "normal": [1.0, 0.0, 0.0],
            "corners": [
                [bbox_max[0], bbox_min[1], bbox_min[2]],  # Bottom-Front
                [bbox_max[0], bbox_min[1], bbox_max[2]],  # Top-Front
                [bbox_max[0], bbox_max[1], bbox_max[2]],  # Top-Back
                [bbox_max[0], bbox_max[1], bbox_min[2]]   # Bottom-Back
            ]
        },
        # Front face (Y = bbox_min[1])
        {
            "name": "Front",
            "normal": [0.0, -1.0, 0.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_min[2]],  # Bottom-Left
                [bbox_max[0], bbox_min[1], bbox_min[2]],  # Bottom-Right
                [bbox_max[0], bbox_min[1], bbox_max[2]],  # Top-Right
                [bbox_min[0], bbox_min[1], bbox_max[2]]   # Top-Left
            ]
        },
        # Back face (Y = bbox_max[1])
        {
            "name": "Back",
            "normal": [0.0, 1.0, 0.0],
            "corners": [
                [bbox_min[0], bbox_max[1], bbox_min[2]],  # Bottom-Left
                [bbox_min[0], bbox_max[1], bbox_max[2]],  # Top-Left
                [bbox_max[0], bbox_max[1], bbox_max[2]],  # Top-Right
                [bbox_max[0], bbox_max[1], bbox_min[2]]   # Bottom-Right
            ]
        },
        # Bottom face (Z = bbox_min[2])
        {
            "name": "Bottom",
            "normal": [0.0, 0.0, -1.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_min[2]],  # Front-Left
                [bbox_max[0], bbox_min[1], bbox_min[2]],  # Front-Right
                [bbox_max[0], bbox_max[1], bbox_min[2]],  # Back-Right
                [bbox_min[0], bbox_max[1], bbox_min[2]]   # Back-Left
            ]
        },
        # Top face (Z = bbox_max[2])
        {
            "name": "Top",
            "normal": [0.0, 0.0, 1.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_max[2]],  # Front-Left
                [bbox_min[0], bbox_max[1], bbox_max[2]],  # Back-Left
                [bbox_max[0], bbox_max[1], bbox_max[2]],  # Back-Right
                [bbox_max[0], bbox_min[1], bbox_max[2]]   # Front-Right
            ]
        }
    ]
    
    selected_face = face_definitions[face_index]
    
    # Get marker size from UI config
    marker_size = config.get("size", 0.021)  # Total marker size (including border)
    
    # marker_size is the total size including border
    # To ensure the entire marker (including border) stays within the bounding box,
    # we need to inset by exactly half the total marker size
    # This ensures the marker center is far enough from the edge that the full marker fits
    inset_distance = marker_size / 2.0
    
    # Calculate inset corners - move corners inward along the face edges
    # For each corner, we need to move it inward along BOTH edges that meet at that corner
    # by at least half the marker size, so the entire marker stays within bounds
    inset_corners = []
    normal = np.array(selected_face["normal"])
    
    # Find the two axes that define the face plane (where normal is 0)
    plane_axes = []
    for axis_idx in range(3):
        if abs(normal[axis_idx]) < 0.5:  # This axis is in the plane
            plane_axes.append(axis_idx)
    
    if len(plane_axes) != 2:
        raise HTTPException(status_code=400, detail="Invalid face normal - must have exactly 2 plane axes")
    
    for corner_pos in selected_face["corners"]:
        corner = np.array(corner_pos)
        inset_corner = corner.copy()
        
        # For each plane axis, move the corner inward by at least half marker size
        # This ensures the marker (which extends marker_size/2 from its center) stays within bounds
        for axis_idx in plane_axes:
            if corner[axis_idx] == bbox_min[axis_idx]:
                # Corner is at min edge, move inward (toward max)
                inset_corner[axis_idx] = bbox_min[axis_idx] + inset_distance
            elif corner[axis_idx] == bbox_max[axis_idx]:
                # Corner is at max edge, move inward (toward min)
                inset_corner[axis_idx] = bbox_max[axis_idx] - inset_distance
            else:
                # Corner is somewhere in between (shouldn't happen for bounding box corners)
                # Move toward center
                if corner[axis_idx] < center[axis_idx]:
                    inset_corner[axis_idx] = corner[axis_idx] + inset_distance
                else:
                    inset_corner[axis_idx] = corner[axis_idx] - inset_distance
        
        # For the face normal axis, keep it at the face position
        for axis_idx in range(3):
            if abs(normal[axis_idx]) > 0.5:  # This is the face normal axis
                # Keep at face position (bbox_min or bbox_max)
                if normal[axis_idx] < 0:
                    inset_corner[axis_idx] = bbox_min[axis_idx]
                else:
                    inset_corner[axis_idx] = bbox_max[axis_idx]
        
        # Final clamp to ensure we're within bounding box
        for axis_idx in range(3):
            inset_corner[axis_idx] = np.clip(inset_corner[axis_idx], bbox_min[axis_idx], bbox_max[axis_idx])
        
        inset_corners.append(inset_corner.tolist())
    
    markers = []
    base_aruco_id = config.get("aruco_id", session_state["next_marker_id"])
    errors = []
    
    # Place 4 markers on the 4 inset corners of the selected face
    for i, corner_pos in enumerate(inset_corners):
        try:
            corner_config = config.copy()
            corner_config.update({
                "aruco_id": base_aruco_id + i,
                "position": {"x": corner_pos[0], "y": corner_pos[1], "z": corner_pos[2]},
                "normal": {"x": selected_face["normal"][0], "y": selected_face["normal"][1], "z": selected_face["normal"][2]}
            })
            result = await add_marker(corner_config)
            # add_marker returns JSONResponse, parse the body
            marker_data = json.loads(result.body)
            markers.append(marker_data)
        except HTTPException as e:
            # If add_marker raised HTTPException (e.g., duplicate ID), log it but continue
            error_msg = f"Marker {i} (ArUco ID {base_aruco_id + i}): {e.detail}"
            errors.append(error_msg)
            print(f"Warning: {error_msg}")
            continue
        except Exception as e:
            # Log other errors but continue with other markers
            error_msg = f"Marker {i} at corner {corner_pos}: {str(e)}"
            errors.append(error_msg)
            print(f"Error: {error_msg}")
            continue
    
    if len(markers) != len(inset_corners):
        warning_msg = f"Only {len(markers)} out of {len(inset_corners)} markers were successfully placed"
        if errors:
            warning_msg += f". Errors: {'; '.join(errors)}"
        print(f"Warning: {warning_msg}")
    
    return JSONResponse({"markers": markers, "warnings": errors if errors else None})

@app.get("/api/export")
async def export_annotations():
    """Export annotations to JSON file with proper format."""
    if len(session_state["markers"]) == 0:
        raise HTTPException(status_code=400, detail="No markers to export")
    
    # Check if CAD model is loaded to get object name
    current_file = session_state.get("current_file")
    if not current_file:
        raise HTTPException(status_code=400, detail="No CAD file name available. Please load a CAD model first.")
    
    first_marker = list(session_state["markers"].values())[0]
    cad_info = session_state["cad_object_info"]
    
    markers_list = []
    for internal_id, marker in session_state["markers"].items():
        # Markers are already stored in object local space, so T_object_to_marker is straightforward
        # Get T_object_to_marker (already in object local space)
        T_object_to_marker = marker.get_T_object_to_marker(cad_info["center"])
        
        markers_list.append({
            "aruco_id": marker.aruco_id,
            "face_type": marker.face_type,
            "surface_normal": marker.face_normal.tolist(),
            "T_object_to_marker": T_object_to_marker
        })
    
    from datetime import datetime
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "model_file": session_state["current_file"],
        "total_markers": len(markers_list),
        "aruco_dictionary": first_marker.dictionary,
        "size": first_marker.size,
        "border_width": first_marker.border_width,
        "cad_object_info": {
            "center": cad_info["center"],
            "dimensions": cad_info["dimensions"],
            "position": cad_info.get("position", [0.0, 0.0, 0.0]),
            "rotation": cad_info.get("rotation", {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            })
        },
        "markers": markers_list,
        "notes": "T_object_to_marker is the complete transform from object center to marker, including full rotation (base + in-plane)."
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    # Extract object name from filename (remove extension)
    object_name = Path(current_file).stem
    
    # Save to data/aruco directory (same location as import-auto looks for)
    aruco_dir = DATA_DIR / "aruco"
    aruco_dir.mkdir(parents=True, exist_ok=True)
    aruco_file = aruco_dir / f"{object_name}_aruco.json"
    
    try:
        with open(aruco_file, 'w') as f:
            f.write(json_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving annotations to {aruco_file}: {str(e)}")
    
    # Also return as downloadable file
    aruco_filename = f"{object_name}_aruco.json"
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={aruco_filename}"}
    )

@app.get("/api/export-wireframe")
async def export_wireframe():
    """Export wireframe data from the loaded mesh to JSON file."""
    if session_state["mesh"] is None:
        raise HTTPException(status_code=400, detail="No mesh loaded. Please load a CAD model first.")
    
    # Check if CAD model is loaded to get object name
    current_file = session_state.get("current_file")
    if not current_file:
        raise HTTPException(status_code=400, detail="No CAD file name available. Please load a CAD model first.")
    
    try:
        # Extract wireframe data from the loaded mesh
        # Note: The mesh is already in meters (converted by CADLoader)
        mesh = session_state["mesh"]
        
        # Extract vertices and edges
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Extract wireframe edges from triangles
        edges = []
        for triangle in triangles:
            for i in range(3):
                v1, v2 = triangle[i], triangle[(i + 1) % 3]
                # Ensure consistent ordering (smaller index first)
                edge = tuple(sorted([int(v1), int(v2)]))
                edges.append(edge)
        
        # Remove duplicates while preserving order
        unique_edges = []
        seen = set()
        for edge in edges:
            if edge not in seen:
                unique_edges.append(edge)
                seen.add(edge)
        
        # Compute mesh info
        mesh_info = {
            'num_vertices': len(vertices),
            'num_edges': len(unique_edges),
            'num_triangles': len(triangles),
            'bounding_box': {
                'min': vertices.min(axis=0).tolist(),
                'max': vertices.max(axis=0).tolist(),
                'center': vertices.mean(axis=0).tolist(),
                'size': (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
            },
            'has_normals': mesh.has_vertex_normals(),
            'has_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable()
        }
        
        # Create wireframe data structure
        wireframe_data = {
            'mesh_info': mesh_info,
            'vertices': vertices.tolist(),
            'edges': [[int(edge[0]), int(edge[1])] for edge in unique_edges],
            'format': 'vector_relation',
            'description': 'Wireframe data with vertices and edge connections'
        }
        
        json_str = json.dumps(wireframe_data, indent=2)
        
        # Extract object name from filename (remove extension)
        object_name = Path(current_file).stem
        
        # Save to data/wireframe directory
        wireframe_dir = DATA_DIR / "wireframe"
        wireframe_dir.mkdir(parents=True, exist_ok=True)
        wireframe_file = wireframe_dir / f"{object_name}_wireframe.json"
        
        try:
            with open(wireframe_file, 'w') as f:
                f.write(json_str)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving wireframe to {wireframe_file}: {str(e)}")
        
        # Also return as downloadable file
        wireframe_filename = f"{object_name}_wireframe.json"
        return Response(
            content=json_str,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={wireframe_filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export wireframe: {str(e)}")

@app.get("/api/marker-image")
async def get_marker_image(
    dictionary: str = Query(..., description="ArUco dictionary name"),
    marker_id: int = Query(..., description="ArUco marker ID"),
    size: int = Query(512, description="Image size in pixels")
):
    """Generate ArUco marker image and return as PNG."""
    try:
        # Generate marker image
        marker_img = aruco_generator.generate_marker(dictionary, marker_id, size)
        
        # Convert to PNG bytes
        _, buffer = cv2.imencode('.png', marker_img)
        img_bytes = buffer.tobytes()
        
        # Return as PNG image
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000"}  # Cache for 1 year
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating marker image: {str(e)}")

@app.post("/api/import")
async def import_annotations(file: UploadFile = File(...)):
    """Import annotations from uploaded JSON file."""
    # Check if CAD model is loaded first
    if session_state.get("cad_object_info") is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded. Please load a CAD model first.")
    
    try:
        content = await file.read()
        data = json.loads(content)
        return await _process_imported_annotations(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing annotations: {str(e)}")


@app.post("/api/import-auto")
async def import_annotations_auto():
    """Automatically import annotations from data folder based on current CAD file."""
    # Check if CAD model is loaded first
    if session_state.get("cad_object_info") is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded. Please load a CAD model first.")
    
    current_file = session_state.get("current_file")
    if not current_file:
        raise HTTPException(status_code=400, detail="No CAD file name available. Please load a CAD model first.")
    
    # Extract object name from filename (remove extension)
    object_name = Path(current_file).stem
    
    # Try to find annotation file in data/aruco directory
    aruco_file = DATA_DIR / "aruco" / f"{object_name}_aruco.json"
    
    if not aruco_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Annotation file not found in data folder: {aruco_file}. Please upload a file or create annotations."
        )
    
    try:
        with open(aruco_file, 'r') as f:
            data = json.load(f)
        return await _process_imported_annotations(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading annotations from {aruco_file}: {str(e)}")


async def _process_imported_annotations(data: dict):
    """Process imported annotation data and update session state."""
    # Clear existing markers
    session_state["markers"] = {}
    session_state["next_marker_id"] = 0
    
    # Load markers
    cad_info = data.get("cad_object_info", {})
    cad_center = np.array(cad_info.get("center", [0, 0, 0]))
    
    # Load CAD pose from imported file if available, otherwise keep defaults
    if session_state.get("cad_object_info") is not None:
        imported_cad_info = data.get("cad_object_info", {})
        
        # Update position if present in imported file
        if "position" in imported_cad_info:
            session_state["cad_object_info"]["position"] = imported_cad_info["position"]
        
        # Update rotation if present in imported file, otherwise keep zero
        if "rotation" in imported_cad_info:
            session_state["cad_object_info"]["rotation"] = imported_cad_info["rotation"]
        else:
            session_state["cad_object_info"]["rotation"] = {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            }
    
    try:
        for marker_data in data.get("markers", []):
            aruco_id = marker_data["aruco_id"]
            
            # Get T_object_to_marker (transformation from object center to marker)
            # This is in object local space
            # Try new format first, fall back to old format for backward compatibility
            T_object_to_marker = marker_data.get("T_object_to_marker")
            if T_object_to_marker is None:
                # Backward compatibility: convert old T_marker_to_object to T_object_to_marker
                T_marker_to_object = marker_data.get("T_marker_to_object", {})
                if T_marker_to_object:
                    rel_pos = T_marker_to_object.get("position", {})
                    rel_rot = T_marker_to_object.get("rotation", {})
                    # Convert from marker frame to object frame
                    relative_pos_marker_frame = np.array([rel_pos.get("x", 0), rel_pos.get("y", 0), rel_pos.get("z", 0)])
                    R_marker_to_object_euler = (
                        rel_rot.get("roll", 0),
                        rel_rot.get("pitch", 0),
                        rel_rot.get("yaw", 0)
                    )
                    R_marker_to_object = Rotation.from_euler('xyz', R_marker_to_object_euler).as_matrix()
                    relative_pos_object_frame = R_marker_to_object @ relative_pos_marker_frame
                    position_local = tuple(cad_center - relative_pos_object_frame)
                    # Invert rotation
                    R_object_to_marker = R_marker_to_object.T
                else:
                    # No transform data, use default
                    position_local = tuple(cad_center)
                    R_object_to_marker = np.eye(3)
            else:
                # New format: T_object_to_marker
                rel_pos = T_object_to_marker.get("position", {})
                rel_rot = T_object_to_marker.get("rotation", {})
                
                # T_object_to_marker.position is in object frame (vector from object center to marker)
                # T_object_to_marker.rotation is rotation from object frame to marker frame
                relative_pos_object_frame = np.array([rel_pos.get("x", 0), rel_pos.get("y", 0), rel_pos.get("z", 0)])
                
                # Get rotation from object frame to marker frame (in object local space)
                R_object_to_marker_euler = (
                    rel_rot.get("roll", 0),
                    rel_rot.get("pitch", 0),
                    rel_rot.get("yaw", 0)
                )
                R_object_to_marker = Rotation.from_euler('xyz', R_object_to_marker_euler).as_matrix()
                
                # Calculate marker position in object local space: marker_pos = object_center + relative_pos
                position_local = tuple(cad_center + relative_pos_object_frame)
            
            # Get surface normal and face type (already in object local space from export)
            surface_normal = tuple(marker_data.get("surface_normal", [0, 0, 1]))
            face_type = marker_data.get("face_type", "unknown")
            
            # Create MarkerData with object-relative position and normal
            marker = MarkerData(
                aruco_id=aruco_id,
                dictionary=data.get("aruco_dictionary", "DICT_4X4_50"),
                size=data.get("size", 0.021),
                border_width=data.get("border_width", 0.05),
                position=position_local,
                face_normal=surface_normal,
                face_type=face_type
            )
            
            # If imported data has translation_offset, we need to adjust initial_position
            # The imported position already includes translation, so we need to extract it
            imported_translation = marker_data.get("translation_offset")
            if imported_translation is not None:
                # Position includes translation, so we need to reverse it to get initial_position
                x_offset = imported_translation.get("x", 0.0)
                y_offset = imported_translation.get("y", 0.0)
                marker.in_plane_translation = np.array([x_offset, y_offset])
                
                # Calculate initial position by reversing the translation
                R_base = marker.base_rotation_matrix
                marker_x_axis = R_base[:, 0]
                marker_y_axis = R_base[:, 1]
                translation_vector = x_offset * marker_x_axis + y_offset * marker_y_axis
                marker.initial_position = marker.position - translation_vector
            else:
                # No translation in imported data, so current position is initial
                marker.in_plane_translation = np.array([0.0, 0.0])
                marker.initial_position = marker.position.copy()
            
            # Extract in-plane rotation from T_object_to_marker if not explicitly provided
            # T_object_to_marker.rotation contains full rotation (base + in-plane)
            # Full rotation: R_object_to_marker = (R_base @ R_inplane).T = R_inplane.T @ R_base.T
            # To extract: R_inplane = (R_object_to_marker @ R_base).T
            in_plane_rotation_deg = marker_data.get("in_plane_rotation_deg")
            if in_plane_rotation_deg is None:
                # Extract from T_object_to_marker rotation
                # R_object_to_marker is the full rotation from object to marker
                R_full = R_object_to_marker
                # R_base is the base rotation from marker to object (aligns marker Z+ with face normal)
                R_base = marker.base_rotation_matrix
                # Extract in-plane rotation: R_inplane = (R_full @ R_base).T
                # This gives us R_inplane in marker frame
                R_inplane = (R_full @ R_base).T
                # Extract Z-axis rotation (in-plane rotation around marker Z+)
                # Convert to Euler and get Z component
                rot_inplane = Rotation.from_matrix(R_inplane)
                euler_inplane = rot_inplane.as_euler('xyz')
                in_plane_rotation_deg = np.rad2deg(euler_inplane[2])  # Z-axis rotation
            else:
                in_plane_rotation_deg = float(in_plane_rotation_deg)
            
            # Set in-plane rotation
            marker.set_in_plane_rotation(in_plane_rotation_deg)
            
            # Use internal marker ID for tracking
            internal_id = session_state["next_marker_id"]
            session_state["markers"][internal_id] = marker
            session_state["next_marker_id"] = max(session_state["next_marker_id"], internal_id + 1)
        
        return JSONResponse({"success": True, "imported": len(session_state["markers"])})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_available_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port starting from {start_port}")

def main():
    """Main entry point for the ArUco Grasp Annotator application."""
    # Find an available port starting from 8000
    port = find_available_port(8000)
    
    print("🚀 Starting ArUco Grasp Annotator Web App...")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print("🎯 Features:")
    print("   • Load and display CAD models (STL, OBJ, PLY)")
    print("   • Place ArUco markers with multiple placement modes")
    print("   • Interactive 3D visualization")
    print("   • Export/import annotations")
    print("🎮 Controls:")
    print("   • Mouse: Left drag to rotate, Right drag to pan, Wheel to zoom")
    print("   • Click placement mode: Click on model surface to place marker")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

