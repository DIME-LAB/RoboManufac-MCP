#!/usr/bin/env python3
"""Web application for ArUco Grasp Annotator using FastAPI + Open3D."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import open3d as o3d
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import uvicorn

# Import your existing modules
from .core.cad_loader import CADLoader
from .core.annotation_manager import AnnotationManager

app = FastAPI(title="ArUco Grasp Annotator Web", version="1.0.0")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
cad_loader = CADLoader()
annotation_manager = AnnotationManager()

# Store active sessions (in production, use a database)
active_sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArUco Grasp Annotator</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .controls { display: flex; gap: 20px; margin-bottom: 20px; }
            .panel { flex: 1; border: 1px solid #ddd; border-radius: 4px; padding: 15px; }
            .viewer { width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 4px; background: #2b2b2b; }
            button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #45a049; }
            .file-input { margin: 10px 0; }
            .status { margin: 10px 0; padding: 10px; background: #e7f3ff; border-radius: 4px; }
            .error { background: #ffebee; color: #c62828; }
            .success { background: #e8f5e8; color: #2e7d32; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 ArUco Grasp Annotator</h1>
                <p>3D CAD annotation tool for robotics applications</p>
            </div>
            
            <div class="controls">
                <div class="panel">
                    <h3>📁 CAD Model</h3>
                    <div class="file-input">
                        <input type="file" id="fileInput" accept=".stl,.obj,.ply,.off">
                        <button onclick="loadModel()">Load CAD File</button>
                    </div>
                    <div id="modelStatus" class="status">No model loaded</div>
                </div>
                
                <div class="panel">
                    <h3>🎯 ArUco Markers</h3>
                    <button onclick="addMarker()">Add Marker</button>
                    <div id="markersList"></div>
                </div>
                
                <div class="panel">
                    <h3>🤖 Grasp Poses</h3>
                    <button onclick="addGraspPose()">Add Grasp Pose</button>
                    <div id="graspsList"></div>
                </div>
            </div>
            
            <div class="viewer" id="viewer"></div>
            
            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 14px;">
                <strong>🖱️ 3D Controls:</strong>
                <span style="margin-left: 15px;">Left Click + Drag: Rotate</span>
                <span style="margin-left: 15px;">Right Click + Drag: Pan</span>
                <span style="margin-left: 15px;">Ctrl + Left Click + Drag: Pan</span>
                <span style="margin-left: 15px;">Mouse Wheel: Zoom</span>
            </div>
            
            <div style="margin-top: 20px; text-align: center;">
                <button onclick="exportAnnotations()">Export Annotations</button>
                <button onclick="downloadAnnotations()">Download JSON</button>
            </div>
        </div>

        <script>
            let scene, camera, renderer, controls;
            let currentMesh = null;
            let markers = [];
            let graspPoses = [];
            let sessionId = null;
            let meshCenter = new THREE.Vector3(0, 0, 0);
            let meshSize = 1;
            let spherical = new THREE.Spherical();
            let target = new THREE.Vector3(0, 0, 0);

            // Initialize Three.js scene
            function initViewer() {
                const container = document.getElementById('viewer');
                
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x2b2b2b);
                
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.set(3, 3, 3);
                
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                container.appendChild(renderer.domElement);
                
                // Add lighting for better material visualization
                const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(10, 10, 5);
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                scene.add(directionalLight);
                
                // Add fill light
                const fillLight = new THREE.DirectionalLight(0x8080ff, 0.3);
                fillLight.position.set(-5, 0, -5);
                scene.add(fillLight);
                
                // Add coordinate axes
                const axesHelper = new THREE.AxesHelper(1);
                scene.add(axesHelper);
                
                // Mouse controls for orbit camera
                let isRotating = false;
                let isPanning = false;
                let previousMousePosition = { x: 0, y: 0 };
                let rotateSpeed = 0.005;
                let panSpeed = 0.002;
                let zoomSpeed = 0.1;
                
                // Set initial spherical coordinates
                target.set(0, 0, 0);
                spherical.setFromVector3(camera.position.clone().sub(target));
                
                renderer.domElement.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    if (e.button === 0) { // Left mouse button
                        if (e.ctrlKey || e.metaKey) {
                            isPanning = true;
                        } else {
                            isRotating = true;
                        }
                    } else if (e.button === 2) { // Right mouse button
                        isPanning = true;
                    }
                    previousMousePosition = { x: e.clientX, y: e.clientY };
                });
                
                renderer.domElement.addEventListener('mouseup', (e) => {
                    isRotating = false;
                    isPanning = false;
                });
                
                renderer.domElement.addEventListener('mouseleave', (e) => {
                    isRotating = false;
                    isPanning = false;
                });
                
                renderer.domElement.addEventListener('mousemove', (e) => {
                    if (!isRotating && !isPanning) return;
                    
                    const deltaX = e.clientX - previousMousePosition.x;
                    const deltaY = e.clientY - previousMousePosition.y;
                    
                    if (isRotating) {
                        // Rotate around target
                        spherical.theta -= deltaX * rotateSpeed;
                        spherical.phi += deltaY * rotateSpeed;
                        
                        // Limit phi to prevent flipping
                        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                        
                        camera.position.setFromSpherical(spherical).add(target);
                        camera.lookAt(target);
                    } else if (isPanning) {
                        // Pan the camera
                        const panOffset = new THREE.Vector3();
                        const cameraRight = new THREE.Vector3();
                        const cameraUp = new THREE.Vector3();
                        
                        camera.getWorldDirection(new THREE.Vector3());
                        cameraRight.crossVectors(camera.up, camera.getWorldDirection(new THREE.Vector3())).normalize();
                        cameraUp.crossVectors(camera.getWorldDirection(new THREE.Vector3()), cameraRight).normalize();
                        
                        panOffset.addScaledVector(cameraRight, -deltaX * panSpeed);
                        panOffset.addScaledVector(cameraUp, deltaY * panSpeed);
                        
                        camera.position.add(panOffset);
                        target.add(panOffset);
                        
                        // Update spherical coordinates
                        spherical.setFromVector3(camera.position.clone().sub(target));
                    }
                    
                    previousMousePosition = { x: e.clientX, y: e.clientY };
                });
                
                renderer.domElement.addEventListener('wheel', (e) => {
                    e.preventDefault();
                    
                    // Zoom in/out by moving camera closer/farther from target
                    const zoomDelta = e.deltaY * zoomSpeed * 0.01;
                    spherical.radius = Math.max(0.1, spherical.radius + zoomDelta);
                    
                    camera.position.setFromSpherical(spherical).add(target);
                    camera.lookAt(target);
                });
                
                // Disable context menu on right click
                renderer.domElement.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                });
                
                animate();
            }
            
            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }
            
            // Load CAD model
            async function loadModel() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    updateStatus('modelStatus', 'Please select a file', 'error');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload-mesh', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // Remove existing mesh
                        if (currentMesh) {
                            scene.remove(currentMesh);
                        }
                        
                        // Create new mesh
                        const geometry = new THREE.BufferGeometry();
                        
                        // Set vertices
                        const vertices = new Float32Array(result.vertices);
                        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                        
                        // Set triangles/indices if available
                        if (result.triangles && result.triangles.length > 0) {
                            const indices = new Uint32Array(result.triangles);
                            geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                        }
                        
                        // Add normals if available, otherwise compute them
                        if (result.normals && result.normals.length > 0) {
                            const normals = new Float32Array(result.normals);
                            geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
                        } else {
                            geometry.computeVertexNormals();
                        }
                        
                        // Force geometry to update
                        geometry.computeBoundingSphere();
                        
                        // Create material with nice appearance - try different materials
                        const material = new THREE.MeshStandardMaterial({ 
                            color: 0x3498db,
                            metalness: 0.1,
                            roughness: 0.7,
                            side: THREE.DoubleSide,
                            flatShading: false
                        });
                        
                        currentMesh = new THREE.Mesh(geometry, material);
                        currentMesh.castShadow = true;
                        currentMesh.receiveShadow = true;
                        scene.add(currentMesh);
                        
                        // Center and scale the mesh
                        geometry.computeBoundingBox();
                        const boundingBox = geometry.boundingBox;
                        meshCenter = boundingBox.getCenter(new THREE.Vector3());
                        
                        // Move mesh to origin
                        currentMesh.position.sub(meshCenter);
                        
                        // Calculate mesh size and scale if needed
                        const size = boundingBox.getSize(new THREE.Vector3());
                        meshSize = Math.max(size.x, size.y, size.z);
                        
                        // Scale mesh to reasonable size (around 2 units)
                        if (meshSize > 5 || meshSize < 0.1) {
                            const scale = 2 / meshSize;
                            currentMesh.scale.setScalar(scale);
                        }
                        
                        // Update camera to view the object properly
                        const distance = meshSize * 2;
                        camera.position.set(distance, distance, distance);
                        camera.lookAt(0, 0, 0);
                        
                        // Update spherical coordinates for orbit controls
                        target.set(0, 0, 0);
                        spherical.setFromVector3(camera.position.clone().sub(target));
                        
                        // Debug info
                        console.log('Mesh loaded:', {
                            vertices: result.vertices.length/3,
                            triangles: result.triangles ? result.triangles.length/3 : 'none',
                            normals: result.normals ? result.normals.length/3 : 'none',
                            hasTriangles: geometry.index !== null,
                            hasNormals: geometry.attributes.normal !== undefined
                        });
                        
                        updateStatus('modelStatus', `Loaded: ${result.vertices.length/3} vertices, ${result.triangles ? result.triangles.length/3 : 0} triangles`, 'success');
                        sessionId = result.session_id;
                    } else {
                        updateStatus('modelStatus', `Error: ${result.detail}`, 'error');
                    }
                } catch (error) {
                    updateStatus('modelStatus', `Error: ${error.message}`, 'error');
                }
            }
            
            // Add ArUco marker
            async function addMarker() {
                if (!sessionId) {
                    alert('Please load a model first');
                    return;
                }
                
                const position = [0, 0, 0]; // Default position
                const size = 0.05;
                
                try {
                    const response = await fetch('/add-marker', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: sessionId,
                            position: position,
                            size: size
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // Add marker to 3D scene
                        const geometry = new THREE.BoxGeometry(size, size, size/10);
                        const material = new THREE.MeshLambertMaterial({ color: 0xff0000 });
                        const marker = new THREE.Mesh(geometry, material);
                        marker.position.set(position[0], position[1], position[2]);
                        scene.add(marker);
                        
                        markers.push({ id: result.marker_id, mesh: marker });
                        updateMarkersList();
                    }
                } catch (error) {
                    console.error('Error adding marker:', error);
                }
            }
            
            // Add grasp pose
            async function addGraspPose() {
                if (!sessionId || markers.length === 0) {
                    alert('Please load a model and add at least one marker first');
                    return;
                }
                
                const markerId = markers[0].id; // Use first marker
                const position = [0, 0, 0.05];
                const orientation = [1, 0, 0, 0]; // Quaternion
                
                try {
                    const response = await fetch('/add-grasp-pose', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: sessionId,
                            marker_id: markerId,
                            name: `Grasp ${graspPoses.length + 1}`,
                            position: position,
                            orientation: orientation
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // Add grasp pose to 3D scene
                        const geometry = new THREE.ConeGeometry(0.01, 0.05, 8);
                        const material = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
                        const grasp = new THREE.Mesh(geometry, material);
                        grasp.position.set(position[0], position[1], position[2]);
                        scene.add(grasp);
                        
                        graspPoses.push({ id: result.grasp_id, mesh: grasp });
                        updateGraspsList();
                    }
                } catch (error) {
                    console.error('Error adding grasp pose:', error);
                }
            }
            
            // Export annotations
            async function exportAnnotations() {
                if (!sessionId) {
                    alert('Please load a model first');
                    return;
                }
                
                try {
                    const response = await fetch(`/export-annotations/${sessionId}`);
                    const result = await response.json();
                    
                    if (response.ok) {
                        updateStatus('modelStatus', 'Annotations exported successfully', 'success');
                    }
                } catch (error) {
                    console.error('Error exporting annotations:', error);
                }
            }
            
            // Download annotations
            async function downloadAnnotations() {
                if (!sessionId) {
                    alert('Please load a model first');
                    return;
                }
                
                try {
                    const response = await fetch(`/download-annotations/${sessionId}`);
                    const blob = await response.blob();
                    
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'annotations.json';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } catch (error) {
                    console.error('Error downloading annotations:', error);
                }
            }
            
            function updateStatus(elementId, message, type) {
                const element = document.getElementById(elementId);
                element.textContent = message;
                element.className = `status ${type}`;
            }
            
            function updateMarkersList() {
                const list = document.getElementById('markersList');
                list.innerHTML = markers.map(m => `<div>Marker ${m.id}</div>`).join('');
            }
            
            function updateGraspsList() {
                const list = document.getElementById('graspsList');
                list.innerHTML = graspPoses.map(g => `<div>Grasp ${g.id}</div>`).join('');
            }
            
            // Initialize when page loads
            window.onload = initViewer;
        </script>
    </body>
    </html>
    """

@app.post("/upload-mesh")
async def upload_mesh(file: UploadFile = File(...)):
    """Upload and process a CAD mesh file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load mesh with Open3D
        mesh = cad_loader.load_file(Path(tmp_file_path))
        mesh_info = cad_loader.get_mesh_info(mesh)
        
        # Create session
        import uuid
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "mesh": mesh,
            "mesh_info": mesh_info,
            "markers": [],
            "grasp_poses": []
        }
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            "session_id": session_id,
            "vertices": np.asarray(mesh.vertices).flatten().tolist(),
            "normals": np.asarray(mesh.vertex_normals).flatten().tolist() if mesh.has_vertex_normals() else [],
            "triangles": np.asarray(mesh.triangles).flatten().tolist(),
            "mesh_info": mesh_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add-marker")
async def add_marker(data: dict):
    """Add an ArUco marker to the session."""
    session_id = data.get("session_id")
    position = data.get("position", [0, 0, 0])
    size = data.get("size", 0.05)
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate marker ID
    marker_id = len(active_sessions[session_id]["markers"])
    
    # Add marker to session
    active_sessions[session_id]["markers"].append({
        "id": marker_id,
        "position": position,
        "size": size
    })
    
    return {"marker_id": marker_id, "success": True}

@app.post("/add-grasp-pose")
async def add_grasp_pose(data: dict):
    """Add a grasp pose to the session."""
    session_id = data.get("session_id")
    marker_id = data.get("marker_id")
    name = data.get("name", "Grasp")
    position = data.get("position", [0, 0, 0])
    orientation = data.get("orientation", [1, 0, 0, 0])
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate grasp ID
    grasp_id = len(active_sessions[session_id]["grasp_poses"])
    
    # Add grasp pose to session
    active_sessions[session_id]["grasp_poses"].append({
        "id": grasp_id,
        "name": name,
        "marker_id": marker_id,
        "position": position,
        "orientation": orientation
    })
    
    return {"grasp_id": grasp_id, "success": True}

@app.get("/export-annotations/{session_id}")
async def export_annotations(session_id: str):
    """Export annotations for a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Prepare export data
    export_data = {
        "session_id": session_id,
        "mesh_info": session["mesh_info"],
        "markers": session["markers"],
        "grasp_poses": session["grasp_poses"],
        "exported_at": "2024-01-01T12:00:00"  # In production, use actual timestamp
    }
    
    return export_data

@app.get("/download-annotations/{session_id}")
async def download_annotations(session_id: str):
    """Download annotations as JSON file."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Prepare export data
    export_data = {
        "session_id": session_id,
        "mesh_info": session["mesh_info"],
        "markers": session["markers"],
        "grasp_poses": session["grasp_poses"],
        "exported_at": "2024-01-01T12:00:00"
    }
    
    return JSONResponse(
        content=export_data,
        headers={"Content-Disposition": "attachment; filename=annotations.json"}
    )

def main():
    """Main entry point for the web application."""
    print("🚀 Starting ArUco Grasp Annotator Web App...")
    print("📱 Open your browser to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
