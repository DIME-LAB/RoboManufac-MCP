#!/usr/bin/env python3
"""
Grasp Candidates Visualization UI
Web-based 3D visualization tool for viewing objects with grasp points overlayed.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
import subprocess
import os

app = FastAPI(
    title="Grasp Candidates Visualizer",
    description="Interactive 3D visualization of objects with grasp points",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GRASP_DIR = DATA_DIR / "grasp"
GRASP_CANDIDATES_DIR = DATA_DIR / "grasp_candidates"
WIREFRAME_DIR = DATA_DIR / "wireframe"
ARUCO_DIR = DATA_DIR / "aruco"


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML interface."""
    return r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grasp Candidates Visualizer</title>
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
                border-right: 1px solid #ddd;
            }
            
            .main-viewer {
                flex: 1;
                position: relative;
                background: #1a1a1a;
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
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 5px;
                color: #2c3e50;
                font-weight: 500;
                font-size: 14px;
            }
            
            .form-group select {
                width: 100%;
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                background: white;
                cursor: pointer;
            }
            
            .form-group select:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            
            .btn-execute {
                width: 100%;
                padding: 12px 20px;
                background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(46, 204, 113, 0.3);
            }
            
            .btn-execute:hover:not(:disabled) {
                background: linear-gradient(135deg, #229954 0%, #27ae60 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(46, 204, 113, 0.4);
            }
            
            .btn-execute:active:not(:disabled) {
                transform: translateY(0);
                box-shadow: 0 2px 4px rgba(46, 204, 113, 0.3);
            }
            
            .btn-execute:disabled {
                background: #95a5a6;
                cursor: not-allowed;
                opacity: 0.6;
            }
            
            .btn-gripper {
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 5px;
                min-width: 120px;
            }
            
            .btn-gripper-open {
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
            }
            
            .btn-gripper-open:hover {
                background: linear-gradient(135deg, #2980b9 0%, #1f6391 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(52, 152, 219, 0.4);
            }
            
            .btn-gripper-close {
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                box-shadow: 0 4px 8px rgba(231, 76, 60, 0.3);
            }
            
            .btn-gripper-close:hover {
                background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(231, 76, 60, 0.4);
            }
            
            .gripper-buttons {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            
            .info-panel {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }
            
            .info-panel h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 16px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }
            
            .info-item {
                margin-bottom: 10px;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                font-size: 13px;
            }
            
            .info-item strong {
                color: #2c3e50;
                display: block;
                margin-bottom: 3px;
            }
            
            .info-item span {
                color: #666;
                font-family: 'Courier New', monospace;
            }
            
            #viewer-container {
                width: 100%;
                height: 100%;
            }
            
            .status-bar {
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 100;
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar">
                <div class="controls-panel">
                    <h3>📦 Object Selection</h3>
                    <div class="form-group">
                        <label for="objectSelect">Select Object:</label>
                        <select id="objectSelect">
                            <option value="">-- Select an object --</option>
                        </select>
                    </div>
                </div>
                
                <div class="controls-panel">
                    <h3>🎯 Grasp Point Selection</h3>
                    <div class="form-group">
                        <label for="graspSelect">Select Grasp Point:</label>
                        <select id="graspSelect" disabled>
                            <option value="">-- Select object first --</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="directionSelect">Select Direction:</label>
                        <select id="directionSelect" disabled>
                            <option value="">-- Select grasp point first --</option>
                        </select>
                    </div>
                    <div class="form-group" id="executeButtonGroup" style="display: none;">
                        <button id="executeButton" class="btn-execute" disabled>Execute Grasp</button>
                    </div>
                    <div class="form-group">
                        <h3>🤖 Gripper Control</h3>
                        <div class="gripper-buttons">
                            <button id="openGripperButton" class="btn-gripper btn-gripper-open">Open Gripper</button>
                            <button id="closeGripperButton" class="btn-gripper btn-gripper-close">Close Gripper</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <h3>🏠 Robot Control</h3>
                        <div class="gripper-buttons">
                            <button id="moveToSafeHeightButton" class="btn-gripper btn-gripper-open">Move to Safe Height</button>
                            <button id="moveHomeButton" class="btn-gripper btn-gripper-close">Move Home</button>
                        </div>
                    </div>
                </div>
                
                <div class="info-panel">
                    <h3>ℹ️ Grasp Point Info</h3>
                    <div id="graspInfo">
                        <div class="info-item">
                            <strong>No grasp point selected</strong>
                            <span>Select a grasp point to view details</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="main-viewer">
                <div id="viewer-container"></div>
                <div class="status-bar" id="statusBar">Ready</div>
            </div>
        </div>
        
        <script>
            // Three.js scene setup
            let scene, camera, renderer, controls;
            let graspPointsGroup = null;
            let currentObjectData = null;
            let currentGraspData = null;
            let currentGraspCandidates = null;
            let selectedGraspPointId = null;  // Selected grasp point ID (e.g., 1, 2, 3)
            let selectedGraspId = null;  // Selected candidate ID (e.g., "1_5")
            let raycaster = new THREE.Raycaster();
            let mouse = new THREE.Vector2();
            
            // Colors
            const GRASP_COLOR_DEFAULT = 0x00ff00;  // Green
            const GRASP_COLOR_SELECTED_POINT = 0xff0000;  // Red for selected grasp point
            const GRASP_COLOR_SELECTED_CANDIDATE = 0x3498db;  // Blue for selected candidate
            const GRIPPER_COLOR_DEFAULT = 0xffaa00;  // Orange
            const GRIPPER_COLOR_SELECTED = 0xff0000;  // Red for selected candidate's gripper
            const WIREFRAME_COLOR = 0x888888;  // Gray
            
            function initScene() {
                // Scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                // Camera
                camera = new THREE.PerspectiveCamera(
                    75,
                    window.innerWidth / window.innerHeight,
                    0.001,
                    1000
                );
                camera.position.set(0.5, -0.5, 0.5);  // Move camera to look at scene from Y-negative
                camera.up.set(0, 0, 1);  // Set Z as up vector
                camera.lookAt(0, 0, 0);
                
                // Renderer
                const container = document.getElementById('viewer-container');
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);
                
                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.minDistance = 0.05;
                controls.maxDistance = 5;
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, -10, 10);  // Light from above in Z-up system
                scene.add(directionalLight);
                
                const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3);
                scene.add(hemisphereLight);
                
                // Grid helper
                const gridHelper = new THREE.GridHelper(1, 1, 0x444444, 0x222222);
                gridHelper.rotateX(Math.PI / 2);  // Rotate 90 degrees to make it horizontal in XY plane
                scene.add(gridHelper);
                
                // Axes helper
                const axesHelper = new THREE.AxesHelper(0.2);
                scene.add(axesHelper);
                
                // Load available objects
                loadObjects();
                
                // Event listeners
                document.getElementById('objectSelect').addEventListener('change', onObjectChanged);
                document.getElementById('graspSelect').addEventListener('change', onGraspPointChanged);
                document.getElementById('directionSelect').addEventListener('change', onDirectionChanged);
                document.getElementById('executeButton').addEventListener('click', onExecuteButtonClick);
                document.getElementById('openGripperButton').addEventListener('click', onOpenGripperClick);
                document.getElementById('closeGripperButton').addEventListener('click', onCloseGripperClick);
                document.getElementById('moveToSafeHeightButton').addEventListener('click', onMoveToSafeHeightClick);
                document.getElementById('moveHomeButton').addEventListener('click', onMoveHomeClick);
                
                // Mouse click handler for grasp point selection
                renderer.domElement.addEventListener('click', onGraspPointClick);
                
                // Window resize
                window.addEventListener('resize', onWindowResize);
                
                // Animation loop
                animate();
            }
            
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            
            function onWindowResize() {
                const container = document.getElementById('viewer-container');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
            
            async function loadObjects() {
                try {
                    const response = await fetch('/api/objects');
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const objects = await response.json();
                    
                    console.log('Loaded objects:', objects);
                    
                    const select = document.getElementById('objectSelect');
                    if (!select) {
                        console.error('objectSelect element not found!');
                        updateStatus('Error: objectSelect element not found');
                        return;
                    }
                    
                    select.innerHTML = '<option value="">-- Select an object --</option>';
                    
                    if (objects && Array.isArray(objects) && objects.length > 0) {
                        objects.forEach(objName => {
                            const option = document.createElement('option');
                            option.value = objName;
                            // Format object name: remove _scaled70, replace _ with space, capitalize words
                            const displayName = objName.replace('_scaled70', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            option.textContent = displayName;
                            select.appendChild(option);
                        });
                        
                        updateStatus(`Loaded ${objects.length} objects`);
                        console.log(`Successfully loaded ${objects.length} objects into dropdown`);
                    } else {
                        updateStatus('No objects found');
                        console.warn('No objects returned from API or empty array:', objects);
                    }
                } catch (error) {
                    console.error('Error loading objects:', error);
                    updateStatus(`Error loading objects: ${error.message}`);
                }
            }
            
            async function onObjectChanged(event) {
                const objectName = event.target.value;
                
                if (!objectName) {
                    clearScene();
                    return;
                }
                
                try {
                    updateStatus(`Loading ${objectName}...`);
                    
                    // Load grasp data (for positions only)
                    const graspResponse = await fetch(`/api/grasp-data/${objectName}`);
                    if (!graspResponse.ok) {
                        throw new Error('Failed to load grasp data');
                    }
                    currentGraspData = await graspResponse.json();
                    
                    // Load grasp candidates (for approach vectors and orientations)
                    const candidatesResponse = await fetch(`/api/grasp-candidates/${objectName}`);
                    if (!candidatesResponse.ok) {
                        throw new Error('Failed to load grasp candidates');
                    }
                    currentGraspCandidates = await candidatesResponse.json();
                    
                    // Load wireframe
                    const wireframeResponse = await fetch(`/api/wireframe/${objectName}`);
                    if (!wireframeResponse.ok) {
                        throw new Error('Failed to load wireframe');
                    }
                    const wireframeData = await wireframeResponse.json();
                    
                    // Load ArUco data
                    const arucoResponse = await fetch(`/api/aruco/${objectName}`);
                    const arucoData = arucoResponse.ok ? await arucoResponse.json() : null;
                    
                    // Merge wireframe and ArUco markers into grasp data
                    currentGraspData.wireframe = wireframeData;
                    
                    // If grasp data doesn't have markers, use ArUco markers
                    if (!currentGraspData.markers && arucoData && arucoData.markers) {
                        // Convert ArUco markers format to grasp markers format
                        currentGraspData.markers = arucoData.markers.map(marker => ({
                            aruco_id: marker.aruco_id,
                            size: marker.size,
                            pose_absolute: marker.pose_absolute
                        }));
                    }
                    
                    // Clear previous visualization
                    clearGraspPoints();
                    
                    // Create group for everything (wireframe, markers, grasp candidates)
                    graspPointsGroup = new THREE.Group();
                    graspPointsGroup.name = "GraspVisualization";
                    
                    // 1. Create wireframe
                    const wireframe = createWireframeFromData(currentGraspData);
                    graspPointsGroup.add(wireframe);
                    
                    // 2. Add ArUco markers
                    if (currentGraspData.markers && Array.isArray(currentGraspData.markers)) {
                        currentGraspData.markers.forEach(markerData => {
                            const markerMesh = createMarkerMesh(markerData);
                            graspPointsGroup.add(markerMesh);
                        });
                    }
                    
                    // 3. Add grasp candidates (use grasp point positions from grasp data, approach vectors from candidates)
                    let totalCandidates = 0;
                    if (currentGraspCandidates && currentGraspCandidates.grasp_candidates) {
                        // Group candidates by grasp_point_id to get unique grasp points
                        const uniqueGraspPoints = new Map();
                        currentGraspCandidates.grasp_candidates.forEach(candidate => {
                            const gpId = candidate.grasp_point_id;
                            if (!uniqueGraspPoints.has(gpId)) {
                                // Find the corresponding grasp point from grasp data for position
                                const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === gpId);
                                if (graspPoint) {
                                    uniqueGraspPoints.set(gpId, graspPoint);
                                }
                            }
                        });
                        
                        // Create visualization for each unique grasp point with all its candidates
                        uniqueGraspPoints.forEach((graspPoint, gpId) => {
                            // Get all candidates for this grasp point
                            const candidatesForPoint = currentGraspCandidates.grasp_candidates.filter(c => c.grasp_point_id === gpId);
                            
                            candidatesForPoint.forEach((candidate, idx) => {
                                // Create sphere using grasp point position, but use candidate's approach quaternion/vector
                                const candidateData = {
                                    id: `${gpId}_${candidate.direction_id}`, // Unique ID: grasp_point_id_direction_id
                                    grasp_point_id: gpId,
                                    direction_id: candidate.direction_id,
                                    position: graspPoint.position, // Use position from grasp data
                                    approach_quaternion: candidate.approach_quaternion, // Use approach quaternion from candidate (new format)
                                    approach_vector: candidate.approach_vector, // Use approach vector from candidate (old format, fallback)
                                    type: graspPoint.type || 'center_point'
                                };
                                
                                const sphere = createGraspPointSphere(candidateData, totalCandidates);
                                graspPointsGroup.add(sphere);
                                totalCandidates++;
                            });
                        });
                    }
                    
                    // Objects are always at world center (0,0,0) - no pose transformation
                    // Grasp points are relative to CAD center which is at world center
                    graspPointsGroup.position.set(0, 0, 0);
                    graspPointsGroup.rotation.set(0, 0, 0);
                    
                    scene.add(graspPointsGroup);
                    
                    // Update grasp point selector
                    updateGraspPointSelector();
                    
                    updateStatus(`Loaded ${objectName} (${totalCandidates} grasp candidates)`);
                } catch (error) {
                    console.error('Error loading object:', error);
                    updateStatus(`Error loading ${objectName}: ${error.message}`);
                }
            }
            
            function createWireframeFromData(data) {
                const vertices = data.wireframe.vertices;
                const edges = data.wireframe.edges;
                
                const lineVertices = [];
                edges.forEach(edge => {
                    const v1 = vertices[edge[0]];
                    const v2 = vertices[edge[1]];
                    lineVertices.push(...v1, ...v2);
                });
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
                
                const material = new THREE.LineBasicMaterial({ 
                    color: 0x4a90e2,
                    linewidth: 2
                });
                
                const wireframe = new THREE.LineSegments(geometry, material);
                wireframe.userData = { 
                    name: data.object_name,
                    type: 'grasp_wireframe',
                    displayName: data.display_name,
                    originalColor: 0x4a90e2,
                    id: generateId()
                };
                
                // Center the wireframe
                wireframe.position.set(0, 0, 0);
                
                return wireframe;
            }
            
            function createMarkerMesh(markerData) {
                const size = markerData.size;
                const geometry = new THREE.BoxGeometry(size, size, size * 0.1);
                const material = new THREE.MeshBasicMaterial({ 
                    color: 0xff6b6b,
                    transparent: true,
                    opacity: 0.7
                });
                
                const marker = new THREE.Mesh(geometry, material);
                
                const pos = markerData.pose_absolute.position;
                marker.position.set(pos.x, pos.y, pos.z);
                
                const rot = markerData.pose_absolute.rotation;
                marker.rotation.set(rot.roll, rot.pitch, rot.yaw);
                
                marker.userData = { 
                    name: `ArUco-${markerData.aruco_id}`,
                    type: 'grasp_marker',
                    arucoId: markerData.aruco_id,
                    displayName: `ArUco ${markerData.aruco_id}`,
                    originalColor: 0xff6b6b,
                    id: generateId()
                };
                
                return marker;
            }
            
            function createGraspPointSphere(graspPoint, index) {
                // Create sphere geometry for grasp point
                // Note: All coordinates are in meters relative to CAD center
                const geometry = new THREE.SphereGeometry(0.003, 16, 16);  // 0.003 m = 3mm
                // Calculate candidate ID once
                const candidateId = graspPoint.id || `${graspPoint.grasp_point_id}_${graspPoint.direction_id}`;
                const graspPointId = graspPoint.grasp_point_id;
                
                // Determine color based on selection state
                let color = GRASP_COLOR_DEFAULT;
                if (selectedGraspPointId === graspPointId) {
                    color = GRASP_COLOR_SELECTED_POINT;  // Red for selected grasp point
                } else if (selectedGraspId === candidateId) {
                    color = GRASP_COLOR_SELECTED_CANDIDATE;  // Blue for selected candidate
                }
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.5,
                    transparent: true,
                    opacity: 0.9
                });
                
                const sphere = new THREE.Mesh(geometry, material);
                
                // Set position from grasp point data (in meters, relative to CAD center)
                // Coordinate flip already applied during export
                const pos = graspPoint.position;
                sphere.position.set(pos.x, pos.y, pos.z);
                
                // Store metadata (candidateId already declared above)
                sphere.userData = {
                    name: `Grasp-${graspPoint.grasp_point_id || graspPoint.id}_${graspPoint.direction_id || ''}`,
                    type: 'grasp_point',
                    graspPointId: graspPointId,  // Store grasp point ID for filtering
                    graspId: graspPoint.grasp_point_id || graspPoint.id,
                    directionId: graspPoint.direction_id,
                    id: candidateId, // Use the unique candidate ID
                    displayName: `Grasp Point ${graspPoint.grasp_point_id || graspPoint.id}${graspPoint.direction_id ? ` - Direction ${graspPoint.direction_id}` : ''}`,
                    originalColor: color,
                    isClickable: true  // Mark as clickable
                };
                
                // Add gripper visualization (parallel-jaw gripper)
                const gripperGroup = new THREE.Group();
                
                // Gripper parameters
                const gripperWidth = 0.008;      // 8mm - distance between jaws
                const gripperLength = 0.012;     // 12mm - length of gripper fingers
                const gripperDepth = 0.004;      // 4mm - depth of gripper fingers
                const gripperThickness = 0.001;  // 1mm - thickness of each jaw
                const approachLength = 0.015;    // 15mm - length of approach indicator
                
                // Material for gripper - color depends on selection
                const isCandidateSelected = selectedGraspId === candidateId;
                const gripperColor = isCandidateSelected ? GRIPPER_COLOR_SELECTED : GRIPPER_COLOR_DEFAULT;
                const gripperMaterial = new THREE.MeshPhongMaterial({ 
                    color: gripperColor,
                    transparent: true,
                    opacity: 0.8
                });
                
                // Create two parallel jaw plates
                const jawGeometry = new THREE.BoxGeometry(gripperThickness, gripperDepth, gripperLength);
                
                // Left jaw
                const leftJaw = new THREE.Mesh(jawGeometry, gripperMaterial);
                leftJaw.position.set(-gripperWidth / 2, 0, 0);
                gripperGroup.add(leftJaw);
                
                // Right jaw
                const rightJaw = new THREE.Mesh(jawGeometry, gripperMaterial);
                rightJaw.position.set(gripperWidth / 2, 0, 0);
                gripperGroup.add(rightJaw);
                
                // Palm/base plate connecting the jaws
                const palmGeometry = new THREE.BoxGeometry(gripperWidth, gripperDepth * 0.6, gripperThickness);
                const palm = new THREE.Mesh(palmGeometry, gripperMaterial);
                palm.position.set(0, 0, -gripperLength / 2);
                gripperGroup.add(palm);
                
                // Approach direction indicator (line from gripper to grasp point)
                const approachGeometry = new THREE.CylinderGeometry(0.0008, 0.0008, approachLength, 8);  // Increased width to 0.8mm
                const approachMaterial = new THREE.MeshPhongMaterial({ 
                    color: 0xffaa00,  // Match gripper color
                    transparent: true,
                    opacity: 0.8
                });
                const approachLine = new THREE.Mesh(approachGeometry, approachMaterial);
                // Rotate to be vertical (pointing down along Z axis)
                approachLine.rotation.x = Math.PI / 2;  // Rotate 90 degrees to be vertical
                approachLine.position.set(0, 0, -gripperLength / 2 - approachLength / 2);
                gripperGroup.add(approachLine);
                
                // Orient gripper based on approach quaternion (new format) or approach vector (old format)
                // The approach quaternion represents the orientation where Z-axis points in approach direction
                // The gripper should point TOWARDS the grasp point (opposite to approach direction)
                let approachVec = new THREE.Vector3(0, 0, -1); // Default direction
                
                if (graspPoint.approach_quaternion) {
                    // New format: use approach quaternion directly
                    const quat = graspPoint.approach_quaternion;
                    const approachQuat = new THREE.Quaternion(quat.x, quat.y, quat.z, quat.w);
                    
                    // Extract Z-axis direction from quaternion (this is the approach direction)
                    const zAxis = new THREE.Vector3(0, 0, 1);
                    zAxis.applyQuaternion(approachQuat);
                    
                    // Negate to get gripper pointing direction (towards grasp point)
                    approachVec = zAxis.negate().normalize();
                    
                    // Create quaternion for gripper orientation (gripper Z points down, opposite to approach)
                    const gripperZAxis = new THREE.Vector3(0, 0, 1); // Gripper Z in local frame
                    const defaultDir = new THREE.Vector3(0, 0, -1); // Default gripper pointing direction
                    const quaternion = new THREE.Quaternion();
                    quaternion.setFromUnitVectors(defaultDir, approachVec);
                    gripperGroup.quaternion.copy(quaternion);
                } else if (graspPoint.approach_vector) {
                    // Old format: use approach vector (backward compatibility)
                    const approach = graspPoint.approach_vector;
                    // Negate the approach vector to point towards the grasp point
                    approachVec = new THREE.Vector3(-approach.x, -approach.y, -approach.z).normalize();
                    
                    // Default gripper points along -Z axis, rotate to match negated approach vector
                    const defaultDir = new THREE.Vector3(0, 0, -1);
                    const quaternion = new THREE.Quaternion();
                    quaternion.setFromUnitVectors(defaultDir, approachVec);
                    gripperGroup.quaternion.copy(quaternion);
                }
                
                // Position gripper offset from sphere along approach direction
                // Offset distance: 8mm (0.008m) - this is just for visualization
                // The actual 0.115m offset is applied at runtime
                const offsetDistance = 0.008; // 8mm for visualization
                const offsetPosition = approachVec.clone().multiplyScalar(offsetDistance);
                gripperGroup.position.copy(offsetPosition);
                sphere.add(gripperGroup);
                
                // Store gripper reference in sphere userData for visibility control
                sphere.userData.gripperGroup = gripperGroup;
                sphere.userData.gripperMaterial = gripperMaterial;  // Store for color updates
                
                // Make gripper group and all its children clickable
                gripperGroup.userData.isGripper = true;
                gripperGroup.traverse((child) => {
                    child.userData.isGripper = true;
                    child.userData.parentSphere = sphere;  // Reference back to sphere
                });
                
                return sphere;
            }
            
            function onGraspPointClick(event) {
                // Calculate mouse position in normalized device coordinates (-1 to +1)
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                // Update raycaster
                raycaster.setFromCamera(mouse, camera);
                
                // Find intersections with grasp points
                const intersects = raycaster.intersectObjects(graspPointsGroup.children, true);
                
                if (intersects.length > 0) {
                    // Find the closest grasp point sphere (not gripper parts)
                    for (let intersect of intersects) {
                        let obj = intersect.object;
                        // Traverse up to find the sphere
                        while (obj && obj.userData) {
                            if (obj.userData.type === 'grasp_point' && obj.userData.isClickable) {
                                const clickedCandidateId = obj.userData.id;  // Full candidate ID (e.g., "6_16")
                                const clickedGraspPointId = obj.userData.graspPointId;
                                const clickedDirectionId = obj.userData.directionId;
                                
                                // If no grasp point is selected, select the grasp point
                                if (selectedGraspPointId === null) {
                                    // Select the clicked grasp point
                                    selectedGraspPointId = clickedGraspPointId;
                                    selectedGraspId = null;  // Clear candidate selection
                                    
                                    // Update dropdowns
                                    document.getElementById('graspSelect').value = clickedGraspPointId.toString();
                                    document.getElementById('directionSelect').value = '';
                                    document.getElementById('directionSelect').disabled = false;
                                    updateDirectionSelector(clickedGraspPointId.toString());
                                    
                                    // Update visualization
                                    updateVisualization();
                                    
                                    return;
                                }
                                
                                // If a grasp point is already selected, check if clicking on a candidate from that grasp point
                                if (selectedGraspPointId === clickedGraspPointId) {
                                    // If clicking the same candidate, deselect it
                                    if (selectedGraspId === clickedCandidateId) {
                                        selectedGraspId = null;
                                        document.getElementById('directionSelect').value = '';
                                        updateVisualization();
                                        updateGraspInfo(null);
                                        // Hide execute button
                                        const executeButtonGroup = document.getElementById('executeButtonGroup');
                                        const executeButton = document.getElementById('executeButton');
                                        if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                                        if (executeButton) executeButton.disabled = true;
                                        return;
                                    }
                                    
                                    // Select the clicked candidate from the selected grasp point
                                    selectedGraspId = clickedCandidateId;
                                    
                                    // Update dropdowns to reflect the selected candidate
                                    document.getElementById('directionSelect').value = clickedDirectionId.toString();
                                    
                                    // Update grasp info and execute button
                                    const candidate = currentGraspCandidates.grasp_candidates.find(
                                        c => c.grasp_point_id === clickedGraspPointId && c.direction_id === clickedDirectionId
                                    );
                                    const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === clickedGraspPointId);
                                    
                                    if (candidate && graspPoint) {
                                        const candidateData = {
                                            id: clickedCandidateId,
                                            grasp_point_id: clickedGraspPointId.toString(),
                                            direction_id: clickedDirectionId.toString(),
                                            position: graspPoint.position,
                                            approach_quaternion: candidate.approach_quaternion,
                                            approach_vector: candidate.approach_vector,
                                            type: graspPoint.type || 'center_point'
                                        };
                                        updateGraspInfo(candidateData);
                                        
                                        // Show and enable execute button
                                        const executeButtonGroup = document.getElementById('executeButtonGroup');
                                        const executeButton = document.getElementById('executeButton');
                                        if (executeButtonGroup) executeButtonGroup.style.display = 'block';
                                        if (executeButton) executeButton.disabled = false;
                                    }
                                    
                                    // Update visualization
                                    updateVisualization();
                                    
                                    return;
                                } else {
                                    // Clicking on a different grasp point - select that grasp point instead
                                    selectedGraspPointId = clickedGraspPointId;
                                    selectedGraspId = null;  // Clear candidate selection
                                    
                                    // Update dropdowns
                                    document.getElementById('graspSelect').value = clickedGraspPointId.toString();
                                    document.getElementById('directionSelect').value = '';
                                    document.getElementById('directionSelect').disabled = false;
                                    updateDirectionSelector(clickedGraspPointId.toString());
                                    
                                    // Update visualization
                                    updateVisualization();
                                    
                                    return;
                                }
                            }
                            obj = obj.parent;
                        }
                    }
                }
            }
            
            function clearGraspPoints() {
                if (graspPointsGroup) {
                    scene.remove(graspPointsGroup);
                    graspPointsGroup = null;
                }
            }
            
            function updateGraspPointSelector() {
                const graspSelect = document.getElementById('graspSelect');
                const directionSelect = document.getElementById('directionSelect');
                
                graspSelect.innerHTML = '<option value="">-- Select a grasp point --</option>';
                directionSelect.innerHTML = '<option value="">-- Select grasp point first --</option>';
                directionSelect.disabled = true;
                
                if (!currentGraspCandidates || !currentGraspCandidates.grasp_candidates) {
                    graspSelect.disabled = true;
                    return;
                }
                
                graspSelect.disabled = false;
                
                // Get unique grasp point IDs
                const uniqueGraspPointIds = new Set();
                currentGraspCandidates.grasp_candidates.forEach(candidate => {
                    uniqueGraspPointIds.add(candidate.grasp_point_id);
                });
                
                // Sort and populate grasp point dropdown
                const sortedGraspPointIds = Array.from(uniqueGraspPointIds).sort((a, b) => a - b);
                sortedGraspPointIds.forEach(gpId => {
                    const option = document.createElement('option');
                    option.value = gpId.toString();
                    option.textContent = `Grasp Point ${gpId}`;
                    graspSelect.appendChild(option);
                });
            }
            
            function updateDirectionSelector(graspPointId) {
                const directionSelect = document.getElementById('directionSelect');
                directionSelect.innerHTML = '<option value="">-- Select a direction --</option>';
                
                if (!graspPointId || !currentGraspCandidates || !currentGraspCandidates.grasp_candidates) {
                    directionSelect.disabled = true;
                    return;
                }
                
                directionSelect.disabled = false;
                
                // Get all directions for this grasp point
                const directions = new Set();
                currentGraspCandidates.grasp_candidates.forEach(candidate => {
                    if (candidate.grasp_point_id === parseInt(graspPointId)) {
                        directions.add(candidate.direction_id);
                    }
                });
                
                // Sort and populate direction dropdown
                const sortedDirections = Array.from(directions).sort((a, b) => a - b);
                sortedDirections.forEach(dirId => {
                    const option = document.createElement('option');
                    option.value = dirId.toString();
                    option.textContent = `Direction ${dirId}`;
                    directionSelect.appendChild(option);
                });
            }
            
            function onGraspPointChanged(event) {
                const graspPointId = event.target.value;
                
                // Clear direction selection when grasp point changes
                const directionSelect = document.getElementById('directionSelect');
                directionSelect.value = '';
                selectedGraspId = null;
                
                if (graspPointId) {
                    selectedGraspPointId = parseInt(graspPointId);
                } else {
                    selectedGraspPointId = null;
                }
                
                // Update direction selector with available directions for this grasp point
                if (graspPointId) {
                    updateDirectionSelector(graspPointId);
                } else {
                    updateDirectionSelector(null);
                }
                
                // Update visualization
                updateVisualization();
            }
            
            function onDirectionChanged(event) {
                const directionId = event.target.value;
                const graspPointId = document.getElementById('graspSelect').value;
                
                if (!graspPointId || !directionId) {
                    selectedGraspId = null;
                    updateVisualization();
                    updateGraspInfo(null);
                    // Hide execute button
                    const executeButtonGroup = document.getElementById('executeButtonGroup');
                    const executeButton = document.getElementById('executeButton');
                    if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                    if (executeButton) executeButton.disabled = true;
                    return;
                }
                
                // Create full candidate ID
                selectedGraspId = `${graspPointId}_${directionId}`;
                const gpId = parseInt(graspPointId);
                const dirId = parseInt(directionId);
                
                // Find grasp candidate data
                const candidate = currentGraspCandidates.grasp_candidates.find(
                    c => c.grasp_point_id === gpId && c.direction_id === dirId
                );
                
                // Find corresponding grasp point for position
                const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === gpId);
                
                if (candidate && graspPoint) {
                    // Create combined data for display
                    const candidateData = {
                        id: selectedGraspId,
                        grasp_point_id: graspPointId,
                        direction_id: directionId,
                        position: graspPoint.position,
                        approach_quaternion: candidate.approach_quaternion, // New format
                        approach_vector: candidate.approach_vector, // Old format (fallback)
                        type: graspPoint.type || 'center_point'
                    };
                    updateGraspInfo(candidateData);
                    
                    // Update visualization
                    updateVisualization();
                    
                    // Show and enable execute button
                    const executeButtonGroup = document.getElementById('executeButtonGroup');
                    const executeButton = document.getElementById('executeButton');
                    if (executeButtonGroup) executeButtonGroup.style.display = 'block';
                    if (executeButton) executeButton.disabled = false;
                }
            }
            
            function updateVisualization() {
                if (!graspPointsGroup) return;
                
                graspPointsGroup.traverse((child) => {
                    if (child.userData && child.userData.type === 'grasp_point') {
                        const graspPointId = child.userData.graspPointId;
                        const candidateId = child.userData.id;
                        
                        // Determine sphere color
                        let sphereColor = GRASP_COLOR_DEFAULT;
                        if (selectedGraspPointId === graspPointId) {
                            sphereColor = GRASP_COLOR_SELECTED_POINT;  // Red for selected grasp point
                        } else if (selectedGraspId === candidateId) {
                            sphereColor = GRASP_COLOR_SELECTED_CANDIDATE;  // Blue for selected candidate
                        }
                        
                        // Update sphere color
                        if (child.material) {
                            child.material.color.setHex(sphereColor);
                            child.material.emissive.setHex(sphereColor);
                        }
                        
                        // Handle visibility and gripper color based on selection state
                        // Priority: candidate selection > grasp point selection > show all
                        if (selectedGraspId !== null) {
                            // A candidate is selected - only show that candidate, hide all others
                            if (candidateId === selectedGraspId) {
                                // Selected candidate: blue sphere, red gripper visible
                                child.visible = true;
                                child.material.opacity = 0.9;
                                
                                // Update gripper to red and make it visible
                                if (child.userData.gripperGroup) {
                                    child.userData.gripperGroup.visible = true;
                                    if (child.userData.gripperMaterial) {
                                        child.userData.gripperMaterial.color.setHex(GRIPPER_COLOR_SELECTED);
                                    }
                                    child.userData.gripperGroup.traverse((gripperChild) => {
                                        if (gripperChild.material) {
                                            gripperChild.material.color.setHex(GRIPPER_COLOR_SELECTED);
                                        }
                                    });
                                }
                            } else {
                                // Hide all other candidates (including from same grasp point)
                                child.visible = false;
                            }
                        } else if (selectedGraspPointId !== null) {
                            // A grasp point is selected (but no candidate selected)
                            // Show all candidates from selected grasp point, hide others
                            if (graspPointId === selectedGraspPointId) {
                                // Show all candidates from selected grasp point
                                child.visible = true;
                                child.material.opacity = 0.9;
                                
                                // Update gripper color - orange for all (no candidate selected)
                                if (child.userData.gripperGroup) {
                                    child.userData.gripperGroup.visible = true;
                                    const gripperColor = GRIPPER_COLOR_DEFAULT;
                                    
                                    // Update gripper material color
                                    if (child.userData.gripperMaterial) {
                                        child.userData.gripperMaterial.color.setHex(gripperColor);
                                    }
                                    
                                    // Update all gripper children colors
                                    child.userData.gripperGroup.traverse((gripperChild) => {
                                        if (gripperChild.material) {
                                            gripperChild.material.color.setHex(gripperColor);
                                        }
                                    });
                                }
                            } else {
                                // Hide candidates from other grasp points
                                child.visible = false;
                            }
                        } else {
                            // Nothing selected: show all with default settings
                            child.visible = true;
                            child.material.opacity = 0.9;
                            
                            if (child.userData.gripperGroup) {
                                child.userData.gripperGroup.visible = true;
                                if (child.userData.gripperMaterial) {
                                    child.userData.gripperMaterial.color.setHex(GRIPPER_COLOR_DEFAULT);
                                }
                                child.userData.gripperGroup.traverse((gripperChild) => {
                                    if (gripperChild.material) {
                                        gripperChild.material.color.setHex(GRIPPER_COLOR_DEFAULT);
                                    }
                                });
                            }
                        }
                    }
                });
            }
            
            function generateId() {
                return Math.random().toString(36).substr(2, 9);
            }
            
            function onExecuteButtonClick() {
                // Execute grasp when button is clicked
                if (selectedGraspId && currentGraspData) {
                    executeGrasp(currentGraspData.object_name, selectedGraspId);
                }
            }
            
            async function executeGrasp(objectName, graspId) {
                try {
                    updateStatus(`Executing grasp for ${objectName} at grasp point ${graspId}...`);
                    
                    const response = await fetch('/api/execute-grasp', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            object_name: objectName,
                            grasp_id: graspId,
                            topic: '/objects_poses_sim',
                            movement_duration: 10.0
                        })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    updateStatus(`✅ ${result.message}`);
                    console.log('Grasp execution started:', result);
                } catch (error) {
                    console.error('Error executing grasp:', error);
                    updateStatus(`❌ Error: ${error.message}`);
                }
            }
            
            function onOpenGripperClick() {
                controlGripper('open');
            }
            
            function onCloseGripperClick() {
                controlGripper('close');
            }
            
            async function controlGripper(command) {
                try {
                    updateStatus(`${command === 'open' ? 'Opening' : 'Closing'} gripper...`);
                    const response = await fetch('/api/gripper-command', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ command: command })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    updateStatus(`✅ ${result.message}`);
                    console.log(`Gripper ${command} command sent:`, result);
                } catch (error) {
                    console.error(`Error ${command === 'open' ? 'opening' : 'closing'} gripper:`, error);
                    updateStatus(`❌ Error: ${error.message}`);
                }
            }
            
            function onMoveToSafeHeightClick() {
                moveToSafeHeight();
            }
            
            function onMoveHomeClick() {
                moveHome();
            }
            
            async function moveToSafeHeight() {
                try {
                    updateStatus('Moving to safe height...');
                    const response = await fetch('/api/move-to-safe-height', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    updateStatus(`✅ ${result.message}`);
                    console.log('Move to safe height command sent:', result);
                } catch (error) {
                    console.error('Error moving to safe height:', error);
                    updateStatus(`❌ Error: ${error.message}`);
                }
            }
            
            async function moveHome() {
                try {
                    updateStatus('Moving to home position...');
                    const response = await fetch('/api/move-home', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    updateStatus(`✅ ${result.message}`);
                    console.log('Move home command sent:', result);
                } catch (error) {
                    console.error('Error moving home:', error);
                    updateStatus(`❌ Error: ${error.message}`);
                }
            }
            
            function updateGraspInfo(candidateData) {
                const infoDiv = document.getElementById('graspInfo');
                
                if (!candidateData) {
                    infoDiv.innerHTML = `
                        <div class="info-item">
                            <strong>No grasp candidate selected</strong>
                            <span>Select a grasp candidate to view details</span>
                        </div>
                    `;
                    return;
                }
                
                const pos = candidateData.position;
                const approachQuat = candidateData.approach_quaternion;
                const approachVec = candidateData.approach_vector || { x: 0, y: 0, z: 1 };
                
                let approachInfo = '';
                if (approachQuat) {
                    approachInfo = `
                        <div class="info-item">
                            <strong>Approach Quaternion:</strong>
                            <span>X: ${approachQuat.x.toFixed(4)}, Y: ${approachQuat.y.toFixed(4)}, Z: ${approachQuat.z.toFixed(4)}, W: ${approachQuat.w.toFixed(4)}</span>
                        </div>
                    `;
                }
                if (approachVec) {
                    approachInfo += `
                        <div class="info-item">
                            <strong>Approach Vector:</strong>
                            <span>X: ${approachVec.x.toFixed(3)}, Y: ${approachVec.y.toFixed(3)}, Z: ${approachVec.z.toFixed(3)}</span>
                        </div>
                    `;
                }
                
                infoDiv.innerHTML = `
                    <div class="info-item">
                        <strong>Grasp Point ID:</strong>
                        <span>${candidateData.grasp_point_id || candidateData.id}</span>
                    </div>
                    <div class="info-item">
                        <strong>Direction ID:</strong>
                        <span>${candidateData.direction_id || 'N/A'}</span>
                    </div>
                    <div class="info-item">
                        <strong>Position:</strong>
                        <span>X: ${pos.x.toFixed(4)}m, Y: ${pos.y.toFixed(4)}m, Z: ${pos.z.toFixed(4)}m</span>
                    </div>
                    ${approachInfo}
                    <div class="info-item">
                        <strong>Type:</strong>
                        <span>${candidateData.type || 'center_point'}</span>
                    </div>
                `;
            }
            
            function clearScene() {
                clearGraspPoints();
                currentObjectData = null;
                currentGraspData = null;
                currentGraspCandidates = null;
                selectedGraspPointId = null;
                selectedGraspId = null;
                
                document.getElementById('graspSelect').innerHTML = '<option value="">-- Select object first --</option>';
                document.getElementById('graspSelect').disabled = true;
                updateGraspInfo(null);
            }
            
            function updateStatus(message) {
                document.getElementById('statusBar').textContent = message;
            }
            
            // Initialize when page loads
            window.addEventListener('load', initScene);
        </script>
    </body>
    </html>
    """


@app.get("/api/objects")
async def get_objects():
    """List available objects from grasp JSON files."""
    objects = []
    
    if not GRASP_DIR.exists():
        return objects
    
    for grasp_file in GRASP_DIR.glob("*_grasp_points_all_markers.json"):
        # Extract object name from filename
        # e.g., "fork_orange_scaled70_grasp_points_all_markers.json" -> "fork_orange_scaled70"
        object_name = grasp_file.stem.replace("_grasp_points_all_markers", "")
        objects.append(object_name)
    
    return sorted(objects)


@app.get("/api/grasp-data/{object_name}")
async def get_grasp_data(object_name: str):
    """Get grasp points data for an object (for positions only)."""
    grasp_file = GRASP_DIR / f"{object_name}_grasp_points_all_markers.json"
    
    if not grasp_file.exists():
        raise HTTPException(status_code=404, detail=f"Grasp data not found for {object_name}")
    
    try:
        with open(grasp_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grasp data: {str(e)}")


@app.get("/api/grasp-candidates/{object_name}")
async def get_grasp_candidates(object_name: str):
    """Get grasp candidates data for an object."""
    candidates_file = GRASP_CANDIDATES_DIR / f"{object_name}_grasp_candidates.json"
    
    if not candidates_file.exists():
        raise HTTPException(status_code=404, detail=f"Grasp candidates not found for {object_name}")
    
    try:
        with open(candidates_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grasp candidates: {str(e)}")


@app.get("/api/aruco/{object_name}")
async def get_aruco_data(object_name: str):
    """Get ArUco marker data for an object."""
    aruco_file = ARUCO_DIR / f"{object_name}_aruco.json"
    
    if not aruco_file.exists():
        return JSONResponse(
            content={"error": f"ArUco data not found for {object_name}"},
            status_code=404
        )
    
    try:
        with open(aruco_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ArUco data: {str(e)}")


@app.get("/api/wireframe/{object_name}")
async def get_wireframe(object_name: str):
    """Get wireframe data for object visualization."""
    wireframe_file = WIREFRAME_DIR / f"{object_name}_wireframe.json"
    
    if not wireframe_file.exists():
        raise HTTPException(status_code=404, detail=f"Wireframe data not found for {object_name}")
    
    try:
        with open(wireframe_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading wireframe data: {str(e)}")


class ExecuteGraspRequest(BaseModel):
    object_name: str
    grasp_id: str  # Changed to str to support format "grasp_point_id_direction_id"
    topic: str = '/objects_poses_sim'
    movement_duration: float = 10.0
    grasp_point_id: Optional[int] = None  # New: explicit grasp point ID
    direction_id: Optional[int] = None  # New: explicit direction ID


@app.post("/api/execute-grasp")
async def execute_grasp(request: ExecuteGraspRequest):
    """Execute visual servo grasp for the specified object and grasp candidate."""
    try:
        object_name = request.object_name
        grasp_id = request.grasp_id  # Format: "grasp_point_id_direction_id" or legacy integer
        topic = request.topic
        movement_duration = request.movement_duration
        grasp_point_id = request.grasp_point_id
        direction_id = request.direction_id
        
        if not object_name:
            raise HTTPException(status_code=400, detail="object_name is required")
        
        # Parse grasp_id if grasp_point_id and direction_id are not explicitly provided
        if grasp_point_id is None or direction_id is None:
            if grasp_id:
                # Try to parse grasp_id as "grasp_point_id_direction_id" format
                if '_' in str(grasp_id):
                    parts = str(grasp_id).split('_')
                    if len(parts) >= 2:
                        try:
                            grasp_point_id = int(parts[0])
                            direction_id = int(parts[1])
                        except ValueError:
                            # Fallback: try to parse as legacy integer grasp_id
                            try:
                                legacy_grasp_id = int(grasp_id)
                                # For legacy mode, we'll use grasp-id parameter
                                grasp_point_id = None
                                direction_id = None
                            except ValueError:
                                raise HTTPException(status_code=400, detail=f"Invalid grasp_id format: {grasp_id}. Expected 'grasp_point_id_direction_id' or integer")
                else:
                    # Try to parse as legacy integer grasp_id
                    try:
                        legacy_grasp_id = int(grasp_id)
                        # For legacy mode, we'll use grasp-id parameter
                        grasp_point_id = None
                        direction_id = None
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid grasp_id format: {grasp_id}. Expected 'grasp_point_id_direction_id' or integer")
            else:
                raise HTTPException(status_code=400, detail="grasp_id or both grasp_point_id and direction_id are required")
        
        # Get the script path
        script_dir = Path(__file__).parent
        script_path = script_dir / "visual_servo_grasp.py"
        
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")
        
        # Map object name from JSON format (fork_yellow_scaled70) to topic format (fork_yellow)
        # Remove _scaled70 suffix if present
        topic_object_name = object_name.replace('_scaled70', '')
        
        # Build command with bash to source ROS2 and use python3.10
        # Use new grasp candidate parameters if available, otherwise fall back to legacy grasp-id
        cmd_parts = [
            "source /opt/ros/humble/setup.bash &&",
            f"python3.10 {script_path}",
            f"--object-name {topic_object_name}",
            f"--topic {topic}",
            f"--movement-duration {movement_duration}"
        ]
        
        if grasp_point_id is not None and direction_id is not None:
            # Use new grasp candidate mode
            cmd_parts.extend([
                f"--grasp-point-id {grasp_point_id}",
                f"--direction-id {direction_id}"
            ])
        else:
            # Use legacy mode with grasp-id
            cmd_parts.append(f"--grasp-id {grasp_id}")
        
        cmd_str = " ".join(cmd_parts)
        
        # Execute in background (non-blocking) using bash
        print(f"🔧 Executing command: {cmd_str}")
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            cwd=str(script_dir.parent.parent),
            shell=True,
            executable='/bin/bash',
            text=True,  # Return strings instead of bytes
            bufsize=1  # Line buffered
        )
        
        # Start a thread to read and log output
        import threading
        
        def log_output(pipe, process_pid):
            """Read output from process and log it."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.strip()
                        if line:
                            print(f"[PID {process_pid}] {line}")
                pipe.close()
            except Exception as e:
                print(f"Error reading process output: {e}")
        
        # Start thread to log output
        output_thread = threading.Thread(
            target=log_output,
            args=(process.stdout, process.pid),
            daemon=True
        )
        output_thread.start()
        
        # Wait a moment to check if process started successfully
        import time
        time.sleep(0.5)
        
        # Check if process is still running or if it exited with error
        return_code = process.poll()
        if return_code is not None:
            # Process exited immediately - read any error output
            error_output = ""
            try:
                if process.stdout:
                    remaining = process.stdout.read()
                    if remaining:
                        error_output = remaining.strip()
            except:
                pass
            
            error_msg = f"Process exited immediately with code {return_code}"
            if error_output:
                error_msg += f". Output: {error_output[:500]}"  # Limit error message length
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"✅ Process started successfully with PID: {process.pid}")
        
        # Create message based on mode
        if grasp_point_id is not None and direction_id is not None:
            message = f"Executing grasp for {topic_object_name} at grasp_point_id {grasp_point_id}, direction_id {direction_id}"
        else:
            message = f"Executing grasp for {topic_object_name} at grasp point {grasp_id} (legacy mode)"
        
        return JSONResponse(content={
            "status": "started",
            "message": message,
            "pid": process.pid,
            "grasp_point_id": grasp_point_id,
            "direction_id": direction_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing grasp: {str(e)}")


class GripperCommandRequest(BaseModel):
    command: str  # 'open' or 'close'


@app.post("/api/gripper-command")
async def gripper_command(request: GripperCommandRequest):
    """Send gripper command (open/close) via ROS2 topic."""
    try:
        command = request.command.lower()
        
        if command not in ['open', 'close']:
            raise HTTPException(status_code=400, detail="Command must be 'open' or 'close'")
        
        # Build ROS2 topic pub command
        cmd_str = (
            f"source /opt/ros/humble/setup.bash && "
            f"ros2 topic pub --once /gripper_command std_msgs/String \"{{data: '{command}'}}\""
        )
        
        # Execute command
        print(f"🔧 Executing gripper command: {cmd_str}")
        result = subprocess.run(
            cmd_str,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=5.0
        )
        
        if result.returncode != 0:
            error_msg = f"Gripper command failed with code {result.returncode}"
            if result.stderr:
                error_msg += f". Error: {result.stderr[:200]}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"✅ Gripper command '{command}' sent successfully")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Gripper {command} command sent successfully",
            "command": command
        })
        
    except subprocess.TimeoutExpired:
        error_msg = "Gripper command timed out"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error sending gripper command: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/move-to-safe-height")
async def move_to_safe_height():
    """Execute move_to_safe_height.py script."""
    try:
        # Get the script path
        script_dir = Path(__file__).parent
        script_path = script_dir / "move_to_safe_height.py"
        
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")
        
        # Build command with bash to source ROS2 and use python3.10
        cmd_str = (
            f"source /opt/ros/humble/setup.bash && "
            f"python3.10 {script_path}"
        )
        
        # Execute in background (non-blocking) using bash
        print(f"🔧 Executing command: {cmd_str}")
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir.parent.parent),
            shell=True,
            executable='/bin/bash',
            text=True,
            bufsize=1
        )
        
        # Start a thread to read and log output
        import threading
        
        def log_output(pipe, process_pid):
            """Read output from process and log it."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.strip()
                        if line:
                            print(f"[PID {process_pid}] {line}")
                pipe.close()
            except Exception as e:
                print(f"Error reading process output: {e}")
        
        # Start thread to log output
        output_thread = threading.Thread(
            target=log_output,
            args=(process.stdout, process.pid),
            daemon=True
        )
        output_thread.start()
        
        # Wait a moment to check if process started successfully
        import time
        time.sleep(0.5)
        
        # Check if process is still running or if it exited with error
        return_code = process.poll()
        if return_code is not None:
            error_output = ""
            try:
                if process.stdout:
                    remaining = process.stdout.read()
                    if remaining:
                        error_output = remaining.strip()
            except:
                pass
            
            error_msg = f"Process exited immediately with code {return_code}"
            if error_output:
                error_msg += f". Output: {error_output[:500]}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"✅ Process started successfully with PID: {process.pid}")
        
        return JSONResponse(content={
            "status": "started",
            "message": "Moving to safe height",
            "pid": process.pid
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing move to safe height: {str(e)}")


@app.post("/api/move-home")
async def move_home():
    """Execute move_home.py script."""
    try:
        # Get the script path
        script_dir = Path(__file__).parent
        script_path = script_dir / "move_home.py"
        
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")
        
        # Build command with bash to source ROS2 and use python3.10
        cmd_str = (
            f"source /opt/ros/humble/setup.bash && "
            f"python3.10 {script_path}"
        )
        
        # Execute in background (non-blocking) using bash
        print(f"🔧 Executing command: {cmd_str}")
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir.parent.parent),
            shell=True,
            executable='/bin/bash',
            text=True,
            bufsize=1
        )
        
        # Start a thread to read and log output
        import threading
        
        def log_output(pipe, process_pid):
            """Read output from process and log it."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.strip()
                        if line:
                            print(f"[PID {process_pid}] {line}")
                pipe.close()
            except Exception as e:
                print(f"Error reading process output: {e}")
        
        # Start thread to log output
        output_thread = threading.Thread(
            target=log_output,
            args=(process.stdout, process.pid),
            daemon=True
        )
        output_thread.start()
        
        # Wait a moment to check if process started successfully
        import time
        time.sleep(0.5)
        
        # Check if process is still running or if it exited with error
        return_code = process.poll()
        if return_code is not None:
            error_output = ""
            try:
                if process.stdout:
                    remaining = process.stdout.read()
                    if remaining:
                        error_output = remaining.strip()
            except:
                pass
            
            error_msg = f"Process exited immediately with code {return_code}"
            if error_output:
                error_msg += f". Output: {error_output[:500]}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"✅ Process started successfully with PID: {process.pid}")
        
        return JSONResponse(content={
            "status": "started",
            "message": "Moving to home position",
            "pid": process.pid
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing move home: {str(e)}")


def main():
    """Main entry point."""
    print("🚀 Starting Grasp Candidates Visualizer...")
    print("📱 Open your browser to: http://localhost:8002")
    print("🎯 Features:")
    print("   • Visualize objects with grasp points overlayed")
    print("   • Select objects and grasp points")
    print("   • Interactive 3D viewer with full rotation/zoom/pan")
    print("   • View grasp point details and positions")
    print("🎮 Controls:")
    print("   • Mouse wheel: zoom")
    print("   • Right-click + drag: rotate")
    print("   • Middle-click + drag: pan")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)


if __name__ == "__main__":
    main()
