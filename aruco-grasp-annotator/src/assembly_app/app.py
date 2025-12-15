#!/usr/bin/env python3
"""
Enhanced Interactive 3D Assembly Web Application v2.0
Complete implementation with precision controls for component assembly
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import os
import socket
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
import numpy as np

app = FastAPI(
    title="Enhanced 3D Assembly App v2.0",
    description="Interactive 3D component assembly tool with precision controls",
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

# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "data"

def get_available_components():
    """Dynamically discover components from wireframe directory."""
    wireframe_dir = DATA_DIR / "wireframe"
    components = []
    
    if wireframe_dir.exists():
        for wireframe_file in wireframe_dir.glob("*_wireframe.json"):
            # Extract component name by removing _wireframe.json suffix
            component_name = wireframe_file.stem.replace("_wireframe", "")
            components.append(component_name)
    
    return sorted(components)

# Global assembly state
assembly_state = {
    "components": {},
    "assemblies": [],
    "next_id": 1
}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the complete enhanced assembly interface with precision controls."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced 3D Assembly App v2.0</title>
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
                width: 500px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 20px;
                overflow-y: auto;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                z-index: 10;
                position: relative;
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
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .component-list {
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            
            .component-item {
                padding: 12px;
                border-bottom: 1px solid #eee;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .component-item:hover {
                background: #f8f9fa;
                transform: translateX(2px);
            }
            
            .component-item:last-child {
                border-bottom: none;
            }
            
            .component-name {
                font-weight: 500;
                color: #2c3e50;
            }
            
            .component-status {
                font-size: 12px;
                padding: 3px 8px;
                border-radius: 12px;
                background: #e0e0e0;
                color: #666;
            }
            
            .component-status.loaded {
                background: #27ae60;
                color: white;
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
                min-width: 100px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
            }
            
            .btn:active {
                transform: translateY(0);
            }
            
            .btn-secondary {
                background: linear-gradient(45deg, #e74c3c, #c0392b);
            }
            
            .btn-secondary:hover {
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
            }
            
            .btn-small {
                padding: 6px 12px;
                font-size: 12px;
                min-width: 60px;
            }
            
            .assembly-controls {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .assembly-controls .btn {
                width: 100%;
                margin: 0;
            }
            
            .precision-controls {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                border: 2px solid #3498db;
            }
            
            .control-group {
                margin-bottom: 20px;
            }
            
            .control-group h4 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .control-row {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                gap: 8px;
                min-width: 0;
            }
            
            .control-label {
                min-width: 20px;
                font-weight: 500;
                color: #34495e;
                font-size: 13px;
            }
            
            .control-input {
                flex: 1;
                padding: 6px 10px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 13px;
                background: white;
            }
            
            .control-input:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            
            .input-group {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 8px;
            }
            
            .input-group label {
                min-width: 60px;
                font-size: 13px;
                color: #555;
                font-weight: 500;
            }
            
            .control-buttons {
                display: flex;
                gap: 4px;
                flex-shrink: 0;
            }
            
            .selected-object {
                background: rgba(52, 152, 219, 0.1);
                border: 2px solid #3498db;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
            }
            
            .selected-object h4 {
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .selected-object p {
                color: #7f8c8d;
                font-size: 12px;
                margin: 0;
            }
            
            .status-bar {
                background: rgba(44, 62, 80, 0.95);
                color: white;
                padding: 12px 20px;
                font-size: 14px;
                backdrop-filter: blur(10px);
                border-top: 1px solid #34495e;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .status-info {
                display: flex;
                gap: 20px;
                font-size: 13px;
            }
            
            .instructions {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                font-size: 13px;
                line-height: 1.4;
                color: #555;
                border-left: 4px solid #3498db;
            }
            
            .instructions h4 {
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .instructions ul {
                margin-left: 15px;
                margin-top: 5px;
            }
            
            .instructions li {
                margin-bottom: 3px;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .error {
                background: #fdf2f2;
                color: #e53e3e;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 4px solid #e53e3e;
                font-size: 13px;
            }
            
            .success {
                background: #f0fff4;
                color: #38a169;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 4px solid #38a169;
                font-size: 13px;
            }
            
            .info {
                background: #ebf8ff;
                color: #2b6cb0;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 4px solid #2b6cb0;
                font-size: 13px;
            }
            
            .scene-objects {
                max-height: 150px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            
            .scene-object-item {
                padding: 8px 12px;
                border-bottom: 1px solid #eee;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 13px;
            }
            
            .scene-object-item:hover {
                background: #f8f9fa;
            }
            
            .scene-object-item.selected {
                background: rgba(52, 152, 219, 0.1);
                border-left: 3px solid #3498db;
            }
            
            .scene-object-item:last-child {
                border-bottom: none;
            }
            
            .object-name {
                font-weight: 500;
                color: #2c3e50;
            }
            
            .object-type {
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 8px;
                background: #ecf0f1;
                color: #7f8c8d;
            }
            
            .quick-actions {
                display: flex;
                gap: 5px;
                margin-top: 10px;
            }
            
            .axis-controls {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 5px;
                margin-top: 8px;
            }
            
            .axis-btn {
                padding: 4px 8px;
                border: 1px solid #bdc3c7;
                background: white;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s;
            }
            
            .axis-btn:hover {
                background: #ecf0f1;
            }
            
            .axis-btn.x { border-color: #e74c3c; color: #e74c3c; }
            .axis-btn.y { border-color: #27ae60; color: #27ae60; }
            .axis-btn.z { border-color: #3498db; color: #3498db; }
            
            .floating-controls {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                backdrop-filter: blur(10px);
                display: none;
                z-index: 1000;
            }
            
            .floating-controls h4 {
                margin: 0 0 10px 0;
                color: #2c3e50;
            }
            
            .floating-controls .controls-grid {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .show-controls-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(52, 152, 219, 0.9);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 16px;
                width: 40px;
                height: 40px;
                z-index: 999;
                transition: all 0.2s;
            }
            
            .show-controls-btn:hover {
                background: rgba(52, 152, 219, 1);
                transform: scale(1.1);
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar sidebar-left">
                <div class="instructions">
                    <h4>Assembly Instructions</h4>
                    <ul>
                        <li><strong>Load:</strong> Click "Load All Components" to load parts</li>
                        <li><strong>Add:</strong> Click component names to add to scene</li>
                        <li><strong>Select:</strong> Click objects in scene or object list</li>
                        <li><strong>Control:</strong> Use precision controls for exact positioning</li>
                        <li><strong>Camera:</strong> Mouse wheel zoom, middle-click pan, right-drag orbit</li>
                        <li><strong>Shortcuts:</strong> Arrow keys move, Q/E rotate Y, Delete removes</li>
                    </ul>
                </div>
                
                <div class="controls-panel">
                    <h3>Components</h3>
                    <div class="assembly-controls">
                        <button class="btn" onclick="loadAllComponents()">Load All Components</button>
                        <button class="btn btn-secondary" onclick="clearScene()">Clear Scene</button>
                    </div>
                    <div id="componentList" class="component-list">
                        <div class="loading">Click "Load All Components" to start</div>
                    </div>
                </div>
                
                <div class="controls-panel">
                    <h3>Assembly Management</h3>
                    <div class="assembly-controls">
                        <input type="file" id="assemblyFileInput" accept=".json" style="display: none;" onchange="loadAssemblyFromFile(event)">
                        <button class="btn" onclick="document.getElementById('assemblyFileInput').click()">Load Assembly</button>
                        <button class="btn" onclick="exportAssembly()">Export Assembly</button>
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 8px;">
                        Load previously saved assembly configurations
                    </div>
                </div>
                
                <div class="controls-panel">
                    <h3>🎯 Grasp Points Visualization</h3>
                    <div class="assembly-controls">
                        <input type="file" id="graspFileInput" accept=".json" style="display: none;" onchange="loadGraspPointsFromFile(event)">
                        <button class="btn" onclick="loadGraspPointsAuto()">Load Grasp Points</button>
                        <button class="btn btn-secondary" onclick="clearGraspPoints()">Clear Grasp Points</button>
                    </div>
                    <div id="graspInfo" style="font-size: 12px; color: #666; margin-top: 8px;">
                        No grasp points loaded
                    </div>
                </div>
                
                <div id="statusMessages"></div>
            </div>
            
            <div class="main-viewer">
                <div class="viewer-container" id="viewer"></div>
                <div class="status-bar" id="statusBar">
                    <span>Ready - Load components to start assembling</span>
                    <div class="status-info">
                        <span id="objectCount">Objects: 0</span>
                        <span id="selectedInfo">Selected: None</span>
                    </div>
                </div>
            </div>
            
            <div class="sidebar sidebar-right">
                <div id="selectedObjectInfo">
                    <div class="controls-panel">
                        <h3>🎛️ Object Transform</h3>
                        <div class="loading" style="padding: 20px; text-align: center; color: #666;">
                            Select an object to view and edit its transform
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Floating control panel -->
        <div id="floatingControls" class="floating-controls">
            <h4>Assembly Tools</h4>
            <div class="controls-grid">
                <button class="btn btn-small" onclick="toggleGrid()">Toggle Grid</button>
                <button class="btn btn-small" onclick="resetCamera()">Reset Camera</button>
                <button class="btn btn-small" onclick="exportAssembly()">Export Assembly</button>
                <button class="btn btn-small" onclick="document.getElementById('assemblyFileInput').click()">Load Assembly</button>
                <button class="btn btn-small btn-secondary" onclick="hideFloatingControls()">Close</button>
            </div>
        </div>
        
        <button id="showControlsBtn" class="show-controls-btn" onclick="showFloatingControls()" title="Assembly Tools">⚙</button>

        <script>
            // Global variables
            let scene, camera, renderer, controls;
            let loadedComponents = {};
            let sceneObjects = [];
            let selectedObject = null;
            let gridVisible = true;
            let gridHelper, axesHelper;
            let graspPointsGroup = null;
            let graspPointsData = null;
            
            // Initialize the 3D scene
            function initScene() {
                const container = document.getElementById('viewer');
                
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                // Camera setup
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.001, 1000);
                camera.position.set(0.3, -0.3, 0.3);  // Move camera closer to scene
                camera.up.set(0, 0, 1);  // Set Z as up vector
                camera.lookAt(0, 0, 0);
                
                // Renderer setup
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                container.appendChild(renderer.domElement);
                
                // Orbit Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = false;  // Disable damping for immediate stop
                controls.minDistance = 0.05;
                controls.maxDistance = 5;
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, -10, 10);  // Light from above in Z-up system
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                scene.add(directionalLight);
                
                const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3);
                scene.add(hemisphereLight);
                
                // Grid - rotate to make it horizontal in XY plane (Z-up)
                gridHelper = new THREE.GridHelper(1, 1, 0x444444, 0x222222);
                gridHelper.rotateX(Math.PI / 2);  // Rotate 90 degrees to make it horizontal in XY plane
                scene.add(gridHelper);
                
                // Custom coordinate frame matching Blender convention (Z-up)
                // Blender: X=Right (Red), Y=Forward (Green), Z=Up (Blue)
                const axesLength = 0.2;
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
                axesHelper = new THREE.LineSegments(axesGeometry, axesMaterial);
                scene.add(axesHelper);
                
                // Event listeners
                setupEventListeners();
                
                // Start animation loop
                animate();
                
                updateStatus("Scene initialized - Ready to load components");
            }
            
            function setupEventListeners() {
                const canvas = renderer.domElement;
                
                // Prevent right-click context menu
                canvas.addEventListener('contextmenu', (e) => e.preventDefault());
                
                // Window resize
                window.addEventListener('resize', onWindowResize);
                
                // Keyboard shortcuts
                document.addEventListener('keydown', onKeyDown);
            }
            
            function onKeyDown(event) {
                if (!selectedObject) return;
                
                const step = event.shiftKey ? 0.1 : 0.01;
                const rotStep = event.shiftKey ? Math.PI / 18 : Math.PI / 36; // 10° or 5°
                
                switch(event.key) {
                    case 'ArrowLeft':
                        selectedObject.position.x -= step;
                        break;
                    case 'ArrowRight':
                        selectedObject.position.x += step;
                        break;
                    case 'ArrowUp':
                        selectedObject.position.y += step;  // Changed from z to y
                        break;
                    case 'ArrowDown':
                        selectedObject.position.y -= step;  // Changed from z to y
                        break;
                    case 'PageUp':
                        selectedObject.position.z += step;  // Now Z moves up/down
                        break;
                    case 'PageDown':
                        selectedObject.position.z -= step;  // Now Z moves up/down
                        break;
                    case 'q':
                    case 'Q':
                        selectedObject.rotation.y -= rotStep;
                        break;
                    case 'e':
                    case 'E':
                        selectedObject.rotation.y += rotStep;
                        break;
                    case 'Delete':
                        deleteSelectedObject();
                        break;
                }
                
                if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'q', 'e', 'Q', 'E'].includes(event.key)) {
                    event.preventDefault();
                    updateSelectedObjectInfo();
                }
            }
            
            function onWindowResize() {
                const container = document.getElementById('viewer');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
            
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            
            // Component loading functions
            async function loadAllComponents() {
                updateStatus("Loading all components...");
                showMessage("Loading components from server...", "info");
                
                try {
                    const response = await fetch('/api/components');
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`HTTP ${response.status}: ${errorText}`);
                    }
                    
                    const components = await response.json();
                    
                    if (Object.keys(components).length === 0) {
                        throw new Error("No components loaded. Check server logs for errors.");
                    }
                    
                    loadedComponents = components;
                    updateComponentList();
                    showMessage(`Loaded ${Object.keys(components).length} components successfully!`, "success");
                    updateStatus("All components loaded - Click on component names to add to scene");
                } catch (error) {
                    console.error("Error loading components:", error);
                    showMessage(`Error loading components: ${error.message}`, "error");
                    updateStatus("Error loading components - Check console for details");
                }
            }
            
            function updateComponentList() {
                const list = document.getElementById('componentList');
                list.innerHTML = '';
                
                Object.keys(loadedComponents).forEach(componentName => {
                    const component = loadedComponents[componentName];
                    const item = document.createElement('div');
                    item.className = 'component-item';
                    item.onclick = () => addComponentToScene(componentName);
                    
                    item.innerHTML = `
                        <span class="component-name">${component.display_name}</span>
                        <span class="component-status loaded">Ready</span>
                    `;
                    
                    list.appendChild(item);
                });
            }
            
            function addComponentToScene(componentName) {
                const component = loadedComponents[componentName];
                if (!component) return;
                
                // Check if this component already exists in the scene
                const existingObject = sceneObjects.find(obj => 
                    obj.userData && 
                    obj.userData.type === 'component' && 
                    obj.userData.name === componentName
                );
                
                if (existingObject) {
                    // Component already exists - just select it
                    selectObject(existingObject);
                    updateStatus(`Selected existing ${component.display_name} in scene`);
                    showMessage(`Selected ${component.display_name} (already in scene)`, "info");
                    return;
                }
                
                // Create wireframe geometry
                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(component.wireframe.vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                // Create edges
                const edges = component.wireframe.edges;
                const lineGeometry = new THREE.BufferGeometry();
                const lineVertices = [];
                
                edges.forEach(edge => {
                    const v1 = component.wireframe.vertices[edge[0]];
                    const v2 = component.wireframe.vertices[edge[1]];
                    lineVertices.push(...v1, ...v2);
                });
                
                lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
                
                // Create material
                const material = new THREE.LineBasicMaterial({ 
                    color: getComponentColor(componentName),
                    linewidth: 1
                });
                
                // Create mesh
                const wireframe = new THREE.LineSegments(lineGeometry, material);
                wireframe.userData = { 
                    name: componentName, 
                    type: 'component',
                    displayName: component.display_name,
                    originalColor: getComponentColor(componentName),
                    id: generateId()
                };
                
                // Position at origin (0, 0, 0)
                wireframe.position.set(0, 0, 0);
                
                scene.add(wireframe);
                sceneObjects.push(wireframe);
                
                // Add ArUco markers if available
                if (component.aruco && component.aruco.markers) {
                    component.aruco.markers.forEach((marker, index) => {
                        addArUcoMarker(marker, wireframe, index);
                    });
                }
                
                updateSceneObjectsList();
                updateStatus(`Added ${component.display_name} to scene`);
                showMessage(`Added ${component.display_name} to assembly`, "success");
                
                // Auto-select the new object
                selectObject(wireframe);
            }
            
            function addArUcoMarker(markerData, parentObject, index) {
                const size = markerData.size;
                const geometry = new THREE.BoxGeometry(size, size, size * 0.1);
                const material = new THREE.MeshBasicMaterial({ 
                    color: 0xff6b6b,
                    transparent: true,
                    opacity: 0.7
                });
                
                const marker = new THREE.Mesh(geometry, material);
                
                // Position relative to parent
                const pos = markerData.pose_absolute.position;
                marker.position.set(pos.x, pos.y, pos.z);
                
                // Apply rotation - convert from roll/pitch/yaw to Three.js Euler angles
                const rot = markerData.pose_absolute.rotation;
                marker.rotation.set(rot.roll, rot.pitch, rot.yaw);
                
                marker.userData = { 
                    name: `ArUco-${markerData.aruco_id}`, 
                    type: 'marker',
                    arucoId: markerData.aruco_id,
                    parentId: parentObject.userData.id,
                    displayName: `ArUco ${markerData.aruco_id}`,
                    originalColor: 0xff6b6b,
                    id: generateId()
                };
                
                // Add marker as child of parent object
                parentObject.add(marker);
                sceneObjects.push(marker);
            }
            
            function selectObject(object) {
                // Deselect previous object
                if (selectedObject && selectedObject.material) {
                    selectedObject.material.color.setHex(selectedObject.userData.originalColor);
                }
                
                selectedObject = object;
                
                if (selectedObject) {
                    // Highlight selected object
                    if (selectedObject.material) {
                        selectedObject.material.color.setHex(0xffff00); // Yellow highlight
                    }
                    updateSelectedObjectInfo();
                    updateStatus(`Selected: ${selectedObject.userData.displayName || selectedObject.userData.name}`);
                } else {
                    updateSelectedObjectInfo();
                    updateStatus("No object selected");
                }
                
                updateSceneObjectsList();
                updateStatusBar();
            }
            
            function toggleObjectVisibility(objectId, event) {
                event.stopPropagation(); // Prevent selection when clicking eye icon
                const targetObject = sceneObjects.find(obj => 
                    obj.userData && obj.userData.id === objectId
                );
                
                if (targetObject) {
                    // Toggle visibility
                    targetObject.visible = !targetObject.visible;
                    // Also hide/show children (ArUco markers, etc.)
                    targetObject.traverse((child) => {
                        child.visible = targetObject.visible;
                    });
                    
                    // Update lists
                    updateSceneObjectsList();
                    updateSelectedObjectInfo();
                }
            }
            window.toggleObjectVisibility = toggleObjectVisibility;
            
            function getSceneObjectsListHTML() {
                // Filter to only show component objects (not markers or grasp points)
                const componentObjects = sceneObjects.filter(obj => 
                    obj.userData && obj.userData.type === 'component'
                );
                
                if (componentObjects.length === 0) {
                    return '<div class="loading" style="padding: 10px; font-size: 12px; color: #666;">No objects in scene</div>';
                }
                
                let html = '<div class="scene-objects" style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; background: white;">';
                
                componentObjects.forEach(obj => {
                    const isSelected = obj === selectedObject;
                    const isVisible = obj.visible !== false; // Default to visible
                    const eyeIcon = isVisible ? '👁️' : '👁️‍🗨️';
                    const eyeStyle = isVisible ? 'opacity: 1;' : 'opacity: 0.4;';
                    html += `
                        <div class="scene-object-item ${isSelected ? 'selected' : ''}" 
                             onclick="selectObjectFromTransformPanel(this)" 
                             data-object-id="${obj.userData.id}"
                             style="cursor: pointer; padding: 8px 12px; border-bottom: 1px solid #eee; transition: all 0.2s; display: flex; align-items: center; justify-content: space-between; ${isSelected ? 'background: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db;' : ''}">
                            <div style="display: flex; align-items: center; gap: 8px; flex: 1;">
                                <span class="object-name" style="font-weight: 500; color: #2c3e50;">${obj.userData.displayName || obj.userData.name}</span>
                                <span class="object-type" style="font-size: 11px; padding: 2px 6px; border-radius: 8px; background: #ecf0f1; color: #7f8c8d;">${obj.userData.type}</span>
                            </div>
                            <button onclick="toggleObjectVisibility('${obj.userData.id}', event)" 
                                    style="background: none; border: none; cursor: pointer; font-size: 16px; padding: 4px 8px; ${eyeStyle} transition: opacity 0.2s; z-index: 10;"
                                    title="${isVisible ? 'Hide object' : 'Show object'}"
                                    onmouseover="this.style.opacity='1'"
                                    onmouseout="this.style.opacity='${isVisible ? '1' : '0.4'}'">
                                ${eyeIcon}
                            </button>
                        </div>
                    `;
                });
                
                html += '</div>';
                return html;
            }
            
            function getArUcoMarkersListHTML() {
                // Filter to only show ArUco markers whose parent component is visible
                const markerObjects = sceneObjects.filter(obj => {
                    if (!obj.userData || obj.userData.type !== 'marker') {
                        return false;
                    }
                    // Check if parent component is visible
                    const parentId = obj.userData.parentId;
                    if (parentId) {
                        const parentObject = sceneObjects.find(p => 
                            p.userData && p.userData.id === parentId
                        );
                        // Only show marker if parent exists and is visible
                        return parentObject && parentObject.visible !== false;
                    }
                    // If no parent, show it (shouldn't happen, but handle gracefully)
                    return true;
                });
                
                if (markerObjects.length === 0) {
                    return '<div class="loading" style="padding: 10px; font-size: 12px; color: #666;">No ArUco markers for visible objects</div>';
                }
                
                let html = '<div class="scene-objects" style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; background: white;">';
                
                markerObjects.forEach(obj => {
                    const isSelected = obj === selectedObject;
                    const markerName = obj.userData.displayName || `ArUco ${obj.userData.arucoId || 'Marker'}`;
                    html += `
                        <div class="scene-object-item ${isSelected ? 'selected' : ''}" 
                             onclick="selectObjectFromTransformPanel(this)" 
                             data-object-id="${obj.userData.id}"
                             style="cursor: pointer; padding: 8px 12px; border-bottom: 1px solid #eee; transition: all 0.2s; ${isSelected ? 'background: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db;' : ''}">
                            <span class="object-name" style="font-weight: 500; color: #2c3e50;">${markerName}</span>
                            <span class="object-type" style="font-size: 11px; padding: 2px 6px; border-radius: 8px; background: #ecf0f1; color: #7f8c8d;">${obj.userData.type}</span>
                        </div>
                    `;
                });
                
                html += '</div>';
                return html;
            }
            
            function selectObjectFromTransformPanel(element) {
                const objectId = element.getAttribute('data-object-id');
                const targetObject = sceneObjects.find(obj => 
                    obj.userData && obj.userData.id === objectId
                );
                
                if (targetObject) {
                    selectObject(targetObject);
                }
            }
            window.selectObjectFromTransformPanel = selectObjectFromTransformPanel;
            
            function updateSelectedObjectInfo() {
                const container = document.getElementById('selectedObjectInfo');
                
                if (!selectedObject) {
                    container.innerHTML = `
                        <div class="controls-panel">
                            <h3>🎛️ Object Transform</h3>
                            <div class="loading" style="padding: 20px; text-align: center; color: #666;">
                                Select an object to view and edit its transform
                            </div>
                            <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                                <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">Scene Objects</h4>
                                ${getSceneObjectsListHTML()}
                            </div>
                            <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                                <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">ArUco Markers</h4>
                                ${getArUcoMarkersListHTML()}
                            </div>
                        </div>
                    `;
                    return;
                }
                
                const obj = selectedObject;
                const pos = obj.position;
                const rot = obj.rotation;
                
                // Calculate quaternion from Euler angles
                const c1 = Math.cos(rot.x / 2);
                const c2 = Math.cos(rot.y / 2);
                const c3 = Math.cos(rot.z / 2);
                const s1 = Math.sin(rot.x / 2);
                const s2 = Math.sin(rot.y / 2);
                const s3 = Math.sin(rot.z / 2);
                const quat = {
                    x: s1 * c2 * c3 - c1 * s2 * s3,
                    y: c1 * s2 * c3 + s1 * c2 * s3,
                    z: c1 * c2 * s3 - s1 * s2 * c3,
                    w: c1 * c2 * c3 + s1 * s2 * s3
                };
                
                container.innerHTML = `
                    <div class="controls-panel">
                        <h3>🎛️ Object Transform</h3>
                        
                        <div class="selected-object" style="margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 4px;">
                            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">${obj.userData.displayName || obj.userData.name}</h4>
                            <p style="margin: 0; font-size: 12px; color: #666;">Type: ${obj.userData.type}</p>
                        </div>
                        
                        <h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #555;">Position (m)</h4>
                        <div class="input-group" style="margin-bottom: 8px;">
                            <label style="min-width: 30px;">X:</label>
                            <input type="number" id="objPosX" class="control-input" step="0.0001" value="${pos.x.toFixed(4)}" 
                                   onchange="setObjectPosition('x', parseFloat(this.value))">
                            <div class="control-buttons">
                                <button class="btn btn-small axis-btn x" onclick="moveObject('x', -0.01)">-</button>
                                <button class="btn btn-small axis-btn x" onclick="moveObject('x', 0.01)">+</button>
                            </div>
                        </div>
                        <div class="input-group" style="margin-bottom: 8px;">
                            <label style="min-width: 30px;">Y:</label>
                            <input type="number" id="objPosY" class="control-input" step="0.0001" value="${pos.y.toFixed(4)}" 
                                   onchange="setObjectPosition('y', parseFloat(this.value))">
                            <div class="control-buttons">
                                <button class="btn btn-small axis-btn y" onclick="moveObject('y', -0.01)">-</button>
                                <button class="btn btn-small axis-btn y" onclick="moveObject('y', 0.01)">+</button>
                            </div>
                        </div>
                        <div class="input-group" style="margin-bottom: 15px;">
                            <label style="min-width: 30px;">Z:</label>
                            <input type="number" id="objPosZ" class="control-input" step="0.0001" value="${pos.z.toFixed(4)}" 
                                   onchange="setObjectPosition('z', parseFloat(this.value))">
                            <div class="control-buttons">
                                <button class="btn btn-small axis-btn z" onclick="moveObject('z', -0.01)">-</button>
                                <button class="btn btn-small axis-btn z" onclick="moveObject('z', 0.01)">+</button>
                            </div>
                        </div>
                        
                        <h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #555;">Rotation (degrees)</h4>
                        <div class="input-group" style="margin-bottom: 8px;">
                            <label style="min-width: 60px;">Roll (X):</label>
                            <input type="number" id="objRotX" class="control-input" step="1" value="${(rot.x * 180 / Math.PI).toFixed(1)}" 
                                   onchange="setObjectRotation('x', this.value * Math.PI / 180)">
                            <div class="control-buttons">
                                <button class="btn btn-small axis-btn x" onclick="rotateObject('x', -Math.PI / 36)">-</button>
                                <button class="btn btn-small axis-btn x" onclick="rotateObject('x', Math.PI / 36)">+</button>
                            </div>
                        </div>
                        <div class="input-group" style="margin-bottom: 8px;">
                            <label style="min-width: 60px;">Pitch (Y):</label>
                            <input type="number" id="objRotY" class="control-input" step="1" value="${(rot.y * 180 / Math.PI).toFixed(1)}" 
                                   onchange="setObjectRotation('y', this.value * Math.PI / 180)">
                            <div class="control-buttons">
                                <button class="btn btn-small axis-btn y" onclick="rotateObject('y', -Math.PI / 36)">-</button>
                                <button class="btn btn-small axis-btn y" onclick="rotateObject('y', Math.PI / 36)">+</button>
                            </div>
                        </div>
                        <div class="input-group" style="margin-bottom: 15px;">
                            <label style="min-width: 60px;">Yaw (Z):</label>
                            <input type="number" id="objRotZ" class="control-input" step="1" value="${(rot.z * 180 / Math.PI).toFixed(1)}" 
                                   onchange="setObjectRotation('z', this.value * Math.PI / 180)">
                            <div class="control-buttons">
                                <button class="btn btn-small axis-btn z" onclick="rotateObject('z', -Math.PI / 36)">-</button>
                                <button class="btn btn-small axis-btn z" onclick="rotateObject('z', Math.PI / 36)">+</button>
                            </div>
                        </div>
                        
                        <button class="btn btn-secondary" onclick="resetObjectPosition(); resetObjectRotation();" style="margin-top: 10px; width: 100%;">Reset to Origin</button>
                        
                        <h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #555;">Current Pose</h4>
                        <div style="background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 12px; font-family: monospace; margin-bottom: 10px;">
                            <div><strong>Position:</strong> <span id="objCurrentPos">(${pos.x.toFixed(4)}, ${pos.y.toFixed(4)}, ${pos.z.toFixed(4)})</span></div>
                            <div style="margin-top: 5px;"><strong>RPY:</strong> <span id="objCurrentRPY">(${(rot.x * 180 / Math.PI).toFixed(1)}°, ${(rot.y * 180 / Math.PI).toFixed(1)}°, ${(rot.z * 180 / Math.PI).toFixed(1)}°)</span></div>
                            <div style="margin-top: 5px;"><strong>Quaternion:</strong> <span id="objCurrentQuat">(${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}, ${quat.w.toFixed(4)})</span></div>
                        </div>
                        
                        <div class="control-group" style="margin-top: 15px;">
                            <h4 style="font-size: 14px; margin-bottom: 8px;">Quick Actions</h4>
                            <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                                <button class="btn btn-small" onclick="resetObjectPosition()">Reset Position</button>
                                <button class="btn btn-small" onclick="resetObjectRotation()">Reset Rotation</button>
                                <button class="btn btn-small btn-secondary" onclick="deleteSelectedObject()" style="flex: 1;">Delete</button>
                            </div>
                        </div>
                        
                        <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                            <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">Scene Objects</h4>
                            ${getSceneObjectsListHTML()}
                        </div>
                        <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                            <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">ArUco Markers</h4>
                            ${getArUcoMarkersListHTML()}
                        </div>
                    </div>
                `;
            }
            
            function updateSceneObjectsList() {
                // Update the right sidebar lists (scene objects and ArUco markers)
                updateSelectedObjectInfo();
            }
            
            function updateStatusBar() {
                document.getElementById('objectCount').textContent = `Objects: ${sceneObjects.length}`;
                document.getElementById('selectedInfo').textContent = 
                    selectedObject ? `Selected: ${selectedObject.userData.displayName || selectedObject.userData.name}` : 'Selected: None';
            }
            
            // Precision control functions
            function setObjectPosition(axis, value) {
                if (!selectedObject) return;
                selectedObject.position[axis] = parseFloat(value);
                updateSelectedObjectInfo();
                updateCurrentPoseDisplay();
                updateStatus(`Set ${selectedObject.userData.displayName} ${axis.toUpperCase()} position: ${value}`);
            }
            
            function setObjectRotation(axis, value) {
                if (!selectedObject) return;
                selectedObject.rotation[axis] = parseFloat(value);
                updateSelectedObjectInfo();
                updateCurrentPoseDisplay();
                updateStatus(`Set ${selectedObject.userData.displayName} ${axis.toUpperCase()} rotation: ${(value * 180 / Math.PI).toFixed(1)}°`);
            }
            
            function moveObject(axis, delta) {
                if (!selectedObject) return;
                selectedObject.position[axis] += delta;
                updateSelectedObjectInfo();
                updateCurrentPoseDisplay();
                updateStatus(`Moved ${selectedObject.userData.displayName} ${axis.toUpperCase()}: ${selectedObject.position[axis].toFixed(4)}`);
            }
            window.moveObject = moveObject;
            
            function rotateObject(axis, delta) {
                if (!selectedObject) return;
                selectedObject.rotation[axis] += delta;
                updateSelectedObjectInfo();
                updateCurrentPoseDisplay();
                updateStatus(`Rotated ${selectedObject.userData.displayName} ${axis.toUpperCase()}: ${(selectedObject.rotation[axis] * 180 / Math.PI).toFixed(1)}°`);
            }
            window.rotateObject = rotateObject;
            
            function updateCurrentPoseDisplay() {
                if (!selectedObject) return;
                
                const pos = selectedObject.position;
                const rot = selectedObject.rotation;
                
                // Calculate quaternion from Euler angles
                const c1 = Math.cos(rot.x / 2);
                const c2 = Math.cos(rot.y / 2);
                const c3 = Math.cos(rot.z / 2);
                const s1 = Math.sin(rot.x / 2);
                const s2 = Math.sin(rot.y / 2);
                const s3 = Math.sin(rot.z / 2);
                const quat = {
                    x: s1 * c2 * c3 - c1 * s2 * s3,
                    y: c1 * s2 * c3 + s1 * c2 * s3,
                    z: c1 * c2 * s3 - s1 * s2 * c3,
                    w: c1 * c2 * c3 + s1 * s2 * s3
                };
                
                // Update display elements if they exist
                const posEl = document.getElementById('objCurrentPos');
                const rpyEl = document.getElementById('objCurrentRPY');
                const quatEl = document.getElementById('objCurrentQuat');
                
                if (posEl) posEl.textContent = `(${pos.x.toFixed(4)}, ${pos.y.toFixed(4)}, ${pos.z.toFixed(4)})`;
                if (rpyEl) rpyEl.textContent = `(${(rot.x * 180 / Math.PI).toFixed(1)}°, ${(rot.y * 180 / Math.PI).toFixed(1)}°, ${(rot.z * 180 / Math.PI).toFixed(1)}°)`;
                if (quatEl) quatEl.textContent = `(${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}, ${quat.w.toFixed(4)})`;
            }
            
            function resetObjectPosition() {
                if (!selectedObject) return;
                selectedObject.position.set(0, 0, 0);
                updateSelectedObjectInfo();
                updateCurrentPoseDisplay();
                updateStatus(`Reset position for ${selectedObject.userData.displayName}`);
            }
            window.resetObjectPosition = resetObjectPosition;
            
            function resetObjectRotation() {
                if (!selectedObject) return;
                selectedObject.rotation.set(0, 0, 0);
                updateSelectedObjectInfo();
                updateCurrentPoseDisplay();
                updateStatus(`Reset rotation for ${selectedObject.userData.displayName}`);
            }
            window.resetObjectRotation = resetObjectRotation;
            
            function deleteSelectedObject() {
                if (!selectedObject) return;
                
                const objectName = selectedObject.userData.displayName || selectedObject.userData.name;
                
                // Remove from scene
                scene.remove(selectedObject);
                
                // Remove from objects array
                const index = sceneObjects.indexOf(selectedObject);
                if (index > -1) {
                    sceneObjects.splice(index, 1);
                }
                
                // Clear selection
                selectedObject = null;
                
                updateSceneObjectsList();
                updateSelectedObjectInfo();
                updateStatusBar();
                updateStatus(`Deleted ${objectName}`);
                showMessage(`Deleted ${objectName}`, "info");
            }
            window.deleteSelectedObject = deleteSelectedObject;
            
            function getComponentColor(componentName) {
                const colors = {
                    'base_scaled70': 0x4CAF50,
                    'fork_orange_scaled70': 0xFF9800,
                    'fork_yellow_scaled70': 0xFFEB3B,
                    'line_brown_scaled70': 0x8D6E63,
                    'line_red_scaled70': 0xF44336
                };
                return colors[componentName] || 0x2196F3;
            }
            
            function generateId() {
                return Math.random().toString(36).substr(2, 9);
            }
            
            // Utility functions
            function clearScene() {
                sceneObjects.forEach(obj => {
                    scene.remove(obj);
                });
                sceneObjects = [];
                selectedObject = null;
                updateSceneObjectsList();
                updateSelectedObjectInfo();
                updateStatusBar();
                updateStatus("Scene cleared");
                showMessage("All components removed from scene", "info");
            }
            window.clearScene = clearScene;
            
            function toggleGrid() {
                gridVisible = !gridVisible;
                gridHelper.visible = gridVisible;
                axesHelper.visible = gridVisible;
                updateStatus(`Grid ${gridVisible ? 'shown' : 'hidden'}`);
            }
            window.toggleGrid = toggleGrid;
            
            function resetCamera() {
                camera.position.set(0.3, -0.3, 0.3);  // Closer to scene
                camera.up.set(0, 0, 1);  // Ensure Z is still up
                camera.lookAt(0, 0, 0);
                controls.reset();
                updateStatus("Camera reset to default position");
            }
            window.resetCamera = resetCamera;
            
            async function exportAssembly() {
                // Export assembly with both RPY (Euler) and quaternion rotations for each object
                // ArUco marker and wireframe data are loaded dynamically from data folders
                
                // Filter to only include components (exclude markers)
                const components = sceneObjects
                    .filter(obj => obj.userData.type === 'component')
                    .map(obj => {
                        const rot = obj.rotation;
                        
                        // Calculate quaternion from Euler angles (Three.js uses XYZ order)
                        const c1 = Math.cos(rot.x / 2);
                        const c2 = Math.cos(rot.y / 2);
                        const c3 = Math.cos(rot.z / 2);
                        const s1 = Math.sin(rot.x / 2);
                        const s2 = Math.sin(rot.y / 2);
                        const s3 = Math.sin(rot.z / 2);
                        const quat = {
                            x: s1 * c2 * c3 - c1 * s2 * s3,
                            y: c1 * s2 * c3 + s1 * c2 * s3,
                            z: c1 * c2 * s3 - s1 * s2 * c3,
                            w: c1 * c2 * c3 + s1 * s2 * s3
                        };
                        
                        return {
                            name: obj.userData.name,
                            position: {
                                x: obj.position.x,
                                y: obj.position.y,
                                z: obj.position.z
                            },
                            rotation: {
                                rpy: {
                                    x: rot.x,
                                    y: rot.y,
                                    z: rot.z
                                },
                                quaternion: {
                                    x: quat.x,
                                    y: quat.y,
                                    z: quat.z,
                                    w: quat.w
                                }
                            }
                        };
                    });
                
                const assembly = {
                    timestamp: new Date().toISOString(),
                    components: components
                };
                
                const blob = new Blob([JSON.stringify(assembly, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `assembly_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                showMessage("Assembly exported successfully!", "success");
                updateStatus("Assembly exported to downloads");
            }
            window.exportAssembly = exportAssembly;
            
            async function loadAssemblyFromFile(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                try {
                    updateStatus("Loading assembly from file...");
                    showMessage("Reading assembly file...", "info");
                    
                    const text = await file.text();
                    const assemblyData = JSON.parse(text);
                    
                    // Validate assembly data structure
                    if (!assemblyData.components || !Array.isArray(assemblyData.components)) {
                        throw new Error("Invalid assembly file format - missing components array");
                    }
                    
                    // Clear current scene
                    clearScene();
                    
                    // Load components first if not already loaded
                    if (Object.keys(loadedComponents).length === 0) {
                        await loadAllComponents();
                    }
                    
                    // Restore assembly components
                    // Note: Simplified format only contains components (no markers)
                    let loadedCount = 0;
                    for (const componentData of assemblyData.components) {
                        await restoreComponentFromAssembly(componentData);
                        loadedCount++;
                    }
                    
                    updateSceneObjectsList();
                    updateStatusBar();
                    showMessage(`Successfully loaded assembly with ${loadedCount} components!`, "success");
                    updateStatus(`Assembly loaded: ${loadedCount} components restored`);
                    
                } catch (error) {
                    showMessage(`Error loading assembly: ${error.message}`, "error");
                    updateStatus("Error loading assembly file");
                    console.error("Assembly loading error:", error);
                }
                
                // Reset file input
                event.target.value = '';
            }
            
            async function restoreComponentFromAssembly(componentData) {
                const componentName = componentData.name;
                const component = loadedComponents[componentName];
                
                if (!component) {
                    console.warn(`Component ${componentName} not found in loaded components`);
                    return;
                }
                
                // Create wireframe geometry (same as addComponentToScene)
                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(component.wireframe.vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                // Create edges
                const edges = component.wireframe.edges;
                const lineGeometry = new THREE.BufferGeometry();
                const lineVertices = [];
                
                edges.forEach(edge => {
                    const v1 = component.wireframe.vertices[edge[0]];
                    const v2 = component.wireframe.vertices[edge[1]];
                    lineVertices.push(...v1, ...v2);
                });
                
                lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
                
                // Create material
                const material = new THREE.LineBasicMaterial({ 
                    color: getComponentColor(componentName),
                    linewidth: 1
                });
                
                // Create mesh
                const wireframe = new THREE.LineSegments(lineGeometry, material);
                wireframe.userData = { 
                    name: componentName, 
                    type: 'component',
                    displayName: component.display_name,
                    originalColor: getComponentColor(componentName),
                    id: generateId()
                };
                
                // Restore position from assembly data
                wireframe.position.set(
                    componentData.position.x,
                    componentData.position.y,
                    componentData.position.z
                );
                
                // Restore rotation from assembly data
                // STRICT: require new format rotation.rpy.{x,y,z}
                if (!componentData.rotation || !componentData.rotation.rpy) {
                    throw new Error(`Assembly component ${componentName} is missing rotation.rpy (expected format: rotation.rpy.{x,y,z})`);
                }
                
                const rotX = componentData.rotation.rpy.x;
                const rotY = componentData.rotation.rpy.y;
                const rotZ = componentData.rotation.rpy.z;
                
                wireframe.rotation.set(rotX, rotY, rotZ);
                
                scene.add(wireframe);
                sceneObjects.push(wireframe);
                
                // Add ArUco markers if available
                if (component.aruco && component.aruco.markers) {
                    component.aruco.markers.forEach((marker, index) => {
                        addArUcoMarker(marker, wireframe, index);
                    });
                }
            }
            
            function showFloatingControls() {
                document.getElementById('floatingControls').style.display = 'block';
                document.getElementById('showControlsBtn').style.display = 'none';
            }
            window.showFloatingControls = showFloatingControls;
            
            function hideFloatingControls() {
                document.getElementById('floatingControls').style.display = 'none';
                document.getElementById('showControlsBtn').style.display = 'block';
            }
            window.hideFloatingControls = hideFloatingControls;
            
            function updateStatus(message) {
                const statusBar = document.getElementById('statusBar');
                const statusSpan = statusBar.querySelector('span');
                statusSpan.textContent = message;
            }
            
            function showMessage(message, type) {
                const container = document.getElementById('statusMessages');
                const div = document.createElement('div');
                div.className = type;
                div.textContent = message;
                container.appendChild(div);
                
                // Auto-remove after 4 seconds
                setTimeout(() => {
                    if (div.parentNode) {
                        div.parentNode.removeChild(div);
                    }
                }, 4000);
            }
            
            // Grasp Points Visualization Functions
            // Load grasp points for a specific target object, relative to its current pose.
            // If clearFirst=true, previous grasp visualization is cleared before attaching new points.
            async function loadGraspPointsFromData(tempData, targetObject, clearFirst = false) {
                if (!targetObject || !targetObject.userData || targetObject.userData.type !== 'component') {
                    throw new Error("No valid target object selected for grasp points (expected a component).");
                }
                
                // Validate grasp points data
                if (!tempData.markers || !Array.isArray(tempData.markers)) {
                    throw new Error("Invalid grasp points data - missing markers array");
                }
                
                if (!tempData.wireframe || !tempData.wireframe.vertices || !tempData.wireframe.edges) {
                    throw new Error("Invalid grasp points data - missing wireframe data");
                }
                
                if (!tempData.grasp_points || !Array.isArray(tempData.grasp_points)) {
                    throw new Error("Invalid grasp points data - missing grasp_points array");
                }
                
                // Optionally clear previous grasp points (only for first object in a batch)
                if (clearFirst) {
                    clearGraspPoints();
                }
                
                // Now set the validated data (last loaded data wins)
                graspPointsData = tempData;
                                
                // Add grasp points as children of the target object (relative to its pose)
                // Grasp points are stored once relative to CAD center (object local frame)
                let totalPoints = 0;
                graspPointsData.grasp_points.forEach((graspPoint, idx) => {
                    const sphere = createGraspPointSphere(graspPoint, idx);
                    // Parent sphere to the target object so it follows its pose
                    targetObject.add(sphere);
                    // Track in sceneObjects for selection/listing purposes
                    sceneObjects.push(sphere);
                    totalPoints++;
                });
                
                // Update info
                const objName = targetObject.userData.displayName || targetObject.userData.name;
                const infoText = `Loaded ${objName}: ${totalPoints} grasp points`;
                document.getElementById('graspInfo').innerHTML = `
                    <span style="color: #27ae60; font-weight: 500;">${infoText}</span>
                `;
                
                showMessage(infoText, "success");
                updateStatus(`Grasp visualization loaded for ${objName}`);
            }
            
            async function loadGraspPointsFromFile(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                try {
                    if (!selectedObject || !selectedObject.userData || selectedObject.userData.type !== 'component') {
                        showMessage("Please select a component in the scene before loading grasp points from file.", "info");
                        return;
                    }
                    
                    updateStatus("Loading grasp points from file...");
                    showMessage("Reading grasp points file...", "info");
                    
                    const text = await file.text();
                    const tempData = JSON.parse(text);
                    
                    // For manual file load, clear previous visualization first
                    await loadGraspPointsFromData(tempData, selectedObject, /* clearFirst */ true);
                } catch (error) {
                    showMessage(`Error loading grasp points: ${error.message}`, "error");
                    updateStatus("Error loading grasp points");
                    console.error("Grasp points loading error:", error);
                }
                
                // Reset file input
                event.target.value = '';
            }
            
            async function loadGraspPointsAuto() {
                try {
                    // Collect all component objects currently in the scene
                    const componentObjects = sceneObjects.filter(obj =>
                        obj.userData && obj.userData.type === 'component'
                    );
                    
                    if (componentObjects.length === 0) {
                        showMessage("No components in scene - load an assembly or add components first.", "info");
                        updateStatus("No components in scene - cannot load grasp points.");
                        return;
                    }
                    
                    updateStatus("Loading grasp points for all scene components from data directory...");
                    showMessage("Loading grasp points for all scene components from server...", "info");
                    
                    // Clear previous grasp visualization once at the start
                    clearGraspPoints();
                    
                    let anyLoaded = false;
                    
                    // Try to auto-load grasp points for each component, attaching them relative to each object's pose
                    for (const obj of componentObjects) {
                        const objectName = obj.userData.name;
                        try {
                            const response = await fetch(`/api/grasp-points/${objectName}`);
                            
                            if (!response.ok) {
                                const errorText = await response.text();
                                console.warn(`Grasp data not found for ${objectName}: ${errorText}`);
                                continue;
                            }
                            
                            const data = await response.json();
                            // Do not clear inside helper; we've already cleared once above
                            await loadGraspPointsFromData(data, obj, /* clearFirst */ false);
                            anyLoaded = true;
                        } catch (innerError) {
                            console.warn(`Grasp points auto-load error for ${objectName}:`, innerError);
                        }
                    }
                    
                    if (!anyLoaded) {
                        showMessage("No grasp data found in data/grasp for any scene components.", "info");
                        updateStatus("No grasp points loaded.");
                    } else {
                        updateStatus("Grasp points auto-loaded for available components.");
                    }
                } catch (error) {
                    console.error("Grasp points auto-load error:", error);
                    showMessage(`Error auto-loading grasp points: ${error.message}`, "error");
                    updateStatus("Error auto-loading grasp points.");
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
                    linewidth: 1
                });
                
                const wireframe = new THREE.LineSegments(geometry, material);
                wireframe.userData = { 
                    name: data.object_name,
                    type: 'grasp_wireframe',
                    displayName: data.display_name,
                    originalColor: 0x4a90e2,
                    id: generateId()
                };
                
                // NOTE: This function is kept for compatibility but is no longer used
                // for grasp visualization in the assembly app. Grasp points are now
                // attached directly to the selected component mesh.
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
                
                sceneObjects.push(marker);
                return marker;
            }
            
            function createGraspPointSphere(graspPoint, index) {
                // Create sphere geometry for grasp point
                // Note: All coordinates are in meters relative to CAD center
                const geometry = new THREE.SphereGeometry(0.003, 16, 16);  // 0.003 m = 3mm
                const material = new THREE.MeshPhongMaterial({
                    color: 0x00ff00,  // Green for grasp points
                    emissive: 0x00ff00,
                    emissiveIntensity: 0.5,
                    transparent: true,
                    opacity: 0.9
                });
                
                const sphere = new THREE.Mesh(geometry, material);
                
                // Set position from grasp point data (in meters, relative to CAD center)
                // Coordinate flip already applied during export
                const pos = graspPoint.position;
                sphere.position.set(pos.x, pos.y, pos.z);
                
                // Store metadata
                sphere.userData = {
                    name: `Grasp-${graspPoint.id}`,
                    type: 'grasp_point',
                    graspId: graspPoint.id,
                    displayName: `Grasp Point ${graspPoint.id}`,
                    originalColor: 0x00ff00,
                    id: generateId()
                };
                
                return sphere;
            }
            
            function clearGraspPoints() {
                // Remove all grasp-related objects (wireframes, markers, points) from scene and tracking
                const toRemove = sceneObjects.filter(obj =>
                    obj.userData &&
                    (
                        obj.userData.type === 'grasp_wireframe' ||
                        obj.userData.type === 'grasp_marker' ||
                        obj.userData.type === 'grasp_point'
                    )
                );
                
                toRemove.forEach(obj => {
                    if (obj.parent) {
                        obj.parent.remove(obj);
                    }
                    const index = sceneObjects.indexOf(obj);
                    if (index > -1) {
                        sceneObjects.splice(index, 1);
                    }
                });
                
                // Logical group is no longer used as a visual parent
                graspPointsGroup = null;
                graspPointsData = null;
                
                // Deselect if a grasp object was selected
                if (selectedObject && (
                    selectedObject.userData.type === 'grasp_wireframe' ||
                    selectedObject.userData.type === 'grasp_marker' ||
                    selectedObject.userData.type === 'grasp_point'
                )) {
                    selectedObject = null;
                }
                
                updateSceneObjectsList();
                document.getElementById('graspInfo').innerHTML = 'No grasp points loaded';
                updateStatus("Grasp points cleared");
                showMessage("Grasp points visualization cleared", "info");
            }
            window.clearGraspPoints = clearGraspPoints;
            window.loadAllComponents = loadAllComponents;
            
            // Initialize when page loads
            window.addEventListener('load', initScene);
        </script>
    </body>
    </html>
    """

def transform_aruco_data(aruco_raw: dict) -> dict:
    """Transform ArUco data from new format (T_object_to_marker) to frontend format (pose_absolute)."""
    if not aruco_raw:
        return None
    
    # Transform ArUco data to match frontend expectations
    # New format has T_object_to_marker, need to convert to pose_absolute
    if 'markers' not in aruco_raw or not isinstance(aruco_raw['markers'], list):
        raise ValueError(f"ArUco data must have a 'markers' array. Got keys: {list(aruco_raw.keys())}")
    
    if len(aruco_raw['markers']) == 0:
        raise ValueError("ArUco data has empty 'markers' array")
    
    transformed_markers = []
    default_size = aruco_raw.get('size', 0.021)  # Get size from top level
    
    for idx, marker in enumerate(aruco_raw['markers']):
        if not isinstance(marker, dict):
            raise ValueError(f"Marker at index {idx} is not a dictionary: {type(marker)}")
        
        if 'T_object_to_marker' not in marker:
            raise ValueError(f"Marker {marker.get('aruco_id', f'at index {idx}')} missing 'T_object_to_marker'. Marker keys: {list(marker.keys())}")
        
        if not isinstance(marker['T_object_to_marker'], dict):
            raise ValueError(f"Marker {marker.get('aruco_id', f'at index {idx}')} has invalid 'T_object_to_marker' type: {type(marker['T_object_to_marker'])}")
        
        # Transform T_object_to_marker to pose_absolute
        transformed_marker = {
            'aruco_id': marker.get('aruco_id'),
            'size': marker.get('size', default_size),  # Use marker size or default
            'face_type': marker.get('face_type'),
            'surface_normal': marker.get('surface_normal'),
        }
        
        # Convert T_object_to_marker to pose_absolute
        t_obj_to_marker = marker['T_object_to_marker']
        if 'position' not in t_obj_to_marker or 'rotation' not in t_obj_to_marker:
            raise ValueError(f"Marker {marker.get('aruco_id', f'at index {idx}')} T_object_to_marker missing 'position' or 'rotation'. Keys: {list(t_obj_to_marker.keys())}")
        
        transformed_marker['pose_absolute'] = {
            'position': t_obj_to_marker.get('position', {'x': 0, 'y': 0, 'z': 0}),
            'rotation': t_obj_to_marker.get('rotation', {'roll': 0, 'pitch': 0, 'yaw': 0})
        }
        
        transformed_markers.append(transformed_marker)
    
    return {
        'markers': transformed_markers,
        'aruco_dictionary': aruco_raw.get('aruco_dictionary', 'DICT_4X4_50'),
        'size': default_size,
        'border_width': aruco_raw.get('border_width', 0.05)
    }

@app.get("/api/components")
async def get_components():
    """Get all available components with their wireframe and ArUco data."""
    components = {}
    
    # Dynamically discover components from wireframe directory
    available_components = get_available_components()
    
    for component_name in available_components:
        try:
            # Load wireframe data
            wireframe_path = DATA_DIR / "wireframe" / f"{component_name}_wireframe.json"
            aruco_path = DATA_DIR / "aruco" / f"{component_name}_aruco.json"
            
            if wireframe_path.exists():
                with open(wireframe_path, 'r') as f:
                    wireframe_data = json.load(f)
                
                # Load ArUco data if available and transform to expected format
                aruco_data = None
                if aruco_path.exists():
                    try:
                        with open(aruco_path, 'r') as f:
                            aruco_raw = json.load(f)
                        aruco_data = transform_aruco_data(aruco_raw)
                    except Exception as e:
                        print(f"Error transforming ArUco data for {component_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue without ArUco data if transformation fails
                        aruco_data = None
                
                components[component_name] = {
                    "wireframe": wireframe_data,
                    "aruco": aruco_data,
                    "name": component_name,
                    "display_name": component_name.replace('_scaled70', '').replace('_', ' ').title()
                }
            else:
                print(f"Warning: Wireframe file not found for {component_name}")
                
        except Exception as e:
            print(f"Error loading component {component_name}: {e}")
            continue
    
    return components

@app.get("/api/components/{component_name}")
async def get_component(component_name: str):
    """Get a specific component's data."""
    available_components = get_available_components()
    if component_name not in available_components:
        raise HTTPException(status_code=404, detail="Component not found")
    
    try:
        wireframe_path = DATA_DIR / "wireframe" / f"{component_name}_wireframe.json"
        aruco_path = DATA_DIR / "aruco" / f"{component_name}_aruco.json"
        
        if not wireframe_path.exists():
            raise HTTPException(status_code=404, detail="Wireframe data not found")
        
        with open(wireframe_path, 'r') as f:
            wireframe_data = json.load(f)
        
        aruco_data = None
        if aruco_path.exists():
            with open(aruco_path, 'r') as f:
                aruco_raw = json.load(f)
            aruco_data = transform_aruco_data(aruco_raw)
        
        return {
            "wireframe": wireframe_data,
            "aruco": aruco_data,
            "name": component_name,
            "display_name": component_name.replace('_scaled70', '').replace('_', ' ').title()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/grasp-points/{object_name}")
async def get_grasp_points(object_name: str):
    """Get grasp points data for an object from the data/grasp directory."""
    grasp_file = DATA_DIR / "grasp" / f"{object_name}_grasp_points_all_markers.json"
    
    if not grasp_file.exists():
        raise HTTPException(status_code=404, detail=f"Grasp data not found for {object_name}")
    
    try:
        with open(grasp_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grasp data: {str(e)}")

@app.post("/api/assembly")
async def save_assembly(assembly_data: dict):
    """Save assembly configuration."""
    assembly_id = assembly_state["next_id"]
    assembly_state["next_id"] += 1
    
    assembly_state["assemblies"].append({
        "id": assembly_id,
        "data": assembly_data,
        "timestamp": "2024-01-01T12:00:00"  # In production, use actual timestamp
    })
    
    return {"assembly_id": assembly_id, "success": True}

@app.get("/api/assemblies")
async def get_assemblies():
    """Get all saved assemblies."""
    return assembly_state["assemblies"]

def find_available_port(start_port=8001, max_attempts=100):
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
    """Main entry point for the enhanced assembly application."""
    # Find an available port starting from 8001
    port = find_available_port(8001)
    
    print("🚀 Starting Enhanced 3D Assembly App v2.0...")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print("🎯 Features:")
    print("   • Load and display wireframe components")
    print("   • Precision position and rotation controls")
    print("   • ArUco marker visualization")
    print("   • Interactive assembly with selection")
    print("   • Export assembly configurations")
    print("🎮 Controls:")
    print("   • Click objects to select")
    print("   • Use control panel for precision movement")
    print("   • Arrow keys for position, Q/E for rotation")
    print("   • Mouse: wheel=zoom, right-drag=orbit")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
