# Simple Isaac Sim script to position assembly objects
# Run this in Isaac Sim script editor after loading the USD files

import omni.kit.commands
from pxr import Gf
import json
import os

# Load assembly data
ASSEMBLY_FILE_PATH = os.path.expanduser("~/Projects/aruco-grasp-annotator/data/fmb_assembly.json")

print("Loading assembly data...")
with open(ASSEMBLY_FILE_PATH, 'r') as f:
    assembly_data = json.load(f)

# Object path mappings
OBJECT_PATHS = {
    "base_scaled70": "/World/base/base/Body1",
    "fork_orange_scaled70": "/World/fork_orange/fork_orange/Body1", 
    "fork_yellow_scaled70": "/World/fork_yellow/fork_yellow/Body1",
    "line_brown_scaled70": "/World/line_brown/line_brown/Body1",
    "line_red_scaled70": "/World/line_red/line_red/Body1"
}

# Position each object
for component in assembly_data['components']:
    if component['type'] == 'component':
        name = component['name']
        position = component['position']
        rotation = component['rotation']
        
        if name in OBJECT_PATHS:
            prim_path = OBJECT_PATHS[name]
            
            print(f"Positioning {name}...")
            
            # Set position
            omni.kit.commands.execute('ChangeProperty',
                                     prop_path=f"{prim_path}.xformOp:translate",
                                     value=Gf.Vec3d(position['x'], position['y'], position['z']),
                                     prev=None)
            
            # Set rotation (convert radians to degrees, ZYX order for Isaac Sim? TODO: check if this is correct)
            rotation_degrees = [
                rotation['x'] * 180.0 / 3.14159,  # Z first
                rotation['y'] * 180.0 / 3.14159,  # Y second  
                rotation['z'] * 180.0 / 3.14159   # X last
            ]
            
            omni.kit.commands.execute('ChangeProperty',
                                     prop_path=f"{prim_path}.xformOp:rotateZYX",
                                     value=Gf.Vec3d(rotation_degrees[0], rotation_degrees[1], rotation_degrees[2]),
                                     prev=None)
            
            # Set scale
            omni.kit.commands.execute('ChangeProperty',
                                     prop_path=f"{prim_path}.xformOp:scale",
                                     value=Gf.Vec3d(0.01, 0.01, 0.01),
                                     prev=None)
            
            print(f"✓ {name} positioned")

print("\n=== Assembly positioning complete! ===")
