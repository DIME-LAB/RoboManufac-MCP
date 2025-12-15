# Isaac Sim script to position assembly objects
# Run this in Isaac Sim script editor after loading the USD files (Add at current selection)

import omni.kit.commands
from pxr import Gf
import json
import os

# Load assembly data
ASSEMBLY_FILE_PATH = os.path.expanduser("~/Projects/aruco-grasp-annotator/data/fmb_assembly1.json")

print("Loading assembly data...")
with open(ASSEMBLY_FILE_PATH, 'r') as f:
    assembly_data = json.load(f)

# Position each object
for component in assembly_data['components']:
    name = component['name']
    position = component['position']
    rotation = component['rotation']
    
    # Construct prim path from component name: /World/{name}/{name}/{name}
    prim_path = f"/World/{name}/{name}/{name}"
    
    print(f"Positioning {name}...")
    
    # Set position
    omni.kit.commands.execute('ChangeProperty',
                             prop_path=f"{prim_path}.xformOp:translate",
                             value=Gf.Vec3d(position['x'], position['y'], position['z']),
                             prev=None)
    
    # Extract rotation from rpy (roll-pitch-yaw)
    rpy = rotation['rpy']
    rpy_values = [rpy['x'], rpy['y'], rpy['z']]
    
    # Set rotation (convert radians to degrees, ZYX order for Isaac Sim)
    rotation_degrees = [
        rpy_values[0] * 180.0 / 3.14159,  # Z first (roll)
        rpy_values[1] * 180.0 / 3.14159,  # Y second (pitch)
        rpy_values[2] * 180.0 / 3.14159   # X last (yaw)
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
