#run in isaacsim script editor using the usd files in the data/usd/folder.
#collected_{base/line_red/line_brown} folders contain processed usd files with the aruco markers added.
#we scale the objects to account for the scale of the object when exported from fusion360.
import omni.usd
import omni.kit.commands
from pxr import UsdGeom, Gf, Sdf, UsdShade
import os
import json

# Configuration - UPDATE THESE PATHS
JSON_FILE_PATH = os.path.expanduser("~/Projects/aruco-grasp-annotator/data/aruco/base1_aruco.json")
ARUCO_PNG_DIR = os.path.expanduser("~/Projects/aruco-grasp-annotator/data/aruco/pngs")
BASE_PRIM_PATH = "/Root/base1/base1"

def create_omni_pbr_and_get_path():
    """Create an OmniPBR material and return its path"""
    out = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=out
    )
    return out[0] if out else None

# Get stage
stage = omni.usd.get_context().get_stage()

# Step 0: Set the scale of the body to 0.01
print(f"Setting scale of {BASE_PRIM_PATH} to 0.01...")
body_prim = stage.GetPrimAtPath(BASE_PRIM_PATH)
if body_prim:
    body_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(0.01, 0.01, 0.01))
    print(f"Body scale set to 0.01")
else:
    print(f"WARNING: Body prim not found at {BASE_PRIM_PATH}")

# Read JSON file
print(f"\nReading JSON file: {JSON_FILE_PATH}")
with open(JSON_FILE_PATH, 'r') as f:
    aruco_data = json.load(f)

total_markers = len(aruco_data['markers'])
print(f"Found {total_markers} markers in JSON")
print(f"PNG directory: {ARUCO_PNG_DIR}")

# Extract dictionary type from JSON (e.g., "DICT_4X4_50" -> "4x4", "DICT_5X5_50" -> "5x5")
aruco_dictionary = aruco_data.get('aruco_dictionary', 'DICT_4X4_50')
# Convert dictionary name to filename format (DICT_4X4_50 -> 4x4, DICT_5X5_50 -> 5x5)
dict_name = aruco_dictionary.replace('DICT_', '').split('_')[0].lower()
print(f"ArUco Dictionary: {aruco_dictionary} (using {dict_name} for filenames)")

# Process each marker from JSON
# ArUco files use T_object_to_marker format
for marker in aruco_data['markers']:
    aruco_id = marker['aruco_id']
    position = marker['T_object_to_marker']['position']
    rotation = marker['T_object_to_marker']['rotation']
    # Size is at top level in aruco files
    marker_size = aruco_data.get('size', 0.021)
    
    # Scale translation values by 100 (scale stays the same)
    scaled_x = position['x'] * 100
    scaled_y = position['y'] * 100
    scaled_z = position['z'] * 100
    
    print(f"\n=== Processing ArUco Marker {aruco_id} ===")
    print(f"Original Position: x={position['x']}, y={position['y']}, z={position['z']}")
    print(f"Scaled Position: x={scaled_x}, y={scaled_y}, z={scaled_z}")
    print(f"Rotation: roll={rotation['roll']}, pitch={rotation['pitch']}, yaw={rotation['yaw']}")
    print(f"Size: {marker_size}m")
    
    # Define paths
    aruco_prim_path = f"{BASE_PRIM_PATH}/aruco_{aruco_id:03d}"
    cube_prim_path = f"{aruco_prim_path}/Cube"
    
    # Step 1: Create cube
    print(f"Creating cube at {cube_prim_path}")
    omni.kit.commands.execute("CreateMeshPrimCommand", 
                            prim_path=cube_prim_path, 
                            prim_type="Cube")
    
    # Get the cube prim and set transforms
    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    
    # Set position from JSON with scaling
    cube_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(scaled_x, scaled_y, scaled_z))
    
    # Convert Euler angles (roll, pitch, yaw) to quaternion
    # Roll = rotation around X, Pitch = rotation around Y, Yaw = rotation around Z
    # Note: Gf.Rotation expects degrees, so convert from radians
    rotation_matrix = Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(1,0,0), rotation['roll'] * 180.0 / 3.14159) * 
                                  Gf.Rotation(Gf.Vec3d(0,1,0), rotation['pitch'] * 180.0 / 3.14159) * 
                                  Gf.Rotation(Gf.Vec3d(0,0,1), rotation['yaw'] * 180.0 / 3.14159))
    quat = Gf.Quatd(rotation_matrix.ExtractRotation().GetQuat())
    cube_prim.GetAttribute("xformOp:orient").Set(quat)
    
    # Set scale to match marker size (NOT scaled - use original value)
    cube_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(marker_size, marker_size, 0.0001))  # Very thin cube
    
    # Step 2: Create material using the proper method
    print(f"Creating material OmniPBR...")
    created_material_path = create_omni_pbr_and_get_path()
    
    if created_material_path:
        print(f"Material created at: {created_material_path}")
        
        # Step 3: Assign texture to material
        # Use dictionary type in filename: aruco_marker_4x4_000.png or aruco_marker_5x5_000.png
        aruco_png_path = os.path.join(ARUCO_PNG_DIR, f"aruco_marker_{dict_name}_{aruco_id:03d}.png")
        
        if os.path.exists(aruco_png_path):
            # Add file: prefix to the path
            aruco_png_path_with_prefix = f"file:{aruco_png_path}"
            print(f"Assigning texture: {aruco_png_path_with_prefix}")
            
            # Get the shader prim
            shader_prim = stage.GetPrimAtPath(created_material_path + "/Shader")
            if shader_prim:
                texture_attr = shader_prim.CreateAttribute('inputs:diffuse_texture', Sdf.ValueTypeNames.Asset)
                texture_attr.Set(Sdf.AssetPath(aruco_png_path_with_prefix))
                print(f"Texture assigned to shader")
            else:
                print(f"WARNING: Shader prim not found at {created_material_path}/Shader")
        else:
            print(f"WARNING: PNG file not found: {aruco_png_path}")
        
        # Step 4: Bind material to cube
        print(f"Binding material to cube")
        omni.kit.commands.execute("BindMaterial",
            prim_path=cube_prim_path,
            material_path=created_material_path
        )
    else:
        print(f"ERROR: Failed to create material for marker {aruco_id}")
    
    print(f"ArUco Marker {aruco_id} complete!")

print("\n=== All markers processed successfully! ===")