#make sure transforms are on on the body1 prim before running this script.
import omni.usd

def fix_rotation_scaling(object_path):
    """
    Reset rotation on object and scaling on Body1.
    
    Args:
        object_path: Path to the object (e.g., "/World/u_brown")
    """
    stage = omni.usd.get_context().get_stage()
    
    # Extract object name from path
    object_name = object_path.split('/')[-1]
    body_path = f"{object_path}/{object_name}/Body1"
    
    # Reset rotation on the parent object
    prim = stage.GetPrimAtPath(object_path)
    if not prim.IsValid():
        print(f"✗ Prim not found at {object_path}")
        return
    
    #===== Reset rotateX to 0 on parent =====
    prim.GetAttribute("xformOp:rotateX:unitsResolve").Set(0.0)
    print(f"✓ Reset xformOp:rotateX:unitsResolve to 0.0 for {object_path}")
    
    # Reset scaling on Body1
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        print(f"✗ Body prim not found at {body_path}")
        return
    
    #===== Reset scaling to 0.01 on Body1 =====
    body_prim.GetAttribute("xformOp:scale").Set((0.01, 0.01, 0.01))
    print(f"✓ Reset scale to (0.01, 0.01, 0.01) for {body_path}")
    
    print(f"✓ Rotation and scaling reset finished\n")


#===== SINGLE ENTRY POINT - ONLY CHANGE THIS =====
OBJECT_PATH = "/World/u_brown"

# Run the fix
fix_rotation_scaling(OBJECT_PATH)