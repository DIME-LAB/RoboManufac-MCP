#run this script in isaacsim first before running set_rotation_scaling_isaacsim.py and set_sdf_collider_isaacsim.py script.
from pxr import UsdPhysics, Usd, UsdGeom, PhysxSchema, UsdShade
import omni.usd

def setup_physics_for_prim(prim_path,
                           restitution=0.1,
                           static_friction=5.0,
                           dynamic_friction=5.0,
                           collision_approximation="convexDecomposition"):
    """
    Complete physics setup for a prim: rigid body, colliders, and physics material.
    
    Args:
        prim_path: Path to the prim (e.g., "/World/Xform/Cube")
        restitution: Bounce factor (0.1 = low bounce for tenga-like behavior)
        static_friction: Friction when stationary
        dynamic_friction: Friction when sliding
        collision_approximation: "convexDecomposition", "convexHull", "boundingCube", etc.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    
    if not root_prim.IsValid():
        print(f"? Prim not found at {prim_path}")
        return
    
    #===== 1: Apply RigidBody API =====
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        root_prim.GetAttribute("physics:rigidBodyEnabled").Set(True)
        print(f"? Applied RigidBodyAPI to: {prim_path}")
    else:
        print(f"? Prim already has RigidBodyAPI: {prim_path}")
    
    #===== 2: Create Physics Material =====
    mat_path = f"{prim_path}_PhysMat"
    
    #===== Check if material already exists =====
    if stage.GetPrimAtPath(mat_path):
        mat_prim = stage.GetPrimAtPath(mat_path)
        print(f"? Physics material already exists at {mat_path}, reusing it")
    else:
        mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
    
    #===== Apply physics material APIs =====
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    
    #===== Set material properties =====
    UsdPhysics.MaterialAPI(mat_prim).CreateRestitutionAttr().Set(restitution)  # FIXED: was CreateRestitutiontAttr()
    UsdPhysics.MaterialAPI(mat_prim).CreateStaticFrictionAttr().Set(static_friction)
    UsdPhysics.MaterialAPI(mat_prim).CreateDynamicFrictionAttr().Set(dynamic_friction)
    
    #===== PhysX-specific settings =====
    PhysxSchema.PhysxMaterialAPI(mat_prim).CreateRestitutionCombineModeAttr().Set(PhysxSchema.Tokens.min)  # FIXED: was CreateRestituationCombineModeAttr()
    PhysxSchema.PhysxMaterialAPI(mat_prim).CreateFrictionCombineModeAttr().Set(PhysxSchema.Tokens.average)
    
    print(f"? Created physics material at: {mat_path}")
    
    #===== 3: Apply Colliders to all mesh descendants =====
    collider_count = 0
    for desc in Usd.PrimRange(root_prim):
        if desc.IsA(UsdGeom.Mesh) or desc.IsA(UsdGeom.Gprim):
            ##########Apply collision APIs
            UsdPhysics.CollisionAPI.Apply(desc).CreateCollisionEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(desc).CreateRestOffsetAttr(0.0)
            UsdPhysics.MeshCollisionAPI.Apply(desc).CreateApproximationAttr().Set(collision_approximation)
            
            ##########Bind material to root prim as well
            UsdShade.MaterialBindingAPI.Apply(desc).Bind(
                UsdShade.Material(mat_prim),
                UsdShade.Tokens.weakerThanDescendants,
                "physics"
            )
            
            collider_count += 1
            print(f"? Collider + material applied to: {desc.GetPath()}")
    
    if collider_count == 0:
        print(f"? No mesh geometry found under {prim_path}")
    else:
        print(f"? Applied ({collider_count}) collider(s) to {prim_path}")
    
    #===== 4: Bind material to root prim as well =====
    UsdShade.MaterialBindingAPI.Apply(root_prim).Bind(
        UsdShade.Material(mat_prim),
        UsdShade.Tokens.weakerThanDescendants,
        "physics"
    )
    
    print(f"? Complete physics setup finished for: {prim_path}\n")

#===== USAGE =====
setup_physics_for_prim("/World/u_brown/u_brown/Body1")