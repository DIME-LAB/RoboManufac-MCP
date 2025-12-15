#run this script after running the set_object_physics_isaacsim.py script.
from pxr import UsdPhysics, UsdGeom, PhysxSchema, Usd
import omni.usd

stage = omni.usd.get_context().get_stage()
prim_path = "/World/u_brown/u_brown/Body1"
root_prim = stage.GetPrimAtPath(prim_path)

if not root_prim.IsValid():
    print(f"Prim not found at {prim_path}")
else:
    for desc in Usd.PrimRange(root_prim):
        if desc.IsA(UsdGeom.Mesh) or desc.IsA(UsdGeom.Gprim):
            ##########Apply collision APIs
            UsdPhysics.CollisionAPI.Apply(desc).CreateCollisionEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(desc).CreateRestOffsetAttr(0.0)
            
            ##########Apply Mesh Collision API and set to SDF
            UsdPhysics.MeshCollisionAPI.Apply(desc).CreateApproximationAttr("sdf")
            
            ##########Apply PhysX SDF Mesh Collision API and set resolution
            meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(desc)
            meshCollision.CreateSdfResolutionAttr(256)
            
            print(f"SDF Collider set on: {desc.GetPath()}")