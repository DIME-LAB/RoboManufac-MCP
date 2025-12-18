from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prompts")

@mcp.prompt()
def identify_valid_grasp_points() -> str:
    """
    Returns a prompt for identifying valid grasp points in a simulation environment.
    """
    return """Initialize:

You are working in a sim environment

Get topics to find the objects available in the scene. Save scene state.

Identify available grasp ids per object.

Your goal is to find the right control gripper mode for each grasp id of each object.

Gripper is in open mode by default. But for some objects in the scene, the gripper needs to be half open before accessing that specific grasp id.

And some grasp ids are not accesible at all.

> To verify if a grasp was successful:

> 1. Record the gripper width command value when you close the gripper
> 2. After moving to safe height, check the current gripper width
> 3. **SUCCESS**: Final gripper width remains the same
> 4. **FAILURE**: Final gripper width is significantly less than (close to 0mm) the close command value - the gripper closed completely with no object between the jaws

Your task is to find which grasp ids are accessible per object and if we need to perform half open before moving to grasp.

Once you've found information about a grasp id for an object save it graph resource. Save both success and failure

Update the same resource as you discover information about the other grasp ids.

Do not use execute python code at any cost

Sequence:

Open/half open gripper if a grasp-id failed. (Default is open state)
Move to grasp the object by selecting a grasp id.
Close gripper.
Move to safe height.
Read gripper width and compare
Restore scene state (reset the positions of the object so you can perform the action again.)"""

if __name__ == "__main__":
    mcp.run()