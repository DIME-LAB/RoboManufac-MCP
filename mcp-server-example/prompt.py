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


@mcp.prompt()
def primitive_def_prompt() -> str:
    return """I will define a set of Motion Primitives that can be achieved using tools available to you.:
Primitive - Grasp: Open or Half-Open the gripper.
Move to Grasp the object by selecting a Grasp ID and Close Gripper. 
Move to safe height.

Primitive - Assemble Component: Translate for Assembly, Rotate for Assembly, Perform Insert, Half Open Gripper.

Primitive - Reorient: Reorient for Assembly

Primitive - Reorient and Regrasp: Reorient for Assembly, Move to Clear Space. 
Move Down and Open Gripper. Move to Safe Height. Hover over Clear Space. Perform Grasp again. 
"""


@mcp.prompt()
def level1_system_prompt():
    return"""Initialize: You are using the sim environment.
You are working in a sim environment. 
Get topics to find the objects available in the scene.
"""

@mcp.prompt()
def level1_grasp_prompt_v2() -> str:
    """
    Returns a prompt for identifying valid grasp points in a simulation environment.
    THIS IS SPECIFIC FOR OLLAMA CLIENT.
    """
    return """The available grasp IDs are: Fork Orange: IDs are 1-9, Fork Yellow: IDs are 1-9, Line Red: ID is 1, Line Brown: ID is 1.

Your goal is to find the right control gripper mode for each grasp id of each object.

Gripper is in open mode by default. But for some objects in the scene, the gripper needs to be half open before accessing that specific grasp id.

And some grasp ids are not accesible at all.

> To verify if a grasp was successful:

> 1. Record the gripper width command value when you close the gripper
> 2. After moving to safe height, check the current gripper width
> 3. **SUCCESS**: Final gripper width remains the same
> 4. **FAILURE**: Final gripper width is significantly less than (close to 0mm) the close command value - the gripper closed completely with no object between the jaws
;
Your task is to find which grasp ids are accessible per object and if we need to perform half open before moving to grasp.

Once you've found information about a grasp id for an object, save it as a graph resource. Save the cases of both success and failure.
Update the same resource as you discover information about the other grasp ids.

Do not use execute python code at any cost.

Sequence:

Open/Half Open gripper if a grasp-id failed. 
Grasp the object using either Open or Half-Open gripper. 
Record the Grasp ID you used and compare to judge success or failure.
"""


@mcp.prompt()
def level2_system_prompt():
    return"""You are using Sim environment.
Before starting any task, get_topics to find the objects available in the scene. 
Then identify available grasp ids per object from your available resources.

Once the overall task is done, use the Assembly Resource to record the sequence in which it has to finish the tasks.
"""

@mcp.prompt()
def level2_assembly_prompt_v1() -> str:
    """
    Returns a prompt with generalized instructions for manipulating and 
    assembling objects using pre-defined motion primitives.
    
    :return: Description
    :rtype: str
    """
    return"""There are a number of assembly objects in the simulation environment. 

Your goal is to assemble the objects with the base in the given sequence.

The object sequence for assembly is:
1) line_red
2) line_brown
3) fork_yellow
4) fork_orange

As you assemble the parts, log your findings in the resources.
"""


if __name__ == "__main__":
    mcp.run()