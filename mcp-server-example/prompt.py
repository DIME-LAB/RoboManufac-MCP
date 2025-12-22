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

Primitive - Assemble Component: Translate for Assembly, Reorient for Assembly, Perform Insert, Half Open Gripper, Move to safe height.

Primitive - Reorient: Reorient for Assembly

Primitive - Reorient and Regrasp: Reorient for Assembly, Move to Clear Space. 
Move Down and Open Gripper. Move to Safe Height. Hover over Clear Space. Perform Grasp. 
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
    return"""
You are using Sim environment.
Before starting any task, get_topics to find the objects available in the scene. Save current scene state.
Then identify available grasp ids per object from your available resources.

Your goal is to assemble the objects with the base.
You are to figure out the sequence of tools to be executed in order for each object in the sequence.

The object sequence for assembly is:
1) line_red
2) line_brown
3) fork_yellow
4) fork_orange

Once you think you've assembled an object, verify final pose assembly.
If your sequence for an object was a success, save scene state after the object is assembled.
If your sequence for an object was a failure, restore scene state to the state before the object was assembled. 
Reason for failure and retry object manipulation with a different tool sequence which you haven't tried yet for this object.

As you assemble the parts, log your findings in the resources - Both success and failure.
"""


@mcp.prompt()
def level2_system_prompt_grasp_id_given():
    return"""
You are using Sim environment.
Before starting any task, get_topics to find the objects available in the scene. Save current scene state.
Then identify available grasp ids per object from your available resources.

Your goal is to assemble the objects with the base.
You are to figure out the sequence of tools to be executed in order for each object in the sequence.

The object sequence for assembly is:
1) line_red: Use Grasp ID 1.
2) line_brown: Use Grasp ID 1.
3) fork_yellow: Use Grasp ID 6.
4) fork_orange: Use Grasp ID 6.

Once you think you've assembled an object, verify final pose assembly.
If your sequence for an object was a success, save scene state after the object is assembled.
If your sequence for an object was a failure, restore scene state to the state before the object was assembled. 
Reason for failure and retry object manipulation with a different tool sequence which you haven't tried yet for this object.

As you assemble the parts, log your findings in the resources - Both success and failure.
"""


@mcp.prompt()
def level2_system_prompt_instruction_following():
    return"""
You are using Sim environment.
Before starting any task, get_topics to find the objects available in the scene. Save current scene state.
Then identify available grasp ids per object from your available resources.

Your goal is to assemble the objects with the base. Start the task by moving to the home position.
Remember to half open the gripper to release the object, once you are done with any placement or insertion task.
Verify positional accuracy after performing each object's sequence. If verification fails, reset the scene to the previous object's successful state.

The object and task sequence for the assembly is:
1) line_red: Use Grasp ID 1. Grasp the object, translate it, reorient the object (if needed) and then translate for assembly and insert it.  
2) line_brown: Use Grasp ID 1. Grasp the object, translate it, reorient the object (if needed) and then translate for assembly and insert it.
3) fork_yellow: Use Grasp ID 6. Grasp the object, reorient it, move to regrasp position and place the object. Grasp the object from new position, then translate and reorient the object for assembly and insert it.
4) fork_orange: Use Grasp ID 6. Grasp the object, reorient it, move to regrasp position and place the object. Grasp the object from new position, then translate and reorient the object for assembly and insert it.

Once you are done with the task and verified, log your assembly results at the end of the verified assembly.
"""



@mcp.prompt()
def phase_4_perform_assembly_sequence(mode: str = "real") -> str:
    """
    Returns a prompt for performing assembly based on assembly log data.

    Args:
        mode: The environment mode - either "sim" or "real" (default: "real")
    """
    environment = "sim environment" if mode == "sim" else "real environment"
    world_context = "simulation" if mode == "sim" else "real world"

    return f"""**Initialize:**

You are working in a {environment}.
Read the assembly log to identify the objects you are dealing with.

**Task**

You are to perform assembly of the objects onto a fixed base in the {world_context} based on the data assembly resource collected using a Digital twin from your previous runs.

**Execution**

Use the available tools to perform assembly using the information of the assembly log in the same order. Do not skip any object assembly sequence.
Follow the sequence of objects and the tool calls with arguements to perform assembly one by one.
When performing assembly onto the base, half open gripper as to not disturb the previously placed parts.
Verify assembly after each object assembly.
Move home after all objects are assembled.

**Verification**

> To verify if an assembly was successful: run verify assembly once you you've ran all tools required to move one object into the fixed base.
> 1.**SUCCESS**: verify assembly returns success.
> 2.**FAILURE**: verify assembly returns failure. Pause and Request for assistance on further instructions.

**Output**

Fully assembled Assembly in the {world_context}."""
 


@mcp.prompt()
def assembly3_grasp_logs() -> str:
    """
    Returns a prompt with generalized instructions for manipulating and 
    assembling objects using pre-defined motion primitives.

    :return: Description
    :rtype: str
    """
    return"""{
  "fork_orange": [
    {
      "grasp_id": 1,
      "gripper_state": "half-open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 2,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 2,
      "gripper_state": "half-open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 3,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 3,
      "gripper_state": "half-open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 4,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 4,
      "gripper_state": "half-open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 5,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 5,
      "gripper_state": "half-open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 6,
      "gripper_state": "open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 7,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 7,
      "gripper_state": "half-open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 8,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 8,
      "gripper_state": "half-open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 9,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 9,
      "gripper_state": "half-open",
      "status": "FAILURE"
    }
  ],
  "fork_yellow": [
    {
      "grasp_id": 1,
      "gripper_state": "open",
      "status": "FAILURE"
    },
    {
      "grasp_id": 1,
      "gripper_state": "half-open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 4,
      "gripper_state": "half-open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 6,
      "gripper_state": "open",
      "status": "SUCCESS"
    },
    {
      "grasp_id": 8,
      "gripper_state": "half-open",
      "status": "SUCCESS"
    }
  ],
  "line_brown": [
    {
      "grasp_id": 1,
      "gripper_state": "open",
      "status": "SUCCESS"
    }
  ],
  "line_red": [
    {
      "grasp_id": 1,
      "gripper_state": "open",
      "status": "SUCCESS"
    }
  ]
}   
"""


if __name__ == "__main__":
    mcp.run()