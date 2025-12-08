from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prompts")

PROMPTS = {
     "detect_prompt_free": (
         "can you read prompt free detection. note that the annotations are labeled wrong. \n"
         "so map what you actually see in the original image to the wrongly labelled prompts \n"
         "and then update yolo prompt with objects of interest \n"
         "(objects to grasp with a robotic arm and box/hand to place the object on) \n"
         "if you see a gripper labeled as chair, then don't add it to the yolo prompt."
     ),
     "poses": (
         "HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY\n"
         "PICK_STATION_POSE = [-0.180, -0.385, 0.481, 0, 180, 0]  # XYZRPY"
     ),
     "yoloe_demo": (
         "peform these steps for a pick and place operation using yoloe\n\n"
         "step 1\n\n"
         "the robot is at home position now\n\n"
         "perform detect \"prompt free instructions txt file\n\n"
         "if you see a gripper labeled as chair, then don't add it to the yolo prompt.\n\n"
         "if you see anything lablled studio or studio shot, then don't add it to the yolo prompt.\n\n"
         "<user instructions will be provided>\n\n"
         "step 2\n\n"
         "read_topic of /objects_poses to know the name of the object as defined in the topic. \n\n"
         "if there are two instances of one object, then ask which one are we talking about before performing the next step\n\n"
         "note down the  object name and position and orientation in quaternion format\n\n"
         "note down the drop position and orientation in quaternion format.\n\n"
         "if you dont see the object to grasp or the drop location, then read object pose topic again.\n\n"
         "if the object is not listed in the object pose topic, then run detect prompt free again.\n\n"
         "add prompt maps of all objects in the scene to the yolo prompt so wrong objects are not detected.\n\n"
         "step 3\n\n"
         "visual servo to the object using visual_servo_yoloe mcp tool\n\n"
         "if visual servo fails, try again.\n\n"
         "step 4\n\n"
         "close gripper using control_gripper\n\n"
         "verify_grasp\n\n"
         "if verify gripper returned that the gripper didnt close (could be due to gripper communication problems), then close the gripper again till the verification works.\n\n"
         "perform control_gripper based on verify_grasp\n\n"
         "if the gripper closed without grasping the object, go back home and read object pose topic again and perform visual servo again and close the gripper again till the verification works.\n\n"
         "step 5\n\n"
         "move to home position\n\n"
         "run the tool again if it didnt work the first time.\n\n"
         "step 6\n\n"
         "the pose of the box will not be visible in the image, so use the position and orientation in quaternion detected for target x,y,z and target x,y,z,w\n\n"
         "move_down with height of 0.22\n\n"
         "try move_down tool again if it didnt work the first time.\n\n"
         "if you are told to perform a stack, use the pose of the target object from earlier but keep the z height of 0.15\n\n"
         "step 7\n\n"
         "open gripper using control_gripper\n\n"
         "verify_grasp\n\n"
         "if the gripper didnt open, keep trying to open the gripper till it opens.\n\n"
         "step 8\n\n"
         "go home\n"
         "HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY\n\n"
     ),
     "pick_and_place_dt_jenga": (
         "## Pick and Place Jenga Block Operation\n"
         "Step 1: List Prims\n"
         "  - List prims in Isaac Sim.\n"
         "\n"
         "Step 2: Read Jenga Pose\n"
         "  - Get jenga object info from /jenga/Xform/Cube as x, y, z, rx, ry, rz.\n"
         "\n"
         "Step 3: Move to Jenga Block\n"
         "  - Calculate pick orientation: UR rz = (Jenga rz).\n"
         "  - Use perform_ik to move to the jenga block position with z = 0.15, orientation [0, 180, (UR rz)], duration 8 seconds.\n"
         "\n"
         "Step 4: Close Gripper\n"
         "  - Close gripper using control_gripper.\n"
         "\n"
         "Step 5: Move UR end effector to x, y, z=0.25\n"
         "  - Move the end effector above the placement location at z=0.25.\n"
         "\n"
         "Step 6: Move UR end effector to final placement pose\n"
         "  - Set end effector orientation to [0, 180, new_jenga_rz] and move to x, y, z=0.25 (drop position).\n"
         "\n"
         "Step 7: Move end effector down for precise placement\n"
         "  - Move end effector z to 0.151 for precise placement.\n"
         "\n"
         "Step 8: Open Gripper\n"
         "  - Open gripper using control_gripper.\n"
         "\n"
         "Step 9: Return Home\n"
         "  - Go back to home position using HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0].\n"
         "Step 10: perform next opertion if any or stay at home position\n"
     ),
     "push_dt_jenga": (
         "## Push Jenga Blocks Operation\n"
         "do not look at this if this is your first time reading this.\n"
         "After picking and placing the firt jenga block, pick and place the second jenga block not on exactly where the final assembly says but place it with an offset of 0.05 meters in the x direction and 0.02 meters in the y direction.\n"
         "you can place the jenga bock either to the right or left with the offset based on where it wouoldnt disturb the first jenga block.\n"
         "now you can slide the second jenga block to the final position (actual final position from the final assembly pose).\n"
         "Step 1: List Prims\n"
         "  - List prims in Isaac Sim.\n"
         "\n"
         "Step 2: Read Jenga Pose\n"    
         "  - Get jenga object info from /jenga/Xform/Cube as x, y, z, rx, ry, rz.\n"
         "\n"
         "Step 3: set gripper width to (200) using control_gripper\n"
         "\n"
         "Step 4: Perform slide on Jenga Block using push_sim\n"
         "  - use push_sim to slide jenga block from inital position to final position with height = 0.15\n"
         " - use final position from user instructions"
         "\n"
         "Step 4: Return Home\n"
         "  - Go back to home position using HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0].\n"
         "Step 5: perform next opertion if any or stay at home position\n"
     ),
     "recreate_dt_jenga": (
         "## Recreate DT Jenga Operation\n"
         "Note go home before performing the next operation be it pick and place or push\n"
         "Step 1: read final assembly pose information\n"
         "\n"
         "Step 2: List prims in Isaac Sim.\n"
         "\n"
         "Step 3: Perform pick and place of one jenga block at a time to recreate the final assembly pose\n"
         "\n"
         "Step 4: after placing all the jenga blocks, read prim info of jenga blocks in Isaac Sim and verify if they match the assembly final pose\n"
         "\n"
         "Step 5: if the jenga blocks do not match the final assembly pose, then stop the scene and play it again. this resets the scene - jenga blocks go back to their inital positions\n"
         "\n"
         "Step 6 : Now go through the other tools available and figure out if you can use any other primitve to achieve the final assembly pose.\n"
         "\n"
         "Step 7: verify again if the jenga blocks match the final assembly pose.\n"
         "Step 8: if they do then read the LLM DT analysis and update the steps of the sequence that worked.\n"
         "\n"
         "Now we are ready to deploy the sequence in the real world.\n"
     ),
     "pick_and_place_real_world_jenga": (
         "## Pick and Place Real World Jenga Operation\n"
         "Step 1: Read topic of /objects_poses to know the position of jenga blocks in the scene.\n"
         "skip this step if you already know the initial position of the object.\n\n"
         "Step 2: Visual servo to the object using visual_servo_yoloe mcp tool\n"
         "If visual servo fails, try again.\n\n"
         "Step 3: Close gripper using control_gripper\n"
         "Verify_grasp\n"
         "If verify gripper returned that the gripper didn't close (could be due to gripper communication problems), then close the gripper again till the verification works.\n"
         "Perform control_gripper based on verify_grasp\n"
         "If the gripper closed without grasping the object, go back home and read object pose topic again and perform visual servo again and close the gripper again till the verification works.\n\n"
         "Step 4: Move to home/pick position\n"
         "Run the tool again if it didn't work the first time.\n\n"
         "Step 5: User will provide the final position - default is 0,-0.5,0.15 with 0 degree yaw.\n"
         "Keep z=0.15 always perform_ik tool to move to the final position\n"
         "Try perform_ik tool again if it didn't work the first time.\n\n"
         "Step 6: Open gripper using control_gripper\n"
         "Verify_grasp\n"
         "If the gripper didn't open, keep trying to open the gripper till it opens.\n\n"
         "Step 7: Go home\n"
         "HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY\n"
         "Step 8: Perform next operation if any or stay at home position\n"
     ),
     "push_real_world_jenga": (
         "## Push Real World Jenga Operation\n"
         "Step 1: Read /objects_poses topic to get the name of the object and the pose available in the real world.\n"
         "Step 2: Set gripper width to 250 using control_gripper for pushing operation\n\n"
         "read gripper width using /gripper_width ros2 topic and verify if the value is 36.\n"
         "retry the operation if the gripper width is not 36.\n"
         "Step 3: User will provide the final position for pushing.\n"
         "Use push_real tool by passing the topic name of the object and the final position and yaw angle as current position with height = 0.15\n"
         "Try push_real tool again if it didn't work the first time.\n"
         "Step 4: move to safe height"
     ),
     "recreate_real_world_jenga": (
         "## Recreate Real World Jenga Operation\n"
         "You are at home position now\n"
         "Step 1: Read the LLM DT analysis from json files.\n"
         "Step 2: Create a plan to recreate the assembly using the steps from LLM DT analysis.\n"
         "Step 3: Move to picking station position [-0.330, -0.385, 0.404, 0, 180, 0] to scan Jenga blocks available for picking.\n\n"
         "Step 4: Read /objects_poses topic to get the name of the object and the pose available in the real world.\n"
         "if any of the jenga blocks are not visible, then ask the user to move the jenga blocks to a visible position."
         "Analyze the required operations (pick and place vs push) for each Jenga block.\n"
         "Step 5: Execute the plan to recreate the assembly using pick and place or push primitive prompts.\n"
         "For each Jenga block:\n"
         "- If pick and place is required: use pick_and_place_real_world_jenga prompt\n"
         "- If push is required: use push_real_world_jenga prompt\n"
         "Step 6: once you are done with executing the plan, return to home position"
         "HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY\n\n"
         "Open gripper using control_gripper"
     ),
    "recreate_fmb_assembly_dt": (
        "## Recreate FMB Assembly Operation\n"
        "Note: dont go to home unnecessarily\n"
        "\n"
        "Step 1: Read /objects_poses_sim topic to get the names of the objects we are dealing with\n"
        "\n"
        "Step 2: Read final assembly steps information and look at the assembly instruction manual images to figure out the assembly tree structure.\n"
        "\n"
        "Step 3: Play scene and restore scene if previous assembly has been completed. Pick the next object to grasp.\n"
        "\n"
        "Step 4: Perform the following to pick and place each object to recreate the final assembly:\n"
        "   - Once you have selected the next object to assemble onto the base, note down its position and rotation and then perform ik (read objects_poses_sim topic first always for perform ik and set ee pose to 0,180,0. If perform ik fails, just try again.) to move above the object with z = 0.3\n"
        "   - You can only take a screenshot if the arm is above the object.Take a screenshot of the /annotated_stream image topic to see the grasp points available for the selected object. You can also read the position of the grasp point from /grasp_points topic\n"
        "   - Pick a grasp id by analysing the instruction manual images, screenshot and previous history of failed attempts and Move to grasp\n"
        "   - Close gripper to pick up object\n"
        "   - Verify grasp: if the gripper didnt close properly, take a photo using the intel_camera_rgb_sim topic. If you can confirm the grasp failed, this is a failed attempt. Log the failed attempt using log_grasp_attempt tool, then reset: Move home first to reset robot position, stop and start the scene in Isaacsim, and restore scene state to the previous state before the current object was manipulated. Pick a different grasp id and repeat from the beginning.\n"
        "   - if the gripper closed properly, move to safe height\n"
        "   - Reorient for assembly (relative to base)\n"
        "   - Read objects_poses_sim topic again to verify if the object has been grabbed properly and reoriented.\n"
        "   - Verify reorientation: If the object is not where it was supposed to be or has moved incorrectly, then the selected grasp id is not correct. Log the failed attempt using log_grasp_attempt tool, then reset: Move home first to reset robot position, stop and start the scene in Isaacsim, and restore scene state to the previous state before the current object was manipulated. Pick a different grasp id and repeat from the beginning.\n"
        "   - If the object has moved more than 85 degrees in any axis, then after performing Move down in sim mode, open gripper, move to grasp the id again, close gripper and move to safe height.\n"
        "   - Perform translate_for_assembly relative to the base.\n"
        "   - Open gripper to release object\n"
        "   - Move to safe height\n"
        "   - Run verify final position of object.\n"
        "   - If the object is where its supposed to be then save scene state using save_scene_state tool by passing the names of the objects whose assembly into the base is already done including base name. Log the successful attempt using log_grasp_attempt tool with status 'success'.\n"
        "   - If the object is not where it was supposed to be, then the selected grasp id is not correct. Log the failed attempt using log_grasp_attempt tool, then reset: Move home first to reset robot position, stop and start the scene in Isaacsim, and restore scene state to the previous state before the current object was manipulated. Pick a different grasp id and repeat from the beginning.\n"
        "\n"
        "Step 5: once you find out that a selected grasp point works, update the fmb_assembly_steps.json file.\n"
        "\n"
        "Step 6: Repeat the process for the next object in the assembly tree.\n"
    ),
}


@mcp.prompt()
def detect_prompt_free() -> str:
    """
    Returns a prompt for performing prompt-free object detection and analysis.
    This function implements a workflow that:
    1. Captures camera images from the scene
    2. Analyzes the image to identify all visible objects without relying on pre-labeled annotations
    3. Maps actual visual content to correct labels, ignoring incorrect pre-existing annotations
    4. Creates a corrected object inventory based on actual visual analysis
    5. Identifies available objects for manipulation vs. already processed objects
    6. Provides accurate scene state description based on visual analysis
    
    This is particularly useful when working with images that have incorrect or misleading annotations,
    as it relies purely on visual analysis rather than assumed labels.
    
    Returns:
        str: The prompt for prompt-free detection operations
    """
    return PROMPTS["detect_prompt_free"]


@mcp.prompt()
def get_poses() -> str:
    """
    Returns the predefined poses for robot operations.
    This function provides the standard home and pick station poses used in robotic operations.
    
    Returns:
        str: The poses with HOME_POSE and PICK_STATION_POSE coordinates
    """
    return PROMPTS["poses"]


@mcp.prompt()
def yoloe_demo(user_instructions: str = "<user instructions will be provided>") -> str:
    """
    Returns a prompt for performing pick and place operations using YOLOe.
    This function implements a complete workflow that:
    1. Starts from home position and performs prompt-free detection
    2. Reads object poses from /objects_poses topic
    3. Handles multiple instances of objects by asking for clarification
    4. Moves to drop station and identifies drop location
    5. Performs visual servo operations using YOLOe
    6. Handles gripper control with grasp verification
    7. Adapts placement strategy based on drop location type (hand vs box)
    8. Returns to home position after completion
    
    Args:
        user_instructions: User-provided instructions for the operation (default: placeholder)
    
    Returns:
        str: The comprehensive prompt for YOLOe pick and place operations
    """
    return PROMPTS["yoloe_demo"].replace("<user instructions will be provided>", user_instructions)


@mcp.prompt()
def pick_and_place_dt_jenga() -> str:
    """
    Returns a prompt for performing a pick and place operation on a Jenga block.
    This function implements an 8-step workflow that:
    1. Lists prims in Isaac Sim
    2. Reads the jenga object pose
    3. Moves to the Jenga block position with z=0.15
    4. Closes the gripper to grasp the block
    5. Moves to home position
    6. Moves to the final drop position
    7. Opens the gripper to release the block
    8. Returns to home position

    Returns:
        str: The prompt for pick and place jenga operations
    """
    return PROMPTS["pick_and_place_dt_jenga"]


@mcp.prompt()
def push_dt_jenga() -> str:
    """
    Returns a prompt for performing push operations on Jenga blocks.
    This function implements a complete workflow that:
    1. Lists available prims in Isaac Sim
    2. Reads the current pose of the target Jenga block
    3. Sets gripper width to 200 for pushing operation
    4. Performs slide operation using push_sim to move block from initial to final position
    5. Returns to home position after completion

    Returns:
        str: The comprehensive prompt for push operations
    """
    return PROMPTS["push_dt_jenga"]


@mcp.prompt()
def recreate_dt_jenga() -> str:
    """
    Returns a prompt for recreating the Digital Twin (DT) Jenga operation.
    This function implements a complete workflow that:
    1. Reads final assembly pose information from the data files
    2. Lists available prims in Isaac Sim to identify current scene state
    3. Performs pick and place operation for Jenga block 1 using final assembly pose information
    4. Performs push operations for Jenga blocks 2 and 3 using final assembly pose information
    5. Returns to home position after completion
    
    This operation combines both pick-and-place and push operations to recreate
    a complete Jenga arrangement based on predefined final assembly poses.
    
    Returns:
        str: The comprehensive prompt for DT Jenga recreation operations
    """
    return PROMPTS["recreate_dt_jenga"]


@mcp.prompt()
def pick_and_place_real_world_jenga(final_x: float = 0.0, final_y: float = -0.5, final_z: float = 0.15) -> str:
    """
    Returns a prompt for performing pick and place operations on real-world Jenga blocks using YOLOE.
    This function implements a complete workflow that:
    1. Reads object poses from /objects_poses topic to identify available Jenga blocks
    2. Handles multiple instances of objects by asking for clarification
    3. Uses visual servo with YOLOE for precise positioning
    4. Performs gripper control with grasp verification
    5. Transports blocks to specified final positions
    6. Returns to home position after completion
    
    Args:
        final_x: X coordinate for the final drop position (default: 0.0)
        final_y: Y coordinate for the final drop position (default: -0.5)
        final_z: Z coordinate for the final drop position (default: 0.15)
    
    Returns:
        str: The comprehensive prompt for real-world Jenga pick and place operations
    """
    return PROMPTS["pick_and_place_real_world_jenga"]


@mcp.prompt()
def push_real_world_jenga(final_x: float = 0.0, final_y: float = -0.5, final_z: float = 0.15) -> str:
    """
    Returns a prompt for performing push operations on real-world Jenga blocks.
    This function implements a complete workflow that:
    1. Reads object poses from /objects_poses topic to identify available Jenga blocks
    2. Handles multiple instances of objects by asking for clarification
    3. Sets gripper width to 200 for pushing operations
    4. Uses visual servo with YOLOE for precise positioning
    5. Performs push operations using push_sim tool
    6. Returns to home position after completion
    
    Args:
        final_x: X coordinate for the final push position (default: 0.0)
        final_y: Y coordinate for the final push position (default: -0.5)
        final_z: Z coordinate for the final push position (default: 0.15)
    
    Returns:
        str: The comprehensive prompt for real-world Jenga push operations
    """
    return PROMPTS["push_real_world_jenga"]


@mcp.prompt()
def recreate_real_world_jenga() -> str:
    """
    Returns a prompt for recreating real-world Jenga assemblies.
    This function implements a complete workflow that:
    1. Scans available Jenga blocks at the picking station
    2. Maps real-world object positions to initial assembly configuration
    3. Reads final assembly pose information from data files
    4. Creates a plan using LLM DT analysis to determine required operations
    5. Executes the plan using appropriate pick and place or push operations
    6. Verifies the final assembly matches the target configuration
    
    This operation combines both pick-and-place and push operations to recreate
    a complete Jenga arrangement in the real world based on predefined final assembly poses.
    
    Returns:
        str: The comprehensive prompt for real-world Jenga recreation operations
    """
    return PROMPTS["recreate_real_world_jenga"]


@mcp.prompt()
def recreate_fmb_assembly_dt() -> str:
    """
    Returns a prompt for recreating FMB (Final Model Build) assembly operations in the digital twin.
    This function implements a complete workflow that:
    1. Lists prims in Isaac Sim to identify available objects
    2. Reads final assembly pose information and analyzes assembly instruction manual images
    3. Creates an assembly tree structure and writes it to fmb_assembly_steps.json
    4. Performs pick and place operations for each object in the assembly tree:
       - Reads object poses from objects_poses_sim topic
       - Moves above objects and captures annotated images to identify grasp points
       - Selects and tests grasp IDs iteratively
       - Reorients objects relative to the base
       - Translates objects to final assembly positions
       - Verifies final positions and updates the JSON file with successful grasp IDs
    5. Repeats the process for all objects in the assembly tree
    
    This operation systematically explores and validates grasp points to ensure successful
    assembly recreation, updating the assembly steps file as valid grasp configurations are found.
    
    Returns:
        str: The comprehensive prompt for FMB assembly recreation operations
    """
    return PROMPTS["recreate_fmb_assembly_dt"]


if __name__ == "__main__":
    mcp.run()