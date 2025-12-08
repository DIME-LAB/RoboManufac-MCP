from mcp.server.fastmcp import FastMCP
import json
from pathlib import Path

mcp = FastMCP("Assembly Poses Management")

# Path to persistent assembly poses data files
FINAL_ASSEMBLY_POSES_FILE = Path(__file__).parent / "data" / "final_assembly_poses.json"
INITIAL_OBJECT_POSES_FILE = Path(__file__).parent / "data" / "initial_object_poses.json"
LLM_DT_ANALYSIS_FILE = Path(__file__).parent / "data" / "llm_dt_analysis.json"
FMB_ASSEMBLY_STEPS_FILE = Path(__file__).parent / "data" / "fmb_assembly_steps.json"

@mcp.tool()
def read_final_assembly_poses_file() -> str:
    """
    Read the final assembly poses file directly and return its raw content
    
    Returns:
        JSON string containing the raw file content
    """
    try:
        with open(FINAL_ASSEMBLY_POSES_FILE, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return json.dumps({
            "error": f"Assembly poses file not found at {FINAL_ASSEMBLY_POSES_FILE}",
            "message": "Please ensure data/final_assembly_poses.json exists."
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to read file: {str(e)}"
        }, indent=2)

@mcp.tool()
def update_object_pose_complete(pose_data: str) -> str:
    """
    Update assembly poses by providing complete pose data as a JSON string.
    This tool will overwrite the entire file with the new data structure.
    
    Expected JSON format:
    {
        "objects": [
            {
                "name": "Object1",
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}
            }
        ]
    }
    
    Args:
        pose_data: JSON string containing the complete pose data structure
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        # Ensure data directory exists
        FINAL_ASSEMBLY_POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse the input JSON string
        new_data = json.loads(pose_data)
        
        # Validate the structure has "objects" key
        if "objects" not in new_data:
            return json.dumps({
                "success": False,
                "error": "Invalid data structure. Must contain 'objects' key.",
                "expected_format": {
                    "objects": [
                        {
                            "name": "Object1",
                            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}
                        }
                    ]
                }
            }, indent=2)
        
        # Overwrite the entire file with the new data
        with open(FINAL_ASSEMBLY_POSES_FILE, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "Updated assembly poses and overwrote file",
            "objects_count": len(new_data["objects"]),
            "file_overwritten": True
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON format: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to update file: {str(e)}"
        }, indent=2)

@mcp.tool()
def read_initial_object_poses_file() -> str:
    """
    Read the initial object poses file directly and return its raw content
    
    Returns:
        JSON string containing the raw file content
    """
    try:
        with open(INITIAL_OBJECT_POSES_FILE, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return json.dumps({
            "error": f"Initial object poses file not found at {INITIAL_OBJECT_POSES_FILE}",
            "message": "Please ensure data/initial_object_poses.json exists."
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to read file: {str(e)}"
        }, indent=2)

@mcp.tool()
def update_initial_object_pose_complete(pose_data: str) -> str:
    """
    Update initial object poses by providing complete pose data as a JSON string.
    This tool will overwrite the entire file with the new data structure.
    
    Expected JSON format:
    {
        "objects": [
            {
                "name": "Object1",
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}
            }
        ]
    }
    
    Args:
        pose_data: JSON string containing the complete pose data structure
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        # Ensure data directory exists
        INITIAL_OBJECT_POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse the input JSON string
        new_data = json.loads(pose_data)
        
        # Validate the structure has "objects" key
        if "objects" not in new_data:
            return json.dumps({
                "success": False,
                "error": "Invalid data structure. Must contain 'objects' key.",
                "expected_format": {
                    "objects": [
                        {
                            "name": "Object1",
                            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}
                        }
                    ]
                }
            }, indent=2)
        
        # Overwrite the entire file with the new data
        with open(INITIAL_OBJECT_POSES_FILE, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "Updated initial object poses and overwrote file",
            "objects_count": len(new_data["objects"]),
            "file_overwritten": True
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON format: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to update file: {str(e)}"
        }, indent=2)

@mcp.tool()
def read_llm_dt_analysis_file() -> str:
    """
    Read the LLM DT analysis file directly and return its raw content
    
    Returns:
        JSON string containing the raw file content
    """
    try:
        with open(LLM_DT_ANALYSIS_FILE, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return json.dumps({
            "error": f"LLM DT analysis file not found at {LLM_DT_ANALYSIS_FILE}",
            "message": "Please ensure data/llm_dt_analysis.json exists."
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to read file: {str(e)}"
        }, indent=2)

@mcp.tool()
def update_llm_dt_analysis_complete(analysis_data: str) -> str:
    """
    Update LLM DT analysis by providing complete analysis data as a JSON string.
    This tool will overwrite the entire file with the new data structure.

    Expected JSON format:
    [
        {
            "block": "jenga_2",
            "operation": "pick_and_place",
            "target_position": {
                "x": 0.0,
                "y": -0.5,
                "z": 0.007
            }
        },
        {
            "block": "jenga_4",
            "operation": "push",
            "target_position": {
                "x": 0.024,
                "y": -0.5,
                "z": 0.007
            },
            "yaw": 0.0
        }
    ]
    
    Args:
        analysis_data: JSON string containing the complete analysis data structure
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        # Ensure data directory exists
        LLM_DT_ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse the input JSON string
        new_data = json.loads(analysis_data)
        
        # Validate the structure is a list
        if not isinstance(new_data, list):
            return json.dumps({
                "success": False,
                "error": "Invalid data structure. Must be a list of analysis entries.",
                "expected_format": [
                    {
                        "block": "jenga_2",
                        "operation": "pick_and_place",
                        "target_position": {
                            "x": 0.0,
                            "y": -0.5,
                            "z": 0.007
                        }
                    }
                ]
            }, indent=2)
        
        # Overwrite the entire file with the new data
        with open(LLM_DT_ANALYSIS_FILE, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "Updated LLM DT analysis and overwrote file",
            "entries_count": len(new_data),
            "file_overwritten": True
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON format: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to update file: {str(e)}"
        }, indent=2)

@mcp.tool()
def read_fmb_assembly_steps_file() -> str:
    """
    Read the FMB assembly steps file directly and return its raw content
    
    Returns:
        JSON string containing the raw file content
    """
    try:
        with open(FMB_ASSEMBLY_STEPS_FILE, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return json.dumps({
            "error": f"FMB assembly steps file not found at {FMB_ASSEMBLY_STEPS_FILE}",
            "message": "Please ensure data/fmb_assembly_steps.json exists."
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to read file: {str(e)}"
        }, indent=2)

@mcp.tool()
def update_fmb_assembly_steps_file(assembly_data: str) -> str:
    """
    Update FMB assembly steps by providing complete assembly data as a JSON string.
    This tool will overwrite the entire file with the new data structure.
    
    Expected JSON format (list of assembly steps in sequence):
    [
        {
            "object_name": "Object1",
            "status": "success",
            "grasp_id": 2,
            "grasp_attempts": [
                {"grasp_id": 1, "status": "failure"},
                {"grasp_id": 2, "status": "success"}
            ]
        },
        {
            "object_name": "Object2",
            "status": "in_progress",
            "grasp_id": null,
            "grasp_attempts": [
                {"grasp_id": 1, "status": "failure"}
            ]
        }
    ]
    
    The sequence in the array represents the order of assembly operations.
    Each entry must contain:
    - object_name: Name of the object to be assembled
    - status: Current status ("pending", "in_progress", "success", "failure")
    - grasp_id: The successful grasp point ID (null if not yet successful)
    - grasp_attempts: List of all grasp attempts with their status
    
    Args:
        assembly_data: JSON string containing the complete assembly steps data structure
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        # Ensure data directory exists
        FMB_ASSEMBLY_STEPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse the input JSON string
        new_data = json.loads(assembly_data)
        
        # Validate the structure is a list
        if not isinstance(new_data, list):
            return json.dumps({
                "success": False,
                "error": "Invalid data structure. Must be a list of assembly steps.",
                "expected_format": [
                    {
                        "object_name": "Object1",
                        "status": "success",
                        "grasp_id": 0,
                        "grasp_attempts": [
                            {"grasp_id": 0, "status": "success"}
                        ]
                    }
                ]
            }, indent=2)
        
        # Validate each entry
        valid_statuses = ["pending", "in_progress", "success", "failure"]
        for i, step in enumerate(new_data):
            if not isinstance(step, dict):
                return json.dumps({
                    "success": False,
                    "error": f"Invalid entry at index {i}. Each step must be an object/dictionary."
                }, indent=2)
            
            if "object_name" not in step:
                return json.dumps({
                    "success": False,
                    "error": f"Missing 'object_name' field at index {i}."
                }, indent=2)
            
            if "status" not in step:
                return json.dumps({
                    "success": False,
                    "error": f"Missing 'status' field at index {i}."
                }, indent=2)
            
            if step["status"] not in valid_statuses:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid 'status' at index {i}. Must be one of: {valid_statuses}"
                }, indent=2)
            
            if "grasp_attempts" not in step:
                return json.dumps({
                    "success": False,
                    "error": f"Missing 'grasp_attempts' field at index {i}."
                }, indent=2)
            
            if not isinstance(step["grasp_attempts"], list):
                return json.dumps({
                    "success": False,
                    "error": f"Invalid 'grasp_attempts' at index {i}. Must be a list."
                }, indent=2)
            
            # Validate grasp_attempts entries
            for j, attempt in enumerate(step["grasp_attempts"]):
                if not isinstance(attempt, dict):
                    return json.dumps({
                        "success": False,
                        "error": f"Invalid grasp_attempt at index {i}, attempt {j}. Must be an object."
                    }, indent=2)
                
                if "grasp_id" not in attempt or "status" not in attempt:
                    return json.dumps({
                        "success": False,
                        "error": f"Missing 'grasp_id' or 'status' in grasp_attempt at index {i}, attempt {j}."
                    }, indent=2)
                
                if attempt["status"] not in ["success", "failure"]:
                    return json.dumps({
                        "success": False,
                        "error": f"Invalid status in grasp_attempt at index {i}, attempt {j}. Must be 'success' or 'failure'."
                    }, indent=2)
            
            # Validate grasp_id (can be null/None for in_progress or failure)
            if "grasp_id" in step:
                if step["grasp_id"] is not None and not isinstance(step["grasp_id"], (int, float)):
                    return json.dumps({
                        "success": False,
                        "error": f"Invalid 'grasp_id' at index {i}. Must be a number or null."
                    }, indent=2)
        
        # Overwrite the entire file with the new data
        with open(FMB_ASSEMBLY_STEPS_FILE, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "Updated FMB assembly steps and overwrote file",
            "steps_count": len(new_data),
            "file_overwritten": True
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON format: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to update file: {str(e)}"
        }, indent=2)

@mcp.tool()
def update_object_status(object_name: str, status: str) -> str:
    """
    Update the status of a specific object in the FMB assembly steps file.
    
    Args:
        object_name: Name of the object to update
        status: New status ("pending", "in_progress", "success", "failure")
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        valid_statuses = ["pending", "in_progress", "success", "failure"]
        if status not in valid_statuses:
            return json.dumps({
                "success": False,
                "error": f"Invalid status. Must be one of: {valid_statuses}"
            }, indent=2)
        
        # Read current file
        if not FMB_ASSEMBLY_STEPS_FILE.exists():
            return json.dumps({
                "success": False,
                "error": "FMB assembly steps file does not exist. Create it first."
            }, indent=2)
        
        with open(FMB_ASSEMBLY_STEPS_FILE, 'r') as f:
            data = json.load(f)
        
        # Find and update the object
        found = False
        for step in data:
            if step.get("object_name") == object_name:
                step["status"] = status
                found = True
                break
        
        if not found:
            return json.dumps({
                "success": False,
                "error": f"Object '{object_name}' not found in assembly steps."
            }, indent=2)
        
        # Write back to file
        with open(FMB_ASSEMBLY_STEPS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return json.dumps({
            "success": True,
            "message": f"Updated status of '{object_name}' to '{status}'"
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON in file: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to update status: {str(e)}"
        }, indent=2)

@mcp.tool()
def log_grasp_attempt(object_name: str, grasp_id: int, attempt_status: str) -> str:
    """
    Log a grasp attempt for a specific object. This adds an entry to the grasp_attempts array.
    If the attempt is successful, it also updates the object's grasp_id and status to "success".
    
    Args:
        object_name: Name of the object
        grasp_id: The grasp point ID that was attempted
        attempt_status: Status of the attempt ("success" or "failure")
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        if attempt_status not in ["success", "failure"]:
            return json.dumps({
                "success": False,
                "error": "Invalid attempt_status. Must be 'success' or 'failure'."
            }, indent=2)
        
        if not isinstance(grasp_id, (int, float)):
            return json.dumps({
                "success": False,
                "error": "grasp_id must be a number."
            }, indent=2)
        
        # Read current file
        if not FMB_ASSEMBLY_STEPS_FILE.exists():
            return json.dumps({
                "success": False,
                "error": "FMB assembly steps file does not exist. Create it first."
            }, indent=2)
        
        with open(FMB_ASSEMBLY_STEPS_FILE, 'r') as f:
            data = json.load(f)
        
        # Find the object
        found = False
        updated_step = None
        for step in data:
            if step.get("object_name") == object_name:
                found = True
                updated_step = step
                
                # Initialize grasp_attempts if it doesn't exist
                if "grasp_attempts" not in step:
                    step["grasp_attempts"] = []
                
                # Add the new attempt
                step["grasp_attempts"].append({
                    "grasp_id": int(grasp_id),
                    "status": attempt_status
                })
                
                # If successful, update grasp_id and status
                if attempt_status == "success":
                    step["grasp_id"] = int(grasp_id)
                    step["status"] = "success"
                elif step.get("status") == "pending":
                    # If first attempt and it failed, set to in_progress
                    step["status"] = "in_progress"
                
                break
        
        if not found:
            return json.dumps({
                "success": False,
                "error": f"Object '{object_name}' not found in assembly steps."
            }, indent=2)
        
        # Write back to file
        with open(FMB_ASSEMBLY_STEPS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return json.dumps({
            "success": True,
            "message": f"Logged grasp attempt for '{object_name}': grasp_id={grasp_id}, status={attempt_status}",
            "updated_object_status": updated_step.get("status") if updated_step else None
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON in file: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to log grasp attempt: {str(e)}"
        }, indent=2)

@mcp.tool()
def reset_fmb_assembly_steps(reset_type: str = "all") -> str:
    """
    Reset the FMB assembly steps file. Can reset all objects to pending status with empty grasp attempts,
    or completely clear the file.
    
    Args:
        reset_type: Type of reset to perform:
            - "all": Reset all objects to pending status with empty grasp_attempts (keeps object names)
            - "clear": Clear the file completely (empty array)
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        if reset_type not in ["all", "clear"]:
            return json.dumps({
                "success": False,
                "error": "Invalid reset_type. Must be 'all' or 'clear'."
            }, indent=2)
        
        # Ensure data directory exists
        FMB_ASSEMBLY_STEPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if reset_type == "clear":
            # Clear the file completely
            with open(FMB_ASSEMBLY_STEPS_FILE, 'w') as f:
                json.dump([], f, indent=2)
            
            return json.dumps({
                "success": True,
                "message": "FMB assembly steps file cleared completely",
                "reset_type": "clear"
            }, indent=2)
        
        else:  # reset_type == "all"
            # Read current file
            if not FMB_ASSEMBLY_STEPS_FILE.exists():
                return json.dumps({
                    "success": False,
                    "error": "FMB assembly steps file does not exist. Nothing to reset."
                }, indent=2)
            
            with open(FMB_ASSEMBLY_STEPS_FILE, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return json.dumps({
                    "success": False,
                    "error": "Invalid file format. Expected a list."
                }, indent=2)
            
            # Reset all objects to pending status with empty grasp attempts
            reset_count = 0
            for step in data:
                if isinstance(step, dict) and "object_name" in step:
                    step["status"] = "pending"
                    step["grasp_id"] = None
                    step["grasp_attempts"] = []
                    reset_count += 1
            
            # Write back to file
            with open(FMB_ASSEMBLY_STEPS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            return json.dumps({
                "success": True,
                "message": f"Reset {reset_count} objects to pending status with empty grasp attempts",
                "reset_type": "all",
                "objects_reset": reset_count
            }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON in file: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to reset file: {str(e)}"
        }, indent=2)


if __name__ == "__main__":
    mcp.run()
