from mcp.server.fastmcp import FastMCP
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

mcp = FastMCP("FMB Assembly Steps and Grasp IDs Management")

# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
# Directories are created lazily when needed by tools
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    RESOURCES_DIR = Path(BASE_OUTPUT_DIR) / "resources"
else:
    RESOURCES_DIR = Path(__file__).parent / "resources"

# Ensure resources directory exists
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# ========== HELPER FUNCTIONS ==========

def get_grasp_log_file(assembly_id: str) -> Path:
    """Get the grasp log file path for a specific assembly"""
    return RESOURCES_DIR / f"Assembly{assembly_id}_grasp_log.json"

def get_assembly_file(assembly_id: str) -> Path:
    """Get the assembly sequence log file path for a specific assembly"""
    return RESOURCES_DIR / f"Assembly{assembly_id}_sequence_log.json"

def get_final_sequence_file(assembly_id: str) -> Path:
    """Get the final sequence file path for a specific assembly"""
    return RESOURCES_DIR / f"Assembly{assembly_id}_final_sequence.json"

def load_grasp_resource(assembly_id: str):
    """Load grasp resource from JSON file for a specific assembly"""
    grasp_file = get_grasp_log_file(assembly_id)
    if not grasp_file.exists():
        return {}

    try:
        with open(grasp_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty dict
        return {}
    except Exception:
        # Any other error - return empty dict
        return {}

def save_grasp_resource(assembly_id: str, data):
    """Save grasp resource to JSON file for a specific assembly"""
    grasp_file = get_grasp_log_file(assembly_id)
    with open(grasp_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_assembly_resource(assembly_id: str):
    """Load assembly resource from JSON file for a specific assembly"""
    assembly_file = get_assembly_file(assembly_id)
    if not assembly_file.exists():
        return []
    try:
        with open(assembly_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty list
        return []
    except Exception:
        # Any other error - return empty list
        return []

def save_assembly_resource(assembly_id: str, data):
    """Save assembly resource to JSON file for a specific assembly"""
    assembly_file = get_assembly_file(assembly_id)
    with open(assembly_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_final_sequence(assembly_id: str):
    """Load final sequence from JSON file for a specific assembly"""
    final_file = get_final_sequence_file(assembly_id)
    if not final_file.exists():
        return []
    try:
        with open(final_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty list
        return []
    except Exception:
        # Any other error - return empty list
        return []

def save_final_sequence(assembly_id: str, data):
    """Save final sequence to JSON file for a specific assembly"""
    final_file = get_final_sequence_file(assembly_id)
    with open(final_file, 'w') as f:
        json.dump(data, f, indent=2)


# ========== RESOURCES ==========

@mcp.resource("Assembly{assembly_id}/object_name/{object_name}/grasp_configs")
def get_object_grasp_configs(assembly_id: str, object_name: str) -> str:
    """Get grasp configurations for an object in a specific assembly (list of grasp_id, gripper_state, and status). Each config includes grasp_id (int), gripper_state ("open" or "half-open"), and status ("SUCCESS" or "FAILURE")."""
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    if isinstance(obj_data, list):
        return json.dumps(obj_data, indent=2)
    else:
        return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/object_name/{object_name}/grasp_configs/{grasp_id}/status")
def get_object_status_for_id(assembly_id: str, object_name: str, grasp_id: str) -> str:
    """Get all configurations (grasp_id, gripper_state, and status) for a specific grasp_id of an object_name in an assembly. Returns all attempts (both SUCCESS and FAILURE) with their modes."""
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    grasp_id_int = int(grasp_id)
    if isinstance(obj_data, list):
        matching_configs = [config for config in obj_data if config.get("grasp_id") == grasp_id_int]
        if matching_configs:
            return json.dumps(matching_configs, indent=2)
    return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/sequence")
def get_assembly_sequence(assembly_id: str) -> str:
    """Get sequence for a specific Assembly ID. Each item has sequence_id (fixed), object_name (fixed), and primitives_trials (list of trials with trial_id, grasp_id, gripper_state, primitives ordered sequence, and status)."""
    sequence = load_assembly_resource(assembly_id)
    return json.dumps(sequence, indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/primitives_trials")
def get_assembly_object_primitives_trials(assembly_id: str, object_name: str) -> str:
    """Get primitives_trials for a specific object in an assembly sequence. Each trial has trial_id, grasp_id, gripper_state, primitives (ordered sequence), and status."""
    sequence = load_assembly_resource(assembly_id)

    for item in sequence:
        if item.get("object_name") == object_name:
            return json.dumps({"primitives_trials": item.get("primitives_trials", [])}, indent=2)

    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/primitives_trials/{trial_id}")
def get_assembly_object_trial(assembly_id: str, object_name: str, trial_id: str) -> str:
    """Get a specific trial for an object in an assembly sequence. Returns trial_id, grasp_id, gripper_state, primitives (ordered sequence), and status."""
    sequence = load_assembly_resource(assembly_id)
    trial_id_int = int(trial_id)
    for item in sequence:
        if item.get("object_name") == object_name:
            trials = item.get("primitives_trials", [])
            for trial in trials:
                if trial.get("trial_id") == trial_id_int:
                    return json.dumps(trial, indent=2)
            return json.dumps({"error": f"Trial {trial_id} not found for object '{object_name}' in Assembly{assembly_id}"}, indent=2)
    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

# ========== TOOLS FOR GRASP RESOURCE (Assembly{id}/object_name/grasp_configs) ==========

@mcp.tool()
def read_grasp_resource(assembly_id: str, object_name: str) -> str:
    """
    Read the grasp configurations for an object in a specific assembly (grasp_configs with grasp_id, gripper_state, and status)
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object

    Returns:
        JSON string containing grasp_configs list with grasp_id, gripper_state, and status (includes both SUCCESS and FAILURE attempts)
        Each config has: {"grasp_id": <int>, "gripper_state": "open"|"half-open", "status": "SUCCESS"|"FAILURE"}
    """
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    if not obj_data:
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": [],
            "message": "Resource not found"
        }, indent=2)
    if isinstance(obj_data, list):
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": obj_data
        }, indent=2)
    else:
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": []
        }, indent=2)

@mcp.tool()
def write_grasp_resource(assembly_id: str, object_name: str, grasp_configs: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Write or update grasp configurations for an object in a specific assembly (grasp_configs list with grasp_id, gripper_state, and status)
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object
        grasp_configs: List of grasp configurations, each with grasp_id, gripper_state, and status.
                       Format: [{"grasp_id": 1, "gripper_state": "open", "status": "SUCCESS"}, ...]
                       - grasp_id: integer (required)
                       - gripper_state: "open" or "half-open" (required)
                       - status: "SUCCESS" or "FAILURE" (required)
                       Both success and failure attempts are stored.

    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_grasp_resource(assembly_id)
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"Error loading resource file: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error loading resource: {str(e)}"}, indent=2)

    if grasp_configs is not None:
        # Handle case where grasp_configs might be a JSON string
        if isinstance(grasp_configs, str):
            try:
                grasp_configs = json.loads(grasp_configs)
            except json.JSONDecodeError as e:
                return json.dumps({"success": False, "error": f"Invalid JSON in grasp_configs: {str(e)}"}, indent=2)

        # Ensure it's a list
        if not isinstance(grasp_configs, list):
            return json.dumps({"success": False, "error": f"grasp_configs must be a list, got: {type(grasp_configs).__name__}"}, indent=2)
        processed_configs = []
        for i, config in enumerate(grasp_configs):
            # Ensure config is a dict
            if not isinstance(config, dict):
                return json.dumps({"success": False, "error": f"Config at index {i} must be a dictionary, got: {type(config).__name__}"}, indent=2)

            # Check for required fields
            if "grasp_id" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'grasp_id' field"}, indent=2)

            if "status" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'status' field"}, indent=2)

            if "gripper_state" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'gripper_state' field"}, indent=2)
            # Reject any extra fields - only allow these three fields
            allowed_fields = {"grasp_id", "status", "gripper_state"}
            extra_fields = set(config.keys()) - allowed_fields
            if extra_fields:
                return json.dumps({"success": False, "error": f"Invalid fields found: {list(extra_fields)}. Only 'grasp_id', 'gripper_state', and 'status' are allowed."}, indent=2)

            # Validate grasp_id is an integer
            try:
                grasp_id = int(config.get("grasp_id"))
            except (ValueError, TypeError):
                return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {config.get('grasp_id')}"}, indent=2)

            # Validate status
            status = config.get("status", "").upper()
            if status not in ["SUCCESS", "FAILURE"]:
                return json.dumps({"success": False, "error": f"Invalid status: {config.get('status')}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)

            # Validate gripper_state
            gripper_state = config.get("gripper_state")
            if gripper_state not in ["open", "half-open"]:
                return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)

            # Store all three fields (only these three, no extra fields)
            processed_configs.append({
                "grasp_id": grasp_id,
                "gripper_state": gripper_state,
                "status": status  # Store as uppercase
            })

        # Store as list - includes both success and failure attempts with all three fields
        data[object_name] = processed_configs

    save_grasp_resource(assembly_id, data)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_grasp_resource(assembly_id: str, object_name: str) -> str:
    """
    Clear/delete a grasp resource for an object in a specific assembly
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object to clear

    Returns:
        JSON string with confirmation or error message
    """
    data = load_grasp_resource(assembly_id)
    if object_name not in data:
        return json.dumps({"success": False}, indent=2)

    del data[object_name]

    save_grasp_resource(assembly_id, data)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def list_grasp_resource(assembly_id: str) -> str:
    """
    List all objects in the grasp resource for a specific assembly

    Args:
        assembly_id: The ID of the assembly

    Returns:
        JSON string containing object names for the assembly
    """
    data = load_grasp_resource(assembly_id)
    return json.dumps({
        "assembly_id": assembly_id,
        "object_names": list(data.keys()),
        "count": len(data)
    }, indent=2)

# ========== TOOLS FOR ASSEMBLY RESOURCE (Assembly{id}/sequence/object_name/grasp_id) ==========

@mcp.tool()
def read_assembly_resource(assembly_id: str) -> str:
    """
    Read the complete resource for an assembly (sequence with sequence_id, object_name, and primitives_trials)
    Args:
        assembly_id: The ID of the assembly

    Returns:
        JSON string containing the assembly sequence with sequence_id (fixed), object_name (fixed), and primitives_trials (list of trials with trial_id, grasp_id, gripper_state, primitives ordered sequence, and status)
    """
    sequence = load_assembly_resource(assembly_id)
    if not sequence:
        return json.dumps({
            "assembly_id": assembly_id,
            "sequence": [],
            "message": "Resource not found"
        }, indent=2)

    return json.dumps({
        "assembly_id": assembly_id,
        "sequence": sequence
    }, indent=2)

@mcp.tool()
def write_assembly_resource(assembly_id: str, sequence: list) -> str:
    """
    Write or update an assembly resource (complete sequence). sequence_id and object_name are fixed and cannot be changed after creation.
    Args:
        assembly_id: The ID of the assembly
        sequence: List of objects in the sequence, each with sequence_id (fixed), object_name (fixed), and primitives_trials
                  Example: [{"sequence_id": 1, "object_name": "line_brown", "primitives_trials": [{"trial_id": 1, "grasp_id": 1, "gripper_state": "half-open", "primitives": ["grasp", "move", "place"], "status": "SUCCESS"}]}, ...]
                  - sequence_id: integer (required, fixed)
                  - object_name: string (required, fixed)
                  - primitives_trials: list of trial objects (required), each with:
                    - trial_id: integer (required)
                    - grasp_id: integer (required, updatable)
                    - gripper_state: "open" or "half-open" (required, updatable)
                    - primitives: ordered list of strings (required, updatable) - sequence of primitive names executed in order
                    - status: "SUCCESS" or "FAILURE" (required)
    Returns:
        JSON string with confirmation or error message
    """
    # Validate sequence items
    for i, item in enumerate(sequence):
        if not isinstance(item, dict):
            return json.dumps({"success": False, "error": f"Item at index {i} must be a dictionary"}, indent=2)
        # Check for required fields
        required_fields = {"sequence_id", "object_name", "primitives_trials"}
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            return json.dumps({"success": False, "error": f"Item at index {i} missing required fields: {list(missing_fields)}"}, indent=2)
        # Validate sequence_id is an integer
        try:
            sequence_id = int(item.get("sequence_id"))
        except (ValueError, TypeError):
            return json.dumps({"success": False, "error": f"sequence_id must be an integer, got: {item.get('sequence_id')}"}, indent=2)
        # Validate primitives_trials is a list
        primitives_trials = item.get("primitives_trials")
        if not isinstance(primitives_trials, list):
            return json.dumps({"success": False, "error": f"primitives_trials must be a list, got: {type(primitives_trials).__name__}"}, indent=2)
        # Validate each trial
        validated_trials = []
        for j, trial in enumerate(primitives_trials):
            if not isinstance(trial, dict):
                return json.dumps({"success": False, "error": f"Trial at index {j} in item {i} must be a dictionary"}, indent=2)
            # Check for required fields in trial
            trial_required_fields = {"trial_id", "grasp_id", "gripper_state", "primitives", "status"}
            trial_missing_fields = trial_required_fields - set(trial.keys())
            if trial_missing_fields:
                return json.dumps({"success": False, "error": f"Trial at index {j} in item {i} missing required fields: {list(trial_missing_fields)}"}, indent=2)
            # Validate trial_id
            try:
                trial_id = int(trial.get("trial_id"))
            except (ValueError, TypeError):
                return json.dumps({"success": False, "error": f"trial_id must be an integer, got: {trial.get('trial_id')}"}, indent=2)
            # Validate grasp_id
            try:
                grasp_id = int(trial.get("grasp_id"))
            except (ValueError, TypeError):
                return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {trial.get('grasp_id')}"}, indent=2)
            # Validate gripper_state
            gripper_state = trial.get("gripper_state")
            if gripper_state not in ["open", "half-open"]:
                return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)
            # Validate primitives is an ordered list (sequence)
            primitives = trial.get("primitives")
            if not isinstance(primitives, list):
                return json.dumps({"success": False, "error": f"primitives must be an ordered list (sequence), got: {type(primitives).__name__}"}, indent=2)
            # Validate each primitive in the sequence is a string
            for k, primitive in enumerate(primitives):
                if not isinstance(primitive, str):
                    return json.dumps({"success": False, "error": f"Primitive at index {k} in trial {j} of item {i} must be a string, got: {type(primitive).__name__}"}, indent=2)
            # Validate status
            status = trial.get("status", "").upper()
            if status not in ["SUCCESS", "FAILURE"]:
                return json.dumps({"success": False, "error": f"Invalid status: {trial.get('status')}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
            # Reject any extra fields in trial
            trial_allowed_fields = {"trial_id", "grasp_id", "gripper_state", "primitives", "status"}
            trial_extra_fields = set(trial.keys()) - trial_allowed_fields
            if trial_extra_fields:
                return json.dumps({"success": False, "error": f"Invalid fields found in trial {j} of item {i}: {list(trial_extra_fields)}. Only 'trial_id', 'grasp_id', 'gripper_state', 'primitives', and 'status' are allowed."}, indent=2)
            validated_trials.append({
                "trial_id": trial_id,
                "grasp_id": grasp_id,
                "gripper_state": gripper_state,
                "primitives": primitives,
                "status": status
            })
        # Reject any extra fields - only allow these three fields
        allowed_fields = {"sequence_id", "object_name", "primitives_trials"}
        extra_fields = set(item.keys()) - allowed_fields
        if extra_fields:
            return json.dumps({"success": False, "error": f"Invalid fields found: {list(extra_fields)}. Only 'sequence_id', 'object_name', and 'primitives_trials' are allowed."}, indent=2)
        # Store validated item
        sequence[i] = {
            "sequence_id": sequence_id,
            "object_name": item.get("object_name"),
            "primitives_trials": validated_trials
        }

    save_assembly_resource(assembly_id, sequence)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def update_assembly_trial(assembly_id: str, object_name: str, trial_id: int, grasp_id: Optional[int] = None, gripper_state: Optional[str] = None, primitives: Optional[List[str]] = None) -> str:
    """
    Update a specific trial for an object in an assembly. sequence_id and object_name are fixed and cannot be changed.
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object (fixed, cannot be changed)
        trial_id: The ID of the trial to update
        grasp_id: Optional new grasp_id to update (if provided)
        gripper_state: Optional new gripper_state to update ("open" or "half-open", if provided)
        primitives: Optional new primitives ordered sequence to update (if provided)
    Returns:
        JSON string with confirmation or error message
    """
    sequence = load_assembly_resource(assembly_id)
    # Find the object
    object_found = False
    for item in sequence:
        if item.get("object_name") == object_name:
            object_found = True
            trials = item.get("primitives_trials", [])
            # Find the trial
            trial_found = False
            for trial in trials:
                if trial.get("trial_id") == trial_id:
                    trial_found = True
                    # Update fields if provided
                    if grasp_id is not None:
                        try:
                            trial["grasp_id"] = int(grasp_id)
                        except (ValueError, TypeError):
                            return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {grasp_id}"}, indent=2)
                    if gripper_state is not None:
                        if gripper_state not in ["open", "half-open"]:
                            return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)
                        trial["gripper_state"] = gripper_state
                    if primitives is not None:
                        if not isinstance(primitives, list):
                            return json.dumps({"success": False, "error": f"primitives must be an ordered list (sequence), got: {type(primitives).__name__}"}, indent=2)
                        # Validate each primitive is a string
                        for k, primitive in enumerate(primitives):
                            if not isinstance(primitive, str):
                                return json.dumps({"success": False, "error": f"Primitive at index {k} must be a string, got: {type(primitive).__name__}"}, indent=2)
                        trial["primitives"] = primitives

                    break

            if not trial_found:
                return json.dumps({"success": False, "error": f"Trial {trial_id} not found for object '{object_name}' in Assembly{assembly_id}"}, indent=2)

            break

    if not object_found:
        return json.dumps({"success": False, "error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

    save_assembly_resource(assembly_id, sequence)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def add_assembly_trial(assembly_id: str, object_name: str, trial_id: int, grasp_id: int, gripper_state: str, primitives: List[str], status: str) -> str:
    """
    Add a new trial to an existing object in an assembly. sequence_id and object_name are fixed.
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object (must already exist)
        trial_id: The ID of the new trial
        grasp_id: The grasp_id for this trial
        gripper_state: "open" or "half-open"
        primitives: Ordered list of primitive names executed in order
        status: "SUCCESS" or "FAILURE"
    Returns:
        JSON string with confirmation or error message
    """
    sequence = load_assembly_resource(assembly_id)
    # Find the object
    object_found = False
    for item in sequence:
        if item.get("object_name") == object_name:
            object_found = True
            trials = item.get("primitives_trials", [])
            # Check if trial_id already exists
            for trial in trials:
                if trial.get("trial_id") == trial_id:
                    return json.dumps({"success": False, "error": f"Trial {trial_id} already exists for object '{object_name}' in Assembly{assembly_id}"}, indent=2)
            # Validate inputs
            try:
                grasp_id = int(grasp_id)
            except (ValueError, TypeError):
                return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {grasp_id}"}, indent=2)

            if gripper_state not in ["open", "half-open"]:
                return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)

            if not isinstance(primitives, list):
                return json.dumps({"success": False, "error": f"primitives must be an ordered list (sequence), got: {type(primitives).__name__}"}, indent=2)

            for k, primitive in enumerate(primitives):
                if not isinstance(primitive, str):
                    return json.dumps({"success": False, "error": f"Primitive at index {k} must be a string, got: {type(primitive).__name__}"}, indent=2)

            status_upper = status.upper()
            if status_upper not in ["SUCCESS", "FAILURE"]:
                return json.dumps({"success": False, "error": f"Invalid status: {status}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)

            # Add the new trial
            trials.append({
                "trial_id": trial_id,
                "grasp_id": grasp_id,
                "gripper_state": gripper_state,
                "primitives": primitives,
                "status": status_upper
            })

            break

    if not object_found:
        return json.dumps({"success": False, "error": f"Object '{object_name}' not found in Assembly{assembly_id}. Use write_assembly_resource to create it first."}, indent=2)

    save_assembly_resource(assembly_id, sequence)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_assembly_resource(assembly_id: str) -> str:
    """
    Clear/delete an assembly resource

    Args:
        assembly_id: The ID of the assembly to clear

    Returns:
        JSON string with confirmation or error message
    """
    assembly_file = get_assembly_file(assembly_id)

    if not assembly_file.exists():
        return json.dumps({"success": False}, indent=2)

    # Delete the file
    try:
        assembly_file.unlink()
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.tool()
def list_assembly_resource() -> str:
    """
    List all assemblies in the assembly resource

    Returns:
        JSON string containing all assembly IDs
    """
    # Find all Assembly{id}_sequence_log.json files
    assembly_files = list(RESOURCES_DIR.glob("Assembly*_sequence_log.json"))
    assembly_ids = []
    for file in assembly_files:
        # Extract assembly ID from filename like "Assembly1_sequence_log.json"
        name = file.stem  # "Assembly1_sequence_log"
        if name.startswith("Assembly") and name.endswith("_sequence_log"):
            assembly_id = name.replace("Assembly", "").replace("_sequence_log", "")
            assembly_ids.append(assembly_id)
    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

# ========== TOOLS FOR FINAL SEQUENCE (Assembly{id}_final_sequence.json) ==========

@mcp.tool()
def read_final_sequence(assembly_id: str) -> str:
    """
    Read the final sequence for an assembly (sequence_id, object_name, grasp_id, gripper_state, and primitives)

    Args:
        assembly_id: The ID of the assembly

    Returns:
        JSON string containing the final sequence with sequence_id, object_name, grasp_id, gripper_state, and primitives (ordered sequence)
    """
    sequence = load_final_sequence(assembly_id)
    if not sequence:
        return json.dumps({
            "assembly_id": assembly_id,
            "sequence": [],
            "message": "Resource not found"
        }, indent=2)
    return json.dumps({
        "assembly_id": assembly_id,
        "sequence": sequence
    }, indent=2)

@mcp.tool()
def write_final_sequence(assembly_id: str, sequence: list) -> str:
    """
    Write or update the final sequence for an assembly
    Args:
        assembly_id: The ID of the assembly
        sequence: List of objects in the final sequence, each with sequence_id, object_name, grasp_id, gripper_state, and primitives
                  Example: [{"sequence_id": 1, "object_name": "line_brown", "grasp_id": 1, "gripper_state": "half-open", "primitives": ["grasp", "move_to_assembly", "reorient", "place"]}, ...]
                  - sequence_id: integer (required)
                  - object_name: string (required)
                  - grasp_id: integer (required)
                  - gripper_state: "open" or "half-open" (required)
                  - primitives: ordered list of strings (required) - sequence of primitive names executed in order
    Returns:
        JSON string with confirmation or error message
    """
    # Validate sequence items
    for i, item in enumerate(sequence):
        if not isinstance(item, dict):
            return json.dumps({"success": False, "error": f"Item at index {i} must be a dictionary"}, indent=2)
        # Check for required fields
        required_fields = {"sequence_id", "object_name", "grasp_id", "gripper_state", "primitives"}
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            return json.dumps({"success": False, "error": f"Item at index {i} missing required fields: {list(missing_fields)}"}, indent=2)
        # Validate sequence_id is an integer
        try:
            sequence_id = int(item.get("sequence_id"))
        except (ValueError, TypeError):
            return json.dumps({"success": False, "error": f"sequence_id must be an integer, got: {item.get('sequence_id')}"}, indent=2)
        # Validate grasp_id is an integer
        try:
            grasp_id = int(item.get("grasp_id"))
        except (ValueError, TypeError):
            return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {item.get('grasp_id')}"}, indent=2)
        # Validate gripper_state
        gripper_state = item.get("gripper_state")
        if gripper_state not in ["open", "half-open"]:
            return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)
        # Validate primitives is an ordered list (sequence)
        primitives = item.get("primitives")
        if not isinstance(primitives, list):
            return json.dumps({"success": False, "error": f"primitives must be an ordered list (sequence), got: {type(primitives).__name__}"}, indent=2)
        # Validate each primitive in the sequence is a string
        for j, primitive in enumerate(primitives):
            if not isinstance(primitive, str):
                return json.dumps({"success": False, "error": f"Primitive at index {j} in item {i} must be a string, got: {type(primitive).__name__}"}, indent=2)
        # Reject any extra fields - only allow these five fields
        allowed_fields = {"sequence_id", "object_name", "grasp_id", "gripper_state", "primitives"}
        extra_fields = set(item.keys()) - allowed_fields
        if extra_fields:
            return json.dumps({"success": False, "error": f"Invalid fields found: {list(extra_fields)}. Only 'sequence_id', 'object_name', 'grasp_id', 'gripper_state', and 'primitives' are allowed."}, indent=2)
        # Store validated item
        sequence[i] = {
            "sequence_id": sequence_id,
            "object_name": item.get("object_name"),
            "grasp_id": grasp_id,
            "gripper_state": gripper_state,
            "primitives": primitives
        }

    save_final_sequence(assembly_id, sequence)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_final_sequence(assembly_id: str) -> str:
    """
    Clear/delete the final sequence for an assembly

    Args:
        assembly_id: The ID of the assembly to clear

    Returns:
        JSON string with confirmation or error message
    """
    final_file = get_final_sequence_file(assembly_id)

    if not final_file.exists():
        return json.dumps({"success": False}, indent=2)

    # Delete the file
    try:
        final_file.unlink()
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

if __name__ == "__main__":
    mcp.run()
 