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

GRASP_RESOURCE_FILE = RESOURCES_DIR / "grasp_resource.json"
ASSEMBLY_RESOURCE_FILE = RESOURCES_DIR / "assembly_resource.json"

# Ensure resources directory exists
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# ========== HELPER FUNCTIONS ==========

def load_grasp_resource():
    """Load grasp resource from JSON file"""
    if not GRASP_RESOURCE_FILE.exists():
        return {}
    
    try:
        with open(GRASP_RESOURCE_FILE, 'r') as f:
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

def save_grasp_resource(data):
    """Save grasp resource to JSON file"""
    with open(GRASP_RESOURCE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_assembly_resource():
    """Load assembly resource from JSON file"""
    if not ASSEMBLY_RESOURCE_FILE.exists():
        return {}
    
    try:
        with open(ASSEMBLY_RESOURCE_FILE, 'r') as f:
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

def save_assembly_resource(data):
    """Save assembly resource to JSON file"""
    with open(ASSEMBLY_RESOURCE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ========== RESOURCES ==========

@mcp.resource("object_name/{object_name}/grasp_configs")
def get_object_grasp_configs(object_name: str) -> str:
    """Get grasp configurations for an object (list of grasp_id, gripper_state, and status) for a specific object_name. Each config includes grasp_id (int), gripper_state ("open" or "half-open"), and status ("SUCCESS" or "FAILURE")."""
    data = load_grasp_resource()
    obj_data = data.get(object_name, {})
    # Support both new format (list) and old format (dict)
    if isinstance(obj_data, list):
        return json.dumps(obj_data, indent=2)
    elif "grasp_configs" in obj_data:
        return json.dumps(obj_data.get("grasp_configs", []), indent=2)
    else:
        # Legacy format - convert to new format
        return json.dumps([], indent=2)

@mcp.resource("object_name/{object_name}/grasp_configs/{grasp_id}/status")
def get_object_status_for_id(object_name: str, grasp_id: str) -> str:
    """Get all configurations (grasp_id, gripper_state, and status) for a specific grasp_id of an object_name. Returns all attempts (both SUCCESS and FAILURE) with their modes."""
    data = load_grasp_resource()
    obj_data = data.get(object_name, {})
    grasp_id_int = int(grasp_id)
    
    # Check if it's a list (new format)
    if isinstance(obj_data, list):
        matching_configs = [config for config in obj_data if config.get("grasp_id") == grasp_id_int]
        if matching_configs:
            return json.dumps(matching_configs, indent=2)
    elif isinstance(obj_data, dict) and "grasp_configs" in obj_data:
        matching_configs = [config for config in obj_data.get("grasp_configs", []) if config.get("grasp_id") == grasp_id_int]
        if matching_configs:
            return json.dumps(matching_configs, indent=2)
    
    return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/sequence")
def get_assembly_sequence(assembly_id: str) -> str:
    """Get sequence for a specific Assembly ID"""
    data = load_assembly_resource()
    assembly_data = data.get(f"Assembly{assembly_id}", {})
    return json.dumps(assembly_data.get("sequence", []), indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/grasp_id")
def get_assembly_object_grasp_id(assembly_id: str, object_name: str) -> str:
    """Get grasp_id for a specific object in an assembly sequence"""
    data = load_assembly_resource()
    assembly_data = data.get(f"Assembly{assembly_id}", {})
    sequence = assembly_data.get("sequence", [])
    
    for item in sequence:
        if item.get("object_name") == object_name:
            result = {"grasp_id": item.get("grasp_id", "")}
            if "gripper_state" in item:
                result["gripper_state"] = item.get("gripper_state", "")
            return json.dumps(result, indent=2)
    
    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/gripper_state")
def get_assembly_object_gripper_state(assembly_id: str, object_name: str) -> str:
    """Get gripper_state for a specific object in an assembly sequence. Valid values are 'open' or 'half-open'."""
    data = load_assembly_resource()
    assembly_data = data.get(f"Assembly{assembly_id}", {})
    sequence = assembly_data.get("sequence", [])
    
    for item in sequence:
        if item.get("object_name") == object_name:
            return json.dumps({"gripper_state": item.get("gripper_state", "")}, indent=2)
    
    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

# ========== TOOLS FOR GRASP RESOURCE (object_name/grasp_configs) ==========

@mcp.tool()
def read_grasp_resource(object_name: str) -> str:
    """
    Read the grasp configurations for an object (grasp_configs with grasp_id, gripper_state, and status)
    
    Args:
        object_name: The name of the object
    
    Returns:
        JSON string containing grasp_configs list with grasp_id, gripper_state, and status (includes both SUCCESS and FAILURE attempts)
        Each config has: {"grasp_id": <int>, "gripper_state": "open"|"half-open", "status": "SUCCESS"|"FAILURE"}
    """
    data = load_grasp_resource()
    obj_data = data.get(object_name, {})
    
    if not obj_data:
        return json.dumps({
            "object_name": object_name,
            "grasp_configs": [],
            "message": "Resource not found"
        }, indent=2)
    
    # Support both new format (list) and old format (dict with grasp_configs)
    if isinstance(obj_data, list):
        return json.dumps({
            "object_name": object_name,
            "grasp_configs": obj_data
        }, indent=2)
    elif isinstance(obj_data, dict) and "grasp_configs" in obj_data:
        return json.dumps({
            "object_name": object_name,
            "grasp_configs": obj_data.get("grasp_configs", [])
        }, indent=2)
    else:
        # Legacy format - return empty
        return json.dumps({
            "object_name": object_name,
            "grasp_configs": []
        }, indent=2)

@mcp.tool()
def write_grasp_resource(object_name: str, grasp_configs: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Write or update grasp configurations for an object (grasp_configs list with grasp_id, gripper_state, and status)
    
    Args:
        object_name: The name of the object
        grasp_configs: List of grasp configurations, each with grasp_id, gripper_state, and status.
                       Format: [{"grasp_id": 1, "gripper_state": "open", "status": "SUCCESS"}, ...]
                       - grasp_id: integer
                       - gripper_state: "open" or "half-open"
                       - status: "SUCCESS" or "FAILURE"
                       Both success and failure attempts are stored.
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_grasp_resource()
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
            
            # Reject any extra fields - only allow exactly these three fields
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
    
    save_grasp_resource(data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_grasp_resource(object_name: str) -> str:
    """
    Clear/delete a grasp resource
    
    Args:
        object_name: The name of the object to clear
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_grasp_resource()
    
    if object_name not in data:
        return json.dumps({"success": False}, indent=2)
    
    del data[object_name]
    save_grasp_resource(data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def list_grasp_resource() -> str:
    """
    List all objects in the grasp resource
    
    Returns:
        JSON string containing all object names
    """
    data = load_grasp_resource()
    return json.dumps({
        "object_names": list(data.keys()),
        "count": len(data)
    }, indent=2)

# ========== TOOLS FOR ASSEMBLY RESOURCE (Assembly{id}/sequence/object_name/grasp_id) ==========

@mcp.tool()
def read_assembly_resource(assembly_id: str) -> str:
    """
    Read the complete resource for an assembly (sequence with object_name, grasp_id, and gripper_state)
    
    Args:
        assembly_id: The ID of the assembly
    
    Returns:
        JSON string containing the assembly sequence with object_name, grasp_id, and optionally gripper_state
    """
    data = load_assembly_resource()
    assembly_key = f"Assembly{assembly_id}"
    assembly_data = data.get(assembly_key, {})
    
    if not assembly_data:
        return json.dumps({
            "assembly_id": assembly_id,
            "sequence": [],
            "message": "Resource not found"
        }, indent=2)
    
    return json.dumps({
        "assembly_id": assembly_id,
        "sequence": assembly_data.get("sequence", [])
    }, indent=2)

@mcp.tool()
def write_assembly_resource(assembly_id: str, sequence: list) -> str:
    """
    Write or update an assembly resource (complete sequence)
    
    Args:
        assembly_id: The ID of the assembly
        sequence: List of objects in the sequence, each with object_name, grasp_id, and optionally gripper_state
                  Example: [{"object_name": "line_brown", "grasp_id": 1, "gripper_state": "half-open"}, ...]
                  gripper_state must be "open" or "half-open" if provided
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_assembly_resource()
    assembly_key = f"Assembly{assembly_id}"
    
    # Validate gripper_state if present in sequence items
    for item in sequence:
        if "gripper_state" in item:
            gripper_state = item["gripper_state"]
            if gripper_state not in ["open", "half-open"]:
                return json.dumps({"success": False}, indent=2)
    
    data[assembly_key] = {
        "sequence": sequence
    }
    
    save_assembly_resource(data)
    
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
    data = load_assembly_resource()
    assembly_key = f"Assembly{assembly_id}"
    
    if assembly_key not in data:
        return json.dumps({"success": False}, indent=2)
    
    del data[assembly_key]
    save_assembly_resource(data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def list_assembly_resource() -> str:
    """
    List all assemblies in the assembly resource
    
    Returns:
        JSON string containing all assembly IDs
    """
    data = load_assembly_resource()
    assembly_ids = [key.replace("Assembly", "") for key in data.keys() if key.startswith("Assembly")]
    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

if __name__ == "__main__":
    mcp.run()
