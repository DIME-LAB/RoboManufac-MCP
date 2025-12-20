"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# isaac_sim_mcp_server.py
import time
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse

# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
# Directories are created lazily when needed by tools
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()

if BASE_OUTPUT_DIR:
    # If BASE_OUTPUT_DIR is set, ensure it's absolute and use it
    BASE_OUTPUT_DIR = os.path.abspath(BASE_OUTPUT_DIR)
    RESOURCES_DIR = os.path.join(BASE_OUTPUT_DIR, "resources")
else:
    # If not set, use relative path (will be relative to where the server is run from)
    RESOURCES_DIR = "resources"

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsaacMCPServer")

@dataclass
class IsaacConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Isaac: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Isaac addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Isaac: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=16384):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(300.0)  # Match the extension's timeout
        
        try:
            while True:
                try:
                    logger.info("Waiting for data from Isaac")
                    #time.sleep(0.5)
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(300.0)  # Match the extension's timeout
            
            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Isaac error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Isaac"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Isaac")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Isaac response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Isaac: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Isaac: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Isaac: {str(e)}")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Isaac: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools
    
    try:
        # Just log that we're starting up
        logger.info("IsaacMCP server starting up")
        
        # Try to connect to Isaac on startup to verify it's available
        try:
            # This will initialize the global connection if needed
            isaac = get_isaac_connection()
            logger.info("Successfully connected to Isaac on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Isaac on startup: {str(e)}")
            logger.warning("Make sure the Isaac addon is running before using Isaac resources or tools")
        
        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _isaac_connection
        if _isaac_connection:
            logger.info("Disconnecting from Isaac Sim on shutdown")
            _isaac_connection.disconnect()
            _isaac_connection = None
        logger.info("Isaac SimMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "IsaacSimMCP",
    description="Isaac Sim integration through the Model Context Protocol",
    lifespan=server_lifespan
)

# Resource endpoints

# Global connection for resources (since resources can't access context)
_isaac_connection = None
# _polyhaven_enabled = False  # Add this global variable

def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals
    
    # If we have an existing connection, check if it's still valid
    if _isaac_connection is not None:
        try:
            
            return _isaac_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _isaac_connection.disconnect()
            except:
                pass
            _isaac_connection = None
    
    # Create a new connection if needed
    if _isaac_connection is None:
        _isaac_connection = IsaacConnection(host="localhost", port=8766)
        if not _isaac_connection.connect():
            logger.error("Failed to connect to Isaac")
            _isaac_connection = None
            raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
        logger.info("Created new persistent connection to Isaac")
    
    return _isaac_connection


# @mcp.tool()
# def execute_script(ctx: Context, code: str) -> str:
#     """
#     SIMULATION CONTROL FUNCTIONS:
#     
#     To stop the simulation:
#     import omni.timeline
#     def stop_simulation():
#         timeline = omni.timeline.get_timeline_interface()
#         if timeline.is_playing() or timeline.is_paused():
#             timeline.stop()
#             print("✓ Simulation stopped")
#         else:
#             print("⚠ Simulation is already stopped")
#     stop_simulation()
#
#     To play/start the simulation:
#     import omni.timeline
#     def play_simulation():
#         timeline = omni.timeline.get_timeline_interface()
#         if timeline.is_stopped():
#             timeline.play()
#             print("✓ Simulation started")
#         elif timeline.is_playing():
#             print("⚠ Simulation is already playing")
#         else:
#             print("⚠ Simulation is paused, resuming...")
#             timeline.play()
#     play_simulation()
#
#     Parameters:
#     - code: The Python code to execute, e.g. "omni.kit.commands.execute("CreatePrim", prim_type="Sphere"), play or stop simulation to reset the scene"
#     """
#     try:
#         # Get the global connection
#         isaac = get_isaac_connection()
#         print("code: ", code)
#         
#         result = isaac.send_command("execute_script", {"code": code})
#         print("result: ", result)
#         return result
#         # return f"Code executed successfully: {result.get('result', '')}"
#     except Exception as e:
#         logger.error(f"Error executing code: {str(e)}")
#         # return f"Error executing code: {str(e)}"
#         return {"status": "error", "error": str(e), "message": "Error executing code"}

    
# @mcp.tool()
# def list_prims(path: str = "/World/Objects") -> str:
#     """
#     List only the direct children of the specified path (default: /World/Objects).
#     Useful for discovering available objects before moving, reading poses, or other operations.
#     
#     Args:
#         path: The prim path to list children from (default: "/World/Objects")
#     
#     Example:
#         list_prims()
#         list_prims("/World/Objects")
#         list_prims("/World")
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("list_prims", {"path": path})
#         
#         if result.get("status") == "success":
#             prims = result.get("prims", [])
#             
#             response = f"Direct children of {path} ({len(prims)}):\n\n"
#             
#             for prim in prims:
#                 prim_path = prim['path']
#                 name = prim['name']
#                 prim_type = prim['type']
#                 
#                 # Format: path (type) or path [name] (type) if name differs
#                 if name and name != prim_path.split('/')[-1]:
#                     response += f"{prim_path} [{name}] ({prim_type})\n"
#                 else:
#                     response += f"{prim_path} ({prim_type})\n"
#             
#             return response
#         else:
#             return f"Error listing prims: {result.get('message', 'Unknown error')}"
#             
#     except Exception as e:
#         logger.error(f"Error listing prims: {str(e)}")
#         return f"Error listing prims: {str(e)}"


# @mcp.tool()
# def stop_scene() -> str:
#     """
#     Stop the simulation timeline.
#     Works exactly like the execute_script stop_simulation example.
    
#     Example:
#         stop_scene()
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("stop_scene")
        
#         if result.get("status") == "success":
#             return result.get("message", "Simulation stopped")
#         else:
#             return f"Error stopping scene: {result.get('message', 'Unknown error')}"
            
#     except Exception as e:
#         logger.error(f"Error stopping scene: {str(e)}")
#         return f"Error stopping scene: {str(e)}"


# @mcp.tool()
# def play_scene() -> str:
#     """
#     Start/resume the simulation timeline.
#     Works exactly like the execute_script play_simulation example.
    
#     Example:
#         play_scene()
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("play_scene")
        
#         if result.get("status") == "success":
#             return result.get("message", "Simulation started")
#         else:
#             return f"Error playing scene: {result.get('message', 'Unknown error')}"
            
#     except Exception as e:
#         logger.error(f"Error playing scene: {str(e)}")
#         return f"Error playing scene: {str(e)}"


# @mcp.tool()
# def reset_scene() -> str:
#     """
#     Reset the simulation by stopping and then starting it again.
#     This tool first stops the simulation timeline, then starts it.
#     
#     Example:
#         reset_scene()
#     """
#     try:
#         # First, stop the scene
#         stop_message = stop_scene()
#         if stop_message.startswith("Error"):
#             return stop_message
#         
#         # Then, start the scene
#         play_message = play_scene()
#         if play_message.startswith("Error"):
#             return play_message
#         
#         return f"{stop_message}\n{play_message}"
#             
#     except Exception as e:
#         logger.error(f"Error resetting scene: {str(e)}")
#         return f"Error resetting scene: {str(e)}"


@mcp.tool()
def save_scene_state() -> str:
    """
    Save scene state (object poses) to a JSON file.
    
    This tool automatically discovers and saves all objects in /World/Objects to a JSON file when a state is verified.
    Use this when objects are in a verified/assembled state that you want to restore later.
    The code automatically finds the prim path from the object name using the pattern /World/Objects/{object_name}/{object_name}/{object_name}.
    
    Example:
        save_scene_state()
    """
    try:
        # Calculate the file path on the server side (where we have access to MCP_CLIENT_OUTPUT_DIR)
        resources_dir = os.path.abspath(RESOURCES_DIR)
        json_file_path = os.path.join(resources_dir, "scene_state.json")
        
        isaac = get_isaac_connection()
        result = isaac.send_command("save_scene_state", {
            "json_file_path": json_file_path
        })
        
        if result.get("status") == "success":
            message = result.get("message", "Scene state saved successfully")
            saved_count = result.get("saved_count")
            failed_names = result.get("failed_names", [])
            
            response = message
            if saved_count is not None:
                response += f"\nSaved: {saved_count} object(s)"
            if failed_names:
                response += f"\nFailed objects: {failed_names}"
            
            return response
        else:
            return f"Error: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in save_scene_state: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def restore_scene_state() -> str:
    """
    Restore scene state (object poses) from a JSON file.
    
    This tool automatically restores all objects from a previously saved JSON file. Move home before using this tool
    to restore objects to their previously verified/assembled state.
    The code automatically finds the prim path from the object name using the pattern /World/Objects/{object_name}/{object_name}/{object_name}.
    
    Example:
        restore_scene_state()
    """
    try:
        # Calculate the file path on the server side (where we have access to MCP_CLIENT_OUTPUT_DIR)
        resources_dir = os.path.abspath(RESOURCES_DIR)
        json_file_path = os.path.join(resources_dir, "scene_state.json")
        
        isaac = get_isaac_connection()
        result = isaac.send_command("restore_scene_state", {
            "json_file_path": json_file_path
        })
        
        if result.get("status") == "success":
            message = result.get("message", "Scene state restored successfully")
            restored_count = result.get("restored_count")
            failed_names = result.get("failed_names", [])
            
            response = message
            if restored_count is not None:
                response += f"\nRestored: {restored_count} object(s)"
            if failed_names:
                response += f"\nFailed objects: {failed_names}"
            
            return response
        else:
            return f"Error: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in restore_scene_state: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def read_scene_state() -> str:
    """
    Read scene state (object poses) from the JSON file without applying them.
    
    This tool allows you to view the saved poses in the scene state file without restoring them to the scene.
    Useful for checking what poses were saved before restoring.
    Reads from the fixed file: scene_state.json
    
    Example:
        read_scene_state()
    """
    try:
        # Calculate the file path on the server side (where we have access to MCP_CLIENT_OUTPUT_DIR)
        resources_dir = os.path.abspath(RESOURCES_DIR)
        json_file_path = os.path.join(resources_dir, "scene_state.json")
        
        isaac = get_isaac_connection()
        result = isaac.send_command("read_scene_state", {
            "json_file_path": json_file_path
        })
        
        if result.get("status") == "success":
            message = result.get("message", "Scene state read successfully")
            json_file_path = result.get("json_file_path", "scene_state.json")
            object_count = result.get("object_count", 0)
            object_names = result.get("object_names", [])
            summary = result.get("summary", [])
            
            response = f"{message}\n"
            response += f"File: {json_file_path}\n"
            response += f"Objects saved: {object_count}\n"
            response += f"Object names: {', '.join(object_names)}\n\n"
            
            # Add detailed pose information for each object
            for item in summary:
                obj_name = item.get("object_name", "unknown")
                pos = item.get("position", [0, 0, 0])
                quat = item.get("quaternion", [1, 0, 0, 0])
                scale = item.get("scale", [1, 1, 1])
                response += f"{obj_name}:\n"
                response += f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
                response += f"  Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]\n"
                response += f"  Scale: [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]\n\n"
            
            return response
        else:
            return f"Error: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in read_scene_state: {str(e)}")
        return f"Error: {str(e)}"


# @mcp.tool()
# def list_scene_states(directory: str = None) -> str:
#     """
#     List all available scene state JSON files in a directory.
#     
#     This tool helps you discover available scene state files that were previously saved.
#     Use this to find the exact filename before using read_scene_state() or restore_scene_state().
#     
#     By default, scans the current working directory (same location where save_scene_state saves files
#     when using a simple filename like "scene_state.json"). If you saved files to a specific directory,
#     provide that same directory path here.
#     
#     Args:
#         directory: Optional directory path to search for scene state files. 
#                    If None, scans the current working directory (same as save_scene_state default location)
#     
#     Example:
#         # List files in current directory (where save_scene_state saves by default)
#         list_scene_states()
#         
#         # List files in a specific directory
#         list_scene_states("./states")
#         list_scene_states("states")
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("list_scene_states", {
#             "directory": directory
#         })
#         
#         if result.get("status") == "success":
#             files = result.get("files", [])
#             file_count = result.get("file_count", 0)
#             
#             response = f"Available scene state files ({file_count}):\n\n"
#             
#             if files:
#                 for file_info in files:
#                     filename = file_info.get("filename", "unknown")
#                     file_path = file_info.get("path", filename)
#                     object_count = file_info.get("object_count", 0)
#                     object_names = file_info.get("object_names", [])
#                     
#                     response += f"{filename}\n"
#                     response += f"  Path: {file_path}\n"
#                     response += f"  Objects: {object_count}"
#                     if object_names:
#                         response += f" ({', '.join(object_names)})"
#                     response += "\n\n"
#             else:
#                 response += "No scene state files found.\n"
#             
#             return response
#         else:
#             return f"Error: {result.get('message', 'Unknown error')}"
#             
#     except Exception as e:
#         logger.error(f"Error in list_scene_states: {str(e)}")
#         return f"Error: {str(e)}"


@mcp.tool()
def clear_scene_state() -> str:
    """
    Clear/delete the scene state JSON file.
    
    This tool deletes the entire scene state file, removing all saved object poses.
    
    Example:
        clear_scene_state()
    """
    try:
        # Calculate the file path on the server side (where we have access to MCP_CLIENT_OUTPUT_DIR)
        resources_dir = os.path.abspath(RESOURCES_DIR)
        json_file_path = os.path.join(resources_dir, "scene_state.json")
        
        isaac = get_isaac_connection()
        result = isaac.send_command("clear_scene_state", {
            "json_file_path": json_file_path
        })
        
        if result.get("status") == "success":
            message = result.get("message", "Scene state cleared successfully")
            return message
        else:
            return f"Error: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in clear_scene_state: {str(e)}")
        return f"Error: {str(e)}"


# @mcp.tool()
# def get_object_info(prim_path: str) -> str:
#     """
#     Get comprehensive information about an object including its pose in all formats.
#     
#     Args:
#         prim_path: Path to the prim/object (use list_prims() to find available paths)
#     
#     Example:
#         get_object_info("/World/UR5e")
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("get_object_info", {"prim_path": prim_path})
#         
#         if result.get("status") == "success":
#             prim_info = result.get("prim_info", {})
#             position = result.get("position", [])
#             quat = result.get("rotation_quaternion", [])
#             rad = result.get("rotation_radians", [])
#             deg = result.get("rotation_degrees", [])
#             scale = result.get("scale", [])
#             
#             response = f"Object Info for {prim_path}:\n\n"
#             
#             # Basic info
#             response += f"Basic Properties:\n"
#             response += f"  Name: {prim_info.get('name', 'N/A')}\n"
#             response += f"  Type: {prim_info.get('type', 'N/A')}\n"
#             response += f"  Active: {prim_info.get('is_active', False)}\n"
#             response += f"  Has Children: {prim_info.get('has_children', False)}\n\n"
#             
#             # Transform info
#             response += f"Transform:\n"
#             response += f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]\n\n"
#             response += f"  Rotation:\n"
#             response += f"    Degrees (r,p,y): [{deg[0]:.3f}, {deg[1]:.3f}, {deg[2]:.3f}]\n"
#             response += f"    Radians (r,p,y): [{rad[0]:.3f}, {rad[1]:.3f}, {rad[2]:.3f}]\n"
#             response += f"    Quaternion (w,x,y,z): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]\n\n"
#             response += f"  Scale: [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]"
#             
#             return response
#         else:
#             return f"Error getting object info: {result.get('message', 'Unknown error')}"
#             
#     except Exception as e:
#         logger.error(f"Error getting object info: {str(e)}")
#         return f"Error getting object info: {str(e)}"


# @mcp.tool()
# def move_prim(prim_path: str, position: list = None, orientation: list = None, 
#               orientation_format: str = "degrees") -> str:
#     """
#     Move a prim to a new pose with flexible orientation input.
#     Reads current pose and allows selective updates to position and orientation.
#     
#     Args:
#         prim_path: Path to the prim/object (use list_prims() to find available paths)
#         position: [x, y, z] new position (optional, keeps current if not specified)
#         orientation: New orientation values (optional, keeps current if not specified)
#         orientation_format: "degrees", "radians", or "quaternion" (default: degrees)
#     
#     Examples:
#         # Move to new position only
#         move_prim("/World/trial", position=[1.0, 2.0, 0.5])
#         
#         # Move with rotation in degrees
#         move_prim("/World/trial", position=[1.0, 2.0, 0.5], orientation=[0, 0, 45])
#         
#         # Move with rotation in radians
#         move_prim("/World/trial", position=[1.0, 2.0, 0.5], orientation=[0, 0, 0.785], orientation_format="radians")
#         
#         # Move with quaternion
#         move_prim("/World/trial", position=[1.0, 2.0, 0.5], orientation=[0.924, 0, 0, 0.383], orientation_format="quaternion")
#         
#         # Just rotate, keep current position
#         move_prim("/World/trial", orientation=[90, 0, 0])
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("move_prim", {
#             "prim_path": prim_path,
#             "position": position,
#             "orientation": orientation,
#             "orientation_format": orientation_format
#         })
#         
#         if result.get("status") == "success":
#             prev_pos = result.get("previous_position", [])
#             prev_quat = result.get("previous_quaternion", [])
#             new_pos = result.get("new_position", [])
#             new_quat = result.get("new_quaternion", [])
#             
#             response = f"Successfully moved {prim_path}:\n\n"
#             
#             response += f"Previous Pose:\n"
#             response += f"  Position: [{prev_pos[0]:.3f}, {prev_pos[1]:.3f}, {prev_pos[2]:.3f}]\n"
#             response += f"  Quaternion: [{prev_quat[0]:.3f}, {prev_quat[1]:.3f}, {prev_quat[2]:.3f}, {prev_quat[3]:.3f}]\n\n"
#             
#             response += f"New Pose:\n"
#             response += f"  Position: [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]\n"
#             response += f"  Quaternion: [{new_quat[0]:.3f}, {new_quat[1]:.3f}, {new_quat[2]:.3f}, {new_quat[3]:.3f}]\n"
#             response += f"  Orientation Format: {result.get('orientation_format', 'degrees')}"
#             
#             return response
#         else:
#             return f"Error moving prim: {result.get('message', 'Unknown error')}"
#             
#     except Exception as e:
#         logger.error(f"Error moving prim: {str(e)}")
#         return f"Error moving prim: {str(e)}"



# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()