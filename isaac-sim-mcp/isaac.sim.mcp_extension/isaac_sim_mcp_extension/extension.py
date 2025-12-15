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

"""Extension module for Isaac Sim MCP."""

import asyncio
import carb
# import omni.ext
# import omni.ui as ui
import omni.usd
import threading
import time
import socket
import json
import traceback

import gc
from pxr import Usd, UsdGeom, Sdf, Gf

import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
from typing import Dict, Any, List, Optional, Union
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
import numpy as np
from omni.isaac.core import World
import requests

# Extension Methods required by Omniverse Kit
# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MCPExtension(omni.ext.IExt):
    def __init__(self) -> None:
        """Initialize the extension."""
        super().__init__()
        self.ext_id = None
        self.running = False
        self.host = None
        self.port = None
        self.socket = None
        self.server_thread = None
        self._usd_context = None
        self._physx_interface = None
        self._timeline = None
        self._window = None
        self._status_label = None
        self._server_thread = None
        self._models = None
        self._settings = carb.settings.get_settings()
        self._image_url_cache = {} # cache for image url
        self._text_prompt_cache = {} # cache for text prompt
        

    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        print("trigger  on_startup for: ", ext_id)
        print("settings: ", self._settings.get("/exts/omni.kit.pipapi"))
        self.port = self._settings.get("/exts/isaac.sim.mcp/server, port") or 8766
        self.host = self._settings.get("/exts/isaac.sim.mcp/server.host") or "localhost"
        if not hasattr(self, 'running'):
            self.running = False

        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        # omni.kit.commands.execute("CreatePrim", prim_type="Sphere")

        # print("sphere created")
        # result = self.execute_script('omni.kit.commands.execute("CreatePrim", prim_type="Cube")')
        # print("script executed", result)  
        self._start()
        # result = self.execute_script('omni.kit.commands.execute("CreatePrim", prim_type="Cube")')
        # print("script executed", result)  
    
    def on_shutdown(self):
        print("trigger  on_shutdown for: ", self.ext_id)
        self._models = {}
        gc.collect()
        self._stop()
    
    def _start(self):
        if self.running:
            print("Server is already running")
            return
            
        self.running = True
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print(f"Isaac Sim MCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.stop()
            
    def _stop(self):
        self.running = False
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Wait for thread to finish
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None
        
        print("Isaac Sim MCP server stopped")

    def _server_loop(self):
        """Main server loop in a separate thread"""
        print("Server thread started")
        self.socket.settimeout(1.0)  # Timeout to allow for stopping
        if not hasattr(self, 'running'):
            self.running = False

        while self.running:
            try:
                # Accept new connection
                try:
                    client, address = self.socket.accept()
                    print(f"Connected to client: {address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Just check running condition
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error in server loop: {str(e)}")
                if not self.running:
                    break
                time.sleep(0.5)
        
        print("Server thread stopped")
    
    def _handle_client(self, client):
        """Handle connected client"""
        print("Client handler started")
        client.settimeout(None)  # No timeout
        buffer = b''
        
        try:
            while self.running:
                # Receive data
                try:
                    data = client.recv(16384)
                    if not data:
                        print("Client disconnected")
                        break
                    
                    buffer += data
                    try:
                        # Try to parse command
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''
                        
                        # Execute command in Isaac Sim's main thread
                        async def execute_wrapper():
                            try:
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                print("response_json: ", response_json)
                                try:
                                    client.sendall(response_json.encode('utf-8'))
                                except:
                                    print("Failed to send response - client disconnected")
                            except Exception as e:
                                print(f"Error executing command: {str(e)}")
                                traceback.print_exc()
                                try:
                                    error_response = {
                                        "status": "error",
                                        "message": str(e)
                                    }
                                    client.sendall(json.dumps(error_response).encode('utf-8'))
                                except:
                                    pass
                            return None
                        # import omni.kit.commands
                        # import omni.kit.async
                        from omni.kit.async_engine import run_coroutine
                        task = run_coroutine(execute_wrapper())
                        # import asyncio
                        # asyncio.ensure_future(execute_wrapper())
                        #time.sleep(30)
                        
    
                        # 
                        # omni.kit.async.get_event_loop().create_task(create_sphere_async())
                        # TODO:Schedule execution in main thread
                        # bpy.app.timers.register(execute_wrapper, first_interval=0.0)
                        # omni.kit.app.get_app().post_to_main_thread(execute_wrapper())
                        # carb.apputils.get_app().get_update_event_loop().post(execute_wrapper)

                        # from omni.kit.async_engine import run_coroutine
                        # run_coroutine(execute_wrapper())
                        # omni.kit.app.get_app().get_update_event_stream().push(0, 0, {"fn": execute_wrapper})
                    except json.JSONDecodeError:
                        # Incomplete data, wait for more
                        pass
                except Exception as e:
                    print(f"Error receiving data: {str(e)}")
                    break
        except Exception as e:
            print(f"Error in client handler: {str(e)}")
        finally:
            try:
                client.close()
            except:
                pass
            print("Client handler stopped")

    # TODO: This is a temporary function to execute commands in the main thread
    def execute_command(self, command):
        """Execute a command in the main thread"""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            # TODO: Ensure we're in the right context
            if cmd_type in ["create_object", "modify_object", "delete_object"]:
                self._usd_context = omni.usd.get_context()
                self._execute_command_internal(command)
            else:
                return self._execute_command_internal(command)
                
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        cmd_type = command.get("type")
        params = command.get("params", {})

        #todo: add a handler for extend simulation method if necessary
        handlers = {
            "execute_script": self.execute_script,
            "list_prims": self.list_prims,
            "get_object_info": self.get_object_info,
            "move_prim": self.move_prim,
            "stop_scene": self.stop_scene,
            "play_scene": self.play_scene,
            "save_scene_state": self.save_scene_state,
            "restore_scene_state": self.restore_scene_state,
            "read_scene_state": self.read_scene_state,
            "list_scene_states": self.list_scene_states,
            "clear_scene_state": self.clear_scene_state,
        }
        
        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"Executing handler for {cmd_type}")
                result = handler(**params)
                print(f"Handler execution complete: /n", result)
                # return result
                if result and result.get("status") == "success":   
                    return {"status": "success", "result": result}
                else:
                    return {"status": "error", "message": result.get("message", "Unknown error")}
            except Exception as e:
                print(f"Error in handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
        

    def execute_script(self, code: str) :
        """Execute a Python script within the Isaac Sim context.
        
        Args:
            code: The Python script to execute.
            
        Returns:
            Dictionary with execution result.
        """
        try:
            # Create a local namespace
            local_ns = {}
            
            # Add frequently used modules to the namespace
            local_ns["omni"] = omni
            local_ns["carb"] = carb
            local_ns["Usd"] = Usd
            local_ns["UsdGeom"] = UsdGeom
            local_ns["Sdf"] = Sdf
            local_ns["Gf"] = Gf
            # code = script["code"]
            
            # Execute the script
            exec(code,  local_ns)
            
            # Get the result if any
            # result = local_ns.get("result", None)
            result = None
            
            
            return {
                "status": "success",
                "message": "Script executed successfully",
                "result": result
            }
        except Exception as e:
            carb.log_error(f"Error executing script: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        

    def list_prims(self, path: str = "/World/Objects") -> Dict[str, Any]:
        """List only the direct children of the specified path (default: /World/Objects)."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
            
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return {
                    "status": "error",
                    "message": "No stage is currently open"
                }
            
            objects_prim = stage.GetPrimAtPath(path)
            if not objects_prim.IsValid():
                return {
                    "status": "error",
                    "message": f"{path} path does not exist"
                }
            
            # Get only direct children
            children = objects_prim.GetChildren()
            
            # Collect direct children prims
            prims_list = []
            for child in children:
                prim_info = {
                    "path": str(child.GetPath()),
                    "name": child.GetName(),
                    "type": child.GetTypeName()
                }
                prims_list.append(prim_info)
            
            return {
                "status": "success",
                "message": f"Found {len(prims_list)} direct children in {path}",
                "prims": prims_list,
                "total_count": len(prims_list)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }

    def stop_scene(self) -> Dict[str, Any]:
        """Stop the simulation timeline.
        
        Works exactly like the execute_script stop_simulation example.
        
        Returns:
            Dictionary with execution result.
        """
        try:
            timeline = omni.timeline.get_timeline_interface()
            
            if timeline.is_playing() or timeline.is_paused():
                timeline.stop()
                print("✓ Simulation stopped")
                return {
                    "status": "success",
                    "message": "Simulation stopped"
                }
            else:
                print("⚠ Simulation is already stopped")
                return {
                    "status": "success",
                    "message": "Simulation is already stopped"
                }
        except Exception as e:
            carb.log_error(f"Error stopping scene: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to stop scene: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def play_scene(self) -> Dict[str, Any]:
        """Start/resume the simulation timeline.
        
        Works exactly like the execute_script play_simulation example.
        
        Returns:
            Dictionary with execution result.
        """
        try:
            timeline = omni.timeline.get_timeline_interface()
            
            if timeline.is_stopped():
                timeline.play()
                print("✓ Simulation started")
                return {
                    "status": "success",
                    "message": "Simulation started"
                }
            elif timeline.is_playing():
                print("⚠ Simulation is already playing")
                return {
                    "status": "success",
                    "message": "Simulation is already playing"
                }
            else:
                print("⚠ Simulation is paused, resuming...")
                timeline.play()
                return {
                    "status": "success",
                    "message": "Simulation resumed"
                }
        except Exception as e:
            carb.log_error(f"Error playing scene: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to play scene: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def get_object_info(self, prim_path: str) -> Dict[str, Any]:
        """Get comprehensive object information including pose using direct USD attribute access."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            import numpy as np
            from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles
            
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.IsValid():
                return {
                    "status": "error",
                    "message": f"Prim at {prim_path} not found"
                }
            
            # Basic prim info
            prim_info = {
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "is_valid": prim.IsValid(),
                "is_active": prim.IsActive(),
                "has_children": bool(prim.GetChildren())
            }
            
            # Direct attribute access using your code
            translate_attr = prim.GetAttribute("xformOp:translate")
            orient_attr = prim.GetAttribute("xformOp:orient")
            scale_attr = prim.GetAttribute("xformOp:scale")
            
            # Get values
            position = [0.0, 0.0, 0.0]
            quaternion = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
            scale = [1.0, 1.0, 1.0]
            
            if translate_attr.IsValid():
                translate_value = translate_attr.Get()
                if translate_value:
                    position = [float(translate_value[0]), float(translate_value[1]), float(translate_value[2])]
            
            if orient_attr.IsValid():
                orient_value = orient_attr.Get()
                if orient_value:
                    quaternion = [float(orient_value.GetReal()), float(orient_value.GetImaginary()[0]), 
                                float(orient_value.GetImaginary()[1]), float(orient_value.GetImaginary()[2])]
            
            if scale_attr.IsValid():
                scale_value = scale_attr.Get()
                if scale_value:
                    scale = [float(scale_value[0]), float(scale_value[1]), float(scale_value[2])]
            
            # Convert quaternion to euler angles using your code
            if abs(quaternion[0] - 1.0) < 1e-6 and abs(quaternion[1]) < 1e-6 and abs(quaternion[2]) < 1e-6 and abs(quaternion[3]) < 1e-6:
                # Identity quaternion - no rotation
                rotation_radians = [0.0, 0.0, 0.0]
                rotation_degrees = [0.0, 0.0, 0.0]
            else:
                # Convert w,x,y,z to x,y,z,w format for Isaac function
                quat_xyzw = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
                rpy_rad = quats_to_euler_angles(quat_xyzw.reshape(1, 4))[0]
                rpy_deg = np.rad2deg(rpy_rad)
                rotation_radians = [float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])]
                rotation_degrees = [float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])]
            
            return {
                "status": "success",
                "message": f"Successfully retrieved info for {prim_path}",
                "prim_info": prim_info,
                "position": position,
                "rotation_quaternion": quaternion,  # [w, x, y, z]
                "rotation_radians": rotation_radians,  # [roll, pitch, yaw]
                "rotation_degrees": rotation_degrees,  # [roll, pitch, yaw]
                "scale": scale
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": f"Failed to get object info: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def move_prim(self, prim_path: str, position: List[float] = None, orientation: List[float] = None, 
                orientation_format: str = "degrees") -> Dict[str, Any]:
        """Move a prim to a new pose with flexible orientation input (degrees/radians/quaternions)."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            import numpy as np
            from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
            
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.IsValid():
                return {
                    "status": "error",
                    "message": f"Prim at {prim_path} not found"
                }
            
            # Read current pose using direct attribute access (same as your working code)
            translate_attr = prim.GetAttribute("xformOp:translate")
            orient_attr = prim.GetAttribute("xformOp:orient")
            
            # Get current values as defaults
            current_position = [0.0, 0.0, 0.0]
            current_quaternion = [1.0, 0.0, 0.0, 0.0]
            
            if translate_attr.IsValid():
                translate_value = translate_attr.Get()
                if translate_value:
                    current_position = [float(translate_value[0]), float(translate_value[1]), float(translate_value[2])]
            
            if orient_attr.IsValid():
                orient_value = orient_attr.Get()
                if orient_value:
                    current_quaternion = [float(orient_value.GetReal()), float(orient_value.GetImaginary()[0]), 
                                        float(orient_value.GetImaginary()[1]), float(orient_value.GetImaginary()[2])]
            
            # Use provided values or keep current ones
            final_position = position if position is not None else current_position
            
            # Handle orientation conversion (same logic as working script)
            if orientation is not None:
                if orientation_format.lower() == "quaternion" and len(orientation) == 4:
                    # Input is quaternion [w, x, y, z]
                    final_quaternion = orientation
                    print(f"Using quaternion orientation: {orientation}")
                elif orientation_format.lower() == "radians" and len(orientation) == 3:
                    # Input is RPY in radians
                    rpy_rad = np.array(orientation)
                    quat_xyzw = euler_angles_to_quats(rpy_rad)
                    final_quaternion = [quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]  # w,x,y,z
                    print(f"Converted radian orientation {orientation} to quaternion {final_quaternion}")
                else:
                    # Default: Input is RPY in degrees
                    rpy_deg = np.array(orientation)
                    rpy_rad = np.deg2rad(rpy_deg)
                    quat_xyzw = euler_angles_to_quats(rpy_rad)
                    final_quaternion = [quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]  # w,x,y,z
                    print(f"Converted degree orientation {orientation} to quaternion {final_quaternion}")
            else:
                # Keep current orientation
                final_quaternion = current_quaternion
                orientation = "unchanged"
            
            # Apply transformations (detect existing precision and match it - same as working script)
            if not prim.IsA(UsdGeom.Xformable):
                return {
                    "status": "error",
                    "message": f"Prim {prim_path} is not transformable"
                }
            
            xform = UsdGeom.Xform(prim)
            
            # Check existing orient operation precision BEFORE clearing
            existing_orient_precision = None
            existing_ops = xform.GetOrderedXformOps()
            for op in existing_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    # Get the precision from existing operation
                    if "quatd" in str(op.GetAttr().GetTypeName()):
                        existing_orient_precision = UsdGeom.XformOp.PrecisionDouble
                    elif "quatf" in str(op.GetAttr().GetTypeName()):
                        existing_orient_precision = UsdGeom.XformOp.PrecisionFloat
                    break
            
            # Clear and recreate with matching precision
            xform.ClearXformOpOrder()
            
            # Apply translation (always double precision for position)
            position_vec = Gf.Vec3d(final_position[0], final_position[1], final_position[2])
            xform.AddTranslateOp().Set(position_vec)
            
            # Apply rotation with matching precision
            if existing_orient_precision == UsdGeom.XformOp.PrecisionFloat:
                quat = Gf.Quatf(final_quaternion[0], final_quaternion[1], final_quaternion[2], final_quaternion[3])
                orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
                print("Using float precision for quaternion")
            else:
                quat = Gf.Quatd(final_quaternion[0], final_quaternion[1], final_quaternion[2], final_quaternion[3])
                orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                print("Using double precision for quaternion")
            
            orient_op.Set(quat)
            
            print(f"Moved {prim_path} to position {final_position}, orientation {orientation}")
            
            return {
                "status": "success",
                "message": f"Successfully moved {prim_path} to new pose",
                "prim_path": prim_path,
                "previous_position": current_position,
                "previous_quaternion": current_quaternion,
                "new_position": final_position,
                "new_quaternion": final_quaternion,
                "orientation_format": orientation_format
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": f"Failed to move prim: {str(e)}",
                "prim_path": prim_path,
                "traceback": traceback.format_exc()
            }

    def _read_prim_pose(self, prim_path: str) -> Dict[str, Any]:
        """Helper method to read pose (position, quaternion, scale) from a prim."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.IsValid():
                return None
            
            # Read attributes
            translate_attr = prim.GetAttribute("xformOp:translate")
            orient_attr = prim.GetAttribute("xformOp:orient")
            scale_attr = prim.GetAttribute("xformOp:scale")
            
            # Get values
            position = [0.0, 0.0, 0.0]
            quaternion = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion [w, x, y, z]
            scale = [1.0, 1.0, 1.0]
            
            if translate_attr.IsValid():
                translate_value = translate_attr.Get()
                if translate_value:
                    position = [float(translate_value[0]), float(translate_value[1]), float(translate_value[2])]
            
            if orient_attr.IsValid():
                orient_value = orient_attr.Get()
                if orient_value:
                    quaternion = [
                        float(orient_value.GetReal()),
                        float(orient_value.GetImaginary()[0]),
                        float(orient_value.GetImaginary()[1]),
                        float(orient_value.GetImaginary()[2])
                    ]
            
            if scale_attr.IsValid():
                scale_value = scale_attr.Get()
                if scale_value:
                    scale = [float(scale_value[0]), float(scale_value[1]), float(scale_value[2])]
            
            return {
                "position": position,
                "quaternion": quaternion,  # [w, x, y, z]
                "scale": scale
            }
        except Exception as e:
            carb.log_error(f"Error reading pose for {prim_path}: {e}")
            return None

    def _write_prim_pose(self, prim_path: str, pose_data: Dict[str, Any]) -> bool:
        """Helper method to write pose (position, quaternion, scale) to a prim.
        Uses the same approach as read_write_pose.py - directly setting properties with ChangeProperty."""
        try:
            import omni.usd
            from pxr import Gf
            
            position = pose_data.get("position", [0.0, 0.0, 0.0])
            quaternion = pose_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
            scale = pose_data.get("scale", [1.0, 1.0, 1.0])
            
            # Set translate (same as read_write_pose.py)
            omni.kit.commands.execute('ChangeProperty',
                                     prop_path=f"{prim_path}.xformOp:translate",
                                     value=Gf.Vec3d(position[0], position[1], position[2]),
                                     prev=None)
            
            # Set orientation using quaternion - USE GfQuatf (float) instead of GfQuatd (double)
            # Same as read_write_pose.py
            quat_gf = Gf.Quatf(quaternion[0], quaternion[1], quaternion[2], quaternion[3])  # w, x, y, z
            omni.kit.commands.execute('ChangeProperty',
                                     prop_path=f"{prim_path}.xformOp:orient",
                                     value=quat_gf,
                                     prev=None)
            
            # Set scale (same as read_write_pose.py)
            omni.kit.commands.execute('ChangeProperty',
                                     prop_path=f"{prim_path}.xformOp:scale",
                                     value=Gf.Vec3d(scale[0], scale[1], scale[2]),
                                     prev=None)
            
            return True
        except Exception as e:
            carb.log_error(f"Error writing pose for {prim_path}: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return False

    def _get_prim_path(self, object_name: str) -> str:
        """Get the full prim path for an object name (same pattern as read_write_pose.py)"""
        return f"/World/Objects/{object_name}/{object_name}/{object_name}"

    def save_scene_state(self, object_names: List[str], json_file_path: str = None) -> Dict[str, Any]:
        """Save scene state (object poses) to a JSON file.
        
        Args:
            object_names: List of object names (e.g., ["fork_orange", "line_red", "base"])
            json_file_path: Optional path to the JSON file (defaults to "object_poses.json")
            
        Returns:
            Dictionary with execution result.
        """
        try:
            import os
            
            # Default JSON file path if not provided
            if json_file_path is None:
                json_file_path = "object_poses.json"
            
            # Save current poses to JSON
            poses = {}
            saved_count = 0
            failed_names = []
            
            for object_name in object_names:
                prim_path = self._get_prim_path(object_name)
                pose = self._read_prim_pose(prim_path)
                if pose:
                    poses[object_name] = pose
                    saved_count += 1
                    print(f"✓ Saved pose for {object_name} ({prim_path})")
                else:
                    failed_names.append(object_name)
                    print(f"⚠ Failed to read pose for {object_name} ({prim_path})")
            
            # Write to JSON file
            # Ensure directory exists
            os.makedirs(os.path.dirname(json_file_path) if os.path.dirname(json_file_path) else ".", exist_ok=True)
            
            with open(json_file_path, 'w') as f:
                json.dump(poses, f, indent=4)
            
            message = f"Saved {saved_count} object pose(s) to {json_file_path}"
            if failed_names:
                message += f". Failed to save {len(failed_names)} object(s): {failed_names}"
            
            return {
                "status": "success",
                "message": message,
                "saved_count": saved_count,
                "failed_names": failed_names,
                "json_file_path": json_file_path
            }
                
        except Exception as e:
            carb.log_error(f"Error in save_scene_state: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to save scene state: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def restore_scene_state(self, object_names: List[str], json_file_path: str = None) -> Dict[str, Any]:
        """Restore scene state (object poses) from a JSON file.
        
        Args:
            object_names: List of object names to restore (e.g., ["fork_orange", "line_red", "base"])
            json_file_path: Optional path to the JSON file (defaults to "object_poses.json")
            
        Returns:
            Dictionary with execution result.
        """
        try:
            import os
            
            # Default JSON file path if not provided
            if json_file_path is None:
                json_file_path = "object_poses.json"
            
            # Restore poses from JSON
            if not os.path.exists(json_file_path):
                return {
                    "status": "error",
                    "message": f"JSON file not found: {json_file_path}"
                }
            
            try:
                with open(json_file_path, 'r') as f:
                    poses = json.load(f)
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "message": f"Invalid JSON file: {str(e)}"
                }
            
            restored_count = 0
            failed_names = []
            
            for object_name in object_names:
                prim_path = self._get_prim_path(object_name)
                if object_name in poses:
                    pose_data = poses[object_name]
                    if self._write_prim_pose(prim_path, pose_data):
                        restored_count += 1
                        print(f"✓ Restored pose for {object_name} ({prim_path})")
                    else:
                        failed_names.append(object_name)
                        print(f"⚠ Failed to restore pose for {object_name} ({prim_path})")
                else:
                    failed_names.append(object_name)
                    print(f"⚠ No saved pose found for {object_name} in JSON file")
            
            message = f"Restored {restored_count} object pose(s) from {json_file_path}"
            if failed_names:
                message += f". Failed to restore {len(failed_names)} object(s): {failed_names}"
            
            return {
                "status": "success",
                "message": message,
                "restored_count": restored_count,
                "failed_names": failed_names,
                "json_file_path": json_file_path
            }
                
        except Exception as e:
            carb.log_error(f"Error in restore_scene_state: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to restore scene state: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def read_scene_state(self, json_file_path: str = None) -> Dict[str, Any]:
        """Read scene state (object poses) from a JSON file without applying them.
        
        Args:
            json_file_path: Optional path to the JSON file (defaults to "object_poses.json")
            
        Returns:
            Dictionary with the saved poses data.
        """
        try:
            import os
            
            # Default JSON file path if not provided
            if json_file_path is None:
                json_file_path = "object_poses.json"
            
            # Check if file exists
            if not os.path.exists(json_file_path):
                return {
                    "status": "error",
                    "message": f"JSON file not found: {json_file_path}"
                }
            
            # Read the JSON file
            try:
                with open(json_file_path, 'r') as f:
                    poses = json.load(f)
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "message": f"Invalid JSON file: {str(e)}"
                }
            
            # Format the response
            object_count = len(poses)
            object_names = list(poses.keys())
            
            # Create a formatted summary
            summary = []
            for object_name, pose_data in poses.items():
                position = pose_data.get("position", [0, 0, 0])
                quaternion = pose_data.get("quaternion", [1, 0, 0, 0])
                scale = pose_data.get("scale", [1, 1, 1])
                summary.append({
                    "object_name": object_name,
                    "position": position,
                    "quaternion": quaternion,
                    "scale": scale
                })
            
            return {
                "status": "success",
                "message": f"Read {object_count} object pose(s) from {json_file_path}",
                "json_file_path": json_file_path,
                "object_count": object_count,
                "object_names": object_names,
                "poses": poses,
                "summary": summary
            }
                
        except Exception as e:
            carb.log_error(f"Error in read_scene_state: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to read scene state: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def list_scene_states(self, directory: str = None) -> Dict[str, Any]:
        """List all available scene state JSON files in a directory.
        
        Args:
            directory: Optional directory path to search for scene state files.
                       If None, scans the current working directory (same as save_scene_state default location)
            
        Returns:
            Dictionary with list of found scene state files and their metadata.
        """
        try:
            import os
            import glob
            
            # Default to current directory if not provided (same as save_scene_state default)
            if directory is None:
                directory = "."
            
            # Check if directory exists
            if not os.path.exists(directory):
                return {
                    "status": "error",
                    "message": f"Directory not found: {directory}"
                }
            
            if not os.path.isdir(directory):
                return {
                    "status": "error",
                    "message": f"Path is not a directory: {directory}"
                }
            
            # Find all JSON files in the directory
            json_pattern = os.path.join(directory, "*.json")
            json_files = glob.glob(json_pattern)
            
            files_info = []
            
            # Read each JSON file to extract metadata
            for json_file_path in json_files:
                try:
                    # Read the JSON file
                    with open(json_file_path, 'r') as f:
                        poses = json.load(f)
                    
                    # Extract metadata
                    object_count = len(poses)
                    object_names = list(poses.keys())
                    
                    # Get just the filename
                    filename = os.path.basename(json_file_path)
                    
                    files_info.append({
                        "filename": filename,
                        "path": json_file_path,
                        "object_count": object_count,
                        "object_names": object_names
                    })
                    
                except json.JSONDecodeError:
                    # Skip invalid JSON files (they might not be scene state files)
                    continue
                except Exception as e:
                    # Skip files that can't be read
                    print(f"⚠ Could not read {json_file_path}: {str(e)}")
                    continue
            
            return {
                "status": "success",
                "message": f"Found {len(files_info)} scene state file(s) in {directory}",
                "directory": directory,
                "file_count": len(files_info),
                "files": files_info
            }
                
        except Exception as e:
            carb.log_error(f"Error in list_scene_states: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to list scene states: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def clear_scene_state(self, object_names: List[str] = None, json_file_path: str = None, clear_all: bool = False) -> Dict[str, Any]:
        """Clear/delete object poses from a JSON file.
        
        Args:
            object_names: Optional list of object names to remove. If None or empty and clear_all=False, clears all.
            json_file_path: Optional path to the JSON file (defaults to "object_poses.json")
            clear_all: If True, deletes the entire file. If False, removes specified objects.
            
        Returns:
            Dictionary with execution result.
        """
        try:
            import os
            
            # Default JSON file path if not provided
            if json_file_path is None:
                json_file_path = "object_poses.json"
            
            # If clear_all is True, delete the entire file
            if clear_all:
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                    return {
                        "status": "success",
                        "message": f"Deleted JSON file: {json_file_path}",
                        "json_file_path": json_file_path
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"JSON file not found: {json_file_path}"
                    }
            
            # Check if file exists
            if not os.path.exists(json_file_path):
                return {
                    "status": "error",
                    "message": f"JSON file not found: {json_file_path}"
                }
            
            # Read the JSON file
            try:
                with open(json_file_path, 'r') as f:
                    poses = json.load(f)
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "message": f"Invalid JSON file: {str(e)}"
                }
            
            # If no object_names provided, clear all
            if not object_names:
                object_names = list(poses.keys())
            
            # Remove specified objects
            removed_count = 0
            not_found = []
            
            for object_name in object_names:
                if object_name in poses:
                    del poses[object_name]
                    removed_count += 1
                    print(f"✓ Removed {object_name} from JSON file")
                else:
                    not_found.append(object_name)
                    print(f"⚠ {object_name} not found in JSON file")
            
            # Write back the updated poses (or delete file if empty)
            if len(poses) == 0:
                os.remove(json_file_path)
                message = f"Removed {removed_count} object(s) and deleted empty JSON file"
            else:
                with open(json_file_path, 'w') as f:
                    json.dump(poses, f, indent=4)
                message = f"Removed {removed_count} object(s) from {json_file_path}. {len(poses)} object(s) remaining."
            
            if not_found:
                message += f" {len(not_found)} object(s) not found: {not_found}"
            
            return {
                "status": "success",
                "message": message,
                "removed_count": removed_count,
                "not_found": not_found,
                "remaining_count": len(poses),
                "json_file_path": json_file_path
            }
                
        except Exception as e:
            carb.log_error(f"Error in clear_scene_state: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to clear scene state: {str(e)}",
                "traceback": traceback.format_exc()
            }


