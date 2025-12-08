# Setting up Various Omnigraphs for UR5e and Gripper
# (1) Action Graphs for UR5e to publish & listen to ROS2 topic
# and
# (2.1) Set up RG2 Gripper action graphs for ROS2 control
# (2.2) Set up Force publisher action graphs for RG2 Gripper

import omni.graph.core as og
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.extensions import enable_extension
import omni.usd

from pxr import UsdGeom, Gf, Sdf


class OmniGraphSetup():
    def __init__(self):
        pass

    async def setup_action_graph(self):
        print("Setting up ROS 2 Action Graph...")

        # Ensure extensions are enabled
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.core.nodes")
        enable_extension("omni.graph.action")

        graph_path = "/World/Graphs/ActionGraph_UR5e"
        (graph, nodes, _, _) = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution"
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("isaac_read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ros2_subscribe_joint_state", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ros2_publish_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("articulation_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ros2_context.inputs:useDomainIDEnvVar", True),
                    ("ros2_context.inputs:domain_id", 0),
                    ("ros2_subscribe_joint_state.inputs:topicName", "/joint_states"),
                    ("ros2_subscribe_joint_state.inputs:nodeNamespace", ""),
                    ("ros2_subscribe_joint_state.inputs:queueSize", 10),
                    ("ros2_publish_clock.inputs:topicName", "/clock"),
                    ("ros2_publish_clock.inputs:nodeNamespace", ""),
                    ("ros2_publish_clock.inputs:qosProfile", "SYSTEM_DEFAULT"),
                    ("ros2_publish_clock.inputs:queueSize", 10),
                    ("isaac_read_simulation_time.inputs:resetOnStop", True),
                    ("articulation_controller.inputs:robotPath", "/World/UR5e"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("on_playback_tick.outputs:tick", "ros2_subscribe_joint_state.inputs:execIn"),
                    ("on_playback_tick.outputs:tick", "ros2_publish_clock.inputs:execIn"),
                    ("on_playback_tick.outputs:tick", "articulation_controller.inputs:execIn"),
                    ("ros2_context.outputs:context", "ros2_subscribe_joint_state.inputs:context"),
                    ("ros2_context.outputs:context", "ros2_publish_clock.inputs:context"),
                    ("isaac_read_simulation_time.outputs:simulationTime", "ros2_publish_clock.inputs:timeStamp"),
                    ("ros2_subscribe_joint_state.outputs:positionCommand", "articulation_controller.inputs:positionCommand"),
                    ("ros2_subscribe_joint_state.outputs:velocityCommand", "articulation_controller.inputs:velocityCommand"),
                    ("ros2_subscribe_joint_state.outputs:effortCommand", "articulation_controller.inputs:effortCommand"),
                    ("ros2_subscribe_joint_state.outputs:jointNames", "articulation_controller.inputs:jointNames"),
                ],
            }
        )

        print("ROS 2 Action Graph setup complete.")



class GripperOmniGraphSetup():
    def __init__(self):
        pass


    def setup_gripper_action_graph(self):
        """Setup gripper action graph for ROS2 control"""
        print("Setting up Gripper Action Graph...")
        
        graph_path = "/World/Graphs/ActionGraph_RG2"
        
        # Delete existing
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(graph_path):
            stage.RemovePrim(graph_path)
        
        keys = og.Controller.Keys
        
        print("Creating nodes...")
        # Create nodes including the new publisher and script_node
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    (f"{graph_path}/tick", "omni.graph.action.OnPlaybackTick"),
                    (f"{graph_path}/context", "isaacsim.ros2.bridge.ROS2Context"),
                    (f"{graph_path}/subscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
                    (f"{graph_path}/script", "omni.graph.scriptnode.ScriptNode"),
                    (f"{graph_path}/script_node", "omni.graph.scriptnode.ScriptNode"),
                    (f"{graph_path}/ros2_publisher", "isaacsim.ros2.bridge.ROS2Publisher")
                ]
            }
        )
        
        print("Adding script attributes...")
        # Add script attributes for command script
        script_node = og.Controller.node(f"{graph_path}/script")
        og.Controller.create_attribute(script_node, "inputs:String", og.Type(og.BaseDataType.TOKEN))
        og.Controller.create_attribute(script_node, "outputs:Integer", og.Type(og.BaseDataType.INT))
        
        # Add script_node attributes for reading gripper state - use proper pattern
        script_node_state = og.Controller.node(f"{graph_path}/script_node")
        og.Controller.create_attribute(
            script_node_state, 
            "gripper_width", 
            og.Type(og.BaseDataType.DOUBLE),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
        )
        print("Created outputs:gripper_width attribute on script_node")
        
        print("Setting values...")
        # Set values with try/catch for each one
        def safe_set(path, value, desc=""):
            try:
                attr = og.Controller.attribute(path)
                if attr.is_valid():
                    og.Controller.set(attr, value)
                    print(f"Set {desc}")
                    return True
                else:
                    print(f"Invalid: {desc}")
                    return False
            except Exception as e:
                print(f"Failed {desc}: {e}")
                return False
        
        # Configure ROS2 Subscriber
        safe_set(f"{graph_path}/subscriber.inputs:messageName", "String", "ROS2 message type")
        safe_set(f"{graph_path}/subscriber.inputs:messagePackage", "std_msgs", "ROS2 package")
        safe_set(f"{graph_path}/subscriber.inputs:topicName", "gripper_command", "ROS2 topic")
        
        # Configure Script Node
        safe_set(f"{graph_path}/script.inputs:usePath", False, "Use inline script")
        
        # Script that handles string processing internally
        script_content = '''from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationActions
import numpy as np
import omni.timeline

gripper_view = None
last_sim_frame = -1

def setup(db):
    global gripper_view
    try:
        # Always create a fresh gripper view in setup
        gripper_view = ArticulationView(prim_paths_expr="/World/RG2_Gripper", name="gripper")
        gripper_view.initialize()
        db.log_info("Gripper initialized successfully")
    except Exception as e:
        db.log_error(f"Gripper setup failed: {e}")
        gripper_view = None

def compute(db):
    global gripper_view, last_sim_frame
    
    try:
        # Get input string from ROS2
        input_str = str(db.inputs.String).strip()
        
        # Handle string replacements in Python
        if input_str == "open":
            processed_str = "1100"
        elif input_str == "close":
            processed_str = "0"
        else:
            processed_str = input_str
        
        # Convert to width in mm
        try:
            width_mm = float(processed_str) / 10.0
        except ValueError:
            db.log_error(f"Invalid input: {input_str}")
            return
        
        # Check if simulation restarted by monitoring frame count
        timeline = omni.timeline.get_timeline_interface()
        current_frame = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
        
        # If frame went backwards, simulation was restarted
        if current_frame < last_sim_frame or last_sim_frame == -1:
            db.log_info("Simulation restart detected, reinitializing gripper...")
            try:
                gripper_view = ArticulationView(prim_paths_expr="/World/RG2_Gripper", name="gripper")
                gripper_view.initialize()
                db.log_info("Gripper reinitialized after restart")
            except Exception as e:
                db.log_error(f"Gripper reinitialization failed: {e}")
                gripper_view = None
        
        last_sim_frame = current_frame
        
        # Check if gripper is available
        if gripper_view is None:
            db.log_warning("Gripper not available")
            return
        
        # Check if simulation is running
        if timeline.is_stopped():
            return
        
        # Try to apply action, with graceful handling of physics not ready
        try:
            # Clamp to valid range
            width_mm = np.clip(width_mm, 0.0, 110.0)
            
            # Map width to joint angle: 0mm = -π/4 (closed), 110mm = π/6 (open)
            ratio = width_mm / 110.0
            joint_angle = -np.pi/4 + ratio * (np.pi/4 + np.pi/6)
            
            # Apply to gripper
            target_positions = np.array([joint_angle, joint_angle])
            action = ArticulationActions(
                joint_positions=target_positions,
                joint_indices=np.array([0, 1])
            )
            gripper_view.apply_action(action)
            
            # Set output
            db.outputs.Integer = int(width_mm)
            
            db.log_info(f"Gripper: \\'{input_str}\\' -> {width_mm:.1f}mm -> {joint_angle:.3f}rad")
            
        except Exception as e:
            # This is expected during the first few frames after restart
            if "Physics Simulation View is not created yet" in str(e):
                # Physics not ready yet, just wait
                db.outputs.Integer = int(np.clip(width_mm, 0.0, 110.0))
            else:
                db.log_warning(f"Action failed: {e}")
                db.outputs.Integer = 0
        
    except Exception as e:
        db.log_error(f"Compute error: {e}")
        db.outputs.Integer = 0

def cleanup(db):
    global gripper_view
    db.log_info("Cleaning up gripper")
    gripper_view = None'''
        
        safe_set(f"{graph_path}/script.inputs:script", script_content, "Python script")
        
        # Configure script_node for reading gripper state
        script_node_content = '''from omni.isaac.core.articulations import ArticulationView
import numpy as np
import omni.timeline

gripper_view = None
last_sim_frame = -1
physics_ready = False

def setup(db):
    global gripper_view, physics_ready
    physics_ready = False
    try:
        gripper_view = ArticulationView(prim_paths_expr="/World/RG2_Gripper", name="gripper")
        gripper_view.initialize()
        db.log_info("Gripper initialized successfully")
    except Exception as e:
        db.log_error(f"Gripper setup failed: {e}")
        gripper_view = None

def compute(db):
    global gripper_view, last_sim_frame, physics_ready
    
    try:
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_stopped():
            return
        
        current_frame = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
        
        # Handle simulation restart
        if current_frame < last_sim_frame or last_sim_frame == -1:
            physics_ready = False
            try:
                gripper_view = ArticulationView(prim_paths_expr="/World/RG2_Gripper", name="gripper")
                gripper_view.initialize()
            except Exception as e:
                db.log_error(f"Gripper reinitialization failed: {e}")
                gripper_view = None
        
        last_sim_frame = current_frame
        
        if gripper_view is None:
            db.outputs.gripper_width = 0.0
            return
        
        # Read actual gripper state every frame
        actual_width_mm = 0.0
        
        try:
            joint_positions = gripper_view.get_joint_positions()
            
            if joint_positions is not None and len(joint_positions) > 0 and joint_positions.shape[1] >= 2:
                physics_ready = True
                
                # Get actual angle
                actual_angle = np.mean(joint_positions[0, :2])
                
                # Convert to width
                actual_ratio = (actual_angle + np.pi/4) / (np.pi/4 + np.pi/6)
                actual_width_mm = actual_ratio * 110.0
                actual_width_mm = np.clip(actual_width_mm, 0.0, 110.0)
                
        except Exception as e:
            pass  # Ignore during initialization
        
        # Always output actual width
        db.outputs.gripper_width = float(actual_width_mm)
        
    except Exception as e:
        db.log_error(f"Compute error: {e}")
        db.outputs.gripper_width = 0.0

def cleanup(db):
    global gripper_view, physics_ready
    gripper_view = None
    physics_ready = False'''
        
        safe_set(f"{graph_path}/script_node.inputs:usePath", False, "Use inline script for script_node")
        safe_set(f"{graph_path}/script_node.inputs:script", script_node_content, "Python script for script_node")
        
        # Configure ROS2 Publisher
        safe_set(f"{graph_path}/ros2_publisher.inputs:messageName", "Float64", "ROS2 message type")
        safe_set(f"{graph_path}/ros2_publisher.inputs:messagePackage", "std_msgs", "ROS2 package")
        safe_set(f"{graph_path}/ros2_publisher.inputs:topicName", "gripper_width_sim", "ROS2 topic")
        
        # Create input attribute on publisher and connect it using OmniGraph API
        try:
            stage = omni.usd.get_context().get_stage()
            publisher_prim = stage.GetPrimAtPath(f"{graph_path}/ros2_publisher")
            if publisher_prim.IsValid():
                # Check if attribute already exists
                existing_attr = publisher_prim.GetAttribute("inputs:data")
                if not existing_attr.IsValid():
                    data_attr = publisher_prim.CreateAttribute("inputs:data", Sdf.ValueTypeNames.Double, custom=True)
                    print("Created inputs:data attribute on ros2_publisher")
                else:
                    print("inputs:data attribute already exists on ros2_publisher")
                
                # Connect script_node output to publisher input using OmniGraph API
                og.Controller.connect(
                    f"{graph_path}/script_node.outputs:gripper_width",
                    f"{graph_path}/ros2_publisher.inputs:data"
                )
                print("Connected script_node.outputs:gripper_width to ros2_publisher.inputs:data")
            else:
                print(f"Warning: Could not find publisher prim at {graph_path}/ros2_publisher")
        except Exception as e:
            print(f"Error creating publisher connection: {e}")
        
        print("Creating connections...")
        # Connections for command handling (subscriber -> script)
        connections = [
            (f"{graph_path}/tick.outputs:tick", f"{graph_path}/subscriber.inputs:execIn", "Tick to subscriber"),
            (f"{graph_path}/context.outputs:context", f"{graph_path}/subscriber.inputs:context", "Context to subscriber"),
            (f"{graph_path}/subscriber.outputs:data", f"{graph_path}/script.inputs:String", "ROS2 data to script"),
            (f"{graph_path}/subscriber.outputs:execOut", f"{graph_path}/script.inputs:execIn", "ROS2 exec to script"),
            # Connections for state reading and publishing (script_node -> publisher)
            (f"{graph_path}/tick.outputs:tick", f"{graph_path}/script_node.inputs:execIn", "Tick to script_node"),
            (f"{graph_path}/script_node.outputs:execOut", f"{graph_path}/ros2_publisher.inputs:execIn", "Script_node exec to publisher"),
            (f"{graph_path}/context.outputs:context", f"{graph_path}/ros2_publisher.inputs:context", "Context to publisher")
        ]
        
        success_count = 0
        for src, dst, desc in connections:
            try:
                og.Controller.edit(graph, {keys.CONNECT: [(src, dst)]})
                print(f"Connected {desc}")
                success_count += 1
            except Exception as e:
                print(f"Failed {desc}: {e}")
        
        print(f"Graph created with {success_count}/7 connections (plus gripper_width connection)!")
        print("Location: /World/Graphs/ActionGraph_RG2")
        print("\nTest commands for gripper control:")
        print("ros2 topic pub /gripper_command std_msgs/String 'data: \"open\"'")
        print("ros2 topic pub /gripper_command std_msgs/String 'data: \"close\"'")
        print("ros2 topic pub /gripper_command std_msgs/String 'data: \"1100\"'")
        print("ros2 topic pub /gripper_command std_msgs/String 'data: \"550\"'")
        print("\nMonitor gripper width:")
        print("ros2 topic echo /gripper_width_sim")

    

    def setup_force_publish_action_graph(self):
        """Setup force publish action graph for ROS2 publishing gripper forces"""
        print("Setting up Force Publish Action Graph...")
        
        # Create the action graph
        graph_path = "/World/Graphs/ActionGraph_RG2_ForcePublish"
        keys = og.Controller.Keys

        # Delete existing graph if it exists
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(graph_path):
            stage.RemovePrim(graph_path)

        # Create nodes with initial values
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    (f"{graph_path}/tick", "omni.graph.action.OnPlaybackTick"),
                    (f"{graph_path}/context", "isaacsim.ros2.bridge.ROS2Context"),
                    (f"{graph_path}/script", "omni.graph.scriptnode.ScriptNode"),
                    (f"{graph_path}/publisher", "isaacsim.ros2.bridge.ROS2Publisher")
                ],
                keys.SET_VALUES: [
                    (f"{graph_path}/script.inputs:usePath", False),
                    (f"{graph_path}/publisher.inputs:messageName", "Float64"),
                    (f"{graph_path}/publisher.inputs:messagePackage", "std_msgs"),
                    (f"{graph_path}/publisher.inputs:topicName", "gripper_force"),
                ],
                keys.CONNECT: [
                    (f"{graph_path}/tick.outputs:tick", f"{graph_path}/script.inputs:execIn"),
                    (f"{graph_path}/script.outputs:execOut", f"{graph_path}/publisher.inputs:execIn"),
                    (f"{graph_path}/context.outputs:context", f"{graph_path}/publisher.inputs:context"),
                ]
            }
        )

        # Script content
        script_content = '''from omni.isaac.core.articulations import ArticulationView
import numpy as np
import omni.timeline

_gripper_view = None

def setup(db):
    global _gripper_view
    try:
        _gripper_view = ArticulationView(prim_paths_expr="/World/RG2_Gripper", name="gripper_force")
        _gripper_view.initialize()
        db.log_info("[FORCE] Force publisher initialized")
        print("[FORCE] Publisher ready")
    except Exception as e:
        db.log_error(f"[FORCE] Setup failed: {e}")
        print(f"[FORCE ERROR] {e}")

def compute(db):
    global _gripper_view
    
    try:
        if _gripper_view is None:
            db.outputs.max_force = 0.0
            return
        
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_stopped():
            db.outputs.max_force = 0.0
            return
        
        try:
            forces = _gripper_view.get_measured_joint_forces()
            
            if forces is not None:
                forces_flat = np.asarray(forces).flatten()
                if len(forces_flat) > 0:
                    max_force = float(np.max(np.abs(forces_flat)))
                    db.outputs.max_force = max_force
                else:
                    db.outputs.max_force = 0.0
            else:
                db.outputs.max_force = 0.0
        
        except Exception as e:
            db.outputs.max_force = 0.0
    
    except Exception as e:
        db.outputs.max_force = 0.0

def cleanup(db):
    global _gripper_view
    try:
        _gripper_view = None
    except:
        pass'''

        # Set script FIRST
        og.Controller.set(og.Controller.attribute(f"{graph_path}/script.inputs:script"), script_content)

        # Now create output attribute on script node using OmniGraph API
        script_node = og.Controller.node(f"{graph_path}/script")
        og.Controller.create_attribute(
            script_node, 
            "max_force", 
            og.Type(og.BaseDataType.DOUBLE),
            attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
        )

        print("Created outputs:max_force attribute on script node")

        # Create input attribute on publisher and connect it using OmniGraph API
        publisher_prim = stage.GetPrimAtPath(f"{graph_path}/publisher")
        data_attr = publisher_prim.CreateAttribute("inputs:data", Sdf.ValueTypeNames.Double, custom=True)

        # Connect script output to publisher input using OmniGraph API
        og.Controller.connect(
            f"{graph_path}/script.outputs:max_force",
            f"{graph_path}/publisher.inputs:data"
        )

        print(f"RG2 Force Publish action graph created at {graph_path}")
        print("Publishing to topic: /gripper_force")