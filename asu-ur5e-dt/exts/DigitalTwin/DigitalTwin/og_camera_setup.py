# 

import omni.graph.core as og


class CameraOmniGraphSetup():
    def __init__(self):
        pass


    def _create_camera_actiongraph(self, camera_prim, width, height, topic, graph_suffix):
        """Helper method to create camera ActionGraph"""
        graph_path = f"/World/Graphs/ActionGraph_{graph_suffix}"
        print(f"Creating ActionGraph: {graph_path}")
        print(f"Camera: {camera_prim}")
        print(f"Resolution: {width}x{height}")
        print(f"ROS2 Topic: {topic}")
        
        # Create ActionGraph
        try:
            og.Controller.create_graph(graph_path)
            print(f"Created ActionGraph at {graph_path}")
        except Exception:
            print(f"ActionGraph already exists at {graph_path}")
        
        # Create nodes
        nodes = [
            ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
            ("isaac_run_one_simulation_frame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
            ("isaac_create_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
            ("ros2_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        
        print("\nCreating nodes...")
        for node_name, node_type in nodes:
            try:
                node_path = f"{graph_path}/{node_name}"
                og.Controller.create_node(node_path, node_type)
                print(f"Created {node_name}")
            except Exception as e:
                print(f"Node {node_name} already exists")
        
        # Set node attributes
        print("\nConfiguring nodes...")
        
        # Configure render product
        try:
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:cameraPrim").set([camera_prim])
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:width").set(width)
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:height").set(height)
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:enabled").set(True)
            print(f"Configured render product: {camera_prim} @ {width}x{height}")
        except Exception as e:
            print(f"Error configuring render product: {e}")
        
        # Configure ROS2 camera helper
        try:
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:topicName").set(topic)
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:frameId").set("camera_link")
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:type").set("rgb")
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:enabled").set(True)
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:queueSize").set(10)
            print(f"Configured ROS2 helper: topic={topic}")
        except Exception as e:
            print(f"Error configuring ROS2 helper: {e}")
        
        # Create connections
        print("\nConnecting nodes...")
        connections = [
            ("on_playback_tick.outputs:tick", "isaac_run_one_simulation_frame.inputs:execIn"),
            ("isaac_run_one_simulation_frame.outputs:step", "isaac_create_render_product.inputs:execIn"),
            ("isaac_create_render_product.outputs:execOut", "ros2_camera_helper.inputs:execIn"),
            ("isaac_create_render_product.outputs:renderProductPath", "ros2_camera_helper.inputs:renderProductPath"),
            ("ros2_context.outputs:context", "ros2_camera_helper.inputs:context"),
        ]
        
        for source, target in connections:
            try:
                og.Controller.connect(f"{graph_path}/{source}", f"{graph_path}/{target}")
                print(f"Connected {source.split('.')[0]} -> {target.split('.')[0]}")
            except Exception as e:
                print(f"Failed to connect {source} -> {target}: {e}")
        
        print(f"\n{graph_suffix} ActionGraph created successfully!")
        print(f"Test with: ros2 topic echo /{topic}")
    

    def setup_camera_action_graph(self):
        """Create ActionGraph for camera ROS2 publishing"""
        # Configuration
        CAMERA_PRIM = "/World/UR5e/wrist_3_link/rsd455/RSD455/Camera_OmniVision_OV9782_Color" 
        IMAGE_WIDTH = 1280
        IMAGE_HEIGHT = 720
        ROS2_TOPIC = "intel_camera_rgb_sim"
        
        graph_path = "/World/Graphs/ActionGraph_Camera"  # Different name to avoid conflicts
        print(f"Creating ActionGraph: {graph_path}")
        print(f"Camera: {CAMERA_PRIM}")
        print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        print(f"ROS2 Topic: {ROS2_TOPIC}")
        
        # Create ActionGraph
        try:
            og.Controller.create_graph(graph_path)
            print(f"Created ActionGraph at {graph_path}")
        except Exception:
            print(f"ActionGraph already exists at {graph_path}")
        
        # Create nodes
        nodes = [
            ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
            ("isaac_run_one_simulation_frame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
            ("isaac_create_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
            ("ros2_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        
        print("\nCreating nodes...")
        for node_name, node_type in nodes:
            try:
                node_path = f"{graph_path}/{node_name}"
                og.Controller.create_node(node_path, node_type)
                print(f"Created {node_name}")
            except Exception as e:
                print(f"Node {node_name} already exists")
        
        # Set node attributes
        print("\nConfiguring nodes...")
        # Configure render product
        try:
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:cameraPrim").set([CAMERA_PRIM])
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:width").set(IMAGE_WIDTH)
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:height").set(IMAGE_HEIGHT)
            og.Controller.attribute(f"{graph_path}/isaac_create_render_product.inputs:enabled").set(True)
            print(f"Configured render product: {CAMERA_PRIM} @ {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        except Exception as e:
            print(f"Error configuring render product: {e}")
        
        # Configure ROS2 camera helper
        try:
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:topicName").set(ROS2_TOPIC)
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:frameId").set("camera_link")
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:type").set("rgb")
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:enabled").set(True)
            og.Controller.attribute(f"{graph_path}/ros2_camera_helper.inputs:queueSize").set(10)
            print(f"Configured ROS2 helper: topic={ROS2_TOPIC}")
        except Exception as e:
            print(f"Error configuring ROS2 helper: {e}")
        
        # Create connections
        print("\nConnecting nodes...")
        connections = [
            ("on_playback_tick.outputs:tick", "isaac_run_one_simulation_frame.inputs:execIn"),
            ("isaac_run_one_simulation_frame.outputs:step", "isaac_create_render_product.inputs:execIn"),
            ("isaac_create_render_product.outputs:execOut", "ros2_camera_helper.inputs:execIn"),
            ("isaac_create_render_product.outputs:renderProductPath", "ros2_camera_helper.inputs:renderProductPath"),
            ("ros2_context.outputs:context", "ros2_camera_helper.inputs:context"),
        ]
        
        for source, target in connections:
            try:
                og.Controller.connect(f"{graph_path}/{source}", f"{graph_path}/{target}")
                print(f"Connected {source.split('.')[0]} -> {target.split('.')[0]}")
            except Exception as e:
                print(f"Failed to connect {source} -> {target}: {e}")
        
        print("\nCamera ActionGraph created successfully!")
        print(f"Test with: ros2 topic echo /{ROS2_TOPIC}")


    def create_additional_camera_actiongraph(self):
        """Create ActionGraph for additional camera ROS2 publishing"""
        # Check which cameras exist and create action graphs accordingly
        is_exocentric = self._exocentric_checkbox.model.get_value_as_bool()
        is_custom = self._custom_checkbox.model.get_value_as_bool()

        if is_exocentric:
            # TODO: Copy other camera properties into exocentric camera (e.g., from Intel camera setup)
            self.cam_og_setup._create_camera_actiongraph(
                "/World/exocentric_camera", 
                1280, 720, 
                "exocentric_camera", 
                "ExocentricCamera"
            )
        
        if is_custom:
            self.cam_og_setup._create_camera_actiongraph(
                "/World/custom_camera", 
                640, 480, 
                "custom_camera", 
                "CustomCamera"
            )