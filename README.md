## RoboManufac-MCP


### TODO
- [ ] Add Install & setup link / instructions for URSim docker.
- [ ] Add networking / ufw instructions for ROS2 / URSim.
- [ ] Install instructions for UR Robot Driver, rosbridge_suite.
- [ ] Install instructions for ROS2 Workspace: containing OnRobot Gripper control, Grasp Points Publisher, Max Camera Localizer setup (and Orbbec Camera?) 
- [ ] Instructions for setting up Isaac Sim extensions?
- [ ] Give locations & setup instructions for all path config files.
- [ ] Give instructions for setting up UR ROS2 driver.
- [ ] Python environments setup instructions?



### Requirements
- Isaac Sim: Instructions for install are here: [Isaac Sim 5.1 install instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html)



### Setup vars
1. Under mcp-client-example/mcp_config.json, set up correct Paths to the MCP Servers and Paths to relevant Python executables.
2. Under ros-mcp-server/SERVER_PATHS_CFG.yaml, set up IP Addresses, ROS WS Paths and Primitive Controllers' paths


### Start MCP-Client

1. Start URSim and/or turn on the robot.
```
docker run --rm -it -p 5900:5900 -p 6080:6080   \
-v ${HOME}/.ursim/urcaps:/urcaps    \
-v ${HOME}/.ursim/programs:/ursim/programs  \
--name ursim --net ursim_net --ip 192.168.56.101 universalrobots/ursim_e-series
```


2. Start the UR ROS2 driver.
```
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py   \
ur_type:=ur5e  robot_ip:=192.168.56.101  use_mock_hardware:=false   \  
activate_joint_controller:=true   \
initial_joint_controller:=scaled_joint_trajectory_controller    
```


4. Start the Rosbridge
```
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```


5. Start Isaac Sim. Using the UI extension "UR5e Digital Twin":
    - Load the Scene and Refresh the Graphs.
    - 'Load the UR5e' and 'Setup UR5e ActionGraph'.
    - 'Import the RG2 Gripper' & 'Attach the RG2 to UR5e'. 
    - Setup the 'Gripper Action Graph' and Setup the 'Force Publish Graph'.
    - 'Import RealSense Camera', 'Attach Camera' to the UR5e robot.
    - Setup the 'Camera Action Graph'.
    - 'Add Objects' and Setup 'Pose Publisher Action Graph'.


6. Run the MCP Client using either OpenAI / Anthropic API keys.
```
# For Claude provider
export ANTHROPIC_API_KEY=your_anthropic_key_here
# For OpenAI provider
export OPENAI_API_KEY=your_openai_key_here
```

Start the Client:

```
cd ~/Desktop/RoboManufac-MCP/mcp-client-example
node dist/bin.js --list-servers
node dist/bin.js --all --provider=openai    # or --provider=claude
```