#!/bin/bash

tab="--tab"
foo=""

# # # UR5e Bringup  
foo+=($tab -e "bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ros2 launch ur_bringup ur5e.launch.py ur_type:=ur5e robot_ip:=192.168.1.111;bash'")

# # # # Max Camera Localizer
foo+=($tab -e "bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ros2 run max_camera_localizer localize_yoloe;bash'")

# # # OnRobot Gripper Control
foo+=($tab -e "bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ros2 run onrobot_ros gripper_control;bash'")

# # # Rosbridge Server
foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ros2 launch rosbridge_server rosbridge_websocket_launch.xml;bash'")

# # Claude
foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ~/Applications/claude-desktop-0.11.6-amd64_5b19eb0add79ec92df059cbddb071cd7.AppImage;bash'")

gnome-terminal "${foo[@]}"

exit 0