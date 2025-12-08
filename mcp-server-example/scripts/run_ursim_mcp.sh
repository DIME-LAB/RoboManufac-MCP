#!/bin/bash

tab="--tab"
foo=""

# # URSim
# foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; docker run --rm -it -p 5900:5900 -p 6080:6080 -v \${HOME}/.ursim/urcaps:/urcaps -v \${HOME}/.ursim/programs:/ursim/programs --name ursim universalrobots/ursim_e-series;bash'")

# # UR5e Bringup  
# foo+=($tab -e "bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ros2 launch ur_bringup ur5e.launch.py ur_type:=ur5e robot_ip:=172.17.0.2;bash'")

# # # Isaac Sim
# foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; source ~/env_isaaclab/bin/activate; isaacsim;bash'")

# # Rosbridge Server
# foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ros2 launch rosbridge_server rosbridge_websocket_launch.xml;bash'")

# # # Claude
foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; ~/Applications/claude-desktop-0.11.6-amd64_5b19eb0add79ec92df059cbddb071cd7.AppImage;bash'")

gnome-terminal "${foo[@]}"

exit 0