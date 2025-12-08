#!/bin/bash

tab="--tab"
foo=""

# # Claude
foo+=($tab -e "bash -c 'eval \"\$(conda shell.bash hook)\"; conda deactivate; source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; source ~/ros2_ws/install/setup.bash; ~/Applications/claude-desktop-0.11.6-amd64_5b19eb0add79ec92df059cbddb071cd7.AppImage;bash'")

gnome-terminal "${foo[@]}"

exit 0