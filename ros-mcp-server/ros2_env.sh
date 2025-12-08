#!/bin/bash
# ROS2 Environment Setup Script

# Source ROS2 environment
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash

# Use system Python for ROS2 (not conda)
export ROS2_PYTHON=/usr/bin/python3

echo "ROS2 Environment Ready!"
echo "Use: $ROS2_PYTHON your_ros2_script.py"
echo "Or just: ros2 topic list, ros2 topic echo /topic_name, etc."
