#!/usr/bin/env python3
"""
Gripper Width Information Reader

This script outputs gripper width information including:
- Gripper fully open width
- Gripper fully closed width  
- Gripper current width

Usage:
    python3 verify_grasp.py [timeout]

Arguments:
    timeout: Maximum time to wait for gripper reading (default: 5 seconds)

Output:
    Three lines with width information
"""

import sys
import subprocess
import re

def get_gripper_width(timeout=5):
    """Get current gripper width from ROS topic."""
    try:
        cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && timeout {timeout} ros2 topic echo /gripper_width --once"
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=timeout + 2
        )
        
        if result.returncode == 0:
            # Extract numeric value from output
            output = result.stdout.strip()
            
            # Look for the data line specifically
            for line in output.split('\n'):
                if 'data:' in line:
                    # Extract the number after 'data:'
                    match = re.search(r'data:\s*([\d.]+)', line)
                    if match:
                        return float(match.group(1))
            
            # Fallback to original method if data: line not found
            numbers = re.findall(r'-?\d+\.?\d*', output)
            if numbers:
                return float(numbers[0])
        
        return None
        
    except Exception:
        return None

def main():
    """Main function to read and output gripper width information."""
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Gripper width constants
    FULLY_OPEN_WIDTH = 110.0
    FULLY_CLOSED_WIDTH = 9.0
    
    current_width = get_gripper_width(timeout)
    
    if current_width is not None:
        print(f"Gripper fully open width: {FULLY_OPEN_WIDTH}")
        print(f"Gripper fully closed width: {FULLY_CLOSED_WIDTH}")
        print(f"Gripper current width: {current_width}")
        print("Typical grasp width is between 25 to 40mm")
        sys.exit(0)
    else:
        print("Error: Could not read gripper width")
        sys.exit(1)

if __name__ == "__main__":
    main()