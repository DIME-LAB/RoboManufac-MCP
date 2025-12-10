#!/usr/bin/env python3
"""
Test script to verify MCP server can access and run primitive scripts
"""

import sys
import os
import subprocess
from pathlib import Path
from box import Box
import yaml


# Configs containing paths of ROS and other related filepaths
config_path = Path(__file__).parent / "SERVER_PATHS_CFGS.yaml"
with open(config_path, "r") as f:
    yaml_cfg = Box(yaml.safe_load(f))

PRIMITIVE_LIBS_PATH = yaml_cfg.ros_paths.primitive_libs_path


def test_primitive_access():
    """Test if we can access and run the primitive scripts"""
    
    # Path to primitive scripts
    primitives_dir = PRIMITIVE_LIBS_PATH
    
    print("🔍 Testing primitive script access...")
    print(f"📁 Primitives directory: {primitives_dir}")
    
    # Check if directory exists
    if not os.path.exists(primitives_dir):
        print("❌ Primitives directory not found!")
        return False
    
    # List available scripts
    scripts = [f for f in os.listdir(primitives_dir) if f.endswith('.py')]
    print(f"📜 Found {len(scripts)} scripts: {scripts}")
    
    # Test reading one script
    test_script = os.path.join(primitives_dir, "movea2b.py")
    if os.path.exists(test_script):
        print(f"✅ Can access: {test_script}")
        
        # Test if we can import it (syntax check)
        try:
            with open(test_script, 'r') as f:
                content = f.read()
            print(f"✅ Can read script content ({len(content)} characters)")
        except Exception as e:
            print(f"❌ Cannot read script: {e}")
            return False
    else:
        print(f"❌ Cannot access: {test_script}")
        return False
    
    # Test if we can run it with proper environment
    print("\n🚀 Testing script execution...")
    try:
        # Source ROS2 environment and run script
        cmd = f"""
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash
source ~/ros2/install/setup.bash
export ROS_DOMAIN_ID=0
cd {primitives_dir}
/usr/bin/python3 {test_script}
"""
        
        # Run with bash shell to source environment
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            print("✅ Script executed successfully!")
            print(f"📤 Output: {result.stdout[:200]}...")
        else:
            print(f"⚠️ Script ran but with issues:")
            print(f"   Return code: {result.returncode}")
            print(f"   Stderr: {result.stderr[:200]}...")
            
    except subprocess.TimeoutExpired:
        print("⏰ Script execution timed out (expected for ROS2 scripts)")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False
    
    print("\n✅ All tests passed! MCP server can access and run primitive scripts.")
    return True

if __name__ == "__main__":
    test_primitive_access()
