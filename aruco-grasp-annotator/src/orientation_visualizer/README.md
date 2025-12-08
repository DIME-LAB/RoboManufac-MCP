# Object Orientation Visualizer

A real-time visualization tool for monitoring object orientations from ROS2 topics compared against final assembly poses.

## Features

- **Split View Display**: Side-by-side comparison of final assembly orientation (left) and current live orientation (right)
- **ROS2 Integration**: Subscribes to TFMessage topics for real-time pose updates
- **Assembly JSON Support**: Auto-loads final object positions from FMB assembly JSON files
- **3D Model Visualization**: Displays actual object geometry using OBJ models
- **Object Selection**: Dropdown menu to select which object to monitor
- **Visual Orientation Display**: 3D coordinate frames and object meshes showing orientations
- **Live Updates**: Real-time updates as objects move in the physical world (10 Hz)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- PyQt6 (>=6.5.0)
- Open3D (>=0.17.0)
- NumPy (>=1.24.0)
- rclpy (ROS2 Python client library)
- tf2_msgs (ROS2 TF2 messages)

## Usage

### Running the Application

```bash
python app.py
```

Or from the project root:

```bash
python -m orientation_visualizer.app
```

### Workflow

1. **Auto-loaded on Startup**: The app automatically loads `data/fmb_assembly.json` and the first object

2. **Select Object**: Choose which object to monitor from the dropdown menu (e.g., base_scaled70, fork_orange_scaled70, etc.)

3. **ROS Topic**: Default is `/objects_poses_sim` (can be changed via dropdown or manual entry)

4. **Monitor Orientation**: 
   - **Left panel**: Shows the final assembly orientation (static reference from JSON)
   - **Right panel**: Shows the current live orientation from ROS2 (updates in real-time)
   - Both panels display the actual 3D object model with orientation

### ROS2 Topic Format

The application expects a `tf2_msgs/msg/TFMessage` topic with transforms containing:

```
transforms:
  - header:
      stamp: {sec: 0, nanosec: 0}
      frame_id: "World"
    child_frame_id: "object_name"
    transform:
      translation: {x: 0.0, y: 0.0, z: 0.0}
      rotation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
```

### Supported Objects

Objects must be:
1. Defined as components (not markers) in the assembly JSON
2. Published on the ROS2 topic with matching names

Common objects:
- base_scaled70
- fork_orange_scaled70
- fork_yellow_scaled70
- line_brown_scaled70
- line_red_scaled70

## Display Information

For each orientation, the application shows:
- **Quaternion**: [x, y, z, w] representation
- **Euler Angles**: Roll, Pitch, Yaw in degrees
- **3D Coordinate Frame**: Visual representation with RGB axes (X=Red, Y=Green, Z=Blue)
- **3D Object Model**: The actual object mesh rendered with the current orientation

## Troubleshooting

### ROS2 Not Available

If you see "ROS2 not available" error:
1. Ensure ROS2 is installed and sourced
2. Install Python ROS2 packages: `pip install rclpy tf2_msgs`
3. Source your ROS2 workspace: `source /opt/ros/<distro>/setup.bash`

### No Data Appearing

1. Verify ROS2 topic is publishing: `ros2 topic echo /objects_poses`
2. Check topic name matches in the application
3. Ensure object names in ROS2 match assembly JSON names
4. Click "Reconnect ROS" to restart the subscriber

### Coordinate System

- Assembly JSON uses Z-up coordinate system
- ROS2 topics should use the same coordinate system
- Rotations are represented as quaternions in ROS2
- Assembly JSON uses Euler angles (XYZ order) in radians

## Architecture

- **app.py**: Main PyQt6 application with GUI
- **ros_subscriber.py**: ROS2 subscriber thread for real-time pose updates
- **Open3DWidget**: Custom widget for 3D visualization using Open3D

## Notes

- The final assembly pose remains static (left panel)
- The current pose updates in real-time as objects move (right panel)
- Updates occur approximately 10 times per second
- 3D visualizations are rendered as images for better Qt integration

