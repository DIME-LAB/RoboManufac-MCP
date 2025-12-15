# ArUco Localizer

OpenCV-based ArUco detection and object pose estimation with mesh overlay

## Components

- `core/` shared utilities:
  - `kalman_filter.py` – Quaternion Kalman + tuning constants
  - `pose_math.py` – pose estimation, quat/rvec helpers, RPY
  - `model_io.py` – load wireframe/aruco/assembly JSON, model discovery
  - `mesh_ops.py` – mesh transforms + projection/drawing
  - `camera_streamer.py` – camera detect/preview/stream (import via `core`)
- Scripts:
  - `object_pose_estimator_camera.py` (desktop)
  - `object_pose_estimator_ros2.py` (ROS2)
  - `assembly_pose_detector_camera.py` (desktop, multi-object)
  - `assembly_pose_detector_ros2.py` (ROS2, multi-object)

## Capabilities

- Multi-dictionary support (4x4 and 5x5 in the same stream for assembly/ROS2).
- Per-model best-marker selection (confirmed-preferred) with Kalman smoothing.
- Mesh overlay per object (one wireframe per object) using shared pose math.
- Pose text overlay shows position and RPY for each detected model.
- Camera selection/streaming via `core.camera_streamer`.

## Install
```bash
pip install -r requirements.txt
```

## Usage (desktop)
- Object pose (single model):  
  `python src/aruco_localizer/object_pose_estimator_camera.py --model <model_name>`
- Assembly pose (all models, mixed dicts):  
  `python src/aruco_localizer/assembly_pose_detector_camera.py`
- Camera utility:  
  `python src/aruco_localizer/camera_streamer.py`  (shim)  
  or `python -m aruco_localizer.core.camera_streamer`

## Usage (ROS2)
- Object pose:  
  `python src/aruco_localizer/object_pose_estimator_ros2.py --model <model_name> --camera-topic /camera/image_raw`
- Assembly pose (multi-object):  
  `python src/aruco_localizer/assembly_pose_detector_ros2.py --camera-topic /camera/image_raw`

## Controls
- Window: `q` to quit, `s` to toggle mesh (desktop scripts).
- Camera selection (core streamer): any key to skip, `ESC` to select.

## Data
- ArUco annotations: `data/aruco/<model>_aruco.json`
- Wireframes: `data/wireframe/<model>_wireframe.json`
- Assembly: `data/fmb_assembly*.json`

## Notes
- Mixed-dictionary detection is automatic for assembly scripts; marker IDs are keyed by `(dict, id)` to avoid collisions.
- `aruco_mesh_overlay.py` is retained only for reference and is not required.