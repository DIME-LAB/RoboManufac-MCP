import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from .kalman_filter import BLEND_FACTOR


def rvec_to_quat(rvec):
    """Convert OpenCV rotation vector to quaternion [x, y, z, w]."""
    rot, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot).as_quat()


def quat_to_rvec(quat):
    """Convert quaternion [x, y, z, w] to OpenCV rotation vector."""
    rot = R.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rot)
    return rvec


def slerp_quat(q1, q2, blend=BLEND_FACTOR):
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + blend * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * blend
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q1 + s1 * q2


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix."""
    r, p, y = roll, pitch, yaw

    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    return rz @ ry @ rx


def rotation_matrix_to_euler(rotation_matrix):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in radians."""
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])


def estimate_object_pose_from_marker(marker_pose, aruco_annotation):
    """Estimate the 6D pose of the object center from an ArUco marker pose."""
    marker_tvec, marker_rvec = marker_pose

    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    marker_tvec = marker_tvec.flatten()

    if "T_object_to_marker" not in aruco_annotation:
        raise ValueError(
            f"Invalid JSON format! Marker ID {aruco_annotation.get('aruco_id', 'unknown')} missing required 'T_object_to_marker' field. "
            f"Available keys: {list(aruco_annotation.keys())}"
        )

    obj_to_marker_data = aruco_annotation["T_object_to_marker"]

    t_obj_to_marker = np.array(
        [
            obj_to_marker_data["position"]["x"],
            obj_to_marker_data["position"]["y"],
            obj_to_marker_data["position"]["z"],
        ]
    )

    obj_to_marker_rot = obj_to_marker_data["rotation"]
    quat = obj_to_marker_rot["quaternion"]
    quat_array = np.array([quat["x"], quat["y"], quat["z"], quat["w"]])  # scipy uses x, y, z, w
    r_obj_to_marker = R.from_quat(quat_array).as_matrix()

    r_marker_to_obj = r_obj_to_marker.T
    t_marker_to_obj = -r_marker_to_obj @ t_obj_to_marker

    t_camera_to_marker = np.eye(4)
    t_camera_to_marker[:3, :3] = marker_rotation_matrix
    t_camera_to_marker[:3, 3] = marker_tvec

    t_marker_to_object = np.eye(4)
    t_marker_to_object[:3, :3] = r_marker_to_obj
    t_marker_to_object[:3, 3] = t_marker_to_obj

    t_camera_to_object = t_camera_to_marker @ t_marker_to_object

    object_rotation_matrix = t_camera_to_object[:3, :3]
    object_tvec = t_camera_to_object[:3, 3]

    object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)

    return object_tvec, object_rvec


def pose_to_world(object_tvec_cam, object_rvec_cam, camera_quat_world):
    """Lift a pose from camera frame to a nominal world frame using a fixed camera quaternion."""
    cam_rot = R.from_quat(camera_quat_world)
    rot_cam, _ = cv2.Rodrigues(object_rvec_cam)

    world_rotation_matrix = cam_rot.as_matrix() @ rot_cam
    world_rvec, _ = cv2.Rodrigues(world_rotation_matrix)
    quat_world = R.from_matrix(world_rotation_matrix).as_quat()
    rpy_world = rotation_matrix_to_euler(world_rotation_matrix)
    tvec_world = cam_rot.apply(object_tvec_cam.reshape(3,))

    return tvec_world, world_rvec, quat_world, rpy_world


__all__ = [
    "rvec_to_quat",
    "quat_to_rvec",
    "slerp_quat",
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "estimate_object_pose_from_marker",
    "pose_to_world",
]

