import cv2
import numpy as np

from .pose_math import estimate_object_pose_from_marker


def transform_mesh_to_camera_frame(vertices, object_pose):
    """Transform mesh vertices from object center frame to camera frame."""
    object_tvec, object_rvec = object_pose

    rotation_matrix, _ = cv2.Rodrigues(object_rvec)

    transformed_vertices = []
    for vertex in vertices:
        vertex_cam = rotation_matrix @ np.array(vertex) + object_tvec
        transformed_vertices.append(vertex_cam)

    return np.array(transformed_vertices)


def transform_mesh_from_marker(vertices, marker_pose, aruco_annotation):
    """Transform mesh vertices using marker pose and annotation (legacy convenience)."""
    object_pose = estimate_object_pose_from_marker(marker_pose, aruco_annotation)
    return transform_mesh_to_camera_frame(vertices, object_pose)


def project_vertices_to_image(vertices, camera_matrix, dist_coeffs):
    """Project 3D vertices (camera frame) to 2D image coordinates."""
    if len(vertices) == 0:
        return np.array([])

    projected_points, _ = cv2.projectPoints(
        vertices.astype(np.float32),
        np.zeros((3, 1)),  # Already in camera frame
        np.zeros((3, 1)),
        camera_matrix,
        dist_coeffs,
    )

    return projected_points.reshape(-1, 2).astype(np.int32)


def draw_wireframe(frame, projected_vertices, edges, color=(0, 255, 0), thickness=2):
    """Draw wireframe on the image."""
    if len(projected_vertices) == 0:
        return

    height, width = frame.shape[:2]
    valid_vertices = []
    valid_indices = []

    for i, vertex in enumerate(projected_vertices):
        x, y = vertex
        if 0 <= x < width and 0 <= y < height:
            valid_vertices.append(vertex)
            valid_indices.append(i)

    if len(valid_vertices) == 0:
        return

    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}

    for edge in edges:
        if len(edge) >= 2:
            start_idx, end_idx = edge[0], edge[1]
            if start_idx in index_map and end_idx in index_map:
                start_point = tuple(valid_vertices[index_map[start_idx]])
                end_point = tuple(valid_vertices[index_map[end_idx]])
                cv2.line(frame, start_point, end_point, color, thickness)

    for vertex in valid_vertices:
        cv2.circle(frame, tuple(vertex), 3, (255, 0, 0), -1)


__all__ = [
    "transform_mesh_to_camera_frame",
    "transform_mesh_from_marker",
    "project_vertices_to_image",
    "draw_wireframe",
]

