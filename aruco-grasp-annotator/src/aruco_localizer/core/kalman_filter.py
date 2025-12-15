import cv2
import numpy as np

# =============================================================================
# KALMAN FILTER CONFIGURATION - ADJUSTABLE VARIABLES
# =============================================================================

# Temporal filtering parameters
MAX_MOVEMENT_THRESHOLD = 0.05  # meters - maximum allowed movement between frames
HOLD_REQUIRED_FRAMES = 2       # frames - required stable detections before confirmation
GHOST_TRACKING_FRAMES = 15     # frames - continue tracking when marker lost
BLEND_FACTOR = 0.99            # 0.0-1.0 - trust in measurements vs predictions

# Kalman filter noise parameters
PROCESS_NOISE_POSITION = 1e-4   # Process noise for position (x,y,z)
PROCESS_NOISE_QUATERNION = 1e-3 # Process noise for quaternion (qx,qy,qz,qw)
PROCESS_NOISE_VELOCITY = 1e-4   # Process noise for velocity (vx,vy,vz)
MEASUREMENT_NOISE_POSITION = 1e-4 # Measurement noise for position
MEASUREMENT_NOISE_QUATERNION = 1e-4 # Measurement noise for quaternion


class QuaternionKalman:
    """Kalman filter for 6D pose estimation with quaternions."""

    def __init__(self):
        # 10 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz]
        self.kf = cv2.KalmanFilter(10, 7)

        dt = 1.0  # Time step (assuming 1 frame = 1 time unit)

        # A: Transition matrix (10x10)
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        for i in range(3):  # x += vx*dt, y += vy*dt, z += vz*dt
            self.kf.transitionMatrix[i, i + 7] = dt

        # H: Measurement matrix (7x10) - we measure position and quaternion
        self.kf.measurementMatrix = np.zeros((7, 10), dtype=np.float32)
        self.kf.measurementMatrix[0:7, 0:7] = np.eye(7)

        # Q: Process noise covariance
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-6
        for i in range(3):  # position noise
            self.kf.processNoiseCov[i, i] = PROCESS_NOISE_POSITION
        for i in range(3, 7):  # quaternion noise
            self.kf.processNoiseCov[i, i] = PROCESS_NOISE_QUATERNION
        for i in range(7, 10):  # velocity noise
            self.kf.processNoiseCov[i, i] = PROCESS_NOISE_VELOCITY

        # R: Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        for i in range(3):  # position measurement noise
            self.kf.measurementNoiseCov[i, i] = MEASUREMENT_NOISE_POSITION
        for i in range(3, 7):  # quaternion measurement noise
            self.kf.measurementNoiseCov[i, i] = MEASUREMENT_NOISE_QUATERNION

        # Initial error covariance
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((10, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion

    def correct(self, tvec, rvec):
        """Update filter with new measurement."""
        from .pose_math import rvec_to_quat

        quat = rvec_to_quat(rvec)
        measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
        self.kf.correct(measurement)

    def predict(self):
        """Predict next state."""
        from .pose_math import quat_to_rvec

        pred = self.kf.predict()
        pred_tvec = pred[0:3].flatten()
        pred_quat = pred[3:7].flatten()
        # Normalize quaternion to prevent drift
        pred_quat /= np.linalg.norm(pred_quat)
        pred_rvec = quat_to_rvec(pred_quat).flatten()
        return pred_tvec, pred_rvec


__all__ = [
    "QuaternionKalman",
    "MAX_MOVEMENT_THRESHOLD",
    "HOLD_REQUIRED_FRAMES",
    "GHOST_TRACKING_FRAMES",
    "BLEND_FACTOR",
    "PROCESS_NOISE_POSITION",
    "PROCESS_NOISE_QUATERNION",
    "PROCESS_NOISE_VELOCITY",
    "MEASUREMENT_NOISE_POSITION",
    "MEASUREMENT_NOISE_QUATERNION",
]

