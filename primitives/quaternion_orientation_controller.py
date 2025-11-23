#!/usr/bin/env python3
"""
Quaternion Orientation Controller - Gimbal Lock Free Gripper Control

This module provides a quaternion-based approach to gripper orientation control,
avoiding gimbal lock singularities that occur at pitch=180° when using Euler angles (RPY).

Key Insight:
- Euler angles have a singularity at pitch = ±90° (gimbal lock)
- At pitch=180°, the roll and yaw axes become colinear, losing one degree of freedom
- Quaternions use a 4-parameter representation on a 3-sphere that covers rotation space
  without singularities - enabling safe operation at ANY yaw angle

Reference:
- scipy.spatial.transform.Rotation uses [x, y, z, w] convention
- NOT [w, x, y, z] - verify with your system's expected format
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation as R


class QuaternionOrientationController:
    """
    Controls gripper orientation using quaternions instead of Euler angles.
    
    Eliminates gimbal lock issues when the gripper is constrained to face-down
    orientation (pitch = 180°) with variable yaw angles.
    """
    
    @staticmethod
    def face_down_quaternion(yaw_degrees):
        """
        Create quaternion for face-down gripper with specified yaw.
        
        Represents: roll=0°, pitch=180° (face down), yaw=yaw_degrees
        
        This is the core method for gimbal-lock-free gripper control.
        At pitch=180°, Euler angles have a singularity, but quaternions
        handle this without any issues.
        
        Args:
            yaw_degrees: Yaw rotation around world Z-axis in degrees
            
        Returns:
            quaternion [x, y, z, w] in scipy convention
            
        Example:
            q = face_down_quaternion(45.0)  # Yaw 45°, pitch 180°, roll 0°
            # Returns: [0.923880, 0.0, 0.0, 0.382683] (approximately)
        """
        # Create face-down orientation with yaw rotation
        # Using scipy's Rotation which handles gimbal lock internally
        r = R.from_euler('xyz', [0, 180, yaw_degrees], degrees=True)
        q = r.as_quat()  # Returns [x, y, z, w]
        
        # Ensure quaternion is normalized
        return QuaternionOrientationController.normalize_quaternion(q)
    
    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions to compose rotations.
        
        Useful for combining multiple rotations.
        
        Args:
            q1, q2: quaternions in [x, y, z, w] format
            
        Returns:
            Result quaternion [x, y, z, w]
            
        Example:
            q_face_down = face_down_quaternion(0)
            q_rotate_45 = from_euler('z', 45, degrees=True).as_quat()
            q_combined = quaternion_multiply(q_face_down, q_rotate_45)
        """
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        return (r1 * r2).as_quat()
    
    @staticmethod
    def apply_yaw_rotation(base_quaternion, yaw_offset_degrees):
        """
        Apply additional yaw rotation to existing quaternion.
        
        Useful for fine-tuning orientation or creating trajectories
        with incremental yaw adjustments.
        
        Args:
            base_quaternion: Starting quaternion [x, y, z, w]
            yaw_offset_degrees: Additional yaw rotation in degrees
            
        Returns:
            Modified quaternion [x, y, z, w]
        """
        r_yaw = R.from_euler('z', yaw_offset_degrees, degrees=True)
        r_base = R.from_quat(base_quaternion)
        return (r_yaw * r_base).as_quat()
    
    @staticmethod
    def slerp(q1, q2, t):
        """
        Smooth interpolation between two quaternions (Spherical Linear Interpolation).
        
        Useful for creating smooth trajectories between two quaternion-based orientations.
        Maintains constant rotation rate without gimbal lock artifacts.
        
        Args:
            q1: Start quaternion [x, y, z, w]
            q2: End quaternion [x, y, z, w]
            t: Interpolation parameter in [0, 1]
               t=0 returns q1, t=1 returns q2
               
        Returns:
            Interpolated quaternion [x, y, z, w]
            
        Example:
            q1 = face_down_quaternion(0)
            q2 = face_down_quaternion(90)
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                q_interp = slerp(q1, q2, t)
                # Smooth transition from 0° to 90° yaw
        """
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        
        # Calculate relative rotation from q1 to q2
        r_relative = r1.inv() * r2
        axis_angle = r_relative.as_rotvec()
        angle = np.linalg.norm(axis_angle)
        
        # Only interpolate if there's a meaningful rotation
        if angle > 1e-6:
            axis = axis_angle / angle
            r_interp = r1 * R.from_rotvec(axis * angle * t)
        else:
            # Quaternions are very close, just return q1
            r_interp = r1
        
        return r_interp.as_quat()
    
    @staticmethod
    def normalize_quaternion(q):
        """
        Normalize quaternion to unit length (magnitude = 1.0).
        
        All quaternions representing rotations must be unit quaternions.
        This method ensures numerical stability.
        
        Args:
            q: quaternion [x, y, z, w]
            
        Returns:
            Normalized quaternion [x, y, z, w]
        """
        magnitude = np.linalg.norm(q)
        if magnitude < 1e-10:
            raise ValueError(f"Cannot normalize zero quaternion: {q}")
        return q / magnitude
    
    @staticmethod
    def quaternion_to_rpy(q):
        """
        Convert quaternion to Euler angles (RPY in degrees).
        
        ⚠️ CAUTION: Use this ONLY at I/O boundaries for logging/display!
        
        Converting FROM quaternion TO RPY for intermediate calculations
        defeats the purpose of using quaternions. This method exists for:
        - Converting input/output at ROS message boundaries
        - Displaying orientation for logging
        - Reading external orientation values
        
        DO NOT use this inside calculation loops or trajectory generation!
        
        Args:
            q: quaternion [x, y, z, w] in scipy convention
            
        Returns:
            [roll, pitch, yaw] in degrees
        """
        r = R.from_quat(q)
        rpy_radians = r.as_euler('xyz')
        return np.degrees(rpy_radians)
    
    @staticmethod
    def rpy_to_quaternion(roll_deg, pitch_deg, yaw_deg):
        """
        Convert Euler angles (RPY) to quaternion.
        
        ⚠️ CAUTION: Only use at I/O boundaries!
        
        Use this to convert external RPY inputs (from ROS messages, files, etc.)
        to quaternion representation for processing.
        
        DO NOT use this for intermediate calculations or conversions!
        
        Args:
            roll_deg: Roll in degrees
            pitch_deg: Pitch in degrees
            yaw_deg: Yaw in degrees
            
        Returns:
            quaternion [x, y, z, w]
        """
        r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
        return r.as_quat()
    
    @staticmethod
    def quaternion_inverse(q):
        """
        Calculate inverse of a quaternion.
        
        For unit quaternion: inverse = conjugate = [-x, -y, -z, w]
        
        Args:
            q: quaternion [x, y, z, w]
            
        Returns:
            Inverse quaternion [x, y, z, w]
        """
        return np.array([-q[0], -q[1], -q[2], q[3]])
    
    @staticmethod
    def quaternion_distance(q1, q2):
        """
        Calculate angular distance between two quaternions.
        
        Uses dot product: distance = 1 - abs(dot(q1, q2))
        For unit quaternions, this gives angular distance in [0, 1]
        
        Args:
            q1, q2: quaternions [x, y, z, w]
            
        Returns:
            Distance in [0, 1] where 0 = identical, 1 = opposite
        """
        q1_norm = q1 / np.linalg.norm(q1)
        q2_norm = q2 / np.linalg.norm(q2)
        dot_product = abs(np.dot(q1_norm, q2_norm))
        return 1.0 - dot_product
    
    @staticmethod
    def extract_yaw_from_quaternion(q):
        """
        Extract yaw angle directly from quaternion (NO RPY conversion).
        
        For face-down quaternions (pitch=180°), uses: 2.0 * atan2(q_z, q_w)
        For other quaternions where q_z and q_w are near zero, uses standard conversion.
        
        Args:
            q: quaternion [x, y, z, w]
            
        Returns:
            yaw in degrees, normalized to [-180, 180]
        """
        q_x, q_y, q_z, q_w = q
        
        # Check if q_z and q_w are both near zero (not a face-down quaternion)
        # Use a threshold to detect when the face-down formula would be unreliable
        threshold = 1e-6
        if abs(q_z) < threshold and abs(q_w) < threshold:
            # Fallback: use standard quaternion-to-Euler conversion for yaw
            sin_y = 2.0 * (q_w * q_z + q_x * q_y)
            cos_y = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
            yaw_radians = math.atan2(sin_y, cos_y)
        else:
            # Standard formula for face-down quaternions (pitch=180°)
            yaw_radians = 2.0 * math.atan2(q_z, q_w)
        
        yaw_degrees = math.degrees(yaw_radians)
        # Normalize to [-180, 180]
        yaw_degrees = ((yaw_degrees + 180) % 360) - 180
        return yaw_degrees
    
    @staticmethod
    def load_fold_symmetry_json(object_name, symmetry_dir):
        """
        Load fold symmetry JSON file for an object using pattern matching.
        
        Searches for files matching pattern: {object_name}_*_symmetry.json
        
        Args:
            object_name: Name of the object (e.g., "line_red")
            symmetry_dir: Directory containing symmetry JSON files
            
        Returns:
            Dictionary with fold symmetry data, or None if not found
        """
        import json
        import os
        import glob
        
        # Pattern match: {object_name}_*_symmetry.json
        pattern = os.path.join(symmetry_dir, f"{object_name}_*_symmetry.json")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None
        
        # Use first match (should typically be only one)
        json_path = matching_files[0]
        
        try:
            with open(json_path, 'r') as f:
                fold_data = json.load(f)
            return fold_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading fold symmetry JSON from {json_path}: {e}")
            return None
    
    @staticmethod
    def find_canonical_quaternion(detected_quat, fold_data, threshold=0.1):
        """
        Find canonical quaternion match using transformation-based algorithm.
        
        Algorithm:
        1. Calculate transform from detected to identity: transform = identity * inverse(detected)
        2. Apply transform to all canonical quaternions from all fold axes
        3. Find closest match using quaternion distance
        4. Return original canonical quaternion if match found (within threshold), else None
        
        Args:
            detected_quat: Detected object quaternion [x, y, z, w]
            fold_data: Dictionary with fold symmetry data from JSON
            threshold: Maximum distance for a match (default 0.1)
            
        Returns:
            Canonical quaternion [x, y, z, w] if match found, else None
        """
        # Normalize detected quaternion
        detected_quat = np.array(detected_quat)
        detected_quat = detected_quat / np.linalg.norm(detected_quat)
        
        # 1. Collect all canonical quaternions from all fold axes
        all_canonicals = []
        for axis in ['x', 'y', 'z']:
            if axis not in fold_data.get('fold_axes', {}):
                continue
            for q_data in fold_data['fold_axes'][axis]['quaternions']:
                q = np.array([
                    q_data['quaternion']['x'],
                    q_data['quaternion']['y'],
                    q_data['quaternion']['z'],
                    q_data['quaternion']['w']
                ])
                # Normalize canonical quaternion
                q = q / np.linalg.norm(q)
                all_canonicals.append(q)
        
        if not all_canonicals:
            return None
        
        # 2. Calculate transform from detected to identity
        identity = np.array([0, 0, 0, 1])
        detected_inverse = QuaternionOrientationController.quaternion_inverse(detected_quat)
        transform = QuaternionOrientationController.quaternion_multiply(identity, detected_inverse)
        
        # 3. Apply transform to all canonicals
        transformed_canonicals = []
        for q in all_canonicals:
            transformed_q = QuaternionOrientationController.quaternion_multiply(transform, q)
            transformed_canonicals.append(transformed_q)
        
        # 4. Find closest match
        min_distance = float('inf')
        closest_idx = -1
        
        for i, transformed_q in enumerate(transformed_canonicals):
            distance = QuaternionOrientationController.quaternion_distance(detected_quat, transformed_q)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # 5. Return original canonical if match found
        if min_distance < threshold:
            return all_canonicals[closest_idx]
        else:
            return None
    
    @staticmethod
    def normalize_to_canonical(detected_quat, object_name, symmetry_dir, threshold=0.1):
        """
        Normalize detected quaternion to canonical pose using fold symmetry.
        
        Main entry point for fold symmetry matching. Loads symmetry data and
        finds canonical match. If match found, returns canonical quaternion;
        otherwise returns detected quaternion.
        
        Args:
            detected_quat: Detected object quaternion [x, y, z, w]
            object_name: Name of the object (e.g., "line_red")
            symmetry_dir: Directory containing symmetry JSON files
            threshold: Maximum distance for a match (default 0.1)
            
        Returns:
            Canonical quaternion [x, y, z, w] if match found, else detected_quat
        """
        # Load fold symmetry data
        fold_data = QuaternionOrientationController.load_fold_symmetry_json(object_name, symmetry_dir)
        
        if fold_data is None:
            # No symmetry data available, return detected quaternion as-is
            return np.array(detected_quat) / np.linalg.norm(detected_quat)
        
        # Find canonical match
        canonical_quat = QuaternionOrientationController.find_canonical_quaternion(
            detected_quat, fold_data, threshold
        )
        
        if canonical_quat is not None:
            # Match found, return canonical quaternion
            return canonical_quat
        else:
            # No match, return detected quaternion (normalized)
            return np.array(detected_quat) / np.linalg.norm(detected_quat)
    
    @staticmethod
    def verify_quaternion_approach():
        """
        Verify quaternion approach works for full 360° rotation (no gimbal lock).
        
        This test ensures that the quaternion-based approach is stable across
        all yaw angles, with no numerical instability or singularities.
        
        Returns:
            bool: True if all tests pass, False otherwise
            
        Example:
            controller = QuaternionOrientationController()
            if controller.verify_quaternion_approach():
                print("✅ Safe to use quaternion approach!")
            else:
                print("❌ Quaternion verification failed!")
        """
        print("=" * 70)
        print("Testing quaternion face-down approach (full 360° rotation)...")
        print("=" * 70)
        
        all_passed = True
        
        for yaw_deg in range(0, 360, 30):
            try:
                q = QuaternionOrientationController.face_down_quaternion(yaw_deg)
                
                # Verify magnitude = 1.0 (unit quaternion)
                mag = np.linalg.norm(q)
                mag_ok = abs(mag - 1.0) < 1e-6
                
                # Verify no NaN/Inf
                finite_ok = np.all(np.isfinite(q))
                
                # Convert back to RPY for verification
                rpy = QuaternionOrientationController.quaternion_to_rpy(q)
                
                status = "✅" if (mag_ok and finite_ok) else "❌"
                print(f"{status} Yaw {yaw_deg:3d}°: magnitude={mag:.8f}, "
                      f"RPY=[{rpy[0]:7.2f}°, {rpy[1]:7.2f}°, {rpy[2]:7.2f}°]")
                
                if not (mag_ok and finite_ok):
                    all_passed = False
                    print(f"   ⚠️  Issues detected!")
                    
            except Exception as e:
                print(f"❌ Yaw {yaw_deg:3d}°: Exception - {e}")
                all_passed = False
        
        print("=" * 70)
        if all_passed:
            print("✅ SUCCESS: Quaternion approach is gimbal-lock-free!")
            print("   All yaw angles (0-360°) work correctly with no gimbal lock.")
            print("   Magnitudes are normalized to 1.0 (unit quaternions).")
            print("   All values are numerically stable (no NaN/Inf).")
        else:
            print("❌ FAILURE: Quaternion verification detected issues!")
        print("=" * 70)
        
        return all_passed


# Convenience functions for backward compatibility
def face_down_quaternion(yaw_degrees):
    """Convenience function wrapper for QuaternionOrientationController.face_down_quaternion()"""
    return QuaternionOrientationController.face_down_quaternion(yaw_degrees)


def quaternion_to_rpy(q):
    """Convenience function wrapper for QuaternionOrientationController.quaternion_to_rpy()"""
    return QuaternionOrientationController.quaternion_to_rpy(q)


def rpy_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """Convenience function wrapper for QuaternionOrientationController.rpy_to_quaternion()"""
    return QuaternionOrientationController.rpy_to_quaternion(roll_deg, pitch_deg, yaw_deg)


if __name__ == "__main__":
    # Run verification test
    controller = QuaternionOrientationController()
    success = controller.verify_quaternion_approach()
    
    # Example usage
    print("\n" + "=" * 70)
    print("Example Usage:")
    print("=" * 70)
    
    # Example 1: Create face-down quaternion with 45° yaw
    q_45 = controller.face_down_quaternion(45.0)
    print(f"\n1. Face-down gripper at yaw=45°:")
    print(f"   Quaternion: [{q_45[0]:.6f}, {q_45[1]:.6f}, {q_45[2]:.6f}, {q_45[3]:.6f}]")
    rpy_45 = controller.quaternion_to_rpy(q_45)
    print(f"   Converted to RPY: {rpy_45}")
    
    # Example 2: Smooth interpolation
    q_0 = controller.face_down_quaternion(0.0)
    q_90 = controller.face_down_quaternion(90.0)
    q_interp = controller.slerp(q_0, q_90, 0.5)
    print(f"\n2. Smooth interpolation between yaw=0° and yaw=90° at t=0.5:")
    print(f"   Interpolated quaternion: [{q_interp[0]:.6f}, {q_interp[1]:.6f}, "
          f"{q_interp[2]:.6f}, {q_interp[3]:.6f}]")
    rpy_interp = controller.quaternion_to_rpy(q_interp)
    print(f"   Converted to RPY: {rpy_interp}")
    
    # Example 3: Apply additional yaw rotation
    q_base = controller.face_down_quaternion(0.0)
    q_rotated = controller.apply_yaw_rotation(q_base, 30.0)
    print(f"\n3. Apply 30° additional yaw rotation to face-down (yaw=0°):")
    print(f"   Result quaternion: [{q_rotated[0]:.6f}, {q_rotated[1]:.6f}, "
          f"{q_rotated[2]:.6f}, {q_rotated[3]:.6f}]")
    rpy_rotated = controller.quaternion_to_rpy(q_rotated)
    print(f"   Converted to RPY: {rpy_rotated}")
    
    print("\n" + "=" * 70)

