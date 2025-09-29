from typing import List, Any, Protocol, Dict, Optional
import json


class Publisher(Protocol):
    def send(self, message: dict) -> None:
        ...
    
    def receive_binary(self) -> bytes:
        ...


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid float value: {value}")


class PoseStamped:
    def __init__(self, publisher: Publisher, topic: str = "/object_pose"):
        self.publisher = publisher
        self.topic = topic

    def publish(self, position: List[float], orientation: List[float], frame_id: str = "base_link"):
        """
        Publish a PoseStamped message.
        
        Args:
            position: [x, y, z] position in meters
            orientation: [x, y, z, w] quaternion OR [roll, pitch, yaw] in degrees
            frame_id: Reference frame (default: "base_link")
        """
        position_f = [to_float(val) for val in position]
        
        # Handle both quaternion and RPY input
        if len(orientation) == 4:
            # Quaternion input [x, y, z, w]
            quat = [to_float(val) for val in orientation]
        elif len(orientation) == 3:
            # RPY input [roll, pitch, yaw] in degrees - convert to quaternion
            import math
            roll, pitch, yaw = [math.radians(to_float(val)) for val in orientation]
            
            # Convert RPY to quaternion
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            
            quat = [qx, qy, qz, qw]
        else:
            raise ValueError("Orientation must be [x,y,z,w] quaternion or [roll,pitch,yaw] degrees")

        msg = {
            "op": "publish",
            "topic": self.topic,
            "msg": {
                "header": {
                    "stamp": {"sec": 0, "nanosec": 0},  # Will be filled by ROS
                    "frame_id": frame_id
                },
                "pose": {
                    "position": {
                        "x": position_f[0],
                        "y": position_f[1], 
                        "z": position_f[2]
                    },
                    "orientation": {
                        "x": quat[0],
                        "y": quat[1],
                        "z": quat[2],
                        "w": quat[3]
                    }
                }
            }
        }
        self.publisher.send(msg)
        return msg

    def subscribe(self, timeout: float = 2.0) -> Optional[Dict]:
        """
        Subscribe to PoseStamped topic and return parsed data.
        
        Returns:
            Dictionary with parsed position, orientation (quaternion + RPY), and metadata
        """
        subscribe_msg = {
            "op": "subscribe",
            "topic": self.topic,
            "type": "geometry_msgs/PoseStamped"
        }
        self.publisher.send(subscribe_msg)
        
        raw = self.publisher.receive_binary()
        if not raw:
            return None
            
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
                
            msg_data = json.loads(raw)
            
            # Extract the ROS message
            if "msg" in msg_data:
                ros_msg = msg_data["msg"]
            else:
                ros_msg = msg_data
                
            # Parse the pose data
            if "pose" in ros_msg:
                pose = ros_msg["pose"]
                
                result = {
                    "topic": self.topic,
                    "frame_id": ros_msg.get("header", {}).get("frame_id", ""),
                    "timestamp": ros_msg.get("header", {}).get("stamp", {}),
                }
                
                # Extract position
                if "position" in pose:
                    pos = pose["position"]
                    result["position"] = {
                        "x": pos["x"],
                        "y": pos["y"], 
                        "z": pos["z"]
                    }
                
                # Extract orientation
                if "orientation" in pose:
                    quat = pose["orientation"]
                    result["orientation_quaternion"] = {
                        "x": quat["x"],
                        "y": quat["y"],
                        "z": quat["z"],
                        "w": quat["w"]
                    }
                    
                    # Convert quaternion to RPY
                    import math
                    qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]
                    
                    # Quaternion to Euler conversion
                    sinr_cosp = 2 * (qw * qx + qy * qz)
                    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                    roll = math.atan2(sinr_cosp, cosr_cosp)
                    
                    sinp = 2 * (qw * qy - qz * qx)
                    if abs(sinp) >= 1:
                        pitch = math.copysign(math.pi / 2, sinp)
                    else:
                        pitch = math.asin(sinp)
                    
                    siny_cosp = 2 * (qw * qz + qx * qy)
                    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                    yaw = math.atan2(siny_cosp, cosy_cosp)
                    
                    result["orientation_rpy_degrees"] = {
                        "roll": math.degrees(roll),
                        "pitch": math.degrees(pitch),
                        "yaw": math.degrees(yaw)
                    }
                    
                    result["orientation_rpy_radians"] = {
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw
                    }
                
                return result
            
            return ros_msg
            
        except Exception as e:
            print(f"[PoseStamped] Failed to parse: {e}")
            return None
