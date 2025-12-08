"""ROS2 subscriber for object poses from TFMessage topics."""

import threading
from typing import Dict, Callable, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from tf2_msgs.msg import TFMessage
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. Install rclpy and tf2_msgs to use ROS2 features.")


class PoseSubscriber(Node):
    """ROS2 node that subscribes to object pose topics."""
    
    def __init__(self, topic_name: str = '/objects_poses', callback: Optional[Callable] = None):
        """Initialize the pose subscriber.
        
        Args:
            topic_name: Name of the ROS2 topic to subscribe to
            callback: Callback function to call when new poses are received
        """
        super().__init__('orientation_visualizer_subscriber')
        self.topic_name = topic_name
        self.callback = callback
        self.latest_poses: Dict[str, Dict] = {}
        
        # Create subscription
        self.subscription = self.create_subscription(
            TFMessage,
            topic_name,
            self.pose_callback,
            10
        )
        self.get_logger().info(f'Subscribed to {topic_name}')
    
    def pose_callback(self, msg: 'TFMessage') -> None:
        """Handle incoming pose messages.
        
        Args:
            msg: TFMessage containing object transforms
        """
        for transform in msg.transforms:
            object_name = transform.child_frame_id
            
            # Extract translation
            trans = transform.transform.translation
            translation = np.array([trans.x, trans.y, trans.z])
            
            # Extract rotation (quaternion)
            rot = transform.transform.rotation
            quaternion = np.array([rot.x, rot.y, rot.z, rot.w])
            
            # Store the pose
            self.latest_poses[object_name] = {
                'translation': translation,
                'quaternion': quaternion,
                'timestamp': transform.header.stamp
            }
        
        # Call the callback if provided
        if self.callback:
            self.callback(self.latest_poses)
    
    def get_pose(self, object_name: str) -> Optional[Dict]:
        """Get the latest pose for a specific object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Dictionary containing translation and quaternion, or None if not found
        """
        return self.latest_poses.get(object_name)
    
    def get_all_objects(self) -> list:
        """Get list of all tracked object names.
        
        Returns:
            List of object names
        """
        return list(self.latest_poses.keys())


class ROS2Thread(threading.Thread):
    """Thread for running the ROS2 subscriber."""
    
    def __init__(self, topic_name: str = '/objects_poses', callback: Optional[Callable] = None):
        """Initialize the ROS2 thread.
        
        Args:
            topic_name: Name of the ROS2 topic to subscribe to
            callback: Callback function to call when new poses are received
        """
        super().__init__(daemon=True)
        self.topic_name = topic_name
        self.callback = callback
        self.subscriber: Optional[PoseSubscriber] = None
        self.running = False
    
    def run(self) -> None:
        """Run the ROS2 subscriber in a separate thread."""
        if not ROS2_AVAILABLE:
            print("Error: ROS2 not available. Cannot start subscriber.")
            return
        
        try:
            rclpy.init()
            self.subscriber = PoseSubscriber(self.topic_name, self.callback)
            self.running = True
            
            while self.running and rclpy.ok():
                rclpy.spin_once(self.subscriber, timeout_sec=0.1)
        except Exception as e:
            print(f"Error in ROS2 thread: {e}")
        finally:
            if self.subscriber:
                self.subscriber.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    
    def stop(self) -> None:
        """Stop the ROS2 subscriber thread."""
        self.running = False
    
    def get_pose(self, object_name: str) -> Optional[Dict]:
        """Get the latest pose for a specific object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Dictionary containing translation and quaternion, or None if not found
        """
        if self.subscriber:
            return self.subscriber.get_pose(object_name)
        return None
    
    def get_all_objects(self) -> list:
        """Get list of all tracked object names.
        
        Returns:
            List of object names
        """
        if self.subscriber:
            return self.subscriber.get_all_objects()
        return []

