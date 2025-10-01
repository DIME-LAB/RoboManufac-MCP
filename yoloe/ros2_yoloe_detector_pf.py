#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from ultralytics import YOLO

class YOLOEDetectorNode(Node):
    def __init__(self):
        super().__init__('yoloe_detector')
        
        # Initialize YOLOE prompt-free model
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'yoloe-11s-seg-pf.pt')
        self.model = YOLO(model_path)
        self.get_logger().info("✅ Loaded YOLOE prompt-free model")
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        
        # OpenCV window
        cv2.namedWindow("YOLOE Live Detection", cv2.WINDOW_AUTOSIZE)
        
        # Subscribe to rgb_lego topic
        self.rgb_subscription = self.create_subscription(
            Image, 
            '/intel_camera_rgb_raw', 
            self.rgb_callback, 
            10
        )
        
        # Publish annotated image with bounding boxes
        self.annotated_publisher = self.create_publisher(
            Image, 
            '/yoloe_annotated_image', 
            10
        )
        
        self.get_logger().info("🤖 YOLOE Detector started - Press 'q' to quit")
    
    def rgb_callback(self, msg):
        """Process incoming RGB images from /rgb_lego topic"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLOE detection with optimizations
            results = self.model(cv_image, verbose=False, conf=0.3)
            
            # Get inference time and calculate FPS
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time  # Convert to FPS
            fps_text = f'FPS: {fps:.1f}'
            
            # Display results with FPS
            annotated_image = results[0].plot(boxes=True, masks=False)
            
            # Add FPS text to frame (top-right corner)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
            text_x = annotated_image.shape[1] - text_size[0] - 10  # 10 pixels from the right
            text_y = text_size[1] + 10  # 10 pixels from the top
            cv2.putText(annotated_image, fps_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Publish annotated image to ROS2 topic
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                annotated_msg.header = msg.header  # Copy original header
                self.annotated_publisher.publish(annotated_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish annotated image: {str(e)}")
            
            cv2.imshow("YOLOE Live Detection", annotated_image)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("🛑 'q' pressed, shutting down...")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"❌ Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = YOLOEDetectorNode()
        node.get_logger().info("🚀 Starting YOLOE detection...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down YOLOE Detector Node...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()