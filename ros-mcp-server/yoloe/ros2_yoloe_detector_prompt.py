#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import signal
import sys
import atexit

class YOLOEPromptDetectorNode(Node):
    def __init__(self):
        super().__init__('yoloe_prompt_detector')
        
        # Shutdown flag
        self.shutdown_requested = False
        
        # Initialize YOLOE prompt-based model
        from ultralytics import YOLOE
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'yoloe-11s-seg.pt')
        self.model = YOLOE(model_path)
        self.get_logger().info("✅ Loaded YOLOE prompt-based model")
        
        # Define working prompts based on debug results
        self.names = ["first-aid kit"]
        # self.names = ["bookmark", "lamp shade", "neon light "]
        self.get_logger().info(f"🎯 Detection targets: {self.names}")
        
        # Set the classes and get text embeddings for prompt-based detection
        self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        self.get_logger().info("✅ Set text embeddings for prompt-based detection")
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Processing control
        self.last_process_time = 0
        self.process_interval = 1.0  # Process every 2 seconds to avoid overload
        
        # NMS parameters for reducing overlapping detections
        self.nms_threshold = 0.3  # IoU threshold for Non-Maximum Suppression (more aggressive)
        self.conf_threshold = 0.4  # Minimum confidence threshold (balanced)
        
        # Initialize filtered detections storage
        self.filtered_detections = {'boxes': [], 'scores': [], 'class_ids': []}
        
        # OpenCV window
        cv2.namedWindow("YOLOE Prompt Detection", cv2.WINDOW_AUTOSIZE)
        
        # Subscribe to rgb_lego topic
        self.rgb_subscription = self.create_subscription(
            Image, 
            '/intel_camera_rgb_raw', 
            self.rgb_callback, 
            10
        )
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
        self.get_logger().info("🤖 YOLOE Prompt Detector started - Press 'q' to quit or Ctrl+C to exit")
    
    def cleanup(self):
        """Cleanup function to properly close OpenCV windows and resources"""
        try:
            self.get_logger().info("🧹 Cleaning up resources...")
            cv2.destroyAllWindows()
            self.get_logger().info("✅ Cleanup completed")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        self.shutdown_requested = True
        self.get_logger().info("🛑 Shutdown requested")
    
    def apply_nms(self, boxes, scores, class_ids):
        """Apply Non-Maximum Suppression to reduce overlapping detections"""
        if len(boxes) == 0:
            return [], [], []
        
        # Convert boxes to the format expected by cv2.dnn.NMSBoxes
        # boxes should be in format [x, y, width, height]
        boxes_nms = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            boxes_nms.append([x1, y1, w, h])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_nms, 
            scores, 
            self.conf_threshold, 
            self.nms_threshold
        )
        
        # Extract the filtered results
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_scores = [scores[i] for i in indices]
            filtered_class_ids = [class_ids[i] for i in indices]
            return filtered_boxes, filtered_scores, filtered_class_ids
        else:
            return [], [], []
    
    def rgb_callback(self, msg):
        """Process incoming RGB images from /rgb_lego topic"""
        # Check if shutdown was requested
        if self.shutdown_requested:
            return
            
        import time
        current_time = time.time()
        
        # Rate limiting - only process every 2 seconds
        if current_time - self.last_process_time < self.process_interval:
            return
            
        self.last_process_time = current_time
        self.get_logger().info("📸 Processing image from /rgb_lego topic")
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info(f"🖼️  Image converted: {cv_image.shape}")
            
            # Run YOLOE detection with text prompts
            self.get_logger().info("🔍 Running YOLOE detection...")
            results = self.model.predict(cv_image, verbose=False, conf=0.4)
            self.get_logger().info("✅ YOLOE detection completed")
            
            # Debug: Check if any detections were found
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Extract detection data
                boxes_raw = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores_raw = results[0].boxes.conf.cpu().numpy()
                class_ids_raw = results[0].boxes.cls.cpu().numpy().astype(int)
                
                self.get_logger().info(f"🔍 Raw detections: {len(results[0].boxes)}")
                
                # Apply NMS to reduce overlapping detections
                filtered_boxes, filtered_scores, filtered_class_ids = self.apply_nms(
                    boxes_raw.tolist(), 
                    scores_raw.tolist(), 
                    class_ids_raw.tolist()
                )
                
                if len(filtered_boxes) > 0:
                    self.get_logger().info(f"🎯 After NMS: {len(filtered_boxes)} detections!")
                    for i, (box, conf, cls) in enumerate(zip(filtered_boxes, filtered_scores, filtered_class_ids)):
                        class_name = self.names[cls] if cls < len(self.names) else f"class_{cls}"
                        self.get_logger().info(f"  Detection {i+1}: {class_name} (confidence: {conf:.3f})")
                else:
                    self.get_logger().info("❌ No detections after NMS filtering")
                    
                # Store filtered detections for manual visualization
                self.filtered_detections = {
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'class_ids': filtered_class_ids
                }
            else:
                self.get_logger().debug("No detections found")
            
            # Get inference time and calculate FPS
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time  # Convert to FPS
            fps_text = f'FPS: {fps:.1f}'
            
            # Create annotated image manually with filtered detections
            annotated_image = cv_image.copy()
            
            # Draw filtered detections manually
            if hasattr(self, 'filtered_detections') and len(self.filtered_detections['boxes']) > 0:
                for box, score, class_id in zip(
                    self.filtered_detections['boxes'], 
                    self.filtered_detections['scores'], 
                    self.filtered_detections['class_ids']
                ):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.names[class_id] if class_id < len(self.names) else f"class_{class_id}"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add FPS text to frame (top-right corner)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
            text_x = annotated_image.shape[1] - text_size[0] - 10  # 10 pixels from the right
            text_y = text_size[1] + 10  # 10 pixels from the top
            cv2.putText(annotated_image, fps_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add detection targets info (top-left corner)
            targets_text = f"Targets: {', '.join(self.names)}"
            cv2.putText(annotated_image, targets_text, (10, 30), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add NMS info (bottom-left corner)
            nms_text = f"NMS: IoU={self.nms_threshold}, Conf={self.conf_threshold}"
            cv2.putText(annotated_image, nms_text, (10, annotated_image.shape[0] - 10), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imshow("YOLOE Prompt Detection", annotated_image)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("🛑 'q' pressed, shutting down...")
                self.request_shutdown()
                return
                
        except Exception as e:
            self.get_logger().error(f"❌ Error processing image: {str(e)}")
            import traceback
            self.get_logger().error(f"❌ Full error: {traceback.format_exc()}")

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    print("\n🛑 Received interrupt signal (Ctrl+C), shutting down gracefully...")
    if 'node' in globals():
        node.request_shutdown()
    sys.exit(0)

def main(args=None):
    global node
    rclpy.init(args=args)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    node = None
    try:
        node = YOLOEPromptDetectorNode()
        node.get_logger().info("🚀 Starting YOLOE prompt detection...")
        
        # Main loop with proper shutdown handling
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
            
        node.get_logger().info("🛑 Shutdown requested, cleaning up...")
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup sequence
        try:
            if node is not None:
                node.get_logger().info("🧹 Destroying node...")
                node.destroy_node()
        except Exception as e:
            print(f"❌ Error destroying node: {e}")
        
        try:
            if rclpy.ok():
                node.get_logger().info("🧹 Shutting down ROS2...")
                rclpy.shutdown()
        except Exception as e:
            print(f"❌ Error shutting down ROS2: {e}")
        
        try:
            print("🧹 Closing OpenCV windows...")
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"❌ Error closing OpenCV windows: {e}")
        
        print("✅ Cleanup completed, exiting...")

if __name__ == '__main__':
    main()
