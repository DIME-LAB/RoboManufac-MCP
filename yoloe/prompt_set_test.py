#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

class WhiteBoxPromptTester(Node):
    def __init__(self):
        super().__init__('white_box_prompt_tester')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Processing control
        self.last_process_time = 0
        self.process_interval = 3.0  # Process every 3 seconds
        self.image_captured = False
        self.test_results = {}
        
        # Subscribe to camera topic
        self.image_subscription = self.create_subscription(
            Image, 
            '/intel_camera_rgb_raw',  # Adjust this topic name as needed
            self.image_callback, 
            10
        )
        
        self.get_logger().info("🤖 White Box Prompt Tester started")
        self.get_logger().info("📸 Waiting for image from camera topic...")
    
    def image_callback(self, msg):
        """Process incoming images and run prompt tests"""
        current_time = time.time()
        
        # Rate limiting - only process every 3 seconds
        if current_time - self.last_process_time < self.process_interval:
            return
            
        self.last_process_time = current_time
        
        if self.image_captured:
            return
            
        self.get_logger().info("📸 Capturing image for white box detection...")
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info(f"🖼️  Image captured: {cv_image.shape}")
            
            # Save the captured image
            image_path = '/tmp/captured_white_box_test.jpg'
            cv2.imwrite(image_path, cv_image)
            self.get_logger().info(f"💾 Saved image to {image_path}")
            
            # Run the prompt tests
            self.test_white_box_prompts(cv_image)
            self.image_captured = True
            
        except Exception as e:
            self.get_logger().error(f"❌ Error processing image: {str(e)}")
    
    def test_white_box_prompts(self, image):
        """Test different prompts for white box detection"""
        self.get_logger().info("🧪 Testing white box detection prompts...")
        
        # Define white box specific prompts to test
        prompt_sets = [
            # Basic white object prompts
            ["white box", "white object", "white cube"],
            
            # Descriptive prompts
            ["white rectangular box", "white square box", "white container"],
            
            # Color-focused prompts
            ["white", "white thing", "white item"],
            
            # Shape-focused prompts
            ["box", "cube", "rectangular object"],
            
            # Generic object prompts
            ["object", "item", "thing"],
            
            # Very specific prompts
            ["white cardboard box", "white plastic box", "white container box"],
            
            # Alternative descriptions
            ["white box on table", "white object in scene", "white box visible"],
            
            # Simple single-word prompts
            ["box", "white", "object"]
        ]
        
        # Test with YOLOE if available
        self.test_yoloe_prompts(image, prompt_sets)
    
    def test_yoloe_prompts(self, image, prompt_sets):
        """Test YOLOE prompt-free model with different prompts"""
        try:
            # Use the prompted model in the same directory
            model_path = os.path.join(os.path.dirname(__file__), 'yoloe-11s-seg.pt')
            
            if not os.path.exists(model_path):
                self.get_logger().warn(f"❌ YOLOE prompted model not found at {model_path}")
                return
            
            try:
                from ultralytics import YOLOE
                self.get_logger().info("✅ YOLOE imported successfully")
            except ImportError:
                self.get_logger().warn("❌ YOLOE not available in ultralytics")
                return
            
            # Load YOLOE prompted model
            model = YOLOE(model_path)
            self.get_logger().info(f"✅ Loaded YOLOE prompted model")
            
            # Test each prompt set with the prompted model
            for i, prompts in enumerate(prompt_sets):
                self.get_logger().info(f"\n🔍 Testing prompt set {i+1}: {prompts}")
                
                try:
                    # Set the prompts for this test
                    model.set_classes(prompts, model.get_text_pe(prompts))
                    self.get_logger().info("✅ Set text embeddings for prompt-based detection")
                    
                    # Run prediction with the prompted model
                    results = model.predict(image, verbose=False, conf=0.3)
                    
                    # Check results
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        detections = len(results[0].boxes)
                        self.get_logger().info(f"🎯 Found {detections} detections!")
                        
                        for j, box in enumerate(results[0].boxes):
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            if cls < len(prompts):
                                class_name = prompts[cls]
                            else:
                                class_name = f"class_{cls}"
                            self.get_logger().info(f"  ✅ {class_name} (conf: {conf:.3f})")
                        
                        # Store successful result
                        self.test_results[f"prompt_set_{i+1}"] = {
                            'prompts': prompts,
                            'detections': detections,
                            'success': True
                        }
                        
                        # Save result image in the same folder as the script
                        annotated = results[0].plot()
                        output_path = os.path.join(os.path.dirname(__file__), f'yoloe_prompt_set_{i+1}_result.jpg')
                        cv2.imwrite(output_path, annotated)
                        self.get_logger().info(f"💾 Saved result to {output_path}")
                    else:
                        self.get_logger().info(f"❌ No detections with prompts: {prompts}")
                        self.test_results[f"prompt_set_{i+1}"] = {
                            'prompts': prompts,
                            'detections': 0,
                            'success': False
                        }
                        
                except Exception as e:
                    self.get_logger().error(f"❌ Error with prompt set {i+1}: {str(e)}")
                    self.test_results[f"prompt_set_{i+1}"] = {
                        'prompts': prompts,
                        'detections': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Print summary of results
            self.print_results_summary()
            
            # Exit after YOLOE testing is complete
            self.get_logger().info("✅ YOLOE prompt testing completed!")
            self.get_logger().info("🛑 Shutting down...")
            rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f"❌ YOLOE test failed: {str(e)}")
            rclpy.shutdown()
    
    
    def print_results_summary(self):
        """Print a summary of the prompted model results"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("📊 YOLOE PROMPTED MODEL RESULTS")
        self.get_logger().info("="*60)
        
        if not self.test_results:
            self.get_logger().info("❌ No test results available")
            return
        
        successful_prompts = []
        failed_prompts = []
        
        for key, result in self.test_results.items():
            if result.get('success', False):
                successful_prompts.append((key, result.get('prompts', []), result.get('detections', 0)))
            else:
                failed_prompts.append((key, result.get('prompts', [])))
        
        if successful_prompts:
            self.get_logger().info("✅ SUCCESSFUL PROMPTS:")
            for key, prompts, detections in successful_prompts:
                self.get_logger().info(f"  {key}: {prompts} -> {detections} detections")
        else:
            self.get_logger().info("❌ No successful prompts found")
        
        if failed_prompts:
            self.get_logger().info("\n❌ FAILED PROMPTS:")
            for key, prompts in failed_prompts:
                self.get_logger().info(f"  {key}: {prompts}")
        
        self.get_logger().info("\n💡 RECOMMENDATIONS:")
        if successful_prompts:
            best_prompt = max(successful_prompts, key=lambda x: x[2])
            self.get_logger().info(f"  🏆 Best performing prompt: {best_prompt[1]}")
            self.get_logger().info(f"  📈 Detections: {best_prompt[2]}")
        else:
            self.get_logger().info("  🔧 Try adjusting confidence thresholds or image quality")
            self.get_logger().info("  🔧 Consider using different prompt combinations")
        
        self.get_logger().info("="*60)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WhiteBoxPromptTester()
        node.get_logger().info("🚀 Starting White Box Prompt Tester...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down White Box Prompt Tester...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
