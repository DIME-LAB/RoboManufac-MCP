#!/usr/bin/env python3
"""
ROS2 Image Topic Video Recorder

Records multiple ROS2 image topics to MP4 video files.
Each recording session is saved in a new folder with timestamp.

Edit the TOPICS list below to specify which topics to record.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import queue
import time

# ============================================================================
# CONFIGURATION: Edit this list to specify which topics to record
# ============================================================================
TOPICS = [
    # "/camera/image_raw",
    "/intel_camera_rgb_sim",
    "/annotated_stream",
    # "/isometric_camera/image_raw",
    # Add your topic names here as strings
]
# ============================================================================


class TopicRecorder:
    """Handles recording for a single topic"""
    
    def __init__(self, topic_name: str, output_path: Path, fps: float = None):
        self.topic_name = topic_name
        self.output_path = output_path
        self.fps = fps  # None means calculate dynamically
        self.bridge = CvBridge()
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.recording = False
        self.thread: Optional[threading.Thread] = None
        self.frame_size = None
        self.frame_count = 0
        # For dynamic FPS calculation
        self.timestamps = []
        self.last_timestamp = None
        self.calculated_fps = 30.0  # Default, will be updated
        self.fps_calculation_frames = 10  # Calculate FPS from first N frames
        self.min_frames_for_writer = 2  # Create writer after at least this many frames
        self.frames_received = 0
        self.writer_creation_start_time = None
        
    def start(self):
        """Start the recording thread"""
        self.recording = True
        self.thread = threading.Thread(target=self._write_frames, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop recording and finalize video"""
        self.recording = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.writer:
            self.writer.release()
            self.writer = None
        print(f"[{self.topic_name}] Stopped recording. Saved {self.frame_count} frames to {self.output_path}")
        
    def add_frame(self, cv_image, timestamp: float = None):
        """Add a frame to the queue for writing"""
        if not self.recording:
            return
        
        current_time = timestamp if timestamp is not None else time.time()
        self.frames_received += 1
        
        # Track when we started waiting for writer creation
        if self.writer is None and self.writer_creation_start_time is None:
            self.writer_creation_start_time = time.time()
        
        # Calculate FPS dynamically if not set
        if self.fps is None:
            if self.last_timestamp is not None:
                dt = current_time - self.last_timestamp
                if dt > 0:
                    self.timestamps.append(dt)
                    # Calculate FPS once we have enough samples
                    if len(self.timestamps) >= self.fps_calculation_frames:
                        avg_dt = sum(self.timestamps) / len(self.timestamps)
                        self.calculated_fps = 1.0 / avg_dt if avg_dt > 0 else 30.0
                        # Clamp FPS to reasonable range
                        self.calculated_fps = max(1.0, min(120.0, self.calculated_fps))
                    # Also calculate FPS with fewer samples if we have at least 2
                    elif len(self.timestamps) >= 2:
                        avg_dt = sum(self.timestamps) / len(self.timestamps)
                        self.calculated_fps = 1.0 / avg_dt if avg_dt > 0 else 30.0
                        self.calculated_fps = max(1.0, min(120.0, self.calculated_fps))
            
            self.last_timestamp = current_time
            fps_to_use = self.calculated_fps
        else:
            fps_to_use = self.fps
            
        # Initialize writer after calculating FPS (or on first frame if FPS is set)
        if self.writer is None:
            # Determine if we should create the writer now
            should_create_writer = False
            
            if self.fps is not None:
                # FPS is fixed, create writer immediately
                should_create_writer = True
            elif len(self.timestamps) >= self.min_frames_for_writer:
                # We have enough timestamps to estimate FPS
                should_create_writer = True
            elif self.frames_received >= self.min_frames_for_writer:
                # We have minimum frames, create writer with current FPS estimate
                should_create_writer = True
            elif self.writer_creation_start_time and (time.time() - self.writer_creation_start_time) > 2.0:
                # Timeout: create writer after 2 seconds even if we don't have enough samples
                should_create_writer = True
                if len(self.timestamps) == 0:
                    # No timestamps yet, use default FPS
                    self.calculated_fps = 30.0
            
            if should_create_writer:
                h, w = cv_image.shape[:2]
                self.frame_size = (w, h)
                # Sanitize topic name for filename
                safe_name = self.topic_name.replace('/', '_').replace(' ', '_').strip('_')
                video_path = self.output_path / f"{safe_name}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    fps_to_use,
                    self.frame_size
                )
                if not self.writer.isOpened():
                    print(f"[{self.topic_name}] ERROR: Failed to open video writer for {video_path}")
                    self.writer = None
                    return
                print(f"[{self.topic_name}] Started recording to {video_path} (FPS: {fps_to_use:.2f})")
                self.writer_creation_start_time = None  # Reset since writer is created
        
        # Add frame to queue (non-blocking, drop if full)
        # Always queue frames - writer will be created when FPS is calculated
        try:
            self.frame_queue.put_nowait(cv_image.copy())
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(cv_image.copy())
            except queue.Empty:
                pass
                
    def _write_frames(self):
        """Background thread to write frames to video file"""
        while self.recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.writer and frame is not None:
                    self.writer.write(frame)
                    self.frame_count += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.topic_name}] Error writing frame: {e}")


class VideoRecorderNode(Node):
    """ROS2 Node that records multiple image topics"""
    
    def __init__(self, topic_names: List[str], output_dir: Path):
        super().__init__('video_recorder')
        self.topic_names = topic_names
        self.output_dir = output_dir
        self.bridge = CvBridge()
        self.recorders: Dict[str, TopicRecorder] = {}
        self.topic_subscriptions = []
        self.shutdown_requested = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Recording to directory: {self.output_dir}")
        
        # Create recorder for each topic
        for topic_name in topic_names:
            recorder = TopicRecorder(topic_name, self.output_dir)
            recorder.start()
            self.recorders[topic_name] = recorder
            
            # Subscribe to topic
            subscription = self.create_subscription(
                Image,
                topic_name,
                lambda msg, t=topic_name: self.image_callback(msg, t),
                10
            )
            self.topic_subscriptions.append(subscription)
            print(f"Subscribed to topic: {topic_name}")
            
        print(f"\n{'='*60}")
        print(f"Recording {len(topic_names)} topic(s)...")
        print(f"Press Ctrl+C to stop recording and save files")
        print(f"{'='*60}\n")
        
    def image_callback(self, msg: Image, topic_name: str):
        """Callback for image messages"""
        if self.shutdown_requested:
            return
            
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Get timestamp from message header
            timestamp = None
            if msg.header.stamp:
                # Convert ROS time to seconds
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Add frame to recorder
            if topic_name in self.recorders:
                self.recorders[topic_name].add_frame(cv_image, timestamp)
            else:
                self.get_logger().warn(f"Received frame from {topic_name} but no recorder found")
        except Exception as e:
            self.get_logger().error(f"Error processing image from {topic_name}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
    def stop_recording(self):
        """Stop all recorders"""
        self.shutdown_requested = True
        print("\nStopping recording...")
        for topic_name, recorder in self.recorders.items():
            recorder.stop()
        print("All recordings saved successfully!")


def main():
    """Main function"""
    # Get topic names from the TOPICS list defined at the top of the file
    topic_names = [topic for topic in TOPICS if topic.strip()]  # Filter out empty strings
    
    if not topic_names:
        print("ERROR: No topics specified!")
        print("Please edit the TOPICS list at the top of this file to add topic names.")
        print("\nExample:")
        print('  TOPICS = [')
        print('      "/camera/image_raw",')
        print('      "/intel_camera_rgb",')
        print('  ]')
        sys.exit(1)
    
    # Create output directory with timestamp
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "recordings" / f"recording_{timestamp}"
    
    # Initialize ROS2
    rclpy.init()
    
    recorder_node = None
    shutdown_called = False
    
    try:
        # Create recorder node
        recorder_node = VideoRecorderNode(topic_names, output_dir)
        
        # Set up signal handler for Ctrl+C
        def signal_handler(sig, frame):
            """Handle Ctrl+C gracefully"""
            nonlocal shutdown_called
            print("\n\nReceived interrupt signal (Ctrl+C)")
            if recorder_node:
                recorder_node.stop_recording()
            if not shutdown_called:
                shutdown_called = True
                rclpy.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Spin the node
        rclpy.spin(recorder_node)
        
    except KeyboardInterrupt:
        # This should be handled by signal handler, but just in case
        if recorder_node:
            recorder_node.stop_recording()
    except SystemExit:
        # Re-raise SystemExit from signal handler
        raise
    except Exception as e:
        print(f"Error: {e}")
        if recorder_node:
            recorder_node.stop_recording()
    finally:
        if recorder_node:
            recorder_node.stop_recording()
        if not shutdown_called:
            try:
                rclpy.shutdown()
            except RuntimeError:
                # Context already shut down, ignore
                pass


if __name__ == '__main__':
    main()

