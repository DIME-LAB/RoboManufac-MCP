#!/usr/bin/env python3
"""
Gripper Control with Verification

Controls gripper and verifies movement using gripper width readings (both sim and real).
Supports "open", "close", or numeric values 0-110 (representing 0-11cm width).

Usage:
    python3 control_gripper.py open [--mode sim|real]
    python3 control_gripper.py close [--mode sim|real]
    python3 control_gripper.py 55 [--mode sim|real]  # 5.5cm width
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Float32
import argparse
import time
import sys
import threading

# Width thresholds for verification
WIDTH_OPEN_THRESHOLD = 100.0  # When open, width should be > 100
WIDTH_CLOSE_THRESHOLD = 20.0  # When closed, width should be < 20
GRASP_SUCCESS_THRESHOLD = 35.0  # If width is 20-35 when closing, likely grasped an object

MAX_RETRIES = 3
RETRY_DELAY = 0.5  # Seconds to wait between retries

class GripperController(Node):
    def __init__(self, command, mode='sim'):
        super().__init__('gripper_controller')
        
        self.command = command
        self.mode = mode
        self.current_width = None
        self.width_received = False
        
        # Threading synchronization
        self.monitoring_lock = threading.Lock()
        self.verification_complete = False
        self.verification_result = None
        
        # Publisher for gripper commands
        self.gripper_pub = self.create_publisher(String, '/gripper_command', 10)
        
        # Subscriber based on mode - both use width now
        if self.mode == 'sim':
            # Sim mode: use gripper width sim
            self.width_sub = self.create_subscription(
                Float64,
                '/gripper_width_sim',
                self.width_callback,
                10
            )
            self.get_logger().info("Using SIM mode: monitoring /gripper_width_sim")
        else:
            # Real mode: use gripper width (Float32)
            self.width_sub = self.create_subscription(
                Float32,
                '/gripper_width',
                self.width_callback,
                10
            )
            self.get_logger().info("Using REAL mode: monitoring /gripper_width")
        
        # Determine target state
        if command.lower() == "open":
            self.target_state = "open"
            self.ros_command = "open"
            self.numeric_value = 1100
        elif command.lower() == "close":
            self.target_state = "close"
            self.ros_command = "close"
            self.numeric_value = 0
        else:
            # Numeric value 0-110 (convert to 0-1100 for ROS)
            try:
                value = float(command)
                if not (0 <= value <= 110):
                    self.get_logger().error(f"Value {value} out of range. Use 0-110.")
                    sys.exit(1)
                self.target_state = "numeric"
                self.numeric_value = int(value * 10)  # Convert 0-110 to 0-1100
                self.ros_command = str(self.numeric_value)
            except ValueError:
                self.get_logger().error(f"Invalid command '{command}'. Use 'open', 'close', or 0-110.")
                sys.exit(1)
        
        self.get_logger().info(f"Target state: {self.target_state}, ROS command: {self.ros_command}, Mode: {self.mode}")
    
    def width_callback(self, msg):
        """Callback for gripper width readings"""
        self.current_width = msg.data
        self.width_received = True
    
    def get_current_width(self, timeout=2.0):
        """Get current gripper width reading"""
        self.width_received = False
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.width_received:
                return self.current_width
        return None
    
    def is_at_target_state(self, width):
        """Check if gripper width indicates target state has been reached"""
        if width is None:
            return False
        if self.target_state == "close":
            return width < GRASP_SUCCESS_THRESHOLD
        else:  # open or numeric
            return width > WIDTH_OPEN_THRESHOLD
    
    def verify_gripper_state(self, initial_value=None):
        """Verify gripper has reached target state using width readings"""
        if initial_value is None:
            initial_value = self.get_current_width(timeout=0.5)
        
        # Quick check: if already at target state, return immediately
        if self.is_at_target_state(initial_value):
            state_str = "closed" if self.target_state == "close" else "open"
            self.get_logger().info(f"✓ Gripper already {state_str} (width: {initial_value:.2f})")
            return True
        
        # Start monitoring immediately
        self.get_logger().info("Monitoring gripper movement...")
        if initial_value is not None:
            self.get_logger().info(f"Starting from width: {initial_value:.2f}, target: {'close' if self.target_state == 'close' else 'open'}")
        
        # Monitoring parameters
        max_wait_time = 15.0
        early_retry_time = 5.0
        check_interval = 0.2
        no_change_threshold = 0.3
        required_stable_checks = 3
        required_stable_no_change = 5
        
        start_time = time.time()
        last_value = initial_value
        stable_count = 0
        no_change_count = 0
        last_change_time = None
        
        while (time.time() - start_time) < max_wait_time:
            elapsed = time.time() - start_time
            current_value = self.get_current_width(timeout=0.3)
            
            # Don't check target state here - let it go through the stability checks below
            # This ensures we wait for the gripper to stop moving
            
            # Early retry check: if no movement detected within 5 seconds
            if elapsed >= early_retry_time and initial_value is not None and last_change_time is None:
                if self.is_at_target_state(current_value):
                    state_str = "closed" if self.target_state == "close" else "open"
                    self.get_logger().info(f"✓ Gripper at target state ({state_str}, width: {current_value:.2f}), no retry needed")
                    return True
                # Check if value hasn't changed (gripper not responding)
                if current_value is not None and abs(current_value - initial_value) < 0.5:
                    self.get_logger().warn(f"No gripper movement detected within {early_retry_time}s (width unchanged: {current_value:.2f}). Retrying...")
                    return False
                self.get_logger().warn(f"No gripper movement detected within {early_retry_time}s. Retrying...")
                return False
            
            # If movement stopped for too long, retry
            if elapsed >= early_retry_time and last_change_time is not None:
                time_since_last_change = time.time() - last_change_time
                if time_since_last_change >= 3.0:
                    if self.is_at_target_state(current_value):
                        state_str = "closed" if self.target_state == "close" else "open"
                        self.get_logger().info(f"✓ Gripper reached target state ({state_str}, width: {current_value:.2f}), no retry needed")
                        return True
                    self.get_logger().warn(f"Gripper movement stopped for {time_since_last_change:.1f}s. Retrying...")
                    return False
            
            if current_value is not None:
                # Detect movement
                if last_value is not None and abs(current_value - last_value) > 0.5:
                    if last_change_time is None:
                        self.get_logger().info(f"Gripper movement detected: {current_value:.2f} (was {last_value:.2f})")
                    stable_count = 0
                    no_change_count = 0
                    last_change_time = time.time()
                elif last_value is None:
                    self.get_logger().info(f"Monitoring width... current: {current_value:.2f}")
                elif abs(current_value - last_value) <= no_change_threshold:
                    # Value is stable
                    no_change_count += 1
                    if no_change_count >= required_stable_no_change:
                        if self.is_at_target_state(current_value):
                            state_str = "closed" if self.target_state == "close" else "open"
                            self.get_logger().info(f"✓ Gripper stabilized and {state_str} (width: {current_value:.2f})")
                            return True
                else:
                    no_change_count = 0
                
                # Check if we've reached target threshold AND gripper has stabilized
                # Only return success if gripper is at target AND has stopped moving
                if self.is_at_target_state(current_value):
                    # Check if gripper has stabilized (not changing) - require last_value to be set
                    if last_value is not None and abs(current_value - last_value) <= no_change_threshold:
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            if self.target_state == "close":
                                if current_value < WIDTH_CLOSE_THRESHOLD:
                                    self.get_logger().info(f"✓ Gripper closed verified (width: {current_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
                                else:
                                    self.get_logger().info(f"✓ Gripper closed and likely grasped object (width: {current_value:.2f})")
                            else:
                                self.get_logger().info(f"✓ Gripper open verified (width: {current_value:.2f} > {WIDTH_OPEN_THRESHOLD})")
                            return True
                    else:
                        # At target but still moving or no previous value - reset stable count
                        stable_count = 0
                else:
                    stable_count = 0
                
                last_value = current_value
            
            time.sleep(check_interval)
        
        # Timeout reached - check final state
        self.get_logger().info("Timeout reached, checking final state...")
        time.sleep(1.0)
        
        # Get final reading
        final_value = self.get_current_width(timeout=0.5)
        
        if self.is_at_target_state(final_value):
            state_str = "closed" if self.target_state == "close" else "open"
            if self.target_state == "close" and final_value < WIDTH_CLOSE_THRESHOLD:
                self.get_logger().info(f"✓ Gripper closed (width: {final_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
            elif self.target_state == "close":
                self.get_logger().info(f"✓ Gripper closed and likely grasped object (width: {final_value:.2f})")
            else:
                self.get_logger().info(f"✓ Gripper open (width: {final_value:.2f} > {WIDTH_OPEN_THRESHOLD})")
            return True
        
        final_value_str = f"{final_value:.2f}" if final_value is not None else "N/A"
        self.get_logger().warn(f"✗ Gripper verification failed (final width: {final_value_str})")
        return False
    
    def send_gripper_command(self):
        """Send gripper command via ROS2 topic"""
        msg = String()
        msg.data = self.ros_command
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"Sent gripper command: {self.ros_command}")
        time.sleep(0.1)  # Small delay to ensure message is sent
    
    def verify_gripper_state_threaded(self, initial_value):
        """Verify gripper state in a separate thread - starts monitoring immediately"""
        result = self.verify_gripper_state(initial_value)
        with self.monitoring_lock:
            self.verification_result = result
            self.verification_complete = True
        return result
    
    def control_with_verification(self, initial_value=None):
        """Control gripper with verification and retry logic"""
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                self.get_logger().info(f"Retry attempt {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            
            # Reset verification state
            with self.monitoring_lock:
                self.verification_complete = False
                self.verification_result = None
            
            # Start monitoring thread before sending command (parallel execution)
            monitoring_thread = threading.Thread(
                target=self.verify_gripper_state_threaded,
                args=(initial_value,),
                daemon=True
            )
            monitoring_thread.start()
            time.sleep(0.01)  # Small delay to ensure thread starts
            
            # Send command (monitoring is already active)
            self.send_gripper_command()
            
            # Wait for verification to complete
            monitoring_thread.join(timeout=20.0)
            
            # Check result
            with self.monitoring_lock:
                if self.verification_complete and self.verification_result:
                    self.get_logger().info(f"Gripper control successful after {attempt} attempt(s)!")
                    return True
                elif not self.verification_complete:
                    self.get_logger().warn("Verification thread did not complete in time")
            
            if attempt < MAX_RETRIES:
                self.get_logger().warn(f"Verification failed, retrying... (attempt {attempt}/{MAX_RETRIES})")
        
        self.get_logger().error(f"Gripper control failed after {MAX_RETRIES} attempts")
        return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Control gripper with verification')
    parser.add_argument('command', type=str, help='Gripper command: "open", "close", or 0-110 (width in cm)')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Mode: "sim" for simulation (uses /gripper_width_sim), "real" for real robot (uses /gripper_width). Default: sim')
    
    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    controller = GripperController(known_args.command, known_args.mode)
    
    # Wait a moment for subscriptions to establish
    time.sleep(0.5)
    
    # Get initial reading (both modes use width now)
    initial_value = controller.get_current_width(timeout=2.0)
    if initial_value is not None:
        controller.get_logger().info(f"Initial gripper width: {initial_value:.2f}")
    
    # Control with verification
    success = controller.control_with_verification(initial_value)
    
    # Get final reading (both modes use width now)
    final_value = controller.get_current_width(timeout=2.0)
    if final_value is not None:
        controller.get_logger().info(f"Final gripper width: {final_value:.2f}")
        
        # For close command, check if grasp was successful
        if known_args.command.lower() == "close" and final_value is not None:
            if final_value < WIDTH_CLOSE_THRESHOLD:
                controller.get_logger().info(f"Grasp verification: Gripper fully closed (width: {final_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
            elif final_value < GRASP_SUCCESS_THRESHOLD:
                controller.get_logger().info(f"Grasp verification: Likely grasped object (width: {final_value:.2f} between {WIDTH_CLOSE_THRESHOLD} and {GRASP_SUCCESS_THRESHOLD})")
            else:
                controller.get_logger().warn(f"Grasp verification: Gripper may not have closed properly (width: {final_value:.2f} > {GRASP_SUCCESS_THRESHOLD})")
    
    controller.destroy_node()
    rclpy.shutdown()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

