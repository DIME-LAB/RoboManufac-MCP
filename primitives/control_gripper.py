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
from std_msgs.msg import String, Float64
import argparse
import time
import sys

# Width thresholds for verification (both sim and real mode)
FULLY_OPEN_WIDTH = 110.0
FULLY_CLOSED_WIDTH = 9.0
WIDTH_OPEN_THRESHOLD = 100.0  # When open, width should be > 100
WIDTH_CLOSE_THRESHOLD = 20.0  # When closed, width should be < 20
# For grasp verification: if closing and width doesn't reach fully closed, likely grasped an object
GRASP_SUCCESS_THRESHOLD = 30.0  # If width is between 20-30 when trying to close, likely grasped something

VERIFICATION_TIMEOUT = 5.0  # Seconds to wait for verification
MAX_RETRIES = 3
RETRY_DELAY = 0.5  # Seconds to wait between retries

class GripperController(Node):
    def __init__(self, command, mode='sim'):
        super().__init__('gripper_controller')
        
        self.command = command
        self.mode = mode  # 'sim' or 'real'
        self.verification_complete = False
        self.verification_success = False
        self.current_width = None
        self.width_received = False
        
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
            # Real mode: use gripper width
            self.width_sub = self.create_subscription(
                Float64,
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
        """Callback for gripper width readings (both sim and real mode)"""
        self.current_width = msg.data
        self.width_received = True
    
    def get_current_width(self, timeout=2.0):
        """Get current gripper width reading (both sim and real mode)"""
        self.width_received = False
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.width_received:
                return self.current_width
        
        return None
    
    def verify_gripper_state(self, initial_value=None):
        """Verify gripper has reached target state using width readings (both sim and real mode)"""
        # Wait longer for gripper to physically move (2-3 seconds)
        self.get_logger().info("Waiting for gripper to move...")
        time.sleep(2.0)
        
        # Monitor over time and wait for it to reach target threshold
        max_wait_time = 15.0  # Increased timeout to allow slow movement and stabilization
        early_retry_time = 5.0  # If no movement detected within 5 seconds, retry early
        check_interval = 0.2  # Check every 200ms
        start_time = time.time()
        last_value = None
        stable_count = 0
        required_stable_checks = 3  # Need 3 consecutive readings meeting threshold
        first_value = None
        moving_in_right_direction = False
        no_change_count = 0  # Count consecutive readings with no significant change
        no_change_threshold = 0.3  # Consider stable if change is less than 0.3mm
        required_stable_no_change = 5  # Need 5 consecutive readings with no change to consider stable
        last_change_time = None  # Track when value last changed
        
        while (time.time() - start_time) < max_wait_time:
            elapsed = time.time() - start_time
            current_value = self.get_current_width(timeout=0.3)
            
            # Early retry check: if no movement detected within 5 seconds, retry
            # Only check this if we've been monitoring for at least 5 seconds and have a first value
            if elapsed >= early_retry_time and first_value is not None and last_change_time is None:
                self.get_logger().warn(f"No gripper movement detected within {early_retry_time}s. Retrying command...")
                return False
            
            # If movement was detected but stopped for too long, also retry
            if elapsed >= early_retry_time and last_change_time is not None:
                time_since_last_change = time.time() - last_change_time
                if time_since_last_change >= 3.0:  # No change for 3 seconds after initial movement
                    self.get_logger().warn(f"Gripper movement stopped for {time_since_last_change:.1f}s. Retrying command...")
                    return False
            
            if current_value is not None:
                if first_value is None:
                    first_value = current_value
                    self.get_logger().info(f"Monitoring width... current: {current_value:.2f}")
                elif abs(current_value - last_value) > 0.5:  # Value is changing significantly
                    self.get_logger().info(f"Width changing: {current_value:.2f} (was {last_value:.2f})")
                    stable_count = 0  # Reset stable count if value is changing
                    no_change_count = 0  # Reset no-change count
                    last_change_time = time.time()  # Update last change time
                    
                    # Check if moving in the right direction
                    if self.target_state == "close":
                        moving_in_right_direction = (current_value < last_value)  # Should be decreasing
                    else:  # open or numeric
                        moving_in_right_direction = (current_value > last_value)  # Should be increasing
                elif abs(current_value - last_value) <= no_change_threshold:
                    # Value is stable (not changing much)
                    no_change_count += 1
                    
                    # If gripper has stabilized and we're in a reasonable state, accept it
                    if no_change_count >= required_stable_no_change:
                        if self.target_state == "close":
                            if current_value < GRASP_SUCCESS_THRESHOLD:
                                self.get_logger().info(f"✓ Gripper stabilized and closed (width: {current_value:.2f})")
                                return True
                        else:  # open
                            if current_value > 80.0:
                                self.get_logger().info(f"✓ Gripper stabilized and open (width: {current_value:.2f})")
                                return True
                else:
                    # Small change but not significant
                    no_change_count = 0
                
                # Check if we've reached target threshold
                if self.target_state == "close":
                    # For closing: check if width is below threshold
                    # If width is between WIDTH_CLOSE_THRESHOLD and GRASP_SUCCESS_THRESHOLD, likely grasped an object
                    if current_value < WIDTH_CLOSE_THRESHOLD:
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            self.get_logger().info(f"✓ Gripper closed verified (width: {current_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
                            return True
                    elif current_value < GRASP_SUCCESS_THRESHOLD:
                        # Width is between thresholds - likely grasped an object
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            self.get_logger().info(f"✓ Gripper closed and likely grasped object (width: {current_value:.2f} between {WIDTH_CLOSE_THRESHOLD} and {GRASP_SUCCESS_THRESHOLD})")
                            return True
                    else:
                        stable_count = 0
                else:  # open or numeric
                    if current_value > WIDTH_OPEN_THRESHOLD:
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            self.get_logger().info(f"✓ Gripper open verified (width: {current_value:.2f} > {WIDTH_OPEN_THRESHOLD})")
                            return True
                    else:
                        stable_count = 0
                
                last_value = current_value
            
            time.sleep(check_interval)
        
        # If we get here, we timed out - wait a bit more and check if gripper has stabilized
        self.get_logger().info("Initial timeout reached, checking if gripper has stabilized...")
        time.sleep(1.0)  # Wait 1 second to see if gripper continues moving
        
        # Check stability over a few more readings
        stability_check_count = 0
        stability_values = []
        for _ in range(5):  # Check 5 more readings
            current_value = self.get_current_width(timeout=0.3)
            if current_value is not None:
                stability_values.append(current_value)
            time.sleep(0.2)
        
        if len(stability_values) >= 3:
            # Check if values are stable (not changing much)
            value_range = max(stability_values) - min(stability_values)
            final_value = stability_values[-1]
            
            if value_range <= 1.0:  # Values are stable (within 1mm)
                self.get_logger().info(f"Gripper stabilized at width: {final_value:.2f} (range: {value_range:.2f}mm)")
            else:
                self.get_logger().warn(f"Gripper still moving (range: {value_range:.2f}mm), waiting longer for stabilization...")
                # Wait longer and check again - keep checking until stable or timeout
                max_stabilization_wait = 5.0
                stabilization_start = time.time()
                last_check_value = final_value
                stable_checks = 0
                
                while (time.time() - stabilization_start) < max_stabilization_wait:
                    time.sleep(0.5)
                    check_value = self.get_current_width(timeout=0.3)
                    if check_value is not None:
                        if abs(check_value - last_check_value) <= 0.5:  # Stable
                            stable_checks += 1
                            if stable_checks >= 3:  # Stable for 3 checks
                                final_value = check_value
                                self.get_logger().info(f"Gripper stabilized after additional wait: {final_value:.2f}")
                                break
                        else:
                            stable_checks = 0
                        last_check_value = check_value
                
                if stable_checks < 3:
                    # Still not stable, get final reading
                    final_value = self.get_current_width(timeout=0.5)
        else:
            final_value = self.get_current_width(timeout=0.5)
        
        if final_value is not None:
            # Only accept if gripper has reached a good state (not just progress)
            # Progress-based acceptance removed - only accept if actually at target state
            
            # Fallback to original threshold checks
            self.get_logger().warn(f"Verification timeout. Final width: {final_value:.2f}")
            if self.target_state == "close":
                if final_value < WIDTH_CLOSE_THRESHOLD:
                    self.get_logger().info(f"✓ Gripper closed (width: {final_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
                    return True
                elif final_value < GRASP_SUCCESS_THRESHOLD:
                    self.get_logger().info(f"✓ Gripper closed and likely grasped object (width: {final_value:.2f} between {WIDTH_CLOSE_THRESHOLD} and {GRASP_SUCCESS_THRESHOLD})")
                    return True
            else:
                if final_value > WIDTH_OPEN_THRESHOLD:
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
    
    def control_with_verification(self, initial_value=None):
        """Control gripper with verification and retry logic - keeps retrying until success"""
        attempt = 0
        while True:
            attempt += 1
            if attempt > 1:
                self.get_logger().info(f"Retry attempt {attempt}")
                time.sleep(RETRY_DELAY)
            
            # Send command
            self.send_gripper_command()
            
            # Verify (with longer wait time built in)
            if self.verify_gripper_state(initial_value):
                self.get_logger().info(f"Gripper control successful after {attempt} attempt(s)!")
                return True
            
            # If verification failed, we'll retry (infinite loop until success)
            self.get_logger().warn(f"Verification failed, retrying... (attempt {attempt})")


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

