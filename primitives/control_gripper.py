#!/usr/bin/env python3
"""
Gripper Control with Verification

Controls gripper and verifies movement using gripper force (sim) or width (real) readings.
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

# Force thresholds for verification (sim mode)
FORCE_OPEN_THRESHOLD = 10.0  # When open, force should be < 10
FORCE_CLOSE_THRESHOLD = 20.0  # When closed, force should be > 20

# Width thresholds for verification (real mode)
FULLY_OPEN_WIDTH = 110.0
FULLY_CLOSED_WIDTH = 9.0
WIDTH_OPEN_THRESHOLD = 100.0  # When open, width should be > 100
WIDTH_CLOSE_THRESHOLD = 20.0  # When closed, width should be < 20

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
        self.current_force = None
        self.current_width = None
        self.force_received = False
        self.width_received = False
        
        # Publisher for gripper commands
        self.gripper_pub = self.create_publisher(String, '/gripper_command', 10)
        
        # Subscriber based on mode
        if self.mode == 'sim':
            # Sim mode: use gripper force
            self.force_sub = self.create_subscription(
                Float64,
                '/gripper_force',
                self.force_callback,
                10
            )
            self.get_logger().info("Using SIM mode: monitoring /gripper_force")
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
    
    def force_callback(self, msg):
        """Callback for gripper force readings (sim mode)"""
        self.current_force = msg.data
        self.force_received = True
    
    def width_callback(self, msg):
        """Callback for gripper width readings (real mode)"""
        self.current_width = msg.data
        self.width_received = True
    
    def get_current_force(self, timeout=2.0):
        """Get current gripper force reading (sim mode)"""
        self.force_received = False
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.force_received:
                return self.current_force
        
        return None
    
    def get_current_width(self, timeout=2.0):
        """Get current gripper width reading (real mode)"""
        self.width_received = False
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.width_received:
                return self.current_width
        
        return None
    
    def verify_gripper_state(self, initial_value=None):
        """Verify gripper has reached target state using force (sim) or width (real) readings"""
        # Wait longer for gripper to physically move (2-3 seconds)
        self.get_logger().info("Waiting for gripper to move...")
        time.sleep(2.0)
        
        # Monitor over time and wait for it to reach target threshold
        max_wait_time = 5.0  # Maximum time to wait for target state
        check_interval = 0.2  # Check every 200ms
        start_time = time.time()
        last_value = None
        stable_count = 0
        required_stable_checks = 3  # Need 3 consecutive readings meeting threshold
        
        while (time.time() - start_time) < max_wait_time:
            if self.mode == 'sim':
                current_value = self.get_current_force(timeout=0.3)
                value_type = "force"
            else:
                current_value = self.get_current_width(timeout=0.3)
                value_type = "width"
            
            if current_value is not None:
                if last_value is None:
                    self.get_logger().info(f"Monitoring {value_type}... current: {current_value:.2f}")
                elif abs(current_value - last_value) > 0.5:  # Value is changing
                    self.get_logger().info(f"{value_type.capitalize()} changing: {current_value:.2f} (was {last_value:.2f})")
                    stable_count = 0  # Reset stable count if value is changing
                
                # Check if we've reached target threshold
                if self.mode == 'sim':
                    # Sim mode: use force thresholds
                    if self.target_state == "close":
                        if current_value > FORCE_CLOSE_THRESHOLD:
                            stable_count += 1
                            if stable_count >= required_stable_checks:
                                self.get_logger().info(f"✓ Gripper closed verified (force: {current_value:.2f} > {FORCE_CLOSE_THRESHOLD})")
                                return True
                        else:
                            stable_count = 0
                    else:  # open or numeric
                        if current_value < FORCE_OPEN_THRESHOLD:
                            stable_count += 1
                            if stable_count >= required_stable_checks:
                                self.get_logger().info(f"✓ Gripper open verified (force: {current_value:.2f} < {FORCE_OPEN_THRESHOLD})")
                                return True
                        else:
                            stable_count = 0
                else:
                    # Real mode: use width thresholds
                    if self.target_state == "close":
                        if current_value < WIDTH_CLOSE_THRESHOLD:
                            stable_count += 1
                            if stable_count >= required_stable_checks:
                                self.get_logger().info(f"✓ Gripper closed verified (width: {current_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
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
        
        # If we get here, we timed out - get final reading
        if self.mode == 'sim':
            final_value = self.get_current_force(timeout=0.5)
            value_type = "force"
            if final_value is not None:
                self.get_logger().warn(f"Verification timeout. Final force: {final_value:.2f}")
                if self.target_state == "close":
                    if final_value > FORCE_CLOSE_THRESHOLD:
                        self.get_logger().info(f"✓ Gripper closed (force: {final_value:.2f} > {FORCE_CLOSE_THRESHOLD})")
                        return True
                else:
                    if final_value < FORCE_OPEN_THRESHOLD:
                        self.get_logger().info(f"✓ Gripper open (force: {final_value:.2f} < {FORCE_OPEN_THRESHOLD})")
                        return True
        else:
            final_value = self.get_current_width(timeout=0.5)
            value_type = "width"
            if final_value is not None:
                self.get_logger().warn(f"Verification timeout. Final width: {final_value:.2f}")
                if self.target_state == "close":
                    if final_value < WIDTH_CLOSE_THRESHOLD:
                        self.get_logger().info(f"✓ Gripper closed (width: {final_value:.2f} < {WIDTH_CLOSE_THRESHOLD})")
                        return True
                else:
                    if final_value > WIDTH_OPEN_THRESHOLD:
                        self.get_logger().info(f"✓ Gripper open (width: {final_value:.2f} > {WIDTH_OPEN_THRESHOLD})")
                        return True
        
        final_value_str = f"{final_value:.2f}" if final_value is not None else "N/A"
        self.get_logger().warn(f"✗ Gripper verification failed (final {value_type}: {final_value_str})")
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
                       help='Mode: "sim" for simulation (uses /gripper_force), "real" for real robot (uses /gripper_width). Default: sim')
    
    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    controller = GripperController(known_args.command, known_args.mode)
    
    # Wait a moment for subscriptions to establish
    time.sleep(0.5)
    
    # Get initial reading based on mode
    if known_args.mode == 'sim':
        initial_value = controller.get_current_force(timeout=2.0)
        if initial_value is not None:
            controller.get_logger().info(f"Initial gripper force: {initial_value:.2f}")
    else:
        initial_value = controller.get_current_width(timeout=2.0)
        if initial_value is not None:
            controller.get_logger().info(f"Initial gripper width: {initial_value:.2f}")
    
    # Control with verification
    success = controller.control_with_verification(initial_value)
    
    # Get final reading based on mode
    if known_args.mode == 'sim':
        final_value = controller.get_current_force(timeout=2.0)
        if final_value is not None:
            controller.get_logger().info(f"Final gripper force: {final_value:.2f}")
    else:
        final_value = controller.get_current_width(timeout=2.0)
        if final_value is not None:
            controller.get_logger().info(f"Final gripper width: {final_value:.2f}")
    
    controller.destroy_node()
    rclpy.shutdown()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

