import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

import os
import time
import datetime
import numpy as np


# This function edits the "Text Color" or "Background Color" of terminal text using ANSI codes
def colorize(color_code, message):
    # TEXT::: color_code = 31<Red>, 32<Green>, 93<Yellow>, 34<Blue>, 36<Cyan>, 95<BrtMagenta>
    # BACKGRD:: color_code = 41<Red>, 42<Green>, 103<Yellow>, 44<Blue>, 46<Cyan>, 105<BrtMagenta>
    # \033[  <Initiate ANSI escape code>, and  \033[0m  <Reset formatting to default>
    return f"\033[{color_code}m{message}\033[0m"


class ROSBagRecorder(Node):
    def __init__(self):
        super().__init__('rosbag_recorder')
        
        self.episode_active = False
        self.recording = False

        self.jt_vels = [0.0] * 6
        self.jt_vels_np = None
        self.jts_moving = True

        self.bag_writer = None
        self.bag_directory = os.path.expanduser('/home/abhara13/Desktop/MCP-UNIFIED/RoboManufac-MCP/rosbag_record')
        os.makedirs(self.bag_directory, exist_ok=True)

        # Topics to Read
        self.img_topic = '/intel_camera_rgb_sim'


        self.subscribers = {
            # DATA SUBSCRIBERS
            "/joint_state": self.create_subscription(Float64MultiArray,
                                                    ),
            '/obs/image': self.create_subscription(Image, self.img_topic,
                                                   self.img_state_callback, 15),
            
            # RECORDING STATE SUBSCRIBERS
            '/episode_start_topic': self.create_subscription(Bool, "/episode_start_topic",
                                                            self.episode_callback, 15),
            '/start_recording': self.create_subscription(Bool, "/start_recording",
                                                            self.record_callback, 15),
        }
        self.pub_start_record = self.create_publisher(Bool, 'start_recording', 20)

    def start_new_episode(self):
        # ROSBAG INIT
        timestamp = datetime.datetime.now().strftime("%m_%d-%H-%M-%S")
        bag_path = os.path.join(self.bag_directory, f"episode_{timestamp}")
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        self.bag_writer = SequentialWriter()
        self.bag_writer.open(storage_options, converter_options)
       
        for i, (topic, msg_type, _) in enumerate([('/obs/image', 'sensor_msgs/msg/Image', Image),
                                                  ('/obs/state', 'std_msgs/msg/Float64MultiArray', 
                                                               Float64MultiArray),
                                                 ]
                                       ):
            topic_metadata = TopicMetadata(name=topic, type=msg_type, 
                                           serialization_format="cdr")
            self.bag_writer.create_topic(topic_metadata)
            
        self.get_logger().info(colorize(42, "Started new episode recording."))
        self.episode_active = True
        self.recording = False
    
    def stop_episode(self):
        if self.bag_writer:
            self.bag_writer = None
            self.get_logger().info(colorize(41, "Stopped episode recording and saved rosbag."))
        self.episode_active = False
        self.recording = False

    def episode_callback(self, msg):
        if msg.data == True:
            self.start_new_episode()
        elif msg.data == False:
            self.stop_episode()
    
    def record_callback(self, msg):
        if self.episode_active and self.recording != msg.data:
            self.recording = msg.data  # Only toggle based on user input
            status = "Started" if msg.data else "Paused"
            self.get_logger().info(colorize(103, 
                                    f"{status} recording based on /start_recording topic."))

    def record_message(self, topic, msg):
        if not hasattr(self, "prev_jts_moving"):
            self.prev_jts_moving = self.jts_moving

        # -----> MOVE3D RECORDING
        # If Recording and Joints are moving, then save data
        if self.recording==True:
            if self.bag_writer and self.primitive_selected in ["Move 3D", "2D Push"]:
                if self.jts_moving==True: #and self.prev_jts_moving==False:
                    '''rec_stop_msg= Bool(); rec_stop_msg.data = True
                    self.pub_start_record.publish(rec_stop_msg)'''
                    print("@@@@@@@@> JOINTS MOVING:", self.jt_vels)
                    self.bag_writer.write(topic, serialize_message(msg), 
                                        self.get_clock().now().nanoseconds)
                    #print(self.jts_moving, self.recording)
                elif self.jts_moving==False and self.prev_jts_moving==True:
                    #time.sleep(3)
                    rec_stop_msg= Bool(); rec_stop_msg.data = False
                    self.pub_start_record.publish(rec_stop_msg)
                    self.get_logger().info(colorize(103,
                                           "Joints stopped moving. Paused recording."))

            # -----> GRIPPER RECORDING
            elif self.bag_writer and self.primitive_selected in ["Open Gripper", 
                                                                "Half Open Gripper", 
                                                                "Close Gripper"]:
                print("########> Wrote Gripper action. ", self.action)
                self.bag_writer.write(topic, serialize_message(msg), 
                                    self.get_clock().now().nanoseconds)
                rec_stop_msg= Bool(); rec_stop_msg.data = False
                self.pub_start_record.publish(rec_stop_msg)
        
        self.prev_jts_moving = self.jts_moving
    
    def img_state_callback(self, msg):
        self.record_message(self.img_topic, msg)


def main(args=None):
    rclpy.init(args=args)
    recorder = ROSBagRecorder()
    rclpy.spin(recorder)
    recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()