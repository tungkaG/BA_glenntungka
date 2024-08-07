import rclpy
from rclpy.node import Node
import math
import numpy as np
import pandas as pd
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

# Helper function to convert quaternion to euler angles
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    based on https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = math.sqrt(1 + 2 * (w * y - x * z))
    cosp = math.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp);   

    return roll, pitch, yaw # in radians

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        
        # Set wether it's a simulation or car rosbag
        simulation = True
        
        # Create Subscriptions
        if simulation:
            self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        elif not simulation:
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        else:
            raise ValueError('Invalid simulation flag value')
        self.pf_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.pf_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 10)
        
        self.data = {
            'timestamp': [],
            'x': [],
            'y': [],
            'orientation': [],
            'speed': [],
            'angular_speed': [],
            'speed_cmd': [],
            'steering_angle': []
        }
        
        # just count variables for logger information
        self.count_pf = 0
        self.count_odom = 0
        self.count_drive = 0
        
        if simulation:
            self.get_logger().info('Node started to record data from simulation rosbag.')
        else:  
            self.get_logger().info('Node started to record data from car rosbag.')

    def pf_callback(self, msg):
        self.data['timestamp'].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.data['x'].append(msg.pose.pose.position.x)
        self.data['y'].append(msg.pose.pose.position.y)
        self.data['orientation'].append(euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)[2])
        self.data['speed'].append(np.nan)
        self.data['angular_speed'].append(np.nan)
        self.data['speed_cmd'].append(np.nan)
        self.data['steering_angle'].append(np.nan)
        self.count_pf += 1
        if self.count_pf % 500 == 0:
            self.get_logger().info(f'pf messages received: {self.count_pf}')
        

    def odom_callback(self, msg):
        self.data['timestamp'].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.data['x'].append(np.nan)
        self.data['y'].append(np.nan)
        self.data['orientation'].append(np.nan)
        self.data['speed'].append(msg.twist.twist.linear.x)
        self.data['angular_speed'].append(msg.twist.twist.angular.z)
        self.data['speed_cmd'].append(np.nan)
        self.data['steering_angle'].append(np.nan)
        self.count_odom += 1
        if self.count_odom % 500 == 0:
            self.get_logger().info(f'odom messages received: {self.count_odom}')


    def drive_callback(self, msg):
        self.data['timestamp'].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.data['x'].append(np.nan)
        self.data['y'].append(np.nan)
        self.data['orientation'].append(np.nan)
        self.data['speed'].append(np.nan)
        self.data['angular_speed'].append(np.nan)
        self.data['speed_cmd'].append(msg.drive.speed)
        self.data['steering_angle'].append(msg.drive.steering_angle)
        self.count_drive += 1
        if self.count_drive % 500 == 0:
            self.get_logger().info(f'drive messages received: {self.count_drive}')
    
    def on_shutdown(self):
        # Save data to csv
        self.get_logger().info(f'pf messages received: {self.count_pf}, odom messages received: {self.count_odom}, drive messages received: {self.count_drive}')
        filename = input("Enter CSV file name to save (press Enter for default 'data.csv'): ").strip() or 'data'
        self.get_logger().info(f'Saving data to {filename}.csv')
        df = pd.DataFrame(self.data)
        df.sort_values(by='timestamp', inplace=True)
        df.to_csv('/home/glenn/BA/Diff_MPC_ws/data/' + filename + '.csv', index=False)

def main(args=None):
    rclpy.init(args=args)
    data_subscriber = DataSubscriber()
    
    # Register on_shutdown callback
    rclpy.get_default_context().on_shutdown(data_subscriber.on_shutdown)

    try:
        rclpy.spin(data_subscriber)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()