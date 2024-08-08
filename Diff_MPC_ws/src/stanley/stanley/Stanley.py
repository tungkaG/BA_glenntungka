import time
import yaml
import numpy as np
from argparse import Namespace
import math
from numba import njit
import signal

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion
import sys, os

###################################### YAML ######################################
config_file = os.path.join(os.getcwd() + '/src/stanley/config', 'Stanley.yaml')
with open(config_file) as file:
    CONFIG_PARAM = yaml.load(file, Loader=yaml.FullLoader)
###################################### YAML ######################################

# Simulation paramter
DT = CONFIG_PARAM['dt']                               # time step [s]

# Vehicle parameters
# LENGTH = 0.58                       # Length of the vehicle [m]
# WIDTH = 0.31                        # Width of the vehicle [m]
WB = CONFIG_PARAM['wheelbase']                           # Wheelbase [m]

K_Stanley = CONFIG_PARAM['k_stanley']                     # Stanley gain
K_pid = CONFIG_PARAM['k_pid']                         # PID gain

@njit(fastmath=False, cache=True)
def zero_2_2pi(angle):
    if angle < 0:
        return angle + 2 * math.pi
    else:
        return angle
    
class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class Path:
    """
    class for the path
    """
    def __init__(self, cx, cy, cyaw, ck, cv):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.cv = cv

class Controller:

    def __init__(self):
        self.odelta = None
        self.oa = None
        self.ind_old = 0

    def front_wheel_feedback_control(self, state, ref_path):
        """
        front wheel feedback controller
        :param state: current information
        :param ref_path: reference path: x, y, yaw, curvature
        :return: optimal steering angle
        """

        theta_e, ef, target_index = self.calc_theta_e_and_ef(state, ref_path)
        delta = theta_e + math.atan2(K_Stanley * ef, state.v)

        return delta, target_index

    def pi_2_pi(self, angle):
        if angle > math.pi:
            return angle - 2.0 * math.pi
        if angle < -math.pi:
            return angle + 2.0 * math.pi

        return angle

    def pid_control(self, target_v, v):
        """
        PID controller and design speed profile.
        :param target_v: target speed
        :param v: current speed
        """
        a = K_pid * (target_v - v)
        v = a * DT + v

        return v

    def stanley_control(self, state, ref_path):
        """
        Stanley steering control
        :param state: current information
        :param ref_path: reference path: x, y, yaw, curvature
        :return: optimal steering angle
        """
        path = Path(ref_path[0], ref_path[1], ref_path[2], ref_path[3], ref_path[4])

        steering_angle, target_index = self.front_wheel_feedback_control(state, path)
        speed = self.pid_control(path.cv[target_index], state.v)

        return speed, steering_angle, path.cx[target_index], path.cy[target_index]

    def calc_theta_e_and_ef(self, state, ref_path):
        """
        calc theta_e and ef.
        theta_e = theta_car - theta_path
        ef = lateral distance in frenet frame (front wheel)

        :param state: current information of vehicle
        :return: theta_e and ef
        """

        fx = state.x + WB * math.cos(state.yaw)
        fy = state.y + WB * math.sin(state.yaw)

        dx = [fx - x for x in ref_path.cx]
        dy = [fy - y for y in ref_path.cy]

        target_index = int(np.argmin(np.hypot(dx, dy)))

        if(abs(target_index - self.ind_old) > 5): # 5 is the threshold of the index difference that tells us we are in a new lap
            self.ind_old = target_index
        else:
            target_index = max(self.ind_old, target_index)
            self.ind_old = max(self.ind_old, target_index)

        front_axle_vec_rot_90 = np.array([[math.cos(state.yaw - math.pi / 2.0)],
                                          [math.sin(state.yaw - math.pi / 2.0)]])

        vec_target_2_front = np.array([[dx[target_index]],
                                       [dy[target_index]]])

        ef = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)

        theta = state.yaw
        theta_p = ref_path.cyaw[target_index]
        theta_e = self.pi_2_pi(theta_p - theta)

        return theta_e, ef, target_index
    
class LatticePlanner:

    def __init__(self, conf):
        self.conf = conf                        # Current configuration for the gym based on the maps
        self.waypoints = np.loadtxt(conf.wpt_path, 
                                    delimiter=conf.wpt_delim, 
                                    skiprows=conf.wpt_rowskip) # Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        
    def plan(self):
        """
        Loading the individual data from the global, optimal raceline and creating one list
        """
        cx = self.waypoints[:, 1]       # X-Position of Raceline
        cy = self.waypoints[:, 2]       # Y-Position of Raceline
        cyaw = self.waypoints[:, 3]     # Heading on Raceline
        ck = self.waypoints[:, 4]       # Curvature of Raceline
        cv = self.waypoints[:, 5]       # velocity on Raceline

        global_raceline = [cx, cy, cyaw, ck, cv]

        return global_raceline

class Stanley_Node(Node):
    
        def __init__(self):
            super().__init__('Stanley_Node')

            map_yaml_filename = CONFIG_PARAM['map_yaml_filename']
            # Construct the absolute path to the YAML file
            map_file = os.path.join(os.getcwd() + '/maps', map_yaml_filename)

            with open(map_file) as file:
                self.conf_dict = yaml.load(file, Loader=yaml.FullLoader)
            self.conf = Namespace(**self.conf_dict)
    
            # Creating the Motion planner and Controller object that is used in Gym
            self.controller = Controller()
            self.planner = LatticePlanner(self.conf)

            # Load global raceline to create a path variable that includes all reference path information
            self.path = self.planner.plan()

            # Initialize attributes
            self.control_count = CONFIG_PARAM['control_frequency']

            self.obs_pose_x = 0.0
            self.obs_pose_y = 0.0
            self.obs_pose_theta = 0.0
            self.obs_linear_vel_x = 0.0
            self.obs_linear_vel_y = 0.0
            self.obs_yawrate = 0.0

            self.speed = 0.0
            self.steer = 0.0

            # Subscribers
            self.odom_sub = self.create_subscription(Odometry, CONFIG_PARAM['odom_topic'], self.odom_callback, 1)
            
            # Publishers
            self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
            self.target_pub = self.create_publisher(Marker, '/target_point', 1)
        
        def odom_callback(self, msg):
            # Convert the Quaternion message to a list
            quaternion = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]

            # Convert the quaternion to Euler angles
            euler = euler_from_quaternion(quaternion)

            # # self.get_logger().info('Odometry: x=%f, y=%f, theta=%f, linear_vel_x=%f, linear_vel_y=%f, yawrate=%f' % (msg.pose.pose.position.x, msg.pose.pose.position.y, 2 * math.asin(msg.pose.pose.orientation.z), msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z))
            self.obs_pose_x = msg.pose.pose.position.x
            self.obs_pose_y = msg.pose.pose.position.y
            # self.obs_theta = 2 * math.asin(msg.pose.pose.orientation.z) # ToDO CHECK IF THIS IS CORRECT
            self.obs_pose_theta = zero_2_2pi(euler[2])
            self.obs_linear_vel_x = msg.twist.twist.linear.x
            self.obs_yawrate = msg.twist.twist.angular.z
            
            if(self.control_count >= CONFIG_PARAM['control_frequency']):
                self.control_count = 0
                self.publish_control()
                self.publish_target_point()

            self.control_count = self.control_count + 1 

        def publish_control(self):
            self.speed, self.steer, self.target_x, self.target_y = self.controller.stanley_control(State( x = self.obs_pose_x, 
                                                                            y = self.obs_pose_y,
                                                                            yaw = self.obs_pose_theta,
                                                                            v = self.obs_linear_vel_x),
                                                                    self.path)
                
            # Prepare the drive message with the steering angle and corresponding speed
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'laser'
            drive_msg.drive.steering_angle = self.steer
            drive_msg.drive.speed = self.speed

            # Publish the drive command
            self.drive_pub.publish(drive_msg)

            self.get_logger().info('Control: speed=%f, steer=%f' % (self.speed, self.steer))
        
        def publish_target_point(self):
            # Prepare the target point marker message
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "target_point"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x = self.target_x, y = self.target_y, z = 0.0)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            # Publish the target point marker
            self.target_pub.publish(marker)
        def cleanup(self):
            # Cleanup the resources
            print("Shutting down the Stanley Node...")


def main(args=None):
    # Main function to initialize the ROS node and spin
    rclpy.init(args=args)

    node = Stanley_Node()

    # Signal handler to catch the Ctrl-C signal and cleanup the resources
    def signal_handler(sig, frame):
        node.get_logger().info("Ctrl-C caught, shutting down.")
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
    signal.signal(signal.SIGINT, signal_handler)

    # Start the ROS node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # Entry point for the script
    main()