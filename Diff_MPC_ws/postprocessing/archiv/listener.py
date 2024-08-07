import time
import yaml
import numpy as np
from argparse import Namespace
import math
from numba import njit
import cvxpy
import torch
import pickle
from scipy.spatial.transform import Rotation as R
import signal
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from tf_transformations import euler_from_quaternion

import pickle
import os

######################## GLOBAL VARIABLES ########################
LOG_ODOM_READY = False
LOG_DRIVE_READY = False
LOG_PF_ODOM_READY = False

map_yaml_filename = 'config_FTM_Halle.yaml'
##################################################################

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.
    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.
        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)
    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

@njit(fastmath=False, cache=True)
def zero_2_2pi(angle):
    if angle < 0:
        return angle + 2 * math.pi
    else:
        return angle
    
class Datalogger:
    """
    This is the class for logging vehicle data in the F1TENTH Gym
    """
    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def __init__(self, conf):
        self.conf = conf                    # Current configuration for the gym based on the maps
        self.load_waypoints(conf)           # Waypoints of the raceline
        self.vehicle_position_x = []        # Current vehicle position X (rear axle) on the map
        self.vehicle_position_y = []        # Current vehicle position Y (rear axle) on the map
        self.vehicle_position_heading = []  # Current vehicle heading on the map
        self.vehicle_velocity_x = []        # Current vehicle velocity - Longitudinal
        self.control_velocity = []          # Desired vehicle velocity based on control calculation
        self.steering_angle = []            # Steering angle based on control calculation
        self.control_raceline_x = []        # Current Control Path X-Position on Raceline
        self.control_raceline_y = []        # Current Control Path y-Position on Raceline
        self.control_raceline_heading = []  # Current Control Path Heading on Raceline
        self.yawrate = []                   # Current State of the yawrate in rad\s

    def logging(self, pose_x, pose_y, pose_theta, current_velocity_x, control_veloctiy, control_steering, yawrate):
        self.vehicle_position_x.append(pose_x)
        self.vehicle_position_y.append(pose_y)
        self.vehicle_position_heading.append(pose_theta)
        self.vehicle_velocity_x .append(current_velocity_x)
        self.control_velocity.append(control_veloctiy)
        self.steering_angle.append(control_steering)
        self.yawrate.append(yawrate)

    def logging2(self, raceline_x, raceline_y, raceline_theta):
        self.control_raceline_x.append(raceline_x)
        self.control_raceline_y.append(raceline_y)
        self.control_raceline_heading.append(raceline_theta)


class listenerNode(Node):
    def __init__(self):
        super().__init__('listener_node')

        self.speed = 0
        self.steer = 0

        # Construct the absolute path to the YAML file
        map_file = os.path.join(os.getcwd() + '/src/mpc/maps', map_yaml_filename)
        with open(map_file) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)
        self.waypoints = np.loadtxt(self.conf.wpt_path, delimiter=self.conf.wpt_delim, skiprows=self.conf.wpt_rowskip)


        self.logger = Datalogger(self.conf)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.odom_callback, 1)
        # self.ego_odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.ego_odom_callback, 1)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 1)

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
        obs_pose_x = msg.pose.pose.position.x
        obs_pose_y = msg.pose.pose.position.y
        # self.obs_theta = 2 * math.asin(msg.pose.pose.orientation.z) # ToDO CHECK IF THIS IS CORRECT
        obs_pose_theta = zero_2_2pi(euler[2])
        obs_linear_vel_x = msg.twist.twist.linear.x
        obs_yawrate = msg.twist.twist.angular.z

        self.logger.logging(obs_pose_x, obs_pose_y, obs_pose_theta, obs_linear_vel_x, self.speed, self.steer,obs_yawrate)
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        vehicle_state = np.array([obs_pose_x, obs_pose_y])
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(vehicle_state, wpts)
        self.logger.logging2(nearest_point[0], nearest_point[1], self.waypoints[:, 3][i])
    
    def drive_callback(self, msg):

        self.speed = msg.drive.speed
        self.steer = msg.drive.steering_angle

    def ego_odom_callback(self, msg):
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
        obs_ego_pose_x = msg.pose.pose.position.x
        obs_ego_pose_y = msg.pose.pose.position.y
        # self.obs_theta = 2 * math.asin(msg.pose.pose.orientation.z) # ToDO CHECK IF THIS IS CORRECT
        obs_ego_pose_theta = zero_2_2pi(euler[2])
        obs_ego_linear_vel_x = msg.twist.twist.linear.x
        obs_ego_yawrate = msg.twist.twist.angular.z

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        with open('postprocessing/mpc_locuslab.p', 'wb') as file:
            pickle.dump(self.logger, file)
        self.get_logger().info("Done")

def main(args=None):
    rclpy.init(args=args)
    node = listenerNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Ctrl-C caught, shutting down.")
        node.cleanup()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    node.get_logger().info("Starting the listener node...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
