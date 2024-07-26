import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from geometry_msgs.msg import Pose

import pickle
import os

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

        self.logger = Datalogger()

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
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
            self.obs_pose_x = msg.pose.pose.position.x
            self.obs_pose_y = msg.pose.pose.position.y
            # self.obs_theta = 2 * math.asin(msg.pose.pose.orientation.z) # ToDO CHECK IF THIS IS CORRECT
            self.obs_pose_theta = zero_2_2pi(euler[2])
            self.obs_linear_vel_x = msg.twist.twist.linear.x
            self.obs_yawrate = msg.twist.twist.angular.z
            
            if self.conf_dict['logging'] == 'True':
                self.logger.logging(self.obs_pose_x, self.obs_pose_y, self.obs_pose_theta, self.obs_linear_vel_x,
                                self.speed, self.steer, self.obs_yawrate)
                
                wpts = np.vstack((self.planner.waypoints[:, self.conf.wpt_xind], self.planner.waypoints[:, self.conf.wpt_yind])).T
                vehicle_state = np.array([self.obs_pose_x, self.obs_pose_y])
                nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(vehicle_state, wpts)
                self.logger.logging2(nearest_point[0], nearest_point[1], self.path[2][i])

            if(self.control_count == 10):
                self.control_count = 0
                self.publish_control()

            self.control_count = self.control_count + 1 

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        with open('postprocessing/mpc_locuslab.p', 'wb') as file:
            pickle.dump(self.logger, file)
        self.get_logger().info("Done")

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Ctrl-C caught, shutting down.")
        node.cleanup()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
