import rclpy
from rclpy.node import Node
import tf2_geometry_msgs
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from tf2_geometry_msgs import PointStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import csv
import numpy as np
from visualization_msgs.msg import Marker
import tf2_ros
import time
import sys, os
sys.path.append(os.getcwd() + '/src')

raceline_path = os.getcwd() + '/maps/FTM_Halle.csv'

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_pf_raceline_data')
        
        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print("Waiting for Buffer to be filled")
        time.sleep(3)
        #self.tf_buffer.wait_for_transform_async('laser', 'map', rclpy.time.Time())

        # Publisher for control commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Parameters
        # self.wheel_base = 0.35
        # self.max_steering_angle = 0.36 # in radian
        # self.speed_percentage = 0.6 # percentage of the calculated best speed
        # self.min_dist_point_reached = 0.4  # minimum value for dist_point_reached
        # self.max_dist_point_reached = 1.5  # minimum value for dist_point_reached
        self.wheel_base = 0.3302
        self.max_steering_angle = np.deg2rad(24) # in radian
        self.speed_percentage = 0.6 # percentage of the calculated best speed
        self.min_dist_point_reached = 0.4  # minimum value for dist_point_reached
        self.max_dist_point_reached = 1.5  # maximum value for dist_point_reached

        # Initialize variables
        self.current_pose = None
        self.waypoints = []
        self.next_waypoint_index = None
        self.closest_waypoint_index = None
        self.target_speed = None
        self.max_curvature = None
        self.dist_point_reached = self.min_dist_point_reached

        # Read raceline_data from CSV file
        self.raceline_data = np.genfromtxt(raceline_path, delimiter=";", comments='#')
        self.waypoints = self.raceline_data[:, 1:3]
        self.max_curvature = np.max(abs(self.raceline_data[:, 4]))
        
        # Initialize Marker to visualize current waypoint target
        self.target_marker_pub = self.create_publisher(Marker, '/target_point', 5)
        self.target_marker = Marker()
        self.target_marker.ns ='TargetVisualization'
        self.target_marker.id = 0
        self.target_marker.header.frame_id = 'map'
        self.target_marker.header.stamp = self.get_clock().now().to_msg()
        self.target_marker.type = Marker.SPHERE
        self.target_marker.action = Marker.ADD
        self.target_marker.scale.x = 0.55
        self.target_marker.scale.y = 0.55
        self.target_marker.scale.z = 0.55
        self.target_marker.color.a = 1.0
        self.target_marker.color.r = 1.0
        
        # Subscribe to current pose
        self.pose_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10)
        # self.pose_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.initalpose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.intialpose_callback, 10)
    
    def p2p_dist (self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
    
    def intialpose_callback(self, msg):
        # Triggers calculation of initial waypoint index inside  calculate control commands if the pose had to be reset with pose estimation in rviz
        self.get_logger().info("Start index resetted")
        self.next_waypoint_index = None 
        
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        
        # Finde closest point to current pose
        self.closest_waypoint_index = np.argmin(self.p2p_dist(self.current_pose.position.x, self.current_pose.position.y, self.raceline_data[:,1], self.raceline_data[:,2]))
        self.target_speed = self.raceline_data[self.closest_waypoint_index, 5]
        # Calculate control commands
        ackermann_cmd = self.calculate_control_commands()
        
        # Publish control commands
        self.drive_pub.publish(ackermann_cmd)

    def calculate_control_commands(self):
        ackermann_cmd = AckermannDriveStamped()

        # Find the next waypoint and mark it
        if self.current_pose and self.waypoints.any():
            # Check if this is the first iteration and if so initialize index
            if self.next_waypoint_index is None:
                self.set_initial_waypoint_index()
            next_waypoint = self.waypoints[self.next_waypoint_index]
            self.target_marker.pose.position.x = next_waypoint[0]
            self.target_marker.pose.position.y = next_waypoint[1]
            self.target_marker_pub.publish(self.target_marker)

            # Calculate distance and angle to the next waypoint
            L = np.sqrt((next_waypoint[0] - self.current_pose.position.x)**2 + (next_waypoint[1] - self.current_pose.position.y)**2)
            
            # create PointStamped from waypoint for transformation
            map_point = PointStamped()
            map_point.header.frame_id = "map"
            map_point.point.x = next_waypoint[0]
            map_point.point.y = next_waypoint[1]
            map_point.point.z = 0.0

            # Transform the point into the new frame
            while(True):
                try:
                    frames_string = self.tf_buffer.all_frames_as_string()
                    #self.get_logger().info("Available TF frames: \n%s" %frames_string)
                    #print("test")
                    #print(f'Coordinates before transform: x:{map_point.point.x}  y:{map_point.point.y}')
                    transform = self.tf_buffer.lookup_transform("laser", "map", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)) # laser is the frame published by the particle filter and not by the sim
                    car_point = tf2_geometry_msgs.do_transform_point(map_point, transform)
                    #print(f'Coordinates after transform: x:{car_point.point.x}  y:{car_point.point.y}')
                    break                    
                except (Exception, TypeError) as ex:
                    self.get_logger().error(str(ex))
                    self.get_logger().info("Available TF frames: \n%s" %frames_string)
            
            
            # Calculate steering angle and speed
            y = car_point.point.y
            Radius = L**2 / (2 * np.abs(y))
            if y<0:
                ackermann_cmd.drive.steering_angle = -self.wheel_base / Radius
            else:
                ackermann_cmd.drive.steering_angle = self.wheel_base / Radius
            ackermann_cmd.drive.steering_angle = np.clip(ackermann_cmd.drive.steering_angle, -self.max_steering_angle, self.max_steering_angle)
            
            self.calculate_point_reached_dist(ackermann_cmd.drive.steering_angle)
            print(f"Reach distance: {self.dist_point_reached:.5f}")
            ackermann_cmd.drive.speed = self.speed_percentage * self.target_speed #* (1 - 0.4 * abs(ackermann_cmd.drive.steering_angle / self.max_steering_angle))
            #print(f"Steering angle: {ackermann_cmd.drive.steering_angle:.5f}     Speed: {ackermann_cmd.drive.speed:.1f} ")
            # Check if the current waypoint is reached
            if L < self.dist_point_reached:
                self.next_waypoint_index = (self.next_waypoint_index + 1) % len(self.waypoints)  # Move to the next waypoint or wrap around if at the end
            # Set the time of the message after the calculation
            ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
            

        return ackermann_cmd

    def calculate_point_reached_dist(self, steering_angle):
        self.dist_point_reached = self.min_dist_point_reached + (1 - min(abs(steering_angle) / self.max_steering_angle + (abs(self.raceline_data[self.closest_waypoint_index, 4]) + abs(self.raceline_data[self.next_waypoint_index, 4])) / 2 / self.max_curvature, 1))  \
            * (self.max_dist_point_reached - self.min_dist_point_reached)
    
        

    def set_initial_waypoint_index(self):
        # Find the index of the closest waypoint to the initial position of the vehicle
        if self.current_pose:
             # create PointStamped from waypoint for transformation
            map_point = PointStamped()
            map_point.header.frame_id = "map"
            map_point.point.x = 0.0
            map_point.point.y = 0.0
            map_point.point.z = 0.0
            min_distance = float('inf')
            closest_index = None
            for i, waypoint in enumerate(self.waypoints):
                map_point.point.x = waypoint[0]
                map_point.point.y = waypoint[1]
                transform = self.tf_buffer.lookup_transform("laser", "map", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)) # laser is the frame published by the particle filter and not by the sim
                transformed_point = tf2_geometry_msgs.do_transform_point(map_point, transform)
                if transformed_point.point.x > 1.0 and np.arctan(transformed_point.point.y /transformed_point.point.x) < self.max_steering_angle: # only consider points in front of vehicle
                    distance = transformed_point.point.x**2 + transformed_point.point.y**2 #dont need to calculate square root since it's not necessary to know how close the point actually is
                    if distance < min_distance:
                        min_distance = distance
                        closest_index = i
            if closest_index == None:
                raise ValueError('No suitable inital waypoint found')
            print(f'Initial Waypoint coordinates: {self.waypoints[closest_index][0]}       {self.waypoints[closest_index][1]}')
            self.next_waypoint_index = closest_index
        

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_controller = PurePursuitController()
    rclpy.spin(pure_pursuit_controller)
    pure_pursuit_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
