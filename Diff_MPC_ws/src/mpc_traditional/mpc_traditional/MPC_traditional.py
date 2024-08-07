##########################################################
# DESCRIPTION: ROS Node that outputs the control commands to f1tenth_gym_ros
# USAGE: Run simulation env on sim_ws
#        go to bash and navigate to /home/glenn/BA/Johannes_MPC_ws
#        run source install/setup.bash
#        run ros2 launch mpc mpc_launch.py
##########################################################
import yaml
import numpy as np
from argparse import Namespace
import math
from numba import njit
import torch
from scipy.spatial.transform import Rotation as R
import signal
import cvxpy

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
import sys, os
###################################### YAML ######################################
config_file = os.path.join(os.getcwd() + '/src/mpc_traditional/config', 'MPC_traditional.yaml')
with open(config_file) as file:
    CONFIG_PARAM = yaml.load(file, Loader=yaml.FullLoader)
###################################### YAML ######################################

# System config
NX = 4          # state vector: z = [x, y, v, yaw]
NU = 2          # input vector: u = [accel, steer]
T = CONFIG_PARAM['t']           # finite time horizon length

# MPC parameters
R = np.diag([CONFIG_PARAM['ctrl_penalty_a'] , CONFIG_PARAM['ctrl_penalty_steer']])  # input cost matrix, penalty for inputs - [accel, steer]
Rd = np.diag([CONFIG_PARAM['ctrl_penalty_a'] , CONFIG_PARAM['ctrl_penalty_steer']])             # input difference cost matrix, penalty for change of inputs - [accel, steer]
Q = np.diag([CONFIG_PARAM['goal_weight_x'], CONFIG_PARAM['goal_weight_y'],
              CONFIG_PARAM['goal_weight_v'], CONFIG_PARAM['goal_weight_theta']])       # state cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
Qf = np.diag([CONFIG_PARAM['goal_weight_x'], CONFIG_PARAM['goal_weight_y'],
              CONFIG_PARAM['goal_weight_v'], CONFIG_PARAM['goal_weight_theta']])      # state final matrix, penalty  for the final state constraints: [x, y, v, yaw]

# MPC prediction paramter
N_IND_SEARCH = CONFIG_PARAM['n_ind_search']                        # Search index number
DT = CONFIG_PARAM['dt']                               # time step [s]
dl = CONFIG_PARAM['dl']                               # dist step [m]

# Vehicle parameters
WB = CONFIG_PARAM['wheelbase']                           # Wheelbase [m]
MAX_STEER = np.deg2rad(CONFIG_PARAM['max_steer'])        # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(CONFIG_PARAM['max_dsteer'])       # maximum steering speed [rad/s]
MAX_SPEED = CONFIG_PARAM['max_speed']                    # maximum speed [m/s]
MIN_SPEED = CONFIG_PARAM['min_speed']                       # minimum backward speed [m/s]
MAX_ACCEL = CONFIG_PARAM['max_accel']                     # maximum acceleration [m/ss]
SPEED_FACTOR = CONFIG_PARAM['speed_factor']                  # published speed = SPEED_FACTOR * speed

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
    
@njit(fastmath=False, cache=True)
def get_kinematic_model_matrix(v, phi, delta):
    """
    ************ Single Track Model: Linear - Kinematic ********
    Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
    Linear System: Xdot = Ax +Bu + C
    State vector: x=[x, y, v, yaw]
    :param v: speed
    :param phi: heading angle of the vehicle
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """

    # State (or system) matrix A, 4x4
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    # Input Matrix B; 4x2
    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C
    
class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class MPC_Pos_Node(Node):
    def __init__(self):
        super().__init__('mpc_traditional_pos_node')
        self.mpc_position_pub = self.create_publisher(MarkerArray, '/mpc_pos', 1)

    def publish_mpc_pos(self, ox, oy):
        mpc_pos = MarkerArray()
        for i in range(len(ox)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "mpc_pos"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x = ox[i], y = oy[i], z = 0.0)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            mpc_pos.markers.append(marker)
        self.mpc_position_pub.publish(mpc_pos)

class Controller():

    def __init__(self):
        self.mpc_initialize = 0
        self.target_ind = 0
        self.odelta = None
        self.oa = None
        self.origin_switch = 1
        self.u_init = None

        ################################## DEBUG ##################################
        self.mpc_position_node = MPC_Pos_Node()
        ################################## DEBUG ##################################

    def calc_nearest_index(self, state, cx, cy, cyaw, pind, N):
        """
        calc index of the nearest ref trajector in N steps
        :param node: path information X-Position, Y-Position, current index.
        :return: nearest index,
        """

        if pind == len(cx)-1:
            dx = [state.x - icx for icx in cx[0:(0 + N)]]
            dy = [state.y - icy for icy in cy[0:(0 + N)]]

            d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
            mind = min(d)
            ind = d.index(mind) + 0

        else:
            dx = [state.x - icx for icx in cx[pind:(pind + N)]]
            dy = [state.y - icy for icy in cy[pind:(pind + N)]]

            d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
            mind = min(d)
            ind = d.index(mind) + pind

        mind = math.sqrt(mind)
        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, dl, pind):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        ind, mind = self.calc_nearest_index(state, cx, cy, cyaw, pind, CONFIG_PARAM['n_ind_search'])

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0                # steer operational point should be 0

        # Initialize Parameter
        travel = 0.0
        self.origin_switch = 1

        for i in range(T + 1):
            travel += abs(state.v) * DT     # Travel Distance into the future based on current velocity: s= v * t

            if CONFIG_PARAM['varying_dl']:
                dl_mod = dl + state.v * CONFIG_PARAM['dl_factor_num']/CONFIG_PARAM['dl_factor_denom']
                dind = int(round(travel / dl_mod))  # Number of distance steps we need to look into the future
            else:
                dind = int(round(travel / dl)) # Number of distance steps we need to look into the future


            if (ind + dind) < ncourse:
                ref_traj[0, i] = cx[ind + dind]
                ref_traj[1, i] = cy[ind + dind]
                ref_traj[2, i] = sp[ind + dind]

                # IMPORTANT: Take Care of Heading Change from 2pi -> 0 and 0 -> 2pi, so that all headings are the same
                if cyaw[ind + dind] -state.yaw > 5:
                    ref_traj[3, i] = abs(cyaw[ind + dind] -2* math.pi)
                elif cyaw[ind + dind] -state.yaw < -5:
                    ref_traj[3, i] = abs(cyaw[ind + dind] + 2 * math.pi)
                else:
                    ref_traj[3, i] = cyaw[ind + dind]

            else:
                # This function takes care about the switch at the origin/ Lap switch
                ref_traj[0, i] = cx[self.origin_switch]
                ref_traj[1, i] = cy[self.origin_switch]
                ref_traj[2, i] = sp[self.origin_switch]
                ref_traj[3, i] = cyaw[self.origin_switch]
                dref[0, i] = 0.0
                self.origin_switch = self.origin_switch +1

        return ref_traj, ind, dref

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * DT
        state.y = state.y + state.v * math.sin(state.yaw) * DT
        state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
        state.v = state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED
        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED

        return state

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def mpc_step(self, ref_path, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if oa is None or od is None:
            oa = [0.0] * T
            od = [0.0] * T

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_optimization(ref_path, path_predict, x0, dref)


        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v

    def mpc_optimization(self, ref_traj, path_predict, x0, dref):
        """
        Create and solve the quadratic optimization problem using cvxpy, solver: OSQP
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        dref: reference steer angle
        :return: optimal acceleration and steering strateg
        """

        # Initialize vectors with CVXPY variables
        x = cvxpy.Variable((NX, T + 1))     # Vehicle State Vector
        u = cvxpy.Variable((NU, T))         # Control Input vector
        objective = 0.0                     # Objective value of the optimization problem, set to zero
        constraints = []                    # Create constraints array

        # Formulate and create the finite-horizon optimal control problem (objective function)
        for t in range(T):
            objective += cvxpy.quad_form(u[:, t], R)

            if t != 0:
                objective += cvxpy.quad_form(ref_traj[:, t] - x[:, t], Q)

            A, B, C = get_kinematic_model_matrix(path_predict[2, t], path_predict[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (T - 1):
                objective += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

        objective+= cvxpy.quad_form(ref_traj[:, T] - x[:, T], Qf)

        # Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        # Save the output of the MPC (States and Input) into specific variables
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            mpc_x = self.get_nparray_from_matrix(x.value[0, :])
            mpc_y = self.get_nparray_from_matrix(x.value[1, :])
            mpc_v = self.get_nparray_from_matrix(x.value[2, :])
            mpc_yaw = self.get_nparray_from_matrix(x.value[3, :])
            mpc_a = self.get_nparray_from_matrix(u.value[0, :])
            mpc_delta = self.get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = None, None, None, None, None, None

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v
    
    def control(self, pose_x, pose_y, pose_theta, velocity, path):
        """
        Control loop for calling the controller
        """
        vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=velocity)

        # Extract information about the trajectory that needs to be followed
        cx = path[0]        # Trajectory x-Position
        cy = path[1]        # Trajectory y-Position
        cyaw = path[2]      # Trajectory Heading angle
        sp = path[4]        # Trajectory Velocity

        # Initialize the MPC parameter
        if self.mpc_initialize == 0:
            # Find nearest index to starting position
            self.target_ind, _ = self.calc_nearest_index(vehicle_state, cx, cy, cyaw, 0, len(cx)-1)

            # Check if the localization is giving unrealistic results
            if self.target_ind >= 21 or self.target_ind <= 15:
                print("Target Index: ", self.target_ind)
                print("vehicle_state: ", vehicle_state.x, vehicle_state.y)
                print("cx, cy: ", cx[self.target_ind], cy[self.target_ind])
                sys.exit() # stop the program

            self.mpc_initialize = 1

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        ref_path, self.target_ind, ref_delta = self.calc_ref_trajectory(vehicle_state, cx, cy, cyaw, sp, CONFIG_PARAM['dl'], self.target_ind)

        # Create state vector based on current vehicle state: x-position, y-position,  velocity, heading
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the Linear MPC Control problem
        self.oa, self.odelta, ox, oy, oyaw, ov = self.mpc_step(ref_path, x0, ref_delta, self.oa, self.odelta)

        ################################## DEBUG ##################################
        # Convert ox and oy to floats
        ox = ox.astype(float)
        oy = oy.astype(float)
        # Publish the MPC position for visualization
        self.mpc_position_node.publish_mpc_pos(ox, oy)
        ################################## DEBUG ##################################

        # Steering Output: First entry of the MPC steering angle output vector in degree
        steer_output = self.odelta[0]
        speed_output= vehicle_state.v + self.oa[0] * DT

        steer_output = np.clip(steer_output, -MAX_STEER, MAX_STEER)
        speed_output = np.clip(speed_output, MIN_SPEED, MAX_SPEED)

        return speed_output, steer_output, ref_path

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

class MPC_Node(Node):
    
        def __init__(self):
            super().__init__('MPC_Node')

            map_yaml_filename = CONFIG_PARAM['map_yaml_filename']
            # Construct the absolute path to the YAML file
            map_file = os.path.join(os.getcwd() + '/maps', map_yaml_filename)
            # Load the configuration map file
            with open(map_file) as file:
                self.conf_dict = yaml.load(file, Loader=yaml.FullLoader)
            self.conf = Namespace(**self.conf_dict)
    
            # Creating the Motion planner and Controller object
            self.controller = Controller() 
            self.planner = LatticePlanner(self.conf)

            # Load global raceline to create a path variable that includes all reference path information
            self.path = self.planner.plan()
            self.ref_path = np.zeros((NX, T + 1))

            # Initialize attributes
            self.control_count = CONFIG_PARAM['control_frequency']

            self.obs_pose_x = 0.0
            self.obs_pose_y = 0.0
            self.obs_pose_theta = 0.0
            self.obs_linear_vel_x = 0.0
            self.obs_linear_vel_y = 0.0
            self.obs_yawrate = 0.0

            self.obs_max_speed = 0.0
            self.max_steering = 0.0

            self.speed = 0.0
            self.steer = 0.0

            # Subscribers
            self.odom_sub = self.create_subscription(Odometry, CONFIG_PARAM['odom_topic'], self.odom_callback, 1)
            
            # Publishers
            self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
            self.ref_path_pub = self.create_publisher(MarkerArray, '/ref_path', 1)
            # self.position_pub = self.create_publisher(Pose, '/current_pose', 1)

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

            self.obs_pose_x = msg.pose.pose.position.x
            self.obs_pose_y = msg.pose.pose.position.y
            self.obs_pose_theta = zero_2_2pi(euler[2])
            self.obs_linear_vel_x = msg.twist.twist.linear.x
            self.obs_yawrate = msg.twist.twist.angular.z

            if self.obs_linear_vel_x > self.obs_max_speed:
                self.obs_max_speed = self.obs_linear_vel_x

            if(self.control_count >= CONFIG_PARAM['control_frequency']):
                self.control_count = 0
                self.publish_control()
                self.publish_ref_path()
                # self.publish_pose()

            self.control_count = self.control_count + 1 

        def publish_control(self):
            self.speed, self.steer, self.ref_path = self.controller.control(self.obs_pose_x, self.obs_pose_y, 
                                                                            self.obs_pose_theta, self.obs_linear_vel_x, 
                                                                            self.path)
            
            # Prepare the drive message with the steering angle and corresponding speed
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'laser'
  
            drive_msg.drive.steering_angle = self.steer.item()
            self.speed = CONFIG_PARAM['speed_factor'] * self.speed
            drive_msg.drive.speed = self.speed.item()

            # Publish the drive command
            self.drive_pub.publish(drive_msg)

            # self.get_logger().info('Control: speed=%f, steer=%f' % (self.speed, self.steer))

        def publish_ref_path(self):
            # Publish the reference path for visualization
            ref_path_msg = MarkerArray()
            for i in range(T+1):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "ref_path"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = Point(x = self.ref_path[0, i], y = self.ref_path[1, i], z = 0.0)
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                ref_path_msg.markers.append(marker)

            self.ref_path_pub.publish(ref_path_msg)

        def cleanup(self):
            print("Max speed: ", self.obs_max_speed)
            print("Max steering: ", self.max_steering*180/math.pi)

    
def main(args=None):
    # Main function to initialize the ROS node and spin
    rclpy.init(args=args)

    node = MPC_Node()

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