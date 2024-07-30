##########################################################
# DESCRIPTION: ROS Node that outputs the control commands to f1tenth_gym_ros
# USAGE: Run simulation env on sim_ws
#        go to bash and navigate to /home/glenn/BA/Johannes_MPC_ws
#        run source install/setup.bash
#        run ros2 launch mpc mpc_launch.py
##########################################################
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

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from tf_transformations import euler_from_quaternion
import sys, os
sys.path.append(os.path.join(os.getcwd() + '/src/mpc/mpc'))
from mpc_local import mpc

########################### ENV CONFIG ###################################
map_yaml_filename = 'config_FTM_Halle.yaml'
# Construct the absolute path to the YAML file
map_file = os.path.join(os.getcwd() + '/src/mpc/maps', map_yaml_filename)
ODOM_TOPIC = '/pf/pose/odom'
# ODOM_TOPIC = '/ego_racecar/odom'
########################### MAP CONFIG ###################################

##########################################################################
# ToDo edit setup.cfg not to use python env
########################################################################## 

#--------------------------- Controller Paramter ---------------------------
# System config
NX = 4          # state vector: z = [x, y, v, yaw]
NU = 2          # input vector: u = [accel, steer]
T = 10           # finite time horizon length
N_BATCH = 1
LQR_ITER = 5

# define P during runtime
GOAL_WEIGHTS = torch.tensor((13.5, 13.5, 5.5, 13.0), dtype=torch.float32)  # nx
CTRL_PENALTY = torch.tensor((0.01, 100), dtype=torch.float32) # nu
q = torch.cat((GOAL_WEIGHTS, CTRL_PENALTY))  # nx + nu
Q = torch.diag(q).repeat(T, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu

# MPC prediction paramter
N_IND_SEARCH = 5                        # Search index number
DT = 0.10                               # time step [s]
dl = 0.20                               # dist step [m]
MAX_DIST = 1.0                          # max allowed distance between previous waypoint and target

# Vehicle parameters
WB = 0.3302                           # Wheelbase [m]
MAX_STEER = np.deg2rad(24.0)        # maximum steering angle [rad]
MAX_SPEED = 2.9                    # maximum speed [m/s]
MIN_SPEED = 0                       # minimum backward speed [m/s]
MAX_ACCEL = 2.5                     # maximum acceleration [m/s]
SPEED_FACTOR = 0.93                  # published speed = SPEED_FACTOR * speed

U_LOWER = torch.tensor([[-np.inf, -MAX_STEER]], dtype=torch.float32).repeat(T, N_BATCH, 1)  # T x B x nu
U_UPPER = torch.tensor([[MAX_ACCEL, MAX_STEER]], dtype=torch.float32).repeat(T, N_BATCH, 1)  # T x B x nu

# #--------------------------- Controller Paramter BEFORE ---------------------------
# # System config
# NX = 4          # state vector: z = [x, y, v, yaw]
# NU = 2          # input vector: u = [accel, steer]
# T = 10           # finite time horizon length
# N_BATCH = 1
# LQR_ITER = 5

# # define P during runtime
# GOAL_WEIGHTS = torch.tensor((13.5, 13.5, 5.5, 13.0), dtype=torch.float32)  # nx
# CTRL_PENALTY = torch.tensor((0.01, 100), dtype=torch.float32) # nu
# q = torch.cat((GOAL_WEIGHTS, CTRL_PENALTY))  # nx + nu
# Q = torch.diag(q).repeat(T, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu

# # MPC prediction paramter
# N_IND_SEARCH = 5                        # Search index number
# DT = 0.10                               # time step [s]
# dl = 0.20                               # dist step [m]

# # Vehicle parameters
# WB = 0.3302                           # Wheelbase [m]
# MAX_STEER = np.deg2rad(24.0)        # maximum steering angle [rad]
# MAX_SPEED = 2.9                    # maximum speed [m/s]
# MIN_SPEED = 0                       # minimum backward speed [m/s]
# MAX_ACCEL = 2.5                     # maximum acceleration [m/s]
# SPEED_FACTOR = 0.93                  # published speed = SPEED_FACTOR * speed

# U_LOWER = torch.tensor([[-np.inf, -MAX_STEER]], dtype=torch.float32).repeat(T, N_BATCH, 1)  # T x B x nu
# U_UPPER = torch.tensor([[MAX_ACCEL, MAX_STEER]], dtype=torch.float32).repeat(T, N_BATCH, 1)  # T x B x nu

###################################################### DEBUG ########################################################
# Create a ROS node for publishing debug messages
class DebugPublisher(Node):
    def __init__(self):
        super().__init__('debug_publisher')
        self.publisher_ = self.create_publisher(String, '/debug', 11)

    def publish(self,str):
        msg = String()
        msg.data = str
        self.publisher_.publish(msg)

DEBUG_PUBLISHER = None # initialized in main
####################################################################################################################
class CarDynamics(torch.nn.Module):
        def forward(self, state, action):
            x = state[:, 0].view(-1, 1)
            y = state[:, 1].view(-1, 1)
            v = state[:, 2].view(-1, 1)
            yaw = state[:, 3].view(-1, 1)

            a = action[:, 0].view(-1, 1)
            delta = action[:, 1].view(-1, 1)

            delta = torch.clamp(delta, -MAX_STEER, MAX_STEER)

            x = x + v * torch.cos(yaw) * DT
            y = y + v * torch.sin(yaw) * DT
            yaw = yaw + v / WB * torch.tan(delta) * DT
            v = v + a * DT

            v = torch.clamp(v, MIN_SPEED, MAX_SPEED)

            state = torch.cat((x,y,v,yaw), dim=1)
            return state
        
        def grad_input(self, state, action):
            """
            ************ Single Track Model: Linear - Kinematic ********
            Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
            Linear System: Xdot = Ax +Bu + C
            State vector: x=[x, y, v, yaw]
            :param v: speed
            :param phi: heading angle of the vehicle
            :param delta: steering angle: delta_bar
            :return: A, B
            """
            v = state[:, 2].squeeze()
            phi = state[:, 3].view(-1, 1).squeeze()
            delta = action[:, 1].view(-1, 1).squeeze()
            
            A = torch.empty(T-1, NX, NX)
            B = torch.empty(T-1, NX, NU)
            for i in range(T-1):
                for j in range(N_BATCH):
                    A_ij = torch.zeros((NX, NX))
                    A_ij[0, 0] = 1.0
                    A_ij[1, 1] = 1.0
                    A_ij[2, 2] = 1.0
                    A_ij[3, 3] = 1.0
                    A_ij[0, 2] = DT * torch.cos(phi[i])
                    A_ij[0, 3] = - DT * v[i] * torch.sin(phi[i])
                    A_ij[1, 2] = DT * torch.sin(phi[i])
                    A_ij[1, 3] = DT * v[i] * torch.cos(phi[i])
                    A_ij[3, 2] = DT * torch.tan(delta[i]) / WB
                    A[i, :, :] = A_ij

                    B_ij = torch.zeros((NX, NU))
                    B_ij[2, 0] = DT
                    B_ij[3, 1] = DT * v[i] / (WB * torch.cos(delta[i]) ** 2)
                    B[i, :, :] = B_ij

            return A, B

""" 
Planner Helpers
"""
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
    
class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class Controller:

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.mpc_initialize = 0
        self.target_ind = 0
        self.odelta = None
        self.oa = None
        self.origin_switch = 1
        self.u_init = None

    def load_waypoints(self, conf):
        # Loading the x and y waypoints in the "..._raceline.vsv" that include the path to follow
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)


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
        ind, mind = self.calc_nearest_index(state, cx, cy, cyaw, pind, N_IND_SEARCH)

        if mind >= MAX_DIST:
            print("Distance between the vehicle and the raceline is too large: ", mind)
            print("Vehicle Position: ", state.x, state.y)
            print("Target Position: ", cx[ind], cy[ind])
            sys.exit()

        #if pind >= ind:
        #    ind = pind

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
            dind = int(round(travel / dl))  # Number of distance steps we need to look into the future

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

    def mpc_control(self, ref_traj, x0):
        """
        Solve the MPC control problem using the linearized kinematic model
        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        x0: initial state
        dref: reference steer angle
        :return: optimal acceleration and steering strategy
        """
        ctrl =  mpc.MPC(NX, NU, T, u_lower=U_LOWER, u_upper=U_UPPER, lqr_iter=LQR_ITER,
                exit_unconverged=False, eps=MAX_ACCEL,
                n_batch=N_BATCH, backprop=False, verbose=0, u_init=self.u_init,
                grad_method=mpc.GradMethods.ANALYTIC)
        
        # Define state
        state = torch.tensor(x0, dtype=torch.float32).view(1, -1)

        # Define the cost function
        p = torch.zeros(T, N_BATCH, NX + NU)
        for i in range(0, T):
            for j in range(0, N_BATCH):
                ref_traj_i = torch.tensor(ref_traj[:, i], dtype=torch.float32)
                px = -GOAL_WEIGHTS * ref_traj_i
                ptau = torch.cat((px, torch.zeros(NU)))
                p[i, j, :] = ptau

        cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

        # compute action based on current state, dynamics, and cost
        x_pred, u_pred, nominal_objs = ctrl(state, cost, CarDynamics())
        # print("x_pred: ", x_pred)
        # print("ref_traj: ", ref_traj)
        self.u_init = torch.cat((u_pred[1:], torch.zeros(1, N_BATCH, NU)), dim=0) # first row out, last row = 0

        # reduce the dimension of the output
        x_pred = x_pred.squeeze()
        u_pred = u_pred.squeeze()

        # only take the first action
        mpc_a = u_pred[:,0].detach().numpy() # T = 0, nu = 0
        mpc_delta = u_pred[:,1].detach().numpy() # T = 0, nu = 1

        mpc_x = x_pred[:,0].detach().numpy() # T = 0, nx = 0
        mpc_y = x_pred[:,1].detach().numpy() # T = 0, nx = 1
        mpc_v = x_pred[:,2].detach().numpy() # T = 0, nx = 2
        mpc_yaw = x_pred[:,3].detach().numpy() # T = 0, nx = 3

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v
    
    def MPC_Controller (self, vehicle_state, path):
        # Extract information about the trajectory that needs to be followed
        cx = path[0]        # Trajectory x-Position
        cy = path[1]        # Trajectory y-Position
        cyaw = path[2]      # Trajectory Heading angle
        sp = path[4]        # Trajectory Velocity

        # Initialize the MPC parameter
        if self.mpc_initialize == 0:
            # Find nearest index to starting position
            self.target_ind, _ = self.calc_nearest_index(vehicle_state, cx, cy, cyaw, 0, len(cx)-1)
            if self.target_ind >= 21 or self.target_ind <= 15:
                print("Target Index: ", self.target_ind)
                print("vehicle_state: ", vehicle_state.x, vehicle_state.y)
                print("cx, cy: ", cx[self.target_ind], cy[self.target_ind])
                sys.exit()
            # self.target_ind = 0
            self.odelta, self.oa = None, None
            self.mpc_initialize = 1

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        ref_path, self.target_ind, ref_delta = self.calc_ref_trajectory(vehicle_state, cx, cy, cyaw, sp, dl, self.target_ind)

        # Create state vector based on current vehicle state: x-position, y-position,  velocity, heading
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the Linear MPC Control problem
        self.oa, self.odelta, ox, oy, oyaw, ov = self.mpc_control(ref_path, x0)

        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]

        ###########################################
        #                    DEBUG
        ##########################################

        # debugplot = 0
        # if debugplot == 1:
        #     plt.cla()
        #     # plt.axis([-40, 2, -10, 10])
        #     plt.axis([vehicle_state.x - 6, vehicle_state.x + 4.5, vehicle_state.y - 2.5, vehicle_state.y  + 2.5])
        #     plt.plot(self.waypoints[:, [1]], self.waypoints[:, [2]], linestyle='solid', linewidth=2, color='#005293', label='Raceline')
        #     plt.plot(vehicle_state.x, vehicle_state.y, marker='o', markersize=10, color='red', label='CoG')
        #     plt.plot(ref_path[0], ref_path[1], linestyle='dotted', linewidth=8, color='purple',label='MPC Input: Ref. Trajectory for T steps')
        #     #plt.plot(cx[self.target_ind], cy[self.target_ind], marker='x', markersize=10, color='green',)
        #     plt.plot(ox, oy, linestyle='dotted', linewidth=5, color='green',label='MPC Output: Trajectory for T steps')
        #     plt.legend()
        #     plt.pause(0.001)
        #     plt.axis('equal')

        # debugplot2 = 0
        # if debugplot2 == 1:
        #     plt.cla()
        #     # Creating the number of subplots
        #     fig, axs = plt.subplots(3, 1)
        #     #  Velocity of the vehicle
        #     axs[0].plot(ov, linestyle='solid', linewidth=2, color='#005293')
        #     axs[0].set_ylim([0, max(ov) + 0.5])
        #     axs[0].set(ylabel='Velocity in m/s')
        #     axs[0].grid(axis="both")

        #     axs[1].plot(self.oa, linestyle='solid', linewidth=2, color='#005293')
        #     axs[1].set_ylim([0, max(self.oa) + 0.5])
        #     axs[1].set(ylabel='Acceleration in m/s')
        #     axs[1].grid(axis="both")
        #     plt.pause(0.001)
        #     plt.axis('equal')

        ###########################################
        #                    DEBUG
        ##########################################


        #------------------- MPC CONTROL Output ---------------------------------
        # Steering Output: First entry of the MPC steering angle output vector in degree
        steer_output = self.odelta[0]
        speed_output= vehicle_state.v + self.oa[0] * DT

        return speed_output, steer_output

class LatticePlanner:

    def __init__(self, conf, controller, wb):
        self.conf = conf                        # Current configuration for the gym based on the maps
        self.controller = controller            # MPC Controller object
        self.load_waypoints(conf)               # Waypoints of the raceline
        self.init_flag = 0                      # Initialization of the states
        self.calcspline = 0                     # Flag for Calculation the Cubic Spline
        self.initial_state = []

    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def plan(self):
        """
        Loading the individual data from the global, optimal raceline and creating one list
        """

        cx = self.waypoints[:, 1]       # X-Position of Raceline
        cy = self.waypoints[:, 2]       # Y-Position of Raceline
        cyaw = self.waypoints[:, 3]     # Heading on Raceline
        # for i in range(len(cyaw)):
        #     cyaw[i] = pi_2_pi(cyaw[i])
        ck = self.waypoints[:, 4]       # Curvature of Raceline
        cv = self.waypoints[:, 5]       # velocity on Raceline

        global_raceline = [cx, cy, cyaw, ck, cv]

        return global_raceline

    def control(self, pose_x, pose_y, pose_theta, velocity, path):
        """
        Control loop for calling the controller
        """

        # -------------------- INITIALIZE Controller ----------------------------------------
        if self.init_flag == 0:
            vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=0.1)
            self.init_flag = 1
        else:
            vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=velocity)

        # -------------------- Call the MPC Controller ----------------------------------------
        speed, steering_angle = self.controller.MPC_Controller(vehicle_state, path)

        return speed, steering_angle

class MPC_Node(Node):
    
        def __init__(self):
            super().__init__('MPC_Node')
            
            # Load the configuration for the desired Racetrack
            self.work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.8125}
            # with open('/home/glenn/BA/Johannes_MPC_KS/config_FTM_Halle.yaml') as file:
            # # with open('/home/glenn/BA/Johannes_MPC_KS/config_Spielberg_map.yaml') as file:
            with open(map_file) as file:
                self.conf_dict = yaml.load(file, Loader=yaml.FullLoader)
            self.conf = Namespace(**self.conf_dict)
    
            # Dictionary for changing vehicle paramters for the vehicle dynamics in the F1TENTH Gym (ToDo: use yaml and declare in init)
            self.params_dict = {'mu': 1.0489,
                        'C_Sf': 4.718,
                        'C_Sr': 5.4562,
                        'lf': 0.15875,
                        'lr': 0.17145,
                        'h': 0.074,
                        'm': 3.74,
                        'I': 0.04712,
                        's_min': -0.4189,
                        's_max': 0.4189,
                        'sv_min': -3.2,
                        'sv_max': 3.2,
                        'v_switch': 7.319,
                        'a_max': 9.51,
                        'v_min': -5.0,
                        'v_max': 20.0}
    
            # Creating the Motion planner and Controller object that is used in Gym
            self.controller = Controller(self.conf, 0.17145 + 0.15875) # 0.3302 the same as in sim_ws
            self.planner = LatticePlanner(self.conf, self.controller, 0.17145 + 0.15875) # 0.3302 the same as in sim_ws

            # Creating a Datalogger object that saves all necessary vehicle data
            self.logger = Datalogger(self.conf)

            # Load global raceline to create a path variable that includes all reference path information
            self.path = self.planner.plan()

            # Initialize attributes
            self.laptime = 0.0
            self.control_count = 10
            self.start = time.time()

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

            # Timers
            # self.control_input_timer = self.create_timer(0.1, self.control_input_callback)

            # Subscribers
            self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 1)
            
            # Publishers
            self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        
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

            if self.obs_linear_vel_x > self.obs_max_speed:
                self.obs_max_speed = self.obs_linear_vel_x
            
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

        def publish_control(self):
            # global DEBUG_PUBLISHER
            self.speed, self.steer = self.planner.control(self.obs_pose_x, self.obs_pose_y, self.obs_pose_theta, self.obs_linear_vel_x, self.path)
            
            if self.steer > self.max_steering:
                self.max_steering = self.steer

            # Prepare the drive message with the steering angle and corresponding speed
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'laser'
  
            drive_msg.drive.steering_angle = self.steer.item()
            self.speed = SPEED_FACTOR * self.speed
            drive_msg.drive.speed = self.speed.item()

            # Publish the drive command
            self.drive_pub.publish(drive_msg)

            # self.get_logger().info('Control: speed=%f, steer=%f' % (self.speed, self.steer))
        
        def cleanup(self):
            print("Max speed: ", self.obs_max_speed)
            print("Max steering: ", self.max_steering*180/math.pi)
            # Save the logged data to a file
            if self.conf_dict['logging'] == 'True':
                with open('src/mpc/mpc/mpc_locuslab.p', 'wb') as f:
                    pickle.dump(self.logger, f)

            self.get_logger().info('Pickle file saved')

    
def main(args=None):
    # global DEBUG_PUBLISHER

    # Main function to initialize the ROS node and spin
    rclpy.init(args=args)

    node = MPC_Node()
    # DEBUG_PUBLISHER = DebugPublisher()

    def signal_handler(sig, frame):
        node.get_logger().info("Ctrl-C caught, shutting down.")
        node.cleanup()
        node.destroy_node()
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
    # Entry point for the script
    main()