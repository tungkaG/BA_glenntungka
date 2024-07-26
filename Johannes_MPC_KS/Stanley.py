import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
from numba import njit
import matplotlib.pyplot as plt
import pickle

DONT_MOVE_CAR = False

# Simulation paramter
DT = 0.10                               # time step [s]

# Vehicle parameters
LENGTH = 0.58                       # Length of the vehicle [m]
WIDTH = 0.31                        # Width of the vehicle [m]
WB = 0.33                           # Wheelbase [m]

K_Stanley = 0.9                     # Stanley gain
K_pid = 0.9                         # PID gain

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
        self.vehicle_velocity_y = []        # Current vehicle velocity - Lateral
        self.control_velocity = []          # Desired vehicle velocity based on control calculation
        self.steering_angle = []            # Steering angle based on control calculation
        self.lapcounter = []                # Current Lap
        self.control_raceline_x = []        # Current Control Path X-Position on Raceline
        self.control_raceline_y = []        # Current Control Path y-Position on Raceline
        self.control_raceline_heading = []  # Current Control Path Heading on Raceline
        self.yawrate = []                   # Current State of the yawrate in rad\s
        self.sideslip_angle = []            # Current Sideslipangle in rad

    def logging(self, pose_x, pose_y, pose_theta, current_velocity_x, current_velocity_y, lap, control_veloctiy, control_steering, yawrate, sideslip_angle):
        self.vehicle_position_x.append(pose_x)
        self.vehicle_position_y.append(pose_y)
        self.vehicle_position_heading.append(pose_theta)
        self.vehicle_velocity_x .append(current_velocity_x)
        self.vehicle_velocity_y.append(current_velocity_y)
        self.control_velocity.append(control_veloctiy)
        self.steering_angle.append(control_steering)
        self.lapcounter.append(lap)
        self.yawrate.append(yawrate)
        self.sideslip_angle.append(sideslip_angle)

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

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.mpc_initialize = 0
        self.target_ind = 0
        self.odelta = None
        self.oa = None
        self.origin_switch = 1
        self.ind_old = 0

    def load_waypoints(self, conf):
        # Loading the x and y waypoints in the "..._raceline.vsv" that include the path to follow
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

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

        return speed, steering_angle

    def calc_theta_e_and_ef(self, state, ref_path):
        """
        calc theta_e and ef.
        theta_e = theta_car - theta_path
        ef = lateral distance in frenet frame (front wheel)

        :param state: current information of vehicle
        :return: theta_e and ef
        """

        fx = state.x + self.wheelbase * math.cos(state.yaw)
        fy = state.y + self.wheelbase * math.sin(state.yaw)

        dx = [fx - x for x in ref_path.cx]
        dy = [fy - y for y in ref_path.cy]

        target_index = int(np.argmin(np.hypot(dx, dy)))
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

    def __init__(self, conf, env, controller, wb):
        self.conf = conf                        # Current configuration for the gym based on the maps
        self.env = env                          # Current environment parameter
        self.controller = controller            # Controller object
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

        # -------------------- Call the Stanley Controller ----------------------------------------
        speed, steering_angle = self.controller.stanley_control(vehicle_state, path)


        return speed, steering_angle


# -------------------------- MAIN SIMULATION  ----------------------------------------

if __name__ == '__main__':
    # Check CVXP Installations
    #print(cvxpy.installed_solvers())

    # Load the configuration for the desired Racetrack
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.8125}
    with open('/home/glenn/BA/Johannes_MPC_KS/config_FTM_Halle.yaml') as file:
    # with open('/home/glenn/BA/Johannes_MPC_KS/config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Dictionary for changing vehicle paramters for the vehicle dynamics in the F1TENTH Gym
    params_dict = {'mu': 1.0489,
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
                   'v_max': 20.0,
                   'width': 0.31,
                   'length': 0.58}

    # Create the simulation environment and inititalize it
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, params=params_dict)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Creating the Motion planner and Controller object that is used in Gym
    controller = Controller(conf, 0.17145 + 0.15875)
    planner = LatticePlanner(conf, env, controller, 0.17145 + 0.15875)

    # Creating a Datalogger object that saves all necessary vehicle data
    logging = Datalogger(conf)

    # Initialize Simulation
    laptime = 0.0
    control_count = 10
    start = time.time()

    # Load global raceline to create a path variable that includes all reference path information
    path = planner.plan()

    max_speed = 0.0

    # -------------------------- SIMULATION LOOP  ----------------------------------------
    while not done:
        if control_count >= 10:

            # Call the function for tracking speed and steering
            # MPC specific: We solve the MPC problem only every 6th timestep of the simultation to decrease the sim time
            speed, steer = planner.control(obs['poses_x'][0], obs['poses_y'][0], zero_2_2pi(obs['poses_theta'][0]),obs['linear_vels_x'][0], path)
            print("Speed:", speed, "Steer:", steer)
            control_count = 0

        # Update the simulation environment

        if obs['linear_vels_x'][0] > max_speed:
            max_speed = obs['linear_vels_x'][0]

        obs, step_reward, done, info = env.step(np.array([[steer, (not DONT_MOVE_CAR) * 0.93 * speed]]))
        laptime += step_reward

        env.render(mode='human_fast')

        # Apply Looging to log information from the waypoints

        if conf_dict['logging'] == 'True':
            logging.logging(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0],
                            obs['linear_vels_y'][0], obs['lap_counts'], speed, steer, env.sim.agents[0].state[5],
                            env.sim.agents[0].state[6])

            wpts = np.vstack((planner.waypoints[:, conf.wpt_xind], planner.waypoints[:, conf.wpt_yind])).T
            vehicle_state = np.array([obs['poses_x'][0], obs['poses_y'][0]])
            nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(vehicle_state, wpts)
            logging.logging2(nearest_point[0], nearest_point[1], path[2][i])

        # Update Asynchronous Counter for the MPC loop
        control_count = control_count + 1

        if obs['lap_counts'] == 1:
            print("The maximum speed is:", max_speed)
            break

    if conf_dict['logging'] == 'True':
        pickle.dump(logging, open("/home/glenn/BA/Johannes_MPC_KS/datalogging_MPC_KS.p", "wb"))

    # Print Statement that simulation is over
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)