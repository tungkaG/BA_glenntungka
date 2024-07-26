import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as pltcol

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
        self.mpc_x = []
        self.mpc_y = []
        self.mpc_yaw = []
        self.mpc_v = []


    def logging(self, pose_x, pose_y, pose_theta, current_velocity_x, current_velocity_y, lap, control_veloctiy, control_steering, yawrate, sideslip_angle, mpc_x, mpc_y, mpc_yaw, mpc_v):
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
        self.mpc_x.append(mpc_x)
        self.mpc_y.append(mpc_y)
        self.mpc_yaw.append(mpc_yaw)
        self.mpc_v.append(mpc_v)

    def logging2(self, raceline_x, raceline_y, raceline_theta):
        self.control_raceline_x.append(raceline_x)
        self.control_raceline_y.append(raceline_y)
        self.control_raceline_heading.append(raceline_theta)


# Load the pickle file data
current_dir = os.path.dirname(os.path.abspath(__file__))
# Load the correct filename
# PurePursuit,Stanley, LQR, MPC
filename = current_dir + '/mpc_locuslab.p'
file_to_read = open(filename, "rb")
data = pickle.load(file_to_read)

# Extract Raceline data
raceline_x = data.waypoints[:,[1]]
raceline_y = data.waypoints[:,[2]]
raceline_heading = data.waypoints[:,[3]]



########### Calculate additional vehicle parameters

y = 0
long_accel = []
long_velocity=np.array(data.vehicle_velocity_x)
for x in range(1,len(long_velocity)):
    v2 =long_velocity[x]
    v1 =long_velocity[y]
    delta_v = v2-v1
    accel =  delta_v  /0.01
    if accel < -50:
        accel =0
    long_accel.append(accel)
    y = y+1

####################  Calculate Lateral Acceleration:

# Curvature Calcultation with Ackerman Steering angle - low speeds, high radius, no side slip
WB = 0.33                           # Wheelbase [m]
curvature = []
for item in data.steering_angle:
    radius = WB/np.tan(item)
    curv = 1/radius
    curvature.append(curv)

curv_arr = np.array(curvature)
vel_arr = np.array(data.vehicle_velocity_x)

# Curvature Calculation based on X-Y Coordinates of the CoG

# Calculate Derivates in each point
x = np.array(data.vehicle_position_x)
y = np.array(data.vehicle_position_y)
x_dot = np.gradient(x)
x_ddot = np.gradient(x_dot)
y_dot = np.gradient(y)
y_ddot = np.gradient(y_dot)

curv_arr2= (x_ddot * y_dot - x_dot * y_ddot) / (x_dot * x_dot + y_dot * y_dot)**1.5

# Calculate Lateral acceleration
lat_accel=(vel_arr*vel_arr)*curv_arr
lat_accel2=(vel_arr*vel_arr)*curv_arr2


#################### Tyre slip Angle

lf=0.15875
lr=0.17145
C_Sf= 4.718
C_Sr= 5.4562

alpha_f = np.arctan((np.sin(np.array(data.sideslip_angle))* np.array(data.vehicle_velocity_x) + np.array(data.yawrate) *lf)/ (np.cos(np.array(data.sideslip_angle))* np.array(data.vehicle_velocity_x))) -np.array(data.steering_angle)
alpha_r = np.arctan((np.sin(np.array(data.sideslip_angle))* np.array(data.vehicle_velocity_x) - np.array(data.yawrate) *lr)/ (np.cos(np.array(data.sideslip_angle))* np.array(data.vehicle_velocity_x)))

Fy_f = C_Sf * alpha_f
Fy_r = C_Sr * alpha_r


###############################################################################################################
################################      Calculate Errors    ##############################################

# Calculate velocity error
velocity_error = np.array(data.control_velocity) -np.array(data.vehicle_velocity_x)
# Calculate heading error
heading_error = np.array(data.control_raceline_heading[1:]) -np.array(data.vehicle_position_heading[:-1])
# Filter Singularities in heading error because of F1TENTH Gym heading issue
heading_error = np.where(heading_error > 0.6, 0, heading_error)
heading_error = np.where(heading_error < -0.6, 0, heading_error)



# Calculate lateral error - deviation from the path
#x_dist = np.array(data.vehicle_position_x[:-1]) -np.array(data.control_raceline_x[1:])  # Be careful: The logging of the raceline has one additional step
#y_dist =np.array(data.vehicle_position_y[:-1]) -np.array(data.control_raceline_y[1:])   # Be careful: The logging of the raceline has one additional step

x_dist = np.array(data.vehicle_position_x) -np.array(data.control_raceline_x)  # Be careful: The logging of the raceline has one additional step
y_dist =np.array(data.vehicle_position_y) -np.array(data.control_raceline_y)   # Be careful: The logging of the raceline has one additional step
lateral_error = np.sqrt(pow(x_dist,2)+pow(y_dist,2))


###############################################################################################################
################################      Visualize Vehicle Data     ##############################################

# Creating the number of subplots
fig, axs = plt.subplots(3,2)

#  Velocity of the vehicle
axs[0,0].plot(data.vehicle_velocity_x, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
axs[0,0].plot(data.control_velocity, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Veloctiy')
axs[0,0].set_ylim([0, max(data.vehicle_velocity_x)+0.5])
axs[0,0].set_title('Vehicle Velocity: Actual Velocity vs. Raceline Velocity')
axs[0,0].set(ylabel='Velocity in m/s')
axs[0,0].grid(axis="both")
axs[0,0].legend()

#  Heading of the Vehicle
axs[1,0].plot(data.vehicle_position_heading , linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Heading')
axs[1,0].plot(data.control_raceline_heading, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Heading')
axs[1,0].set_title('Vehicle Heading: Actual Heading vs. Raceline Heading')
axs[1,0].set(ylabel='Vehicle Heading in rad')
axs[1,0].grid(axis="both")
axs[1,0].legend()


#  Steering Angle
axs[2,0].plot(data.steering_angle, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Heading')
axs[2,0].set_title('Steering angle')
axs[2,0].set(ylabel='Steering angle in degree')
axs[2,0].grid(axis="both")
axs[2,0].legend()


###########################    ERRORS      ###########################

#  Velocity Error of the vehicle
axs[0,1].plot(velocity_error, linestyle ='solid',linewidth=2,color = '#005293', label = 'Veloctiy Error')
axs[0,1].set_ylim([-1.6, max(velocity_error)+0.2])
axs[0,1].set_title('Velocity Error')
axs[0,1].set(ylabel='Velocity in m/s')
axs[0,1].grid(axis="both")
axs[0,1].legend()

#  Heading Error of the vehicle
axs[1,1].plot(heading_error, linestyle ='solid',linewidth=2,color = '#005293', label = 'Heading Error')
axs[1,1].set_ylim([min(heading_error)-0.1, max(heading_error)+0.1])
axs[1,1].set_title('Heading Error')
axs[1,1].set(ylabel='Vehicle Heading in rad')
axs[1,1].grid(axis="both")
axs[1,1].legend()


#  Lateral Error of the vehicle
axs[2,1].plot(lateral_error, linestyle ='solid',linewidth=2,color = '#005293', label = 'Lateral Error')
#axs[2,1].plot(lateral_error_smoothed, linestyle ='solid',linewidth=2,color = '#00FF00', label = 'Actual Veloctiy')
axs[2,1].set_ylim([min(lateral_error)-0.2, max(lateral_error)+0.2])
axs[2,1].set_title('Lateral Error')
axs[2,1].set(ylabel='Distance in m')
axs[2,1].grid(axis="both")
axs[2,1].legend()

plt.show()

#########################        Vehicle Dynamics Plots          ##############################

# Creating the number of subplots
fig2, axs2 = plt.subplots(3,2)
#  Driven Path
axs2[0,0].plot(data.vehicle_position_x,data.vehicle_position_y,linestyle ='solid',linewidth=2, color = '#005293', label = 'Driven Path')
axs2[0,0].set_title('Vehicle Path: Driven Path')
axs2[0,0].set(xlabel='X-Position on track')
axs2[0,0].set(ylabel='Y-Position on track')
axs2[0,0].grid(axis="both")
axs2[0,0].axis('equal')
axs2[0,0].legend()

#  Velocity of the vehicle
axs2[1,0].plot(data.vehicle_velocity_x, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
axs2[1,0].set_ylim([0, max(data.vehicle_velocity_x)+0.5])
axs2[1,0].set_title('Vehicle Velocity')
axs2[1,0].set(ylabel='Velocity in m/s')
axs2[1,0].grid(axis="both")
axs2[1,0].legend()

#  Steering Angle
axs2[2,0].plot(np.rad2deg(data.steering_angle), linestyle ='solid',linewidth=2,color = '#005293', label = 'Steering Angle')
axs2[2,0].set_title('Steering angle')
axs2[2,0].set(ylabel='Steering angle in degree')
axs2[2,0].grid(axis="both")
axs2[2,0].legend()

#  Longitudinal acceleration
axs2[0,1].plot(long_accel, linestyle ='solid',linewidth=2,color = '#005293', label = 'Longitudinal Acceleration')
axs2[0,1].set_ylim([min(long_accel)-0.2, max(long_accel)+0.2])
axs2[0,1].set_title('Longitudinal Vehicle Acceleration')
axs2[0,1].set(ylabel='Acceleration in m/s2')
axs2[0,1].grid(axis="both")
axs2[0,1].legend()

#  Lateral Acceleration
axs2[1,1].plot(lat_accel2, linestyle ='solid',linewidth=2,color = '#005293', label = 'Lateral Acceleration')
axs2[1,1].set_ylim([min(lat_accel)-0.2, max(lat_accel)+0.2])
axs2[1,1].set_title('Lateral Vehicle Acceleration')
axs2[1,1].set(ylabel='Acceleration in m/s2')
axs2[1,1].grid(axis="both")
axs2[1,1].legend()

#  Yawrate of the vehicle
axs2[2,1].plot(data.yawrate, linestyle ='solid',linewidth=2,color = '#005293', label = 'Yawrate')
#axs[2,1].set_ylim([min(data.sideslip_angle), max(data.sideslip_angle)])
axs2[2,1].set_title('Yawrate')
axs2[2,1].set(ylabel='Yawrate in rad/s')
axs2[2,1].grid(axis="both")
axs2[2,1].legend()

plt.show()
###########################################    WHOLE TRACK      ############################################

# fig3, axs3 = plt.subplots(2,2)

# #  Lateral Tire Forces
# axs3[0,0].plot(np.rad2deg(alpha_f),Fy_f, linestyle ='solid',linewidth=2,color = '#005293', label = 'Front Axle/Front Wheels')
# axs3[0,0].plot(np.rad2deg(alpha_r),Fy_r, linestyle ='solid',linewidth=2,color = '#e37222', label = 'Rear Axle/Rear Wheels')
# axs3[0,0].set_title('Lateral Tyre Forces')
# axs3[0,0].set(ylabel='Lateral Force in N')
# axs3[0,0].set(xlabel='Tyre slip angle in degree')
# axs3[0,0].grid(axis="both")
# axs3[0,0].legend()

# axs3[1,0].plot(Fy_f, linestyle ='solid',linewidth=2,color = '#005293', label = 'Front Axle/Front Wheels')
# axs3[1,0].plot(Fy_r, linestyle ='solid',linewidth=2,color = '#e37222', label = 'Rear Axle/Rear Wheels')
# axs3[1,0].set_title('Lateral Tyre Forces')
# axs3[1,0].set(ylabel='Lateral Force in N')
# axs3[1,0].grid(axis="both")
# axs3[1,0].legend()

# axs3[0,1].plot(np.rad2deg(data.sideslip_angle), linestyle ='solid',linewidth=2,color = '#005293', label = 'Sideslip Angle')
# #axs[2,1].set_ylim([min(data.sideslip_angle), max(data.sideslip_angle)])
# axs3[0,1].set_title('Sideslip angle')
# axs3[0,1].set(ylabel='Sideslip in Degree')
# axs3[0,1].grid(axis="both")
# axs3[0,1].legend()

# axs3[1,1].plot(lat_accel,np.rad2deg(data.steering_angle), linestyle ='solid',linewidth=2,color = '#005293', label = 'Steering Angle')
# axs3[1,1].plot(lat_accel,np.rad2deg(data.sideslip_angle), linestyle ='solid',linewidth=2,color = '#e37222', label = 'Sideslip angle')
# axs3[1,1].set_title('Under/ Oversteer')
# axs3[1,1].set(ylabel='Angle in Degree')
# axs3[1,1].set(xlabel='Lateral Acceleration in m/s2')
# axs3[1,1].grid(axis="both")
# axs3[1,1].legend()

# plt.show()

##################################       Additional Plots       #################################

# fig4, axs4= plt.subplots(2,2)

# axs4[0,0].scatter(lat_accel[:-1],np.array(long_accel), linestyle ='solid',linewidth=2,color = '#005293', label = 'Acceleration')
# axs4[0,0].set_title('G-G Diagram')
# axs4[0,0].set(ylabel='Longitudinal Acceleration in m/s2')
# axs4[0,0].set(xlabel='Lateral Acceleration in m/s2')
# axs4[0,0].grid(axis="both")
# axs4[0,0].legend()

# plt.show()

###########################################    WHOLE TRACK      ############################################

# #  Plot driven path of vehicle for all laps
# plt.figure(1)
# plt.plot(data.vehicle_position_x,data.vehicle_position_y,linestyle ='solid',linewidth=2, color = '#005293', label = 'Driven Path')
# #plt.plot(raceline_x,raceline_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Path')
# plt.plot(data.control_raceline_x,data.control_raceline_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Path')

# plt.axis('equal')
# plt.xlabel ('X-Position on track')
# plt.ylabel ('Y-Position on track')
# plt.legend()
# plt.title ('Vehicle Path: Driven Path vs. Raceline Path')
# plt.show()

#  Plot driven path of vehicle for all laps
plt.figure(2)
plt.plot(data.vehicle_position_x,data.vehicle_position_y,linestyle ='solid',linewidth=2, color = '#005293', label = 'Driven Path')
#plt.plot(raceline_x,raceline_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Path')
plt.plot(data.mpc_x,data.mpc_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'MPC Path')

plt.axis('equal')
plt.xlabel ('X-Position on track')
plt.ylabel ('Y-Position on track')
plt.legend()
plt.title ('Vehicle Path: Driven Path vs. MPC Path')
plt.show()

###########################################    PREDICTION ERROR      ############################################
# print("prediction error:", np.sum((np.array(data.mpc_x)-np.array(data.vehicle_position_x))**2 + (np.array(data.mpc_y)-np.array(data.vehicle_position_y))**2))

# print("length of data.vehicle_position_x", len(data.vehicle_position_x))

# Calculate absolute error of MPC_x - vehicle_position_x
mpc_x_error = np.abs(np.array(data.mpc_x) - np.array(data.vehicle_position_x))

# Calculate absolute error of MPC_y - vehicle_position_y
mpc_y_error = np.abs(np.array(data.mpc_y) - np.array(data.vehicle_position_y))

# Calculate absolute error of MPC_v - vehicle_velocity_x
mpc_v_error = np.abs(np.array(data.mpc_v) - np.array(data.vehicle_velocity_x))

# Calculate absolute error of MPC_yaw - vehicle_position_yaw
mpc_yaw_error = np.abs(np.array(data.mpc_yaw) - np.array(data.vehicle_position_heading))

# Creating the number of subplots
fig3, axs3 = plt.subplots(4, 1)

# Plot absolute error of MPC_x - vehicle_position_x
axs3[0].plot(mpc_x_error, linestyle='solid', linewidth=2, color='#005293', label='MPC_x Error')
axs3[0].set_title('Absolute Error of MPC_x - vehicle_position_x')
axs3[0].set(ylabel='Error')
axs3[0].grid(axis="both")
axs3[0].legend()

# Plot absolute error of MPC_y - vehicle_position_y
axs3[1].plot(mpc_y_error, linestyle='solid', linewidth=2, color='#005293', label='MPC_y Error')
axs3[1].set_title('Absolute Error of MPC_y - vehicle_position_y')
axs3[1].set(ylabel='Error')
axs3[1].grid(axis="both")
axs3[1].legend()

# Plot absolute error of MPC_v - vehicle_velocity_x
axs3[2].plot(mpc_v_error, linestyle='solid', linewidth=2, color='#005293', label='MPC_v Error')
axs3[2].set_title('Absolute Error of MPC_v - vehicle_velocity_x')
axs3[2].set(ylabel='Error')
axs3[2].grid(axis="both")
axs3[2].legend()

# Plot absolute error of MPC_yaw - vehicle_position_yaw
axs3[3].plot(mpc_yaw_error, linestyle='solid', linewidth=2, color='#005293', label='MPC_yaw Error')
axs3[3].set_title('Absolute Error of MPC_yaw - vehicle_position_yaw')
axs3[3].set(ylabel='Error')
axs3[3].grid(axis="both")
axs3[3].legend()

plt.show()
