import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import math

######################
# Variables to be set by user
######################

# Paths to define
results_path = '/home/glenn/BA/Diff_MPC_ws/postprocessing/results/diff_mpc_sim_result.csv' # Path to where the metrics per lap should be saved as csv
data_path = '/home/glenn/BA/Diff_MPC_ws/postprocessing/data/diff_mpc_sim.csv' # Path to the pandas dataframe saved as csv with .csv ending
raceline_path = '/home/glenn/BA/Diff_MPC_ws/maps/FTM_Halle.csv' # Path to the raceline with .csv ending
map_path = '/home/glenn/BA/Diff_MPC_ws/maps/FTM_Halle' #without ending

# Define laps you want to plot
selected_laps = [1, 3, 5]

# Initialize vehicle parameters
lf_veh = 0.15875 # distance from center of gravity to front axle
wheelbase = 0.324 # wheelbase of the vehicle

######################
# helper functions
######################
def p2p_dist (x1, y1, x2, y2):
    return np.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
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

def smoothing(arr, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")
    if len(arr) < window_size:
        raise ValueError("Array length must be at least as large as the window size.")
    kernel = np.ones(window_size) / window_size
    smoothed_arr = np.convolve(arr, kernel, mode='same')
    return smoothed_arr

# Load information of raceline
raceline_data = np.genfromtxt(raceline_path, delimiter=";", comments='#')
# Turn heading of raceline by 90 degrees to align with cooring system of the car
raceline_data[:, 3] = raceline_data[:, 3] - np.pi / 2
raceline_data[:, 3] = np.remainder(raceline_data[:, 3], 2 * np.pi)

data_df = pd.read_csv(data_path)

######################
######################
# Data Preprocessing
######################
######################

# Remove a large part of the data where the car is not moving
start_index = None
end_index = None
for index, value in enumerate(data_df['speed']):
    if value > 0 and start_index == None:
        start_index = max(index - 10, 0)
    elif value == 0 and start_index != None:
        end_index = index + 10
        break
if end_index == None or end_index == data_df.index[-1]:
    data_df = data_df[start_index:]
else:
    data_df = data_df[start_index : (end_index + 1)]

# Set timestamps as index
data_df.set_index('timestamp', inplace=True)
# Substracting first index so starting time is 0:00
data_df.index = data_df.index - data_df.index[0]  
# Change data types to numeric
data_df = data_df.apply(pd.to_numeric, errors='coerce')
original_x = data_df['x'].copy()
# Catch missing data in form of a column full of NaN values
for colum in data_df.columns:
    if data_df[colum].isna().all():
        data_df[colum] = 0.0
        print(f" WARNING: Column {colum} is completely NaN and was set to 0.0")
        
# Important: Interpolate before calulating ref indexes otherwise between max index and 0 of the ref index you get max/2 which is incorrect
data_df.interpolate(method='index', inplace=True)
data_df['x'] = original_x
# Drop all rows which have no original pf_pose data (Reducing size of DataFrame)
data_df.dropna(how='any', inplace=True)

# Create 'ref_index' column
data_df['ref_index'] = np.nan
# Calculating the index of raceline data of the closest point for each time step and saving it as reference index for calculating metrics
for index, row in data_df[data_df['x'].notna()].iterrows():
    data_df.loc[index, 'ref_index'] = np.argmin(p2p_dist(row['x'], row['y'], raceline_data[:,1], raceline_data[:,2]))
data_df['ref_index'] = data_df['ref_index'].astype(int)

######################
###################### 
# Calculate metrics
######################
######################

######################
# General calculations over the whole dataset
######################

# Calculate timestamps and metrics for optimum as defined in the raceline
optimal_track_progress_timedeltas = np.concatenate([np.array([0.0], dtype=float) ,np.diff(raceline_data[:, 0])]) / \
        (raceline_data[:,5] + np.concatenate([np.diff(raceline_data[:, 5]), np.array([0.0], dtype=float)]) / 2) # Calculate timedeltas for each ref index segement
optimal_track_progress_timestamps = np.cumsum(optimal_track_progress_timedeltas) # cumulative sum of timedeltas
steering_angle_optimal = np.arctan(wheelbase * raceline_data[:, 4]) # Calculate optimal steering angle
steer_rate_optimal = np.diff(steering_angle_optimal) / optimal_track_progress_timedeltas[1:] # Calculate optimal steering angle rate
steer_rate_optimal = np.concatenate([np.array([0.0], dtype=float) ,steer_rate_optimal]) # Add 0.0 as first value

# Accelerations
a_y = data_df['speed'] * data_df['angular_speed']
a_x = data_df['speed'].diff() / data_df.index.to_series().diff()
a_x.fillna(0.0, inplace=True)
steering_angle_rate = data_df['steering_angle'].diff() / data_df.index.to_series().diff()
steering_angle_rate.fillna(0.0, inplace=True)

# Pose data at front axle
x_front_axle = data_df['x'].values + lf_veh * np.sin(data_df['orientation'].values)
y_front_axle = data_df['y'].values + lf_veh * np.cos(data_df['orientation'].values)
theta_front_axle = np.remainder(data_df['orientation'].values + np.pi,  2 * np.pi)
# Cross Track Error
cross_track_err = [p2p_dist(x_front_axle[i],y_front_axle[i], raceline_data[idx, 1], raceline_data[idx, 2]) for i, idx in enumerate(data_df['ref_index'].values)]
# Heading Error
heading_err = [(raceline_data[idx, 3] - theta_front_axle[i]) for i, idx in enumerate(data_df['ref_index'].values)]
heading_err = np.remainder(np.array(heading_err, dtype=float) + np.pi, 2 * np.pi) - np.pi
# Convert cross_track_err and heading error into a pandas Series
cross_track_err = pd.Series(cross_track_err, index=data_df.index)
heading_err = pd.Series(heading_err, index=data_df.index)

######################
# Laptimes and metrics per lap
######################
# Lap Times
lap_timestamps = []
# Iterate through the DataFrame to identify start and end of laps
for i in range(1, len(data_df)):
    current_index = data_df['ref_index'].iloc[i]
    previous_index = data_df['ref_index'].iloc[i - 1]
    # Detect start of a lap
    # Detect if Rosbags starts with start of a lap
    if previous_index == 0 and i == 1:
        lap_timestamps.append(data_df.index[i-1])
    # Only detect laps with valid laptimes (3.0s minimum)
    elif current_index == 0 and previous_index > 1 and (len(lap_timestamps) == 0 or data_df.index[i] - lap_timestamps[-1] > 3.0):
        lap_timestamps.append(data_df.index[i])
# Calculate lap times
laptimes = np.diff(lap_timestamps)

# Calculate metrics per lap
cte_per_lap = []
cte_mean_per_lap= []
heading_err_per_lap = []
heading_err_mean_per_lap = []
v_avg_per_lap = []
v_max_per_lap = []
a_x_max_per_lap = []
a_x_min_per_lap = []
a_y_max_L_per_lap = []
a_y_max_R_per_lap = []
a_total_max_per_lap = []
steering_per_lap = []
steering_angle_rate_mean_per_lap = []

for i in range(len(lap_timestamps)-1):
    cte_per_lap.append(np.sum(cross_track_err.loc[lap_timestamps[i]:lap_timestamps[i+1]])) 
    cte_mean_per_lap.append(np.mean(cross_track_err.loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    heading_err_per_lap.append(np.sum(abs(heading_err.loc[lap_timestamps[i]:lap_timestamps[i+1]])))
    heading_err_mean_per_lap.append(np.mean(abs(heading_err.loc[lap_timestamps[i]:lap_timestamps[i+1]])))
    v_avg_per_lap.append(np.mean(data_df['speed'].loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    v_max_per_lap.append(np.max(data_df['speed'].loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    a_x_max_per_lap.append(np.max(a_x.loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    a_x_min_per_lap.append(np.min(a_x.loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    a_y_max_L_per_lap.append(np.max(a_y.loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    a_y_max_R_per_lap.append(np.min(a_y.loc[lap_timestamps[i]:lap_timestamps[i+1]]))
    a_total_max_per_lap.append(np.max(np.sqrt(a_x.loc[lap_timestamps[i]:lap_timestamps[i+1]]**2 + a_y.loc[lap_timestamps[i]:lap_timestamps[i+1]]**2)))
    steering_per_lap.append(np.sum(abs(data_df['steering_angle'].loc[lap_timestamps[i]:lap_timestamps[i+1]].diff().fillna(0))))
    steering_angle_rate_mean_per_lap.append(np.mean(abs(steering_angle_rate.loc[lap_timestamps[i]:lap_timestamps[i+1]])))
excess_steering_angle_per_lap = steering_per_lap - np.sum(abs(np.diff(steering_angle_optimal)))  

# Round decimals
laptimes = laptimes.round(2)
cte_per_lap = np.round(cte_per_lap, 2)
cte_mean_per_lap = np.round(cte_mean_per_lap, 4)
heading_err_per_lap = np.round(heading_err_per_lap, 2)
heading_err_mean_per_lap = np.round(heading_err_mean_per_lap, 4)
v_avg_per_lap = np.round(v_avg_per_lap, 4)
v_max_per_lap = np.round(v_max_per_lap, 4)
a_x_max_per_lap = np.round(a_x_max_per_lap, 4)
a_x_min_per_lap = np.round(a_x_min_per_lap, 4)
a_y_max_L_per_lap = np.round(a_y_max_L_per_lap, 4)
a_y_max_R_per_lap = np.round(a_y_max_R_per_lap, 4)
steering_per_lap = np.round(steering_per_lap, 4)
excess_steering_angle_per_lap = np.round(excess_steering_angle_per_lap, 4)
steering_angle_rate_mean_per_lap = np.round(steering_angle_rate_mean_per_lap, 4)

# Save metrics per lap to csv
metrics_df = pd.DataFrame({
    'Lap': range(1, len(laptimes) + 1),
    'Lap Time (s)': laptimes,
    'CTE (m)': cte_per_lap,
    'CTE Mean (m)': cte_mean_per_lap,
    'Heading Error (rad)': heading_err_per_lap,
    'Heading Error Mean (rad)': heading_err_mean_per_lap,
    'Average Speed (m/s)': v_avg_per_lap,
    'Max Speed (m/s)': v_max_per_lap,
    'Max Acceleration (m/s^2)': a_x_max_per_lap,
    'Max Braking (m/s^2)': a_x_min_per_lap,
    'Max Lateral Acceleration Left (m/s^2)': a_y_max_L_per_lap,
    'Max Lateral Acceleration Right (m/s^2)': a_y_max_R_per_lap,
    'Max Total Acceleration (m/s^2)': a_total_max_per_lap,
    'Steering (rad)': steering_per_lap,
    'Excessive Steering (rad)': excess_steering_angle_per_lap,
    'Mean Steering Angle Rate (rad/s)': steering_angle_rate_mean_per_lap
    
})
metrics_df.to_csv(results_path, index=False)

###############################
###############################
# Plotting from here on
###############################
###############################

# Load YAML data
with open(map_path + '.yaml', 'r') as yaml_file:
    yaml_dict = yaml.full_load(yaml_file)

origin = yaml_dict["origin"]
resolution = float(yaml_dict["resolution"])

# Load the PNG image (replace with your actual image path)
image = cv2.imread(map_path + '.png', cv2.IMREAD_GRAYSCALE)

# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = list(contours)
del contours[-1] # Delete the outermost contours since its simply a rectangle drawn around the whole image


speed_fig, speed_axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(12.8, 9.6))
speed_fig.canvas.manager.set_window_title('Speed and CTE and HER')
acc_data_fig =plt.figure(figsize=(14.8, 9.6))
acc_data_fig.canvas.manager.set_window_title('Acceleration Data')
acc_subfigs = acc_data_fig.subfigures(2,1, height_ratios=[1, 0.7])
acc_axs = acc_subfigs[0].subplots(1,2)
# Transform pixel coordinates to global coordinates and plot racetrack boundaries
for contour in contours:
    contours_coords_x = []
    contours_coords_y = []
    for point in contour:
        x, y = point[0]
        global_x = origin[0] + x * resolution  # Compute global x coordinate
        global_y = origin[1] + (image.shape[0] - y) * resolution  # Compute global y coordinate
        contours_coords_x.append(global_x)
        contours_coords_y.append(global_y)
    for i in range(speed_axs.shape[0]):
        for j in range(speed_axs.shape[1]):
            ax = speed_axs[i, j]  # Get the current Axes object
            ax.plot(contours_coords_x, contours_coords_y, color='black', linewidth=3)
    for i in range(acc_axs.shape[0]):
        ax = acc_axs[i]
        ax.plot(contours_coords_x, contours_coords_y, color='black', linewidth=3)

# Subplot with pf data
# Plot the pf pose data with the velocities as colormap
pf_data_ax = speed_axs[0,0]
pf_data_ax.set_aspect('equal')
for i in selected_laps:
    sc = pf_data_ax.scatter(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'], data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'y'], c=data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'speed'], \
        s=np.ones(len(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'])), cmap='plasma', alpha=0.7)
plt.colorbar(sc, label='Speed (m/s)', ax=pf_data_ax)  # Add colorbar to show the mapping of speeds to colors
pf_data_ax.plot(raceline_data[:,1], raceline_data[:,2], color='red', linewidth=1 , label="Raceline")
pf_data_ax.set_title("Trajectory with driven speed")
pf_data_ax.legend(loc='lower right')
pf_data_ax.axis('off') 

# Subplot with cross track error based on pf
cte_ax = speed_axs[0,1]
cte_ax.set_aspect('equal')
for i in selected_laps:
    sc = cte_ax.scatter(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'], data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'y'], c=cross_track_err.loc[lap_timestamps[i-1]:lap_timestamps[i]], \
        s=np.ones(len(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'])), cmap='plasma', alpha=0.7)
plt.colorbar(sc, label='Cross track error (m)', ax=cte_ax)  # Add colorbar to show the mapping of cross track error to colors
cte_ax.plot(raceline_data[:,1], raceline_data[:,2], color='red', linewidth=1 , label="Raceline")
cte_ax.set_title("Trajectory with cross track error")
cte_ax.legend(loc='lower right')
cte_ax.axis('off')

# Subplot with heading error base on pf
herr_ax = speed_axs[1,1]
herr_ax.set_aspect('equal')
for i in selected_laps:
    sc = herr_ax.scatter(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'], data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'y'], c=heading_err.loc[lap_timestamps[i-1]:lap_timestamps[i]], \
        s=np.ones(len(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'])), cmap='plasma', alpha=0.7)
plt.colorbar(sc, label='Heading error (rad)', ax=herr_ax)  # Add colorbar to show the mapping of heading error to colors
herr_ax.plot(raceline_data[:,1], raceline_data[:,2], color='red', linewidth=1 , label="Raceline")
herr_ax.set_title("Trajectory with heading error")
herr_ax.legend(loc='lower right')
herr_ax.axis('off')

# Subplot with speed heat map for optimal racing line
opt_speed_ax = speed_axs[1,0]
opt_speed_ax.set_aspect('equal')
sc = opt_speed_ax.scatter(raceline_data[:,1], raceline_data[:,2], c=raceline_data[:,5], \
    s=np.ones(len(raceline_data)) * 3, cmap='plasma', label=f"Optimal racing line", alpha=1.0)
plt.colorbar(sc, label='Speed (m/s)', ax=opt_speed_ax)  # Add colorbar to show the mapping of speed to colors
opt_speed_ax.set_title("Speed for optimal racing line")
opt_speed_ax.legend(loc='lower right')
opt_speed_ax.axis('off')


# Subplot with acceleration data
# Plot a_y over the trajectory
ay_data_ax = acc_axs[1]
ay_data_ax.set_aspect('equal')
for i in selected_laps:
    sc = ay_data_ax.scatter(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'], data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'y'], c=a_y.loc[lap_timestamps[i-1]:lap_timestamps[i]], \
        s=np.ones(len(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'])), cmap='plasma', alpha=0.7)
plt.colorbar(sc, label='a_y (m/s^2)', ax=ay_data_ax)  # Add colorbar to show the mapping of a_y to colors
ay_data_ax.plot(raceline_data[:,1], raceline_data[:,2], color='red', linewidth=1 , label="Raceline")
ay_data_ax.set_title("Trajectory with lateral acceleration")
ay_data_ax.legend(loc='lower right')
ay_data_ax.axis('off')

# Plot a_x over the trajectory
ax_data_ax = acc_axs[0]
ax_data_ax.set_aspect('equal')
for i in selected_laps:
    sc = ax_data_ax.scatter(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'], data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'y'], c=a_x.loc[lap_timestamps[i-1]:lap_timestamps[i]], \
        s=np.ones(len(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'x'])), cmap='plasma', alpha=0.7)
plt.colorbar(sc, label='a_x (m/s^2)', ax=ax_data_ax)  # Add colorbar to show the mapping of a_x to colors
ax_data_ax.plot(raceline_data[:,1], raceline_data[:,2], color='red', linewidth=1 , label="Raceline")
ax_data_ax.set_title("Trajectory with longitudinal acceleration")
ax_data_ax.legend(loc='lower right')
ax_data_ax.axis('off')


# Plot acceleration scatter with histograms
# Create the main axes, leaving 25% of the figure space at the top and on the right to position marginals.
ax = acc_subfigs[1].add_gridspec(top=0.75, right=0.75).subplots()
# The main axes' aspect can be fixed.
# Create marginal axes, which have 25% of the size of the main axes.  Note that the inset axes are positioned *outside* (on the right and the top) of the
# main axes, by specifying axes coordinates greater than 1.  Axes coordinates less than 0 would likewise specify positions on the left and the bottom of the main axes.
ax.set_xlabel('a_x')
ax.set_ylabel('a_y')
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the scatter plot and marginals with no labels
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
a_x_his_data = np.array([], dtype=float)
a_y_his_data = np.array([], dtype=float)
for i in selected_laps:
    a_x_his_data = np.concatenate((a_x_his_data, a_x.loc[lap_timestamps[i-1]:lap_timestamps[i]].values))
    a_y_his_data = np.concatenate((a_y_his_data, a_y.loc[lap_timestamps[i-1]:lap_timestamps[i]].values))
ax.scatter(a_x_his_data, a_y_his_data)
ax_histx.hist(a_x_his_data, bins=20)
ax_histy.hist(a_y_his_data, bins=20, orientation='horizontal')

# Plot profiles over driven distance
# Create figure and subplots
profile_fig, profile_axs = plt.subplots(2,3, figsize=(16.8, 9.6))
profile_fig.canvas.manager.set_window_title('Profiles')
v_profile_ax = profile_axs[0,0]
a_x_profile_ax = profile_axs[0,1]
distance_ax = profile_axs[0,2]
steer_rate_profile_ax = profile_axs[1,0]
smoothed_steer_rate_profile_ax = profile_axs[1,1]
track_prog_ax = profile_axs[1,2]

for i in selected_laps:
    # Calculate driven distance per lap
    driven_distance = ((data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'speed'] + data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'speed'].diff().fillna(0) / 2) *  \
        data_df.loc[lap_timestamps[i-1]:lap_timestamps[i]].index.to_series().diff().fillna(0)).cumsum()
    # Plot speed profile
    v_profile_ax.plot(driven_distance, data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'speed'], label=f"Lap {i}")
    # Plot a_x profile
    a_x_profile_ax.plot(driven_distance, a_x.loc[lap_timestamps[i-1]:lap_timestamps[i]], label=f"Lap {i}")
    # Plot steering angle rate
    steer_rate_profile_ax.plot(driven_distance, steering_angle_rate.loc[lap_timestamps[i-1]:lap_timestamps[i]], label=f"Lap {i}")
    # Plot smoothed steering angle rate
    smoothed_steer_rate_profile_ax.plot(driven_distance, smoothing(steering_angle_rate.loc[lap_timestamps[i-1]:lap_timestamps[i]], 9), label=f"Lap {i}")
    # Plot driven distance over time
    distance_ax.plot(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i]].index - lap_timestamps[i-1], driven_distance, label=f"Lap {i}")
    # Plot track progress over time
    track_prog_ax.plot(data_df.loc[lap_timestamps[i-1]:lap_timestamps[i]].index - lap_timestamps[i-1], data_df.loc[lap_timestamps[i-1]:lap_timestamps[i], 'ref_index'] / max(data_df['ref_index'].values) * 100, label=f"Lap {i}")   
    

# Plot precalculatet optimal values once
v_profile_ax.plot(raceline_data[:,0], raceline_data[:,5], label="Optimal")
a_x_profile_ax.plot(raceline_data[:,0], raceline_data[:,6], label="Optimal")
steer_rate_profile_ax.plot(raceline_data[:,0], steer_rate_optimal, label="Optimal")
smoothed_steer_rate_profile_ax.plot(raceline_data[:,0], steer_rate_optimal, label="Optimal")
distance_ax.plot(optimal_track_progress_timestamps, raceline_data[:,0], label="Optimal")
track_prog_ax.plot(optimal_track_progress_timestamps, np.arange(len(raceline_data), dtype=float) / (len(raceline_data) - 1) * 100, label="Optimal")
 # Set labels for profile plots 
v_profile_ax.set_title('Speed Profile')
v_profile_ax.set_xlabel('Distance Driven (m)')
v_profile_ax.set_ylabel('Speed (m/s)')
v_profile_ax.legend()
a_x_profile_ax.set_title('a_x Profile')
a_x_profile_ax.set_xlabel('Distance Driven (m)')
a_x_profile_ax.set_ylabel('Acceleration (m/s^2)')
a_x_profile_ax.legend()
steer_rate_profile_ax.set_title('Steering Angle Rate Profile')
steer_rate_profile_ax.set_xlabel('Distance Driven (m)')
steer_rate_profile_ax.set_ylabel('Steering Angle Rate (rad/s)')
steer_rate_profile_ax.legend()
smoothed_steer_rate_profile_ax.set_title('Smoothed steering Angle Rate Profile')
smoothed_steer_rate_profile_ax.set_xlabel('Distance Driven (m)')
smoothed_steer_rate_profile_ax.set_ylabel('Steering Angle Rate (rad/s)')
smoothed_steer_rate_profile_ax.legend()
distance_ax.set_xlabel('Time (s)')
distance_ax.set_ylabel('Distance Driven (m)')
distance_ax.set_title('Distance Driven Profile')
distance_ax.legend()
track_prog_ax.set_xlabel('Time (s)')
track_prog_ax.set_ylabel('Track Progress (%)')
track_prog_ax.set_title('Track Progress Profile')
track_prog_ax.legend()

plt.show()

