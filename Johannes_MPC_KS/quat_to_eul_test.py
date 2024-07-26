import rclpy
from geometry_msgs.msg import Quaternion
from tf_transformations import euler_from_quaternion

# Assume you have a Quaternion message
quaternion_msg = Quaternion()
quaternion_msg.w = 0.609
quaternion_msg.x = 0.0
quaternion_msg.y = 0.0
quaternion_msg.z = 0.792

# Convert the Quaternion message to a list
quaternion = [
    quaternion_msg.x,
    quaternion_msg.y,
    quaternion_msg.z,
    quaternion_msg.w
]

# Convert the quaternion to Euler angles
euler = euler_from_quaternion(quaternion)

# euler is a tuple (roll, pitch, yaw)
roll = euler[0]
pitch = euler[1]
yaw = euler[2]

print("Roll: ", roll)
print("Pitch: ", pitch)
print("Yaw: ", yaw)

# def callback(msg):
#     # Convert the Quaternion message to a list
#     quaternion = [
#         msg.pose.pose.orientation.x,
#         msg.pose.pose.orientation.y,
#         msg.pose.pose.orientation.z,
#         msg.pose.pose.orientation.w
#     ]
#     # Convert the quaternion to Euler angles
#     euler = euler_from_quaternion(quaternion)
#     # euler is a tuple (roll, pitch, yaw)
#     roll = euler[0]
#     pitch = euler[1]
#     yaw = euler[2]
#     print("Roll: ", roll)
#     print("Pitch: ", pitch)
#     print("Yaw: ", yaw)

#     # Publish the euler angle on the /debug topic
#     euler_msg = Quaternion()
#     euler_msg.x = roll
#     euler_msg.y = pitch
#     euler_msg.z = yaw
#     euler_msg.w = 0.0  # Not used in this case
#     debug_publisher.publish(euler_msg)


# def main():
#     rclpy.init()
#     node = rclpy.create_node('odom_listener')

#     # Create a subscriber to the /ego_racecar/odom topic
#     odom_subscriber = node.create_subscription(
#         Odometry,
#         '/ego_racecar/odom',
#         callback
#     )

#     # Create a publisher for the /debug topic
#     debug_publisher = node.create_publisher(
#         Quaternion,
#         '/debug',
#         1
#     )

#     rclpy.spin(node)

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

