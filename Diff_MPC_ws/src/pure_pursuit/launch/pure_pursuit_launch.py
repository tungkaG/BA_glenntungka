from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()

    pure_pursuit_node = Node(
        package='pure_pursuit',
        executable='pure_pursuit',
        name='pure_pursuit_node',
        output='screen',
    )

    # finalize
    ld.add_action(pure_pursuit_node)

    return ld