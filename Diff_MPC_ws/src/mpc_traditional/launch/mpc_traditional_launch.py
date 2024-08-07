import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='mpc_traditional',
            executable='MPC_traditional',
            name='MPC_traditional',
            output='screen'),
  ])