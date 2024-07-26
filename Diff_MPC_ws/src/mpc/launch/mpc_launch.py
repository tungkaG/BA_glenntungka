import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='mpc',
            executable='MPC_Felix',
            name='MPC_Felix',
            output='screen'),
  ])