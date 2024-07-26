# import rclpy
# from rclpy.node import Node
# import pickle
# import time

# logger = []

# class HelloWorldNode(Node):
#     def __init__(self):
#         super().__init__('hello_world_node')
#         self.timer = self.create_timer(1.0, self.log_hello_world)

#     def log_hello_world(self):
#         current_time = time.strftime("%H:%M:%S", time.localtime())
#         logger.append(current_time)
#         self.get_logger().info(current_time)

#     def on_shutdown(self):
#         self.get_logger().info('Shutting down')
#         with open('src/mpc/mpc/hello_world.pickle', 'wb') as file:
#             pickle.dump(logger, file)

# def main(args=None):
#     global logger

#     rclpy.init(args=args)
#     node = HelloWorldNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
import signal
import time
import pickle

class Datalogger:
    def __init__(self):
        self.data = []
        self.data2 = []

    def logging(self,datapoint):
        self.data.append(datapoint)
        self.data2.append('Hello World')

class MyNode(Node):
    def __init__(self):
        super().__init__('hello_world_node')
        self.timer = self.create_timer(1.0, self.log_hello_world)
        self.logger = Datalogger()

    def log_hello_world(self):
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.logger.logging(current_time)
        self.get_logger().info(current_time)

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        with open('postprocessing/hello_world.p', 'wb') as file:
            pickle.dump(self.logger, file)
        self.get_logger().info("Done")

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Ctrl-C caught, shutting down.")
        node.cleanup()
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
    main()
