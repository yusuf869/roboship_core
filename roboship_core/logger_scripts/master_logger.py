#!/usr/bin/env python3
"""
Roboship Combined Logger
────────────────────────
Usage:
    ros2 run roboship_core logger
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

# These imports still look at the parent module
from roboship_core.logger_scripts.path_logger import PathLogger
from roboship_core.logger_scripts.power_logger import PowerLogger

def main(args=None):
    rclpy.init(args=args)

    path_node = PathLogger()
    power_node = PowerLogger()

    executor = MultiThreadedExecutor()
    executor.add_node(path_node)
    executor.add_node(power_node)

    try:
        path_node.get_logger().info("Starting combined loggers from logger_scripts...")
        executor.spin()
    except KeyboardInterrupt:
        path_node.get_logger().info("Keyboard interrupt, shutting down loggers...")
    finally:
        path_node.destroy_node()
        power_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()