#!/usr/bin/env python3
"""
power_logger.py – ROS2 node that logs battery current, voltage,
and RC throttle (RC3) to a timestamped CSV file.

Usage:
    ros2 run <your_package> power_logger
    # or just:
    python3 power_logger.py

Subscribes to:
    /mavros/battery   (sensor_msgs/BatteryState)
    /mavros/rc/in     (mavros_msgs/RCIn)

Outputs:
    ~/power_logs/power_log_<timestamp>.csv
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import BatteryState
from mavros_msgs.msg import RCIn

import csv
import os
from datetime import datetime


class PowerLogger(Node):
    def __init__(self):
        super().__init__("power_logger")

        # ── tunables ──
        self.declare_parameter("throttle_channel", 2)  # 0-indexed, RC3 = index 2
        self.declare_parameter("log_dir", os.path.expanduser("~/power_logs"))

        self.throttle_ch = (
            self.get_parameter("throttle_channel").get_parameter_value().integer_value
        )
        log_dir = (
            self.get_parameter("log_dir").get_parameter_value().string_value
        )

        # ── set up CSV ──
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"power_log_{stamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            ["elapsed_s", "voltage_V", "current_A", "throttle_us"]
        )

        self.get_logger().info(f"Logging to {self.csv_path}")

        # ── state ──
        self.t0 = None  # set on first message
        self.latest_throttle_us = None

        # ── QoS profile to match MAVROS ──
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── subscribers ──
        self.create_subscription(BatteryState, "/mavros/battery", self.battery_cb, mavros_qos)
        self.create_subscription(RCIn, "/mavros/rc/in", self.rc_cb, mavros_qos)

    # ── callbacks ──
    def battery_cb(self, msg: BatteryState):
        now = self.get_clock().now()
        if self.t0 is None:
            self.t0 = now

        elapsed = (now - self.t0).nanoseconds * 1e-9
        voltage = msg.voltage
        current = msg.current  # amps (may be negative depending on driver)

        self.writer.writerow(
            [
                f"{elapsed:.3f}",
                f"{voltage:.3f}",
                f"{abs(current):.3f}",
                self.latest_throttle_us if self.latest_throttle_us is not None else "",
            ]
        )
        self.csv_file.flush()

    def rc_cb(self, msg: RCIn):
        if self.throttle_ch < len(msg.channels):
            self.latest_throttle_us = msg.channels[self.throttle_ch]

    # ── cleanup ──
    def destroy_node(self):
        self.csv_file.close()
        print(f"[power_logger] CSV saved: {self.csv_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PowerLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()