#!/usr/bin/env python3
"""
zn_tuning_logger.py - Log yaw rate and surge velocity for Ziegler-Nichols PID tuning

Logs the data needed to find K_cr (critical gain) and T_cr (oscillation period)
for both ArduRover controllers:

  ATC_STR_RAT (steering rate PID):
    - Observe yaw rate (r) over time
    - Increase ATC_STR_RAT_P until sustained oscillation in r
    - Measure K_cr = that P value, T_cr = oscillation period from the log

  ATC_SPEED (speed PID):
    - Observe surge velocity (u) over time
    - Increase ATC_SPEED_P until sustained oscillation in u
    - Measure K_cr = that P value, T_cr = oscillation period from the log

Run this ALONGSIDE your navigation node (CalmWaterNav, LOS, or ILOS).
It only logs — it doesn't command anything.

Also useful for the Nomoto turning circle test: command a step yaw rate,
log the response, fit T and K_boat from the yaw rate curve.

Usage:
  ros2 run roboship zn_tuning_logger
  # or standalone:
  python3 zn_tuning_logger.py
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from mavros_msgs.msg import State


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ZNTuningLogger(Node):
    def __init__(self):
        super().__init__('zn_tuning_logger')

        self.declare_parameter('log_dir', str(Path.home() / 'roboship_logs'))
        self.declare_parameter('log_rate', 50.0)  # Hz, how fast to sample

        log_dir = Path(self.get_parameter('log_dir').value)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = log_dir / f'zn_tuning_{ts}.csv'

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't',                    # timestamp (s)
            'x', 'y',              # position (ENU, m)
            'psi',                 # heading (rad)
            'u', 'v',             # body-frame surge, sway velocity (m/s)
            'r_actual',           # actual yaw rate from EKF (rad/s)
            'speed',              # total ground speed (m/s)
            'mode', 'armed',      # flight mode and arm state
        ])

        # State
        self.x = self.y = self.psi = 0.0
        self.u = self.v = 0.0
        self.r_actual = 0.0
        self.speed = 0.0
        self.mode = ''
        self.armed = False
        self.have_odom = False
        self.t0 = None

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        self.create_subscription(
            Odometry, '/mavros/local_position/odom', self._odom_cb, sensor_qos)
        self.create_subscription(
            State, '/mavros/state', self._state_cb, 10)

        # Logging timer
        rate = float(self.get_parameter('log_rate').value)
        self.timer = self.create_timer(1.0 / rate, self._log_tick)

        self.get_logger().info(f'ZN tuning logger -> {self.csv_path}')
        self.get_logger().info(
            'Columns: t, x, y, psi, u (surge), v (sway), '
            'r_actual (yaw rate), speed, mode, armed')

    def _state_cb(self, msg: State):
        self.mode = msg.mode
        self.armed = msg.armed

    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.psi = quat_to_yaw(msg.pose.pose.orientation)

        # World-frame velocity -> body frame
        vx_w = msg.twist.twist.linear.x
        vy_w = msg.twist.twist.linear.y
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.u =  c * vx_w + s * vy_w
        self.v = -s * vx_w + c * vy_w
        self.speed = math.sqrt(vx_w * vx_w + vy_w * vy_w)

        # Yaw rate directly from EKF (more reliable than differentiating psi)
        self.r_actual = msg.twist.twist.angular.z

        self.have_odom = True

    def _log_tick(self):
        if not self.have_odom:
            return

        now = self.get_clock().now()
        if self.t0 is None:
            self.t0 = now

        t_sec = (now - self.t0).nanoseconds * 1e-9

        self.csv_writer.writerow([
            f'{t_sec:.4f}',
            f'{self.x:.4f}', f'{self.y:.4f}',
            f'{self.psi:.4f}',
            f'{self.u:.4f}', f'{self.v:.4f}',
            f'{self.r_actual:.4f}',
            f'{self.speed:.4f}',
            self.mode,
            int(self.armed),
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file is not None:
            try:
                self.csv_file.close()
                self.get_logger().info(f'Log saved: {self.csv_path}')
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZNTuningLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()