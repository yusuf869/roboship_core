#!/usr/bin/env python3
"""
los_guidance.py - Line-of-Sight guidance for Roboship USV

LOS baseline for comparison against ilos_guidance.py.
Includes an APF-based polygon boundary that replaces ArduRover's fence
(which requires GPS, >= 30 m radius, no indoor support).

References
----------
[1] Fossen, T. I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
    Control (2nd ed.), Ch. 12. Wiley.
[2] Lekkas, A. M., & Fossen, T. I. (2012). A time-varying lookahead distance
    guidance law for path following. IFAC Proceedings Volumes, 45(27).
[3] Khatib, O. (1986). Real-Time Obstacle Avoidance for Manipulators and
    Mobile Robots. IJRR, 5(1), 90-98.

Architecture: Pattern B (/mavros/setpoint_velocity/cmd_vel_unstamped).
Does NOT start MAVROS (roboship_core.service handles that).
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry

from roboship_core.navigation_scripts.boundary_apf import BoundaryPolygon


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# -----------------------------------------------------------------------------
# Main node
# -----------------------------------------------------------------------------

class LOSGuidance(Node):
    def __init__(self):
        super().__init__('los_guidance')

        # ---- Guidance parameters -------------------------------------------
        self.declare_parameter('waypoints',
                               [0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5] * 3)
        self.declare_parameter('U_desired', 0.20)
        self.declare_parameter('U_min', 0.05)
        self.declare_parameter('lookahead_min', 0.5)
        self.declare_parameter('lookahead_max', 1.0)
        self.declare_parameter('lookahead_gain', 1.0)
        self.declare_parameter('K_psi', 1.5)
        self.declare_parameter('acceptance_radius', 0.2)
        self.declare_parameter('braking_distance', 1.0)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('loop_mission', False)
        # Cap on commanded yaw rate (rad/s). Keeps the inner ATC_STR_RAT
        # loop out of saturation during large heading changes, which limits
        # the unwanted surge produced by the boat's asymmetric thrust.
        self.declare_parameter('r_max', 0.5236) # 30 deg/s

        # ---- Polygon boundary ----------------------------------------------
        default_boundary_file = '/home/roboship/ros2_ws/src/roboship_core/roboship_core/navigation_scripts/mocap_boundary.json'
        self.declare_parameter('boundary_file', default_boundary_file)
        self.declare_parameter('boundary_margin', 0.3)
        self.declare_parameter('boundary_eta', 0.3)

        # ---- Logging -------------------------------------------------------
        self.declare_parameter('log_csv', True)
        self.declare_parameter('log_dir', str(Path.home() / 'roboship_logs'))

        # ---- Load params ---------------------------------------------------
        wp_flat = self.get_parameter('waypoints').value
        if len(wp_flat) < 4 or len(wp_flat) % 2 != 0:
            raise ValueError(f'waypoints must be even-length, got {len(wp_flat)}')
        self.waypoints = [(wp_flat[i], wp_flat[i + 1])
                          for i in range(0, len(wp_flat), 2)]

        self.U_d        = float(self.get_parameter('U_desired').value)
        self.U_min      = float(self.get_parameter('U_min').value)
        self.D_min      = float(self.get_parameter('lookahead_min').value)
        self.D_max      = float(self.get_parameter('lookahead_max').value)
        self.D_gain     = float(self.get_parameter('lookahead_gain').value)
        self.K_psi      = float(self.get_parameter('K_psi').value)
        self.R_accept   = float(self.get_parameter('acceptance_radius').value)
        self.brake_dist = float(self.get_parameter('braking_distance').value)
        self.rate_hz    = float(self.get_parameter('control_rate').value)
        self.loop_mode  = bool(self.get_parameter('loop_mission').value)
        self.r_max      = float(self.get_parameter('r_max').value)

        # ---- Load JSON boundary file ---------------------------------------
        b_margin = float(self.get_parameter('boundary_margin').value)
        eta = float(self.get_parameter('boundary_eta').value)
        boundary_file_path = Path(self.get_parameter('boundary_file').value)
        self.boundary = None
        bv_flat = []

        if boundary_file_path.is_file():
            try:
                with open(boundary_file_path, 'r') as f:
                    data = json.load(f)
                    for pt in data.get('boundary_points', []):
                        bv_flat.extend([pt['x'], pt['y']])
                self.get_logger().info(
                    f'Loaded boundary coordinates from {boundary_file_path.name}')
            except Exception as e:
                self.get_logger().error(
                    f'Failed to parse {boundary_file_path.name}: {e}')
        else:
            self.get_logger().info(
                f'No boundary file found at {boundary_file_path}.')

        if len(bv_flat) >= 6 and len(bv_flat) % 2 == 0:
            verts = [(bv_flat[i], bv_flat[i + 1])
                     for i in range(0, len(bv_flat), 2)]
            self.boundary = BoundaryPolygon(verts, b_margin, eta=eta)
            self.get_logger().info(
                f'Boundary active: {len(verts)} vertices, '
                f'margin={b_margin:.2f} m, eta={eta:.2f}')

            for i, (wx, wy) in enumerate(self.waypoints):
                if not self.boundary.point_inside(wx, wy):
                    self.get_logger().warn(
                        f'WP {i} ({wx:.2f}, {wy:.2f}) OUTSIDE boundary!')
                else:
                    sd, _ = self.boundary.signed_distance(wx, wy)
                    if sd < b_margin:
                        self.get_logger().warn(
                            f'WP {i} ({wx:.2f}, {wy:.2f}) within '
                            f'{b_margin:.2f} m margin (dist={sd:.2f} m).')
        else:
            self.get_logger().warn(
                'Boundary disabled. Valid polygon data not found.')

        # ---- State ----------------------------------------------------------
        self.x = self.y = self.psi = 0.0
        self.u = self.v = 0.0
        self.have_state = False
        self.mode = ''
        self.armed = False

        self.wp_idx = 0
        self._first_wp_idx = 0
        self.start_pose = None
        self.mission_complete = False
        self.last_time = self.get_clock().now()

        # ---- ROS interfaces -------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            Odometry, '/mavros/local_position/odom', self._odom_cb, sensor_qos)
        self.create_subscription(
            State, '/mavros/state', self._state_cb, 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # ---- CSV logging ----------------------------------------------------
        self.log_enabled = bool(self.get_parameter('log_csv').value)
        self.csv_file = None
        if self.log_enabled:
            log_dir = Path(self.get_parameter('log_dir').value)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = log_dir / f'los_{ts}.csv'
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't', 'x', 'y', 'psi', 'u', 'v',
                'wp_idx', 'wp_target_x', 'wp_target_y',
                'alpha_k', 'x_e', 'y_e',
                'lookahead',
                'psi_d', 'psi_err', 'U_cmd', 'r_cmd',
                'bound_fx', 'bound_fy', 'min_wall_dist',
            ])
            self.get_logger().info(f'Logging to {self.csv_path}')

        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f'LOS started. {len(self.waypoints)} WPs, U_d={self.U_d} m/s, '
            f'r_max={self.r_max:.2f} rad/s')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _state_cb(self, msg: State):
        new_mode = msg.mode
        if new_mode != self.mode:
            self.get_logger().info(f'Mode: {self.mode!r} -> {new_mode!r}')
            if new_mode == 'GUIDED':
                self._reset_mission()
            elif self.mode == 'GUIDED':
                self.get_logger().info('Left GUIDED. Halting.')
                self._publish_zero()
        self.mode = new_mode
        self.armed = msg.armed

    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.psi = quat_to_yaw(msg.pose.pose.orientation)
        vx_w = msg.twist.twist.linear.x
        vy_w = msg.twist.twist.linear.y
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.u =  c * vx_w + s * vy_w
        self.v = -s * vx_w + c * vy_w
        self.have_state = True

    # -----------------------------------------------------------------------
    # LOS guidance (non-integral)
    # -----------------------------------------------------------------------

    def _path_errors(self, wp_prev, wp_target):
        xk, yk = wp_prev
        xk1, yk1 = wp_target
        alpha_k = math.atan2(yk1 - yk, xk1 - xk)
        dx = self.x - xk
        dy = self.y - yk
        c, s = math.cos(alpha_k), math.sin(alpha_k)
        x_e =  c * dx + s * dy
        y_e = -s * dx + c * dy
        return alpha_k, x_e, y_e

    def _adaptive_lookahead(self, y_e: float) -> float:
        return ((self.D_max - self.D_min)
                * math.exp(-self.D_gain * abs(y_e))
                + self.D_min)

    def _los_desired_heading(self, alpha_k, y_e, Delta) -> float:
        """Standard LOS: psi_d = alpha_k - atan2(y_e, Delta).
        No integral term -> nonzero steady-state y_e under disturbance."""
        return alpha_k - math.atan2(y_e, Delta)

    # -----------------------------------------------------------------------
    # Mission management
    # -----------------------------------------------------------------------

    def _reset_mission(self):
        self.mission_complete = False

        if self.have_state:
            self.start_pose = (self.x, self.y)
            min_dist = float('inf')
            closest_idx = 0
            for i, (wx, wy) in enumerate(self.waypoints):
                d = math.hypot(wx - self.x, wy - self.y)
                if d < min_dist:
                    min_dist = d
                    closest_idx = i
            self.wp_idx = closest_idx
            self._first_wp_idx = closest_idx
            self.get_logger().info(
                f'Mission reset. Nearest WP {closest_idx} '
                f'({self.waypoints[closest_idx][0]:.2f}, '
                f'{self.waypoints[closest_idx][1]:.2f}), '
                f'dist={min_dist:.2f} m')
        else:
            self.start_pose = None
            self.wp_idx = 0
            self._first_wp_idx = 0
            self.get_logger().info(
                'Mission reset (no state yet, defaulting to WP 0).')

    def _advance_waypoint(self):
        self.wp_idx += 1
        if self.wp_idx >= len(self.waypoints):
            if self.loop_mode:
                self.wp_idx = 0
                self.get_logger().info('Looping to WP 0.')
            else:
                self.mission_complete = True
                self.get_logger().info('Mission complete.')

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    # -----------------------------------------------------------------------
    # Main control loop
    # -----------------------------------------------------------------------

    def _control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        if not self.have_state:
            return
        if self.mode != 'GUIDED' or not self.armed:
            self._publish_zero()
            return
        if self.mission_complete:
            self._publish_zero()
            return
        if self.start_pose is None:
            self.start_pose = (self.x, self.y)

        wp_target = self.waypoints[self.wp_idx]
        wp_prev = (self.start_pose if self.wp_idx == self._first_wp_idx
                   else self.waypoints[self.wp_idx - 1])

        dist_to_target = math.hypot(wp_target[0] - self.x,
                                    wp_target[1] - self.y)

        alpha_k, x_e, y_e = self._path_errors(wp_prev, wp_target)

        if dist_to_target < self.R_accept:
            self.get_logger().info(
                f'Reached WP {(self.wp_idx % 4) + 1} '
                f'({wp_target[0]:.2f}, {wp_target[1]:.2f})')
            self._advance_waypoint()
            return

        # ---- LOS guidance --------------------------------------------------
        Delta = self._adaptive_lookahead(y_e)
        psi_d = self._los_desired_heading(alpha_k, y_e, Delta)
        psi_err = wrap_angle(psi_d - self.psi)

        # ---- Yaw rate command, rate-limited --------------------------------
        # K_psi * psi_err can exceed what the boat can physically deliver
        # (~2 rad/s in our case). Asking for more just pins the ATC_STR_RAT
        # loop in saturation, which under asymmetric thrust produces large
        # unwanted surge during heading changes. Cap it.
        r_cmd = self.K_psi * psi_err
        r_cmd = max(-self.r_max, min(self.r_max, r_cmd))

        # ---- Surge command, smooth cos taper -------------------------------
        # No hard pivot-in-place threshold: the boat cannot do a pure pivot
        # (asymmetric forward/reverse thrust => residual surge whenever the
        # motors are differential), so commanding U=0 during a turn is a lie.
        # Letting the boat curve into the waypoint matches the achievable
        # dynamics. cos(psi_err) tapers to 0 by |psi_err|=90 deg, so the
        # boat still effectively rotates first when very off-heading.
        U_cmd = self.U_d * max(0.0, math.cos(psi_err))

        # Final-waypoint braking.
        is_final = (self.wp_idx == len(self.waypoints) - 1) and not self.loop_mode
        if is_final and dist_to_target < self.brake_dist:
            U_brake = max(self.U_d * (dist_to_target / self.brake_dist),
                          self.U_min)
            U_cmd = min(U_cmd, U_brake)

        # ---- Polygon boundary (APF) ----------------------------------------
        bound_fx = bound_fy = 0.0
        min_wall_dist = float('inf')
        if self.boundary is not None:
            U_cmd, r_cmd, min_wall_dist, bound_fx, bound_fy = \
                self.boundary.apply_to_command(
                    self.x, self.y, self.psi, U_cmd, r_cmd)
            if min_wall_dist <= 0.0:
                self.get_logger().warn(
                    f'OUTSIDE boundary! ({self.x:.2f}, {self.y:.2f})')
            elif min_wall_dist < self.boundary.margin:
                self.get_logger().info(
                    f'APF active: dist={min_wall_dist:.2f} m, '
                    f'F=({bound_fx:.3f}, {bound_fy:.3f})')

        # ---- Publish -------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = float(U_cmd)
        cmd.angular.z = float(r_cmd)
        self.cmd_pub.publish(cmd)

        # ---- Log -----------------------------------------------------------
        if self.log_enabled and self.csv_file is not None:
            t_sec = now.nanoseconds * 1e-9
            self.csv_writer.writerow([
                t_sec, self.x, self.y, self.psi, self.u, self.v,
                self.wp_idx, wp_target[0], wp_target[1],
                alpha_k, x_e, y_e,
                Delta,
                psi_d, psi_err, U_cmd, r_cmd,
                bound_fx, bound_fy, min_wall_dist,
            ])
            self.csv_file.flush()

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    def destroy_node(self):
        try:
            self._publish_zero()
        except Exception:
            pass
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = LOSGuidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()