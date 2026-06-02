#!/usr/bin/env python3
"""
pure_pursuit.py - Canonical Pure Pursuit guidance for Roboship USV

Geometric path-tracking algorithm based on Coulter (1992) [1]. Computes a
circular arc from the vehicle's current position to a goal point on the
desired path located at a fixed lookahead distance ahead.

This is the CANONICAL form: yaw rate is commanded as r = U * kappa, where
kappa is the arc curvature computed from the goal point's lateral offset
in the body frame (Coulter Eq. 2.4). The lookahead distance L_d is the
single primary tuning parameter, exactly as Coulter intended.

Goal-behind handling
--------------------
Canonical PP is ill-conditioned when the goal point falls in the rear
hemisphere (g_x < 0 in the body frame) — kappa becomes small or vanishes
even though the vehicle needs to rotate substantially. When this happens
(typically only at startup with an arbitrary initial heading), the node
switches to a fixed-rate pivot toward the goal side until the goal is
back in the forward hemisphere, then resumes canonical PP.

References
----------
[1] Coulter, R. C. (1992). "Implementation of the Pure Pursuit Path Tracking
    Algorithm." CMU Robotics Institute Technical Report CMU-RI-TR-92-01.
[2] Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and
    Mobile Robots." IJRR 5(1), 90-98.  (Boundary APF.)

Architecture
------------
Pattern B: publishes geometry_msgs/Twist on
/mavros/setpoint_velocity/cmd_vel_unstamped while in GUIDED mode.
ArduRover's inner turn-rate PID and speed controller execute the commands.
EKF3 (fed by Qualisys via /mavros/vision_pose/pose -> ExternalNav) provides
the state on /mavros/local_position/odom.

Uses the SAME inner-loop controller (ArduRover ATC_STR_RAT) as LOS and
ILOS, ensuring the comparison isolates the guidance law only.

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
    """Extract yaw from a geometry_msgs/Quaternion (ENU frame)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


# -----------------------------------------------------------------------------
# Main node
# -----------------------------------------------------------------------------

class PurePursuitGuidance(Node):
    def __init__(self):
        super().__init__('pure_pursuit_guidance')

        # ---- Guidance parameters -------------------------------------------
        # Base waypoints (one lap)
        self.declare_parameter('waypoints', [0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5])
        self.declare_parameter('laps', 3)

        # Surge speed [m/s]
        self.declare_parameter('U_desired', 0.20)
        self.declare_parameter('U_min', 0.05)

        # Lookahead distance [m]
        self.declare_parameter('lookahead_distance', 0.4)

        # Hard caps on yaw rates
        self.declare_parameter('r_max', 0.5236) # 30 deg/s
        self.declare_parameter('pivot_rate_frac', 0.8)

        # Mission parameters
        self.declare_parameter('acceptance_radius', 0.25)
        self.declare_parameter('braking_distance', 1.0)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('loop_mission', False)

        # ---- Polygon boundary parameters -----------------------------------
        default_boundary_file = '/home/roboship/ros2_ws/src/roboship_core/roboship_core/navigation_scripts/mocap_boundary.json'
        self.declare_parameter('boundary_file', default_boundary_file)
        self.declare_parameter('boundary_margin', 0.3)
        self.declare_parameter('boundary_eta', 0.3)

        # ---- Logging -------------------------------------------------------
        self.declare_parameter('log_csv', True)
        self.declare_parameter('log_dir', str(Path.home() / 'roboship_logs'))

        # ---- Load params ---------------------------------------------------
        wp_flat = list(self.get_parameter('waypoints').value)
        laps = int(self.get_parameter('laps').value)
        if len(wp_flat) < 4 or len(wp_flat) % 2 != 0:
            raise ValueError(f'waypoints must be even-length ≥4, got {len(wp_flat)}')

        base_wps = [(wp_flat[i], wp_flat[i + 1]) for i in range(0, len(wp_flat), 2)]
        self.waypoints = base_wps * laps
        self.n_base_wps = len(base_wps)

        self.U_d            = float(self.get_parameter('U_desired').value)
        self.U_min          = float(self.get_parameter('U_min').value)
        self.L_d            = float(self.get_parameter('lookahead_distance').value)
        self.r_max          = float(self.get_parameter('r_max').value)
        self.pivot_rate_frac = float(self.get_parameter('pivot_rate_frac').value)
        self.R_accept       = float(self.get_parameter('acceptance_radius').value)
        self.brake_dist     = float(self.get_parameter('braking_distance').value)
        self.rate_hz        = float(self.get_parameter('control_rate').value)
        self.loop_mode      = bool(self.get_parameter('loop_mission').value)

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
        else:
            self.get_logger().warn('Boundary disabled. Valid polygon data not found.')

        # ---- State ----------------------------------------------------------
        self.x = self.y = self.psi = 0.0
        self.u = self.v = 0.0
        self.have_state = False
        self.mode = ''
        self.armed = False

        self.wp_idx = 0
        self.waypoints_cleared = 0  # Counter ensuring we complete exact lap totals
        self.start_pose = None
        self.mission_complete = False
        self.last_time = self.get_clock().now()

        # ---- QoS for MAVROS topics -----------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- ROS interfaces -------------------------------------------------
        self.create_subscription(Odometry, '/mavros/local_position/odom', self._odom_cb, sensor_qos)
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # ---- CSV logging ----------------------------------------------------
        self.log_enabled = bool(self.get_parameter('log_csv').value)
        self.csv_file = None
        if self.log_enabled:
            log_dir = Path(self.get_parameter('log_dir').value)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = log_dir / f'pp_{ts}.csv'
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't', 'x', 'y', 'psi', 'u', 'v',
                'wp_idx', 'wp_target_x', 'wp_target_y',
                'goal_x', 'goal_y', 'goal_dist', 'g_x', 'g_y',
                'curvature', 'L_d',
                'psi_d', 'psi_err', 'U_cmd', 'r_cmd',
                'bound_fx', 'bound_fy', 'min_wall_dist',
            ])

        # ---- Control timer --------------------------------------------------
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        
        self.get_logger().info('')
        self.get_logger().info('=' * 55)
        self.get_logger().info('  PURE PURSUIT GUIDANCE')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  {len(base_wps)} waypoints × {laps} laps = {len(self.waypoints)} total')
        self.get_logger().info(f'  U={self.U_d} m/s  L_d={self.L_d} m  r_max={self.r_max} rad/s')
        self.get_logger().info('  Waiting for GUIDED + armed ...')
        self.get_logger().info('=' * 55)

    # ------------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------------

    def _state_cb(self, msg: State):
        new_mode = msg.mode
        if new_mode != self.mode:
            self.get_logger().info(f'Mode: {self.mode!r} -> {new_mode!r}')
            if new_mode == 'GUIDED':
                self._reset_mission()
            elif self.mode == 'GUIDED':
                self.get_logger().info('Left GUIDED. Sending zero command.')
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

    # ------------------------------------------------------------------------
    # Pure pursuit core math
    # ------------------------------------------------------------------------

    def _find_goal_point(self, wp_prev, wp_target):
        x1, y1 = wp_prev
        x2, y2 = wp_target
        dx, dy = x2 - x1, y2 - y1

        fx, fy = x1 - self.x, y1 - self.y
        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - self.L_d * self.L_d

        discriminant = b * b - 4.0 * a * c
        goal_x, goal_y = wp_target  

        if discriminant >= 0 and a > 1e-12:
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            t_goal = None
            if 0.0 <= t2 <= 1.0:
                t_goal = t2
            elif 0.0 <= t1 <= 1.0:
                t_goal = t1

            if t_goal is not None:
                goal_x = x1 + t_goal * dx
                goal_y = y1 + t_goal * dy

        return goal_x, goal_y

    def _pure_pursuit_curvature(self, goal_x, goal_y):
        dx_w = goal_x - self.x
        dy_w = goal_y - self.y
        goal_dist = math.hypot(dx_w, dy_w)
        psi_d = math.atan2(dy_w, dx_w)

        c, s = math.cos(self.psi), math.sin(self.psi)
        g_x =  c * dx_w + s * dy_w   
        g_y = -s * dx_w + c * dy_w   

        ell_sq = max(goal_dist * goal_dist, 0.01)
        curvature = 2.0 * g_y / ell_sq

        return curvature, psi_d, goal_dist, g_x, g_y

    # ------------------------------------------------------------------------
    # Mission management
    # ------------------------------------------------------------------------

    def _reset_mission(self):
        self.mission_complete = False
        self.waypoints_cleared = 0

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
            wx, wy = self.waypoints[closest_idx]
            self.get_logger().info(
                f'Mission reset. Nearest WP {closest_idx} '
                f'({wx:.2f}, {wy:.2f}), dist={min_dist:.2f} m')
        else:
            self.start_pose = None
            self.wp_idx = 0
            self.get_logger().info('Mission reset (no state yet, defaulting to WP 0).')

    def _advance_waypoint(self, reason=''):
        tag = f' ({reason})' if reason else ''
        wp = self.waypoints[self.wp_idx]
        
        # Calculate visual lap based on total WPs cleared so it counts up smoothly
        lap = self.waypoints_cleared // self.n_base_wps + 1
        wp_in_lap = self.waypoints_cleared % self.n_base_wps + 1
        
        self.get_logger().info(
            f'  ✓ WP {wp_in_lap}/{self.n_base_wps} lap {lap} '
            f'({wp[0]:.2f}, {wp[1]:.2f}){tag}')

        # Modulo wrapping ensures we loop around the array if we started in the middle
        self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
        self.waypoints_cleared += 1

        if self.waypoints_cleared >= len(self.waypoints):
            if self.loop_mode:
                self.waypoints_cleared = 0
                self.get_logger().info('  ↻ Looping mission')
            else:
                self.mission_complete = True
                self.get_logger().info('  ■ Mission complete')

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------------

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
        
        # Determine previous WP dynamically handling array wrap-around
        if self.waypoints_cleared == 0:
            wp_prev = self.start_pose
        else:
            wp_prev = self.waypoints[(self.wp_idx - 1) % len(self.waypoints)]

        dist_to_target = math.hypot(wp_target[0] - self.x, wp_target[1] - self.y)

        # Along-track overshoot check
        alpha_k = math.atan2(wp_target[1] - wp_prev[1], wp_target[0] - wp_prev[0])
        dx = self.x - wp_prev[0]
        dy = self.y - wp_prev[1]
        x_e = math.cos(alpha_k) * dx + math.sin(alpha_k) * dy
        seg_length = math.hypot(wp_target[0] - wp_prev[0], wp_target[1] - wp_prev[1])

        if dist_to_target < self.R_accept:
            self._advance_waypoint('radius')
            return
        if seg_length > 0.01 and x_e > seg_length:
            self._advance_waypoint('along-track')
            return

        # ---- Pure pursuit guidance -----------------------------------------
        goal_x, goal_y = self._find_goal_point(wp_prev, wp_target)
        curvature, psi_d, goal_dist, g_x, g_y = self._pure_pursuit_curvature(goal_x, goal_y)
        psi_err = wrap_angle(psi_d - self.psi)

        if g_x < 0.0:
            U_cmd = 0.0
            pivot_dir = 1.0 if g_y >= 0.0 else -1.0
            r_cmd = pivot_dir * self.pivot_rate_frac * self.r_max
        else:
            U_cmd = self.U_d
            # Decelerate if we are approaching the final WP of the entire mission
            is_final = (self.waypoints_cleared == len(self.waypoints) - 1) and not self.loop_mode
            if is_final and dist_to_target < self.brake_dist:
                U_cmd = max(self.U_d * (dist_to_target / self.brake_dist), self.U_min)
            r_cmd = U_cmd * curvature

        r_cmd = max(-self.r_max, min(self.r_max, r_cmd))

        # ---- Polygon boundary (APF) ----------------------------------------
        bound_fx = bound_fy = 0.0
        min_wall_dist = float('inf')
        if self.boundary is not None:
            U_cmd, r_cmd, min_wall_dist, bound_fx, bound_fy = \
                self.boundary.apply_to_command(self.x, self.y, self.psi, U_cmd, r_cmd)
            r_cmd = max(-self.r_max, min(self.r_max, r_cmd))
            
            if min_wall_dist <= 0.0:
                self.get_logger().warn(
                    f'OUTSIDE boundary! ({self.x:.2f}, {self.y:.2f}) recovering toward inward normal.')
            elif min_wall_dist < self.boundary.margin:
                self.get_logger().info(
                    f'APF active: dist={min_wall_dist:.2f} m, F=({bound_fx:.3f}, {bound_fy:.3f})')

        # ---- Publish -------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = float(U_cmd)
        cmd.angular.z = float(r_cmd)
        self.cmd_pub.publish(cmd)

        # ---- Log -----------------------------------------------------------
        if self.log_enabled and self.csv_file is not None:
            t_sec = now.nanoseconds * 1e-9
            self.csv_writer.writerow([
                f'{t_sec:.4f}',
                f'{self.x:.4f}', f'{self.y:.4f}', f'{self.psi:.4f}',
                f'{self.u:.4f}', f'{self.v:.4f}',
                self.wp_idx, f'{wp_target[0]:.2f}', f'{wp_target[1]:.2f}',
                f'{goal_x:.4f}', f'{goal_y:.4f}', f'{goal_dist:.4f}', 
                f'{g_x:.4f}', f'{g_y:.4f}',
                f'{curvature:.4f}', f'{self.L_d:.4f}',
                f'{psi_d:.4f}', f'{psi_err:.4f}', f'{U_cmd:.4f}', f'{r_cmd:.4f}',
                f'{bound_fx:.4f}', f'{bound_fy:.4f}', f'{min_wall_dist:.4f}',
            ])
            self.csv_file.flush()

    # ------------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitGuidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()