#!/usr/bin/env python3
"""
ilos_guidance.py — Integral Line-of-Sight guidance for Roboship USV
═══════════════════════════════════════════════════════════════════

References
----------
[1] Lekkas & Fossen (2014). "Integral LOS Path Following for Curved Paths
    Based on a Constant Bearing Guidance Law." IEEE TCST 22(6), 2287–2301.
[2] Fossen, Pettersen & Galeazzi (2015). "Line-of-sight path following for
    Dubins paths with adaptive sideslip compensation." IEEE TCST 23(2).
[3] Lekkas & Fossen (2012). "A time-varying lookahead distance guidance law
    for path following." IFAC Proceedings 45(27).
[4] Fossen (2021). Handbook of Marine Craft Hydrodynamics, 2nd ed, Ch. 12.
[5] Khatib (1986). "Real-Time Obstacle Avoidance for Manipulators and
    Mobile Robots." IJRR 5(1), 90-98.  (Boundary APF.)

Architecture
------------
Publishes geometry_msgs/Twist on /mavros/setpoint_velocity/cmd_vel_unstamped
while in GUIDED mode. ArduRover's inner speed + yaw-rate loops execute the
commands. EKF3 (fed by Qualisys → ExternalNav) provides the state on
/mavros/local_position/odom.

Set `enable_integral:=false` to run as plain LOS (no sideslip estimator)
for a clean A/B comparison in your dissertation.

Mathematical formulation
------------------------
Path segment from waypoint k to k+1:

  α_k  = atan2(y_{k+1} − y_k, x_{k+1} − x_k)            path tangent
  x_e  =  (x−x_k)·cos α_k + (y−y_k)·sin α_k             along-track
  y_e  = −(x−x_k)·sin α_k + (y−y_k)·cos α_k             cross-track

  Δ(y_e) = (Δ_max − Δ_min)·exp(−k_d·|y_e|) + Δ_min      adaptive lookahead

  ψ_d = α_k − atan2(y_e + β̂, Δ)                          desired heading
  dβ̂/dt = κ·Δ·y_e / √(Δ² + (y_e + β̂)²)                  sideslip estimator

  r_cmd = clip( K_ψ · wrap(ψ_d − ψ), ±r_max )           yaw rate command
  U_cmd = U_d · max(0, cos(ψ_err))                       surge with cos taper

Waypoint advancement: distance-based OR along-track overflow (x_e > seg_len)
to prevent the boat sailing past a waypoint it missed laterally.

Boundary
--------
A polygon boundary is loaded from mocap_boundary.json and enforced via the
APF in boundary_apf.py. Inside the margin band the repulsive force is added
to (U_cmd, r_cmd). Outside the polygon the APF takes over and steers the
vessel back inside.

Prerequisites
-------------
  • MAVROS running (roboship_core.service)
  • Mocap pipeline active (Qualisys → motion_capture_tracking → relay)
  • SYSID_MYGCS = 1
  • EKF3 sources = ExternalNav
  • Switch to GUIDED mode and arm on transmitter

Usage
-----
  ros2 run roboship_core ilos_guidance

  # As plain LOS (disable integral term):
  ros2 run roboship_core ilos_guidance --ros-args -p enable_integral:=false

  # Custom waypoints:
  ros2 run roboship_core ilos_guidance --ros-args \
      -p waypoints:=[1.0,0.0,0.0,1.0,-1.0,0.0,0.0,-1.0]
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_yaw(q) -> float:
    """Extract yaw from a geometry_msgs/Quaternion (ENU frame)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def wrap(a: float) -> float:
    """Wrap angle to [−π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class ILOSGuidance(Node):
    def __init__(self):
        super().__init__('ilos_guidance')

        # ── Declare parameters ─────────────────────────────────────────
        self.declare_parameter('waypoints',
                               [0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5])
        self.declare_parameter('laps', 3)
        self.declare_parameter('U_desired', 0.20)          # m/s cruise speed
        self.declare_parameter('U_min', 0.05)              # m/s crawl speed
        self.declare_parameter('lookahead_min', 0.5)       # Δ_min  (m)
        self.declare_parameter('lookahead_max', 1.0)       # Δ_max  (m)
        self.declare_parameter('lookahead_gain', 1.0)      # k_d    (1/m)
        # Integral term: increased from old default (0.05) so the integral
        # actually converges within a single 1-m segment. Lekkas & Fossen
        # (2014) suggest κ ~ U_d / Δ_max as an order-of-magnitude scale.
        self.declare_parameter('kappa', 0.15)              # integral gain
        # Saturate β̂ at 30 cm — needs to be larger than the empirical
        # cross-track bias (≈20 cm seen on previous LOS runs) so the
        # integral isn't clamped before it can fully compensate.
        self.declare_parameter('y_int_sat', 0.30)          # anti-windup (m)
        # K_psi = 1.5 matches LOS; the old default of 2.5 was above the
        # cascaded-loop stability bound and produced wobble.
        self.declare_parameter('K_psi', 1.5)               # heading P-gain
        # Hard cap on yaw rate command — see LOS for rationale.
        self.declare_parameter('r_max', 0.5236)            # 30 deg/s in rad/s
        # Bumped slightly above the empirical 20 cm CTE to comfortably
        # catch WPs during the first segment before β̂ has converged.
        self.declare_parameter('acceptance_radius', 0.25)  # WP capture (m)
        self.declare_parameter('braking_distance', 1.0)    # final WP ramp (m)
        self.declare_parameter('control_rate', 20.0)       # Hz
        self.declare_parameter('loop_mission', True)
        self.declare_parameter('enable_integral', True)    # False = plain LOS

        # ── Boundary parameters ────────────────────────────────────────
        default_boundary_file = '/home/roboship/ros2_ws/src/roboship_core/roboship_core/navigation_scripts/mocap_boundary.json'
        self.declare_parameter('boundary_file', default_boundary_file)
        self.declare_parameter('boundary_margin', 0.3)
        self.declare_parameter('boundary_eta', 0.3)

        # ── Logging ────────────────────────────────────────────────────
        self.declare_parameter('log_csv', True)
        self.declare_parameter('log_dir', str(Path.home() / 'roboship_logs'))

        # ── Read parameters ────────────────────────────────────────────
        wp_flat = list(self.get_parameter('waypoints').value)
        laps = int(self.get_parameter('laps').value)
        if len(wp_flat) < 4 or len(wp_flat) % 2 != 0:
            raise ValueError(f'waypoints must be even-length ≥4, got {len(wp_flat)}')

        base_wps = [(wp_flat[i], wp_flat[i + 1])
                    for i in range(0, len(wp_flat), 2)]
        self.waypoints = base_wps * laps
        self.n_base_wps = len(base_wps)

        self.U_d           = float(self.get_parameter('U_desired').value)
        self.U_min         = float(self.get_parameter('U_min').value)
        self.D_min         = float(self.get_parameter('lookahead_min').value)
        self.D_max         = float(self.get_parameter('lookahead_max').value)
        self.D_gain        = float(self.get_parameter('lookahead_gain').value)
        self.kappa         = float(self.get_parameter('kappa').value)
        self.y_int_sat     = float(self.get_parameter('y_int_sat').value)
        self.K_psi         = float(self.get_parameter('K_psi').value)
        self.r_max         = float(self.get_parameter('r_max').value)
        self.R_accept      = float(self.get_parameter('acceptance_radius').value)
        self.brake_dist    = float(self.get_parameter('braking_distance').value)
        self.rate_hz       = float(self.get_parameter('control_rate').value)
        self.loop_mode     = bool(self.get_parameter('loop_mission').value)
        self.integral_on   = bool(self.get_parameter('enable_integral').value)

        # ── Load polygon boundary ──────────────────────────────────────
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
            self.get_logger().warn(
                'Boundary disabled. Valid polygon data not found.')

        # ── State ──────────────────────────────────────────────────────
        self.x = self.y = self.psi = 0.0
        self.u = self.v = 0.0
        self.have_state = False

        self.mode = ''
        self.armed = False

        self.wp_idx = 0
        self.waypoints_cleared = 0  # Counter ensuring we complete exact lap totals
        self.y_int = 0.0            # sideslip estimator  β̂
        self.start_pose = None
        self.mission_complete = False

        self.last_time = self.get_clock().now()

        # ── QoS (matches mocap / MAVROS defaults) ─────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── Subscriptions ─────────────────────────────────────────────
        self.create_subscription(Odometry, '/mavros/local_position/odom', self._odom_cb, sensor_qos)
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # ── CSV logging ───────────────────────────────────────────────
        self.csv_file = None
        if bool(self.get_parameter('log_csv').value):
            log_dir = Path(self.get_parameter('log_dir').value)
            log_dir.mkdir(parents=True, exist_ok=True)
            tag = 'ilos' if self.integral_on else 'los'
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = log_dir / f'{tag}_{ts}.csv'
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't', 'x', 'y', 'psi', 'u', 'v',
                'wp_idx', 'wp_target_x', 'wp_target_y',
                'alpha_k', 'x_e', 'y_e', 'seg_len',
                'lookahead', 'y_int',
                'psi_d', 'psi_err', 'U_cmd', 'r_cmd',
                'bound_fx', 'bound_fy', 'min_wall_dist',
            ])

        # ── Control timer ─────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)

        mode_str = 'ILOS' if self.integral_on else 'LOS (integral disabled)'
        self.get_logger().info('')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  {mode_str} GUIDANCE')
        self.get_logger().info('=' * 55)
        self.get_logger().info(
            f'  {len(base_wps)} waypoints × {laps} laps = '
            f'{len(self.waypoints)} total')
        self.get_logger().info(
            f'  U={self.U_d} m/s  K_ψ={self.K_psi}  r_max={self.r_max} rad/s')
        self.get_logger().info(
            f'  κ={self.kappa}  y_int_sat={self.y_int_sat} m  '
            f'R_accept={self.R_accept} m')
        self.get_logger().info(
            f'  Δ=[{self.D_min}, {self.D_max}] m')
        self.get_logger().info('  Waiting for GUIDED + armed ...')
        self.get_logger().info('=' * 55)

    # ─────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────

    def _state_cb(self, msg: State):
        new_mode = msg.mode
        if new_mode != self.mode:
            self.get_logger().info(f'Mode: {self.mode!r} → {new_mode!r}')
            if new_mode == 'GUIDED':
                self._reset_mission()
            elif self.mode == 'GUIDED':
                self.get_logger().info('Left GUIDED — stopping.')
                self._send_zero()
        self.mode = new_mode
        self.armed = msg.armed

    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.psi = quat_to_yaw(msg.pose.pose.orientation)
        vx_w = msg.twist.twist.linear.x
        vy_w = msg.twist.twist.linear.y
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.u =  c * vx_w + s * vy_w     # body surge
        self.v = -s * vx_w + c * vy_w     # body sway
        self.have_state = True

    # ─────────────────────────────────────────────────────────────────
    # ILOS math  (Lekkas & Fossen 2014)
    # ─────────────────────────────────────────────────────────────────

    def _path_errors(self, wp_prev, wp_target):
        """Compute path tangent α_k, along-track x_e, cross-track y_e,
        and segment length."""
        xk, yk = wp_prev
        xk1, yk1 = wp_target
        alpha_k = math.atan2(yk1 - yk, xk1 - xk)

        dx = self.x - xk
        dy = self.y - yk
        c, s = math.cos(alpha_k), math.sin(alpha_k)
        x_e =  c * dx + s * dy          # along-track  (+ = past wp_prev)
        y_e = -s * dx + c * dy          # cross-track  (+ = right of path)

        seg_len = math.hypot(xk1 - xk, yk1 - yk)
        return alpha_k, x_e, y_e, seg_len

    def _adaptive_lookahead(self, y_e: float) -> float:
        """Time-varying lookahead: small when far from path (aggressive
        convergence), large when on path (smooth tracking)."""
        return ((self.D_max - self.D_min)
                * math.exp(-self.D_gain * abs(y_e))
                + self.D_min)

    def _desired_heading(self, alpha_k, y_e, Delta, dt) -> float:
        """Compute ψ_d.  If enable_integral is True, also updates the
        sideslip estimate β̂ (y_int).  Otherwise identical to plain LOS."""

        if self.integral_on:
            # Update sideslip estimator  (eq. 15 in Lekkas & Fossen 2014)
            denom = math.sqrt(Delta**2 + (y_e + self.y_int)**2)
            if denom > 1e-6:
                y_int_dot = self.kappa * Delta * y_e / denom
                self.y_int += y_int_dot * dt
                self.y_int = max(-self.y_int_sat,
                                 min(self.y_int_sat, self.y_int))

            psi_d = alpha_k - math.atan2(y_e + self.y_int, Delta)
        else:
            # Plain LOS: no integral term
            psi_d = alpha_k - math.atan2(y_e, Delta)

        return psi_d

    # ─────────────────────────────────────────────────────────────────
    # Mission management
    # ─────────────────────────────────────────────────────────────────

    def _reset_mission(self):
        """Called on entry to GUIDED mode. Picks the nearest waypoint
        so the first segment isn't an arbitrary long line from wherever
        the boat happened to start."""
        self.mission_complete = False
        self.waypoints_cleared = 0
        self.y_int = 0.0

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
        """Move to the next waypoint. Reset the integral on each segment
        transition to avoid carry-over bias."""
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
        self.y_int = 0.0          # reset integral on new segment

        if self.waypoints_cleared >= len(self.waypoints):
            if self.loop_mode:
                self.waypoints_cleared = 0
                self.get_logger().info('  ↻ Looping mission')
            else:
                self.mission_complete = True
                self.get_logger().info('  ■ Mission complete')

    def _send_zero(self):
        self.cmd_pub.publish(Twist())

    # ─────────────────────────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        # ── Pre-checks ─────────────────────────────────────────
        if not self.have_state:
            return
        if self.mode != 'GUIDED' or not self.armed:
            self._send_zero()
            return
        if self.mission_complete:
            self._send_zero()
            return
        if self.start_pose is None:
            self.start_pose = (self.x, self.y)

        # ── Current segment ────────────────────────────────────
        wp_target = self.waypoints[self.wp_idx]
        
        # Determine previous WP dynamically handling array wrap-around
        if self.waypoints_cleared == 0:
            wp_prev = self.start_pose
        else:
            wp_prev = self.waypoints[(self.wp_idx - 1) % len(self.waypoints)]

        # ── Path errors ────────────────────────────────────────
        alpha_k, x_e, y_e, seg_len = self._path_errors(wp_prev, wp_target)
        dist_to_wp = math.hypot(wp_target[0] - self.x, wp_target[1] - self.y)

        # ── Waypoint advancement ───────────────────────────────
        if dist_to_wp < self.R_accept:
            self._advance_waypoint('radius')
            return
        if seg_len > 0.01 and x_e > seg_len:
            self._advance_waypoint('along-track')
            return

        # ── Guidance ───────────────────────────────────────────
        Delta = self._adaptive_lookahead(y_e)
        psi_d = self._desired_heading(alpha_k, y_e, Delta, dt)
        psi_err = wrap(psi_d - self.psi)

        # ── Yaw rate command, rate-limited ─────────────────────
        r_cmd = self.K_psi * psi_err
        r_cmd = max(-self.r_max, min(self.r_max, r_cmd))

        # ── Surge command, smooth cos taper ────────────────────
        U_cmd = self.U_d * max(0.0, math.cos(psi_err))

        # Final-waypoint braking
        is_final = (self.waypoints_cleared == len(self.waypoints) - 1) and not self.loop_mode
        if is_final and dist_to_wp < self.brake_dist:
            U_brake = max(self.U_d * (dist_to_wp / self.brake_dist), self.U_min)
            U_cmd = min(U_cmd, U_brake)

        # ── Polygon boundary (APF) ─────────────────────────────
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

        # ── Publish ────────────────────────────────────────────
        cmd = Twist()
        cmd.linear.x = float(U_cmd)
        cmd.angular.z = float(r_cmd)
        self.cmd_pub.publish(cmd)

        # ── CSV log ────────────────────────────────────────────
        if self.csv_file is not None:
            t_sec = now.nanoseconds * 1e-9
            self.csv_writer.writerow([
                f'{t_sec:.4f}',
                f'{self.x:.4f}', f'{self.y:.4f}', f'{self.psi:.4f}',
                f'{self.u:.4f}', f'{self.v:.4f}',
                self.wp_idx, f'{wp_target[0]:.2f}', f'{wp_target[1]:.2f}',
                f'{alpha_k:.4f}', f'{x_e:.4f}', f'{y_e:.4f}',
                f'{seg_len:.4f}',
                f'{Delta:.4f}', f'{self.y_int:.4f}',
                f'{psi_d:.4f}', f'{psi_err:.4f}',
                f'{U_cmd:.4f}', f'{r_cmd:.4f}',
                f'{bound_fx:.4f}', f'{bound_fy:.4f}', f'{min_wall_dist:.4f}',
            ])
            self.csv_file.flush()

    # ─────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────

    def destroy_node(self):
        try:
            self._send_zero()
        except Exception:
            pass
        if self.csv_file is not None:
            try:
                self.csv_file.close()
                self.get_logger().info(f'  CSV saved → {self.csv_path}')
            except Exception:
                pass
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ILOSGuidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()