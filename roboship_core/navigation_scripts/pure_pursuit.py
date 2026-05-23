#!/usr/bin/env python3
"""
pure_pursuit.py - Pure Pursuit guidance for Roboship USV

Geometric path-tracking algorithm based on Coulter (1992) [1]. Computes a
circular arc from the vehicle's current position to a goal point on the
desired path located at a fixed lookahead distance ahead.

References
----------
[1] Coulter, R. C. (1992). "Implementation of the Pure Pursuit Path Tracking
    Algorithm." CMU Robotics Institute Technical Report CMU-RI-TR-92-01.

Architecture
------------
Pattern B: publishes geometry_msgs/Twist on
/mavros/setpoint_velocity/cmd_vel_unstamped while in GUIDED mode.
ArduRover's inner turn-rate PID and speed controller execute the commands.
EKF3 (fed by Qualisys via /mavros/vision_pose/pose -> ExternalNav) provides
the state on /mavros/local_position/odom.

This uses the SAME inner-loop controller (ArduRover turn-rate PID) as LOS
and ILOS, ensuring the comparison isolates the guidance law only.

Does NOT start MAVROS (roboship_core.service handles that).

Polygon boundary
-----------------
Same boundary system as los_guidance.py and ilos_guidance.py.
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
from geographic_msgs.msg import GeoPointStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry


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


def dist_point_to_segment(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px, py) to line segment (x1,y1)-(x2,y2).
    Returns (distance, nearest_x, nearest_y)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(px - x1, py - y1), x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy), cx, cy


# -----------------------------------------------------------------------------
# Polygon geometry (identical to LOS/ILOS)
# -----------------------------------------------------------------------------

class BoundaryPolygon:
    """Precomputed polygon boundary with signed distance queries."""

    def __init__(self, vertices: list, margin: float):
        self.verts = list(vertices)
        self.n = len(self.verts)
        self.margin = margin

        if self.n < 3:
            raise ValueError(f'Polygon needs >= 3 vertices, got {self.n}')

        area2 = 0.0
        for i in range(self.n):
            x1, y1 = self.verts[i]
            x2, y2 = self.verts[(i + 1) % self.n]
            area2 += x1 * y2 - x2 * y1
        sign = 1.0 if area2 > 0.0 else -1.0

        self.edges = []
        for i in range(self.n):
            x1, y1 = self.verts[i]
            x2, y2 = self.verts[(i + 1) % self.n]
            edx, edy = x2 - x1, y2 - y1
            length = math.sqrt(edx * edx + edy * edy)
            if length < 1e-9:
                continue
            nx = sign * (-edy) / length
            ny = sign * (edx) / length
            self.edges.append((x1, y1, x2, y2, nx, ny))

    def point_inside(self, px: float, py: float) -> bool:
        inside = False
        j = self.n - 1
        for i in range(self.n):
            xi, yi = self.verts[i]
            xj, yj = self.verts[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def signed_distance(self, px: float, py: float):
        min_dist = float('inf')
        nearest_idx = 0
        for i, (x1, y1, x2, y2, _nx, _ny) in enumerate(self.edges):
            d, _, _ = dist_point_to_segment(px, py, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        if not self.point_inside(px, py):
            min_dist = -min_dist
        return min_dist, nearest_idx

    def speed_factor(self, px, py, heading_rad):
        sd, nearest_idx = self.signed_distance(px, py)
        if sd <= 0.0:
            return 0.0, sd, nearest_idx
        if sd >= self.margin:
            return 1.0, sd, nearest_idx
        factor = sd / self.margin
        _, _, _, _, nx, ny = self.edges[nearest_idx]
        hx = math.cos(heading_rad)
        hy = math.sin(heading_rad)
        dot = hx * nx + hy * ny
        if dot < 0.0:
            factor *= (1.0 + dot)
        return max(factor, 0.0), sd, nearest_idx


# -----------------------------------------------------------------------------
# Main node
# -----------------------------------------------------------------------------

class PurePursuitGuidance(Node):
    def __init__(self):
        super().__init__('pure_pursuit_guidance')

        # ---- Guidance parameters -------------------------------------------
        # Waypoints: flat list [x0, y0, x1, y1, ...]
        self.declare_parameter('waypoints',
                               [0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5] * 3)

        # Surge speed [m/s]
        self.declare_parameter('U_desired', 0.20)
        self.declare_parameter('U_min', 0.05)

        # Lookahead distance [m] — the SINGLE key tuning parameter of
        # pure pursuit (Coulter 1992, Section 2.4). Governs the trade-off
        # between tracking tightness and smoothness:
        #   Small L_d -> aggressive corrections, risk of oscillation
        #   Large L_d -> smoother paths, cuts corners
        self.declare_parameter('lookahead_distance', 0.6)

        # Yaw rate gain [1/s] — proportional gain mapping heading error
        # to commanded yaw rate: r_cmd = K_psi * heading_error
        self.declare_parameter('K_psi', 1.5)

        # Waypoint acceptance radius [m] — distance threshold to declare
        # a waypoint reached and advance to the next
        self.declare_parameter('acceptance_radius', 0.2)

        # Braking distance [m] — distance from the FINAL waypoint at which
        # surge speed begins linearly decreasing to U_min
        self.declare_parameter('braking_distance', 1.0)

        # Control loop rate [Hz]
        self.declare_parameter('control_rate', 20.0)

        # Whether to loop back to WP 0 after the last waypoint
        self.declare_parameter('loop_mission', False)

        # ---- Polygon boundary ----------------------------------------------
        default_boundary_file = '/home/roboship/ros2_ws/src/roboship_core/roboship_core/navigation_scripts/mocap_boundary.json'
        self.declare_parameter('boundary_file', default_boundary_file)
        self.declare_parameter('boundary_margin', 0.3)

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
        self.L_d        = float(self.get_parameter('lookahead_distance').value)
        self.K_psi      = float(self.get_parameter('K_psi').value)
        self.R_accept   = float(self.get_parameter('acceptance_radius').value)
        self.brake_dist = float(self.get_parameter('braking_distance').value)
        self.rate_hz    = float(self.get_parameter('control_rate').value)
        self.loop_mode  = bool(self.get_parameter('loop_mission').value)

        # ---- Load JSON boundary file ---------------------------------------
        b_margin = float(self.get_parameter('boundary_margin').value)
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
                    f'Loaded boundary from {boundary_file_path.name}')
            except Exception as e:
                self.get_logger().error(
                    f'Failed to parse {boundary_file_path.name}: {e}')
        else:
            self.get_logger().info(
                f'No boundary file at {boundary_file_path}.')

        if len(bv_flat) >= 6 and len(bv_flat) % 2 == 0:
            verts = [(bv_flat[i], bv_flat[i + 1])
                     for i in range(0, len(bv_flat), 2)]
            self.boundary = BoundaryPolygon(verts, b_margin)
            self.get_logger().info(
                f'Boundary active: {len(verts)} vertices, '
                f'margin={b_margin:.2f} m')
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
        self.origin_set = False
        self.last_time = self.get_clock().now()

        # ---- QoS for MAVROS topics -----------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- ROS interfaces -------------------------------------------------
        self.create_subscription(
            Odometry, '/mavros/local_position/odom', self._odom_cb, sensor_qos)
        self.create_subscription(
            State, '/mavros/state', self._state_cb, 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        self.origin_pub = self.create_publisher(
            GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)

        # ---- Global origin timer (1 Hz until EKF has pose) -----------------
        self.origin_timer = self.create_timer(1.0, self._set_origin)

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
                'goal_x', 'goal_y', 'goal_dist',
                'curvature', 'L_d',
                'psi_d', 'psi_err', 'U_cmd', 'r_cmd',
                'bound_factor', 'min_wall_dist',
            ])
            self.get_logger().info(f'Logging to {self.csv_path}')

        # ---- Control timer --------------------------------------------------
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f'Pure Pursuit started. {len(self.waypoints)} WPs, '
            f'U_d={self.U_d} m/s, L_d={self.L_d} m')


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
        """Find the goal point on the path segment at lookahead distance L_d.

        The goal point is the intersection of a circle of radius L_d centred
        on the vehicle with the line segment from wp_prev to wp_target.

        If no intersection exists (vehicle is too far from the path), the
        goal point falls back to the nearest point on the segment offset
        toward the target. If the target is closer than L_d, the target
        itself is used as the goal point (Coulter 1992, Section 2.3).

        Returns (goal_x, goal_y) in the world frame.
        """
        x1, y1 = wp_prev
        x2, y2 = wp_target

        # Path segment as parametric line: P(t) = (x1,y1) + t*(dx,dy), t in [0,1]
        dx = x2 - x1
        dy = y2 - y1

        # Solve |P(t) - vehicle|^2 = L_d^2 for t
        # Expanding: a*t^2 + b*t + c = 0
        fx = x1 - self.x
        fy = y1 - self.y
        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - self.L_d * self.L_d

        discriminant = b * b - 4.0 * a * c

        goal_x, goal_y = wp_target  # fallback: aim at the target

        if discriminant >= 0 and a > 1e-12:
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # Pick the furthest-along intersection that is on the segment
            # Prefer t2 (further along the path) over t1
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
        """Compute the curvature to reach the goal point.

        Transforms the goal into the vehicle's body frame and applies
        Coulter (1992) Equation 2.4:

            kappa = 2 * g_y / L_d^2

        where g_y is the lateral offset of the goal in the body frame.

        Returns (curvature, psi_d) where psi_d is the desired heading
        toward the goal point (for logging/comparison).
        """
        # Vector from vehicle to goal in world frame
        dx_w = goal_x - self.x
        dy_w = goal_y - self.y
        goal_dist = math.hypot(dx_w, dy_w)

        # Desired heading toward goal (for logging and speed scaling)
        psi_d = math.atan2(dy_w, dx_w)

        # Transform to body frame
        c = math.cos(self.psi)
        s = math.sin(self.psi)
        g_x =  c * dx_w + s * dy_w   # longitudinal (forward)
        g_y = -s * dx_w + c * dy_w   # lateral (left positive)

        # Curvature: kappa = 2 * g_y / L_d^2 (Coulter 1992, Eq 2.4)
        # Use actual goal distance instead of L_d when goal is closer
        # (e.g. when target waypoint is within L_d)
        ell_sq = max(goal_dist * goal_dist, 0.01)  # avoid division by zero
        curvature = 2.0 * g_y / ell_sq

        return curvature, psi_d, goal_dist

    # ------------------------------------------------------------------------
    # Mission management
    # ------------------------------------------------------------------------

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
                self.get_logger().info('Mission looping back to WP 0.')
            else:
                self.mission_complete = True
                self.get_logger().info('Mission complete.')

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
        wp_prev = (self.start_pose if self.wp_idx == self._first_wp_idx
                   else self.waypoints[self.wp_idx - 1])

        dist_to_target = math.hypot(wp_target[0] - self.x,
                                    wp_target[1] - self.y)

        # Along-track overshoot check
        alpha_k = math.atan2(wp_target[1] - wp_prev[1],
                             wp_target[0] - wp_prev[0])
        dx = self.x - wp_prev[0]
        dy = self.y - wp_prev[1]
        x_e = math.cos(alpha_k) * dx + math.sin(alpha_k) * dy
        seg_length = math.hypot(wp_target[0] - wp_prev[0],
                                wp_target[1] - wp_prev[1])

        if dist_to_target < self.R_accept or x_e > seg_length:
            self.get_logger().info(
                f'Reached WP {(self.wp_idx % 4) + 1} '
                f'({wp_target[0]:.2f}, {wp_target[1]:.2f})')
            self._advance_waypoint()
            return

        # ---- Pure pursuit guidance -----------------------------------------
        goal_x, goal_y = self._find_goal_point(wp_prev, wp_target)
        curvature, psi_d, goal_dist = self._pure_pursuit_curvature(
            goal_x, goal_y)

        # Convert curvature to yaw rate: r = U * kappa
        # But also apply proportional heading correction for robustness
        psi_err = wrap_angle(psi_d - self.psi)
        r_cmd = self.K_psi * psi_err

        # Surge speed
        U_cmd = self.U_d
        is_final = (self.wp_idx == len(self.waypoints) - 1) \
                   and not self.loop_mode
        if is_final and dist_to_target < self.brake_dist:
            U_cmd = max(self.U_d * (dist_to_target / self.brake_dist),
                        self.U_min)
        # Reduce speed when heading error is large (avoid driving sideways)
        U_cmd *= max(0.0, math.cos(psi_err))

        # ---- Polygon boundary ----------------------------------------------
        bound_factor = 1.0
        min_wall_dist = float('inf')
        if self.boundary is not None:
            bound_factor, min_wall_dist, _ = self.boundary.speed_factor(
                self.x, self.y, self.psi)
            U_cmd *= bound_factor
            if bound_factor <= 0.0:
                self.get_logger().warn(
                    f'OUTSIDE boundary! ({self.x:.2f}, {self.y:.2f})')
            elif bound_factor < 1.0:
                self.get_logger().info(
                    f'Boundary: dist={min_wall_dist:.2f} m, '
                    f'factor={bound_factor:.2f}')

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
                goal_x, goal_y, goal_dist,
                curvature, self.L_d,
                psi_d, psi_err, U_cmd, r_cmd,
                bound_factor, min_wall_dist,
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