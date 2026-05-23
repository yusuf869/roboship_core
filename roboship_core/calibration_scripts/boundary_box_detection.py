#!/usr/bin/env python3
"""
Mocap Bounding Box Detection — Simple MANUAL Mode
──────────────────────────────────────────────────
Probes outward from wherever the boat is when armed.
All RC overrides in MANUAL mode — no GUIDED, no position setpoints.

Per probe:
    1. ORIENT  — spin in place to face probe direction
    2. EXPLORE — drive forward until mocap drops
    3. RETURN  — reverse back to start position
    4. Repeat for next direction

Start script → switch to MANUAL → ARM on transmitter.
"""

import math
import time
import json
import pathlib
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import OverrideRCIn, State


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class BoundingBoxDetector(Node):

    # ── Tunables ────────────────────────────────────────────────
    NUM_DIRECTIONS = 12
    FWD_THROTTLE = 1520
    REV_THROTTLE = 1480
    STEER_P_GAIN = 2.0
    MAX_STEERING = 150
    YAW_TOLERANCE = math.radians(15)
    HOME_THRESHOLD = 0.4
    QUALISYS_TIMEOUT = 0.4
    MIN_EXPLORE_TIME = 2.0
    RETURN_EXTRA_SEC = 0.5
    SAFETY_MARGIN = 0.3
    WP_THRESHOLD = 0.4

    def __init__(self):
        super().__init__('bbox_detector')

        self.state = 'SETUP'
        self.pose = None
        self.last_pose_time = None
        self.mavros_state = None
        self.start_pos = None
        self.boundary_points = []
        self.current_dir_index = 0
        self.explore_start_time = None
        self.return_start_time = None
        self.last_good_position = None
        self.return_target_reached = False

        self.directions = [
            i * (2 * math.pi / self.NUM_DIRECTIONS)
            for i in range(self.NUM_DIRECTIONS)
        ]

        self.verify_waypoints = []
        self.verify_wp_index = 0
        self.verified_boundary = []
        self.verify_issues = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(
            PoseStamped, '/mavros/vision_pose/pose', self.pose_cb, qos)
        self.create_subscription(
            State, '/mavros/state', self.state_cb, 10)

        self.rc_pub = self.create_publisher(
            OverrideRCIn, '/mavros/rc/override', 10)

        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info(
            f'Bounding box detector — {self.NUM_DIRECTIONS} probes')
        self.get_logger().info('Switch to MANUAL and ARM to begin.')

    # ── Callbacks ───────────────────────────────────────────────

    def pose_cb(self, msg):
        self.pose = msg
        self.last_pose_time = time.time()
        self.last_good_position = (
            msg.pose.position.x, msg.pose.position.y)

    def state_cb(self, msg):
        self.mavros_state = msg

    # ── Helpers ─────────────────────────────────────────────────

    def alive(self):
        if self.last_pose_time is None:
            return False
        return (time.time() - self.last_pose_time) < self.QUALISYS_TIMEOUT

    def pos(self):
        if self.pose is None:
            return None
        return (self.pose.pose.position.x, self.pose.pose.position.y)

    def yaw(self):
        if self.pose is None:
            return 0.0
        return yaw_from_quaternion(self.pose.pose.orientation)

    def dist_to_home(self):
        p = self.pos()
        if p is None or self.start_pos is None:
            return float('inf')
        dx = p[0] - self.start_pos[0]
        dy = p[1] - self.start_pos[1]
        return math.sqrt(dx*dx + dy*dy)

    def send_rc(self, throttle, steering=0):
        rc = OverrideRCIn()
        rc.channels = [0] * 18
        rc.channels[0] = max(1000, min(2000, 1500 + int(steering)))
        rc.channels[2] = max(1000, min(2000, int(throttle)))
        self.rc_pub.publish(rc)

    def stop(self):
        self.send_rc(1500, 0)

    def steer_to(self, target_yaw, throttle=1500):
        error = wrap_pi(target_yaw - self.yaw())
        steer = self.STEER_P_GAIN * error * self.MAX_STEERING / math.pi
        steer = max(-self.MAX_STEERING, min(self.MAX_STEERING, steer))
        turn_factor = max(0.0, math.cos(error))
        adj_throttle = 1500 + (throttle - 1500) * turn_factor
        self.send_rc(adj_throttle, steer)
        return abs(error)

    def steer_to_home(self, throttle):
        p = self.pos()
        if p is None or self.start_pos is None:
            return
        bearing = math.atan2(
            self.start_pos[1] - p[1],
            self.start_pos[0] - p[0])
        self.steer_to(bearing, throttle)

    def build_verify_waypoints(self):
        if len(self.boundary_points) < 3:
            return []
        cx = sum(p['x'] for p in self.boundary_points) / len(self.boundary_points)
        cy = sum(p['y'] for p in self.boundary_points) / len(self.boundary_points)
        wps = []
        for p in self.boundary_points:
            dx, dy = p['x'] - cx, p['y'] - cy
            d = math.sqrt(dx*dx + dy*dy)
            if d < 0.01:
                continue
            s = max(0, (d - self.SAFETY_MARGIN)) / d
            wps.append({'x': cx + dx*s, 'y': cy + dy*s})
        return wps

    # ── State machine ──────────────────────────────────────────

    def loop(self):

        # ── SETUP ──────────────────────────────────────────────

        if self.state == 'SETUP':
            if self.mavros_state is None:
                return
            if not self.alive():
                self.get_logger().info(
                    'Waiting for mocap...', throttle_duration_sec=2.0)
                return
            if self.mavros_state.mode != 'MANUAL':
                self.get_logger().info(
                    f'In {self.mavros_state.mode} — switch to MANUAL.',
                    throttle_duration_sec=2.0)
                return
            if not self.mavros_state.armed:
                self.get_logger().info(
                    'MANUAL active. ARM to begin.',
                    throttle_duration_sec=2.0)
                return

            self.start_pos = self.pos()
            self.current_dir_index = 0
            self.get_logger().info('=' * 50)
            self.get_logger().info(
                f'  Starting from ({self.start_pos[0]:.2f}, '
                f'{self.start_pos[1]:.2f})')
            self.get_logger().info('  PASS 1: RADIAL PROBING')
            self.get_logger().info('=' * 50)
            self.state = 'ORIENT'

        # ── ORIENT ─────────────────────────────────────────────

        elif self.state == 'ORIENT':
            if self.current_dir_index >= len(self.directions):
                self.stop()
                self.state = 'START_VERIFY'
                return

            target = self.directions[self.current_dir_index]
            error = self.steer_to(target)

            if error < self.YAW_TOLERANCE:
                self.stop()
                deg = math.degrees(target)
                self.get_logger().info(
                    f'  Probe {self.current_dir_index + 1}/'
                    f'{self.NUM_DIRECTIONS} ({deg:.0f}°) — exploring')
                self.explore_start_time = time.time()
                self.state = 'EXPLORE'

        # ── EXPLORE ────────────────────────────────────────────

        elif self.state == 'EXPLORE':
            elapsed = time.time() - self.explore_start_time

            if elapsed > self.MIN_EXPLORE_TIME and not self.alive():
                self.stop()
                if self.last_good_position is not None:
                    bx, by = self.last_good_position
                    deg = math.degrees(
                        self.directions[self.current_dir_index])
                    self.boundary_points.append({
                        'x': round(bx, 3),
                        'y': round(by, 3),
                        'direction_deg': round(deg, 1),
                    })
                    self.get_logger().info(
                        f'  BOUNDARY at ({bx:.2f}, {by:.2f})')
                self.return_target_reached = False
                self.return_start_time = None
                self.state = 'RETURN'
                return

            target = self.directions[self.current_dir_index]
            self.steer_to(target, self.FWD_THROTTLE)

            dist = self.dist_to_home()
            self.get_logger().info(
                f'  {dist:.2f}m out, {elapsed:.1f}s',
                throttle_duration_sec=1.0)

        # ── RETURN ─────────────────────────────────────────────

        elif self.state == 'RETURN':
            if not self.alive():
                self.send_rc(self.REV_THROTTLE, 0)
                self.get_logger().info(
                    '  Reversing blind...', throttle_duration_sec=1.0)
                return

            dist = self.dist_to_home()

            if dist < self.HOME_THRESHOLD:
                if not self.return_target_reached:
                    self.return_target_reached = True
                    self.return_start_time = time.time()

            if self.return_target_reached:
                elapsed = time.time() - self.return_start_time
                if elapsed > self.RETURN_EXTRA_SEC:
                    self.stop()
                    self.get_logger().info('  Back home')
                    self.current_dir_index += 1
                    self.state = 'ORIENT'
                    return
                self.send_rc(self.REV_THROTTLE, 0)
            else:
                self.steer_to_home(self.FWD_THROTTLE)
                self.get_logger().info(
                    f'  Returning... {dist:.2f}m',
                    throttle_duration_sec=1.0)

        # ── VERIFICATION ───────────────────────────────────────

        elif self.state == 'START_VERIFY':
            if len(self.boundary_points) < 3:
                self.get_logger().warn('Not enough points — skipping verify')
                self.state = 'SAVE'
                return

            self.verify_waypoints = self.build_verify_waypoints()
            self.verify_wp_index = 0
            self.verified_boundary = list(self.boundary_points)
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info(
                f'  PASS 2: VERIFICATION '
                f'({len(self.verify_waypoints)} waypoints)')
            self.get_logger().info('=' * 50)
            self.state = 'VERIFY'

        elif self.state == 'VERIFY':
            if self.verify_wp_index >= len(self.verify_waypoints):
                self.stop()
                self.state = 'SAVE'
                return

            wp = self.verify_waypoints[self.verify_wp_index]

            if not self.alive():
                self.stop()
                self.get_logger().warn(
                    f'  Tracking LOST near WP {self.verify_wp_index}!')
                if self.last_good_position:
                    self.verify_issues.append({
                        'wp_index': self.verify_wp_index,
                        'last_good_x': round(self.last_good_position[0], 3),
                        'last_good_y': round(self.last_good_position[1], 3),
                        'issue': 'tracking_lost',
                    })
                    if self.verify_wp_index < len(self.verified_boundary):
                        bp = self.verified_boundary[self.verify_wp_index]
                        dx, dy = bp['x'], bp['y']
                        d = math.sqrt(dx*dx + dy*dy)
                        if d > 0.1:
                            bp['x'] = round(dx * 0.7, 3)
                            bp['y'] = round(dy * 0.7, 3)
                self.state = 'VERIFY_RECOVER'
                return

            p = self.pos()
            if p is not None:
                dx = wp['x'] - p[0]
                dy = wp['y'] - p[1]
                dist = math.sqrt(dx*dx + dy*dy)
                bearing = math.atan2(dy, dx)
                self.steer_to(bearing, self.FWD_THROTTLE)
                if dist < self.WP_THRESHOLD:
                    self.get_logger().info(
                        f'  Verify WP {self.verify_wp_index + 1}/'
                        f'{len(self.verify_waypoints)} OK')
                    self.verify_wp_index += 1

        elif self.state == 'VERIFY_RECOVER':
            if not self.alive():
                self.send_rc(self.REV_THROTTLE, 0)
                return
            self.stop()
            self.verify_wp_index += 1
            self.state = 'VERIFY'

        # ── SAVE ───────────────────────────────────────────────

        elif self.state == 'SAVE':
            self.stop()
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('  BOUNDING BOX COMPLETE')
            self.get_logger().info('=' * 50)

            out_path = (pathlib.Path.home() / 'usv_logs'
                        / 'mocap_boundary.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)

            final = (self.verified_boundary
                     if self.verified_boundary
                     else self.boundary_points)

            output = {
                'start_position': {
                    'x': self.start_pos[0],
                    'y': self.start_pos[1]},
                'raw_boundary': self.boundary_points,
                'verified_boundary': final,
                'boundary_points': final,
                'verification_issues': self.verify_issues,
                'num_directions': self.NUM_DIRECTIONS,
                'safety_margin_m': self.SAFETY_MARGIN,
            }

            with open(out_path, 'w') as f:
                json.dump(output, f, indent=2)

            self.get_logger().info(f'  Saved to {out_path}')
            self.get_logger().info(
                f'  Points: {len(self.boundary_points)} | '
                f'Issues: {len(self.verify_issues)}')
            for bp in final:
                self.get_logger().info(
                    f'    {bp.get("direction_deg", "?"):>6}°  →  '
                    f'({bp["x"]:.2f}, {bp["y"]:.2f})')

            self.state = 'FINISHED'
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()