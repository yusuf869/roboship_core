#!/usr/bin/env python3
"""
Motor Auto-Calibration Node 
─────────────────────────────────
Automatically determines motor trim and pushes corrected SERVO
parameters directly to the Pixhawk via MAVROS — no QGC needed.

Safety: The script does NOT arm or change modes automatically.
You must arm the vehicle and set MANUAL mode yourself (via transmitter
or QGC). Once detected, the script navigates to the mocap origin
and then begins calibration.

Test sequence:
  0. WAIT — Wait for user to arm + set MANUAL mode
  0b. GO TO ORIGIN — Navigate to (0,0) in GUIDED mode
  1. FORWARD STRAIGHT — Measure heading drift, find left/right imbalance
  2. REVERSE STRAIGHT — Same in reverse (separate scaling)
  3. SPEED MATCH — Adjust reverse throttle to match forward speed
  4. SPIN TESTS — Verify CW/CCW symmetry and translational drift
  5. APPLY — Compute new SERVO9/10 MIN/MAX/TRIM and push to Pixhawk

Output: ~/usv_logs/motor_calibration.json + params applied to Pixhawk.

Prerequisites:
    - SYSID_MYGCS = 1, EKF3 sources = ExternalNav
    - Mocap pipeline running
    - WP_SPEED ~0.2-0.3

Usage:
    ros2 run roboship_core motor_calibration
"""

import math
import json
import pathlib
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import OverrideRCIn, PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode, ParamGet, ParamSet
from geographic_msgs.msg import GeoPointStamped
from rcl_interfaces.msg import ParameterValue


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def normalise_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class MotorCalibration(Node):

    # ── Tunables ────────────────────────────────────────────────
    BASE_THROTTLE = 1560
    REVERSE_THROTTLE = 1440
    SPIN_THROTTLE = 1560
    RUN_DURATION = 3.0
    SPIN_DURATION = 4.0
    SETTLE_TIME = 2.0
    MAX_ITERATIONS = 10
    HEADING_TOLERANCE = 2.0         # degrees
    SPEED_TOLERANCE = 0.05          # m/s
    TRIM_STEP = 0.02
    ORIGIN_THRESHOLD = 0.3         # meters — how close to origin before starting

    # Which SERVO outputs are your left and right motors
    LEFT_SERVO = 'SERVO9'           # change if yours differ
    RIGHT_SERVO = 'SERVO10'

    # GUIDED position setpoint config
    TYPE_MASK = 3580
    COORD_FRAME = PositionTarget.FRAME_LOCAL_NED

    def __init__(self):
        super().__init__('motor_calibration')

        # Calibration scaling factors (1.0 = no correction)
        self.fwd_left_scale = 1.0
        self.fwd_right_scale = 1.0
        self.rev_left_scale = 1.0
        self.rev_right_scale = 1.0
        self.reverse_throttle_multiplier = 1.0

        # State machine
        self.state = 'WAIT_FOR_ARM'
        self.iteration = 0
        self.test_start_time = None
        self.start_yaw = None
        self.start_pos = None
        self.current_pose = None
        self.mavros_state = None
        self.results = []
        self.last_wait_log = 0.0
        self.origin_set = False

        # Speed measurement
        self.forward_speed = None
        self.reverse_speed = None

        # Current SERVO params (read from Pixhawk)
        self.servo_params = {}

        # Subscribers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PoseStamped, '/mavros/vision_pose/pose',
                                 self.pose_cb, qos)
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)

        # Publishers
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.setpoint_pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.origin_pub = self.create_publisher(
            GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)

        # Services
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.param_get_client = self.create_client(ParamGet, '/mavros/param/get')
        self.param_set_client = self.create_client(ParamSet, '/mavros/param/set')

        # Timers
        self.timer = self.create_timer(0.05, self.loop)
        self.origin_timer = self.create_timer(1.0, self.publish_origin)

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('  MOTOR CALIBRATION')
        self.get_logger().info('=' * 50)
        self.get_logger().info('  Waiting for you to:')
        self.get_logger().info('    1. Arm the vehicle (transmitter or QGC)')
        self.get_logger().info('    2. Set MANUAL mode')
        self.get_logger().info('  The script will NOT arm automatically.')
        self.get_logger().info('=' * 50)

    # ── Callbacks ───────────────────────────────────────────────

    def pose_cb(self, msg):
        self.current_pose = msg

    def state_cb(self, msg):
        self.mavros_state = msg

    def publish_origin(self):
        if self.origin_set:
            return
        msg = GeoPointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position.latitude = 50.3755
        msg.position.longitude = -4.1427
        msg.position.altitude = 0.0
        self.origin_pub.publish(msg)
        if self.current_pose is not None:
            self.origin_set = True
            self.origin_timer.cancel()
            self.get_logger().info('  Global origin set.')

    # ── Motor commands ──────────────────────────────────────────

    def send_differential(self, left_throttle, right_throttle):
        throttle = (left_throttle + right_throttle) / 2.0
        steering = (right_throttle - left_throttle) / 2.0
        rc = OverrideRCIn()
        rc.channels = [0] * 18
        rc.channels[0] = max(1000, min(2000, 1500 + int(steering)))
        rc.channels[2] = max(1000, min(2000, int(throttle)))
        self.rc_pub.publish(rc)

    def stop(self):
        rc = OverrideRCIn()
        rc.channels = [0] * 18
        rc.channels[0] = 1500
        rc.channels[2] = 1500
        self.rc_pub.publish(rc)

    # ── Navigation to origin (GUIDED mode) ─────────────────────

    def publish_position(self, east, north):
        """Publish a position setpoint in ENU. MAVROS converts to NED."""
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = self.COORD_FRAME
        msg.type_mask = self.TYPE_MASK
        msg.position.x = float(east)
        msg.position.y = float(north)
        msg.position.z = 0.0
        self.setpoint_pub.publish(msg)

    def distance_to_origin(self):
        if self.current_pose is None:
            return float('inf')
        pos = self.current_pose.pose.position
        return math.sqrt(pos.x * pos.x + pos.y * pos.y)

    def set_mode(self, mode):
        """Request a mode change (non-blocking)."""
        if self.mode_client.wait_for_service(timeout_sec=1.0):
            req = SetMode.Request()
            req.custom_mode = mode
            self.mode_client.call_async(req)

    # ── Helpers ─────────────────────────────────────────────────

    def record_start(self):
        pos = self.current_pose.pose.position
        self.start_yaw = yaw_from_quaternion(self.current_pose.pose.orientation)
        self.start_pos = (pos.x, pos.y)
        self.test_start_time = self.get_clock().now().nanoseconds / 1e9

    def elapsed(self):
        return self.get_clock().now().nanoseconds / 1e9 - self.test_start_time

    def measure_drift_and_distance(self):
        pos = self.current_pose.pose.position
        yaw = yaw_from_quaternion(self.current_pose.pose.orientation)
        drift_deg = math.degrees(normalise_angle(yaw - self.start_yaw))
        dx = pos.x - self.start_pos[0]
        dy = pos.y - self.start_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return drift_deg, dist

    def measure_spin(self):
        pos = self.current_pose.pose.position
        yaw = yaw_from_quaternion(self.current_pose.pose.orientation)
        total_yaw = normalise_angle(yaw - self.start_yaw)
        dx = pos.x - self.start_pos[0]
        dy = pos.y - self.start_pos[1]
        trans_drift = math.sqrt(dx * dx + dy * dy)
        yaw_rate = math.degrees(total_yaw) / self.SPIN_DURATION
        return total_yaw, yaw_rate, trans_drift

    # ── Param read/write ───────────────────────────────────────

    def get_param(self, name):
        if not self.param_get_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(f'Param get service not available')
            return None
        req = ParamGet.Request()
        req.param_id = name
        future = self.param_get_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None and future.result().success:
            val = future.result().value
            return int(val.integer) if val.integer != 0 else int(val.real)
        self.get_logger().warn(f'Failed to read param {name}')
        return None

    def set_param(self, name, value):
        if not self.param_set_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(f'Param set service not available')
            return False
        req = ParamSet.Request()
        req.param_id = name
        req.value.integer = int(value)
        req.value.real = 0.0
        future = self.param_set_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f'  Set {name} = {int(value)}')
            return True
        self.get_logger().warn(f'  Failed to set {name}')
        return False

    def read_servo_params(self):
        params = {}
        for servo in [self.LEFT_SERVO, self.RIGHT_SERVO]:
            for suffix in ['_MIN', '_MAX', '_TRIM']:
                name = f'{servo}{suffix}'
                val = self.get_param(name)
                if val is not None:
                    params[name] = val
                    self.get_logger().info(f'  Read {name} = {val}')
                else:
                    self.get_logger().warn(f'  Could not read {name} — using default')
                    if '_MIN' in suffix:
                        params[name] = 1000
                    elif '_MAX' in suffix:
                        params[name] = 2000
                    else:
                        params[name] = 1500
        return params

    def compute_new_servo_params(self):
        p = self.servo_params
        new_params = {}
        for servo, fwd_s, rev_s in [
            (self.LEFT_SERVO, self.fwd_left_scale, self.rev_left_scale),
            (self.RIGHT_SERVO, self.fwd_right_scale, self.rev_right_scale),
        ]:
            trim = p[f'{servo}_TRIM']
            old_max = p[f'{servo}_MAX']
            old_min = p[f'{servo}_MIN']
            fwd_range = old_max - trim
            rev_range = trim - old_min
            new_max = int(trim + fwd_range * fwd_s)
            new_min = int(trim - rev_range * rev_s * self.reverse_throttle_multiplier)
            new_max = max(trim + 1, min(2200, new_max))
            new_min = max(800, min(trim - 1, new_min))
            new_params[f'{servo}_MAX'] = new_max
            new_params[f'{servo}_MIN'] = new_min
            new_params[f'{servo}_TRIM'] = trim
        return new_params

    def apply_servo_params(self, new_params):
        self.get_logger().info('')
        self.get_logger().info('  Applying new SERVO parameters to Pixhawk...')
        success = True
        for name, value in new_params.items():
            if not self.set_param(name, value):
                success = False
        return success

    # ── State machine ──────────────────────────────────────────

    def loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        # ── PHASE 0: WAIT FOR USER TO ARM ──────────────────

        if self.state == 'WAIT_FOR_ARM':
            if self.mavros_state is None or self.current_pose is None:
                return
            if not self.origin_set:
                return

            if now - self.last_wait_log > 5.0:
                self.last_wait_log = now
                armed = self.mavros_state.armed
                mode = self.mavros_state.mode
                connected = self.mavros_state.connected
                if not connected:
                    self.get_logger().info('  Waiting for MAVROS connection...')
                elif not armed and mode != 'MANUAL':
                    self.get_logger().info(
                        f'  Waiting... (mode: {mode}, armed: {armed}) '
                        f'→ Please arm and set MANUAL mode')
                elif not armed:
                    self.get_logger().info(
                        f'  Mode is MANUAL ✓ — Please arm the vehicle')
                elif mode != 'MANUAL':
                    self.get_logger().info(
                        f'  Armed ✓ — Please switch to MANUAL mode (current: {mode})')

            if (self.mavros_state.connected
                    and self.mavros_state.armed
                    and self.mavros_state.mode == 'MANUAL'):
                self.get_logger().info('  ✓ Armed in MANUAL mode — detected!')
                self.get_logger().info('  Switching to GUIDED to navigate to origin...')
                self.set_mode('GUIDED')
                self.test_start_time = now
                self.state = 'GOTO_ORIGIN_WAIT_MODE'
            return

        # ── PHASE 0b: NAVIGATE TO ORIGIN ───────────────────

        if self.state == 'GOTO_ORIGIN_WAIT_MODE':
            if self.mavros_state.mode == 'GUIDED':
                dist = self.distance_to_origin()
                self.get_logger().info(
                    f'  In GUIDED mode — navigating to origin '
                    f'(currently {dist:.2f}m away)')
                self.state = 'GOTO_ORIGIN'
            elif (now - self.test_start_time) > 5.0:
                self.get_logger().warn('  Failed to switch to GUIDED — retrying...')
                self.set_mode('GUIDED')
                self.test_start_time = now
            return

        if self.state == 'GOTO_ORIGIN':
            # Publish (0,0) position setpoint — L1 controller handles it
            self.publish_position(0.0, 0.0)
            dist = self.distance_to_origin()

            if now - self.last_wait_log > 3.0:
                self.last_wait_log = now
                self.get_logger().info(f'  Navigating to origin... {dist:.2f}m away')

            if dist < self.ORIGIN_THRESHOLD:
                self.get_logger().info(f'  ✓ At origin ({dist:.2f}m)')
                self.get_logger().info('  Switching to MANUAL for calibration...')
                self.stop()
                self.set_mode('MANUAL')
                self.test_start_time = now
                self.state = 'ORIGIN_SETTLE'
            return

        if self.state == 'ORIGIN_SETTLE':
            if self.mavros_state.mode == 'MANUAL':
                if (now - self.test_start_time) >= self.SETTLE_TIME:
                    self.state = 'FWD_STRAIGHT_START'
                    self.iteration = 0
                    self.get_logger().info('')
                    self.get_logger().info('=' * 50)
                    self.get_logger().info('  CALIBRATION STARTING')
                    self.get_logger().info('=' * 50)
            elif (now - self.test_start_time) > 5.0:
                self.set_mode('MANUAL')
                self.test_start_time = now
            return

        # ── Safety: abort if disarmed during calibration ───

        if self.mavros_state is not None and not self.mavros_state.armed:
            if self.state not in ('WAIT_FOR_ARM', 'FINISHED'):
                self.get_logger().warn('  Vehicle disarmed — calibration aborted!')
                self.stop()
                self.state = 'FINISHED'
                self.timer.cancel()
                return

        if self.current_pose is None:
            return

        # ── PHASE 1: FORWARD STRAIGHT ──────────────────────

        if self.state == 'FWD_STRAIGHT_START':
            self.iteration += 1
            if self.iteration > self.MAX_ITERATIONS:
                self.get_logger().warn('Forward calibration did not converge')
                self.state = 'REV_STRAIGHT_START'
                self.iteration = 0
                return
            self.get_logger().info(
                f'[FWD {self.iteration}] L={self.fwd_left_scale:.3f} R={self.fwd_right_scale:.3f}')
            self.record_start()
            self.state = 'FWD_STRAIGHT_RUN'

        elif self.state == 'FWD_STRAIGHT_RUN':
            if self.elapsed() < self.RUN_DURATION:
                offset = self.BASE_THROTTLE - 1500
                left = 1500 + offset * self.fwd_left_scale
                right = 1500 + offset * self.fwd_right_scale
                self.send_differential(left, right)
            else:
                self.stop()
                self.test_start_time = now
                self.state = 'FWD_STRAIGHT_SETTLE'

        elif self.state == 'FWD_STRAIGHT_SETTLE':
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'FWD_STRAIGHT_ANALYSE'

        elif self.state == 'FWD_STRAIGHT_ANALYSE':
            drift_deg, dist = self.measure_drift_and_distance()
            self.forward_speed = dist / self.RUN_DURATION
            self.results.append({
                'test': 'forward_straight', 'iteration': self.iteration,
                'heading_drift_deg': round(drift_deg, 2),
                'distance_m': round(dist, 3),
                'left_scale': round(self.fwd_left_scale, 4),
                'right_scale': round(self.fwd_right_scale, 4),
            })
            self.get_logger().info(f'  Drift: {drift_deg:.1f}°  Dist: {dist:.2f}m')
            if abs(drift_deg) < self.HEADING_TOLERANCE:
                self.get_logger().info('  ✓ Forward converged!')
                self.state = 'REV_STRAIGHT_START'
                self.iteration = 0
                return
            if drift_deg > 0:
                self.fwd_left_scale -= self.TRIM_STEP
            else:
                self.fwd_right_scale -= self.TRIM_STEP
            self.state = 'FWD_STRAIGHT_START'

        # ── PHASE 2: REVERSE STRAIGHT ──────────────────────

        elif self.state == 'REV_STRAIGHT_START':
            self.iteration += 1
            if self.iteration > self.MAX_ITERATIONS:
                self.get_logger().warn('Reverse calibration did not converge')
                self.state = 'SPEED_MATCH_FWD'
                self.iteration = 0
                return
            self.get_logger().info(
                f'[REV {self.iteration}] L={self.rev_left_scale:.3f} R={self.rev_right_scale:.3f}')
            self.record_start()
            self.state = 'REV_STRAIGHT_RUN'

        elif self.state == 'REV_STRAIGHT_RUN':
            if self.elapsed() < self.RUN_DURATION:
                offset = 1500 - self.REVERSE_THROTTLE
                left = 1500 - offset * self.rev_left_scale
                right = 1500 - offset * self.rev_right_scale
                self.send_differential(left, right)
            else:
                self.stop()
                self.test_start_time = now
                self.state = 'REV_STRAIGHT_SETTLE'

        elif self.state == 'REV_STRAIGHT_SETTLE':
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'REV_STRAIGHT_ANALYSE'

        elif self.state == 'REV_STRAIGHT_ANALYSE':
            drift_deg, dist = self.measure_drift_and_distance()
            self.reverse_speed = dist / self.RUN_DURATION
            self.results.append({
                'test': 'reverse_straight', 'iteration': self.iteration,
                'heading_drift_deg': round(drift_deg, 2),
                'distance_m': round(dist, 3),
                'left_scale': round(self.rev_left_scale, 4),
                'right_scale': round(self.rev_right_scale, 4),
            })
            self.get_logger().info(f'  Drift: {drift_deg:.1f}°  Dist: {dist:.2f}m')
            if abs(drift_deg) < self.HEADING_TOLERANCE:
                self.get_logger().info('  ✓ Reverse converged!')
                self.state = 'SPEED_MATCH_FWD'
                self.iteration = 0
                return
            if drift_deg > 0:
                self.rev_left_scale -= self.TRIM_STEP
            else:
                self.rev_right_scale -= self.TRIM_STEP
            self.state = 'REV_STRAIGHT_START'

        # ── PHASE 3: SPEED MATCHING ────────────────────────

        elif self.state == 'SPEED_MATCH_FWD':
            self.get_logger().info('[SPEED MATCH] Measuring forward speed...')
            self.record_start()
            self.state = 'SPEED_MATCH_FWD_RUN'

        elif self.state == 'SPEED_MATCH_FWD_RUN':
            if self.elapsed() < self.RUN_DURATION:
                offset = self.BASE_THROTTLE - 1500
                left = 1500 + offset * self.fwd_left_scale
                right = 1500 + offset * self.fwd_right_scale
                self.send_differential(left, right)
            else:
                self.stop()
                _, dist = self.measure_drift_and_distance()
                self.forward_speed = dist / self.RUN_DURATION
                self.get_logger().info(f'  Forward speed: {self.forward_speed:.3f} m/s')
                self.test_start_time = now
                self.state = 'SPEED_MATCH_FWD_SETTLE'

        elif self.state == 'SPEED_MATCH_FWD_SETTLE':
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'SPEED_MATCH_REV'

        elif self.state == 'SPEED_MATCH_REV':
            self.iteration += 1
            if self.iteration > self.MAX_ITERATIONS:
                self.get_logger().warn('Speed matching did not converge')
                self.state = 'SPIN_START'
                self.iteration = 0
                return
            self.get_logger().info(
                f'[SPEED REV {self.iteration}] mult={self.reverse_throttle_multiplier:.3f}')
            self.record_start()
            self.state = 'SPEED_MATCH_REV_RUN'

        elif self.state == 'SPEED_MATCH_REV_RUN':
            if self.elapsed() < self.RUN_DURATION:
                base_offset = 1500 - self.REVERSE_THROTTLE
                adj = base_offset * self.reverse_throttle_multiplier
                left = 1500 - adj * self.rev_left_scale
                right = 1500 - adj * self.rev_right_scale
                self.send_differential(left, right)
            else:
                self.stop()
                _, dist = self.measure_drift_and_distance()
                self.reverse_speed = dist / self.RUN_DURATION
                self.test_start_time = now
                self.state = 'SPEED_MATCH_REV_SETTLE'

        elif self.state == 'SPEED_MATCH_REV_SETTLE':
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'SPEED_MATCH_ANALYSE'

        elif self.state == 'SPEED_MATCH_ANALYSE':
            diff = self.forward_speed - self.reverse_speed
            self.results.append({
                'test': 'speed_match', 'iteration': self.iteration,
                'forward_speed': round(self.forward_speed, 4),
                'reverse_speed': round(self.reverse_speed, 4),
                'multiplier': round(self.reverse_throttle_multiplier, 4),
            })
            self.get_logger().info(
                f'  Rev: {self.reverse_speed:.3f} m/s (fwd: {self.forward_speed:.3f}, diff: {diff:.3f})')
            if abs(diff) < self.SPEED_TOLERANCE:
                self.get_logger().info('  ✓ Speed match converged!')
                self.state = 'SPIN_START'
                self.iteration = 0
                return
            if self.reverse_speed < self.forward_speed:
                self.reverse_throttle_multiplier += 0.05
            else:
                self.reverse_throttle_multiplier -= 0.05
            self.state = 'SPEED_MATCH_REV'

        # ── PHASE 4: SPIN TESTS ────────────────────────────

        elif self.state == 'SPIN_START':
            self.iteration += 1
            if self.iteration > 2:
                self.state = 'READ_PARAMS'
                return
            direction = 'CW' if self.iteration == 1 else 'CCW'
            self.get_logger().info(f'[SPIN {direction}]')
            self.record_start()
            self.state = 'SPIN_RUN'

        elif self.state == 'SPIN_RUN':
            if self.elapsed() < self.SPIN_DURATION:
                offset = self.SPIN_THROTTLE - 1500
                if self.iteration == 1:
                    left = 1500 + offset * self.fwd_left_scale
                    right = 1500 - offset * self.reverse_throttle_multiplier * self.rev_right_scale
                else:
                    left = 1500 - offset * self.reverse_throttle_multiplier * self.rev_left_scale
                    right = 1500 + offset * self.fwd_right_scale
                self.send_differential(left, right)
            else:
                self.stop()
                self.test_start_time = now
                self.state = 'SPIN_SETTLE'

        elif self.state == 'SPIN_SETTLE':
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'SPIN_ANALYSE'

        elif self.state == 'SPIN_ANALYSE':
            total_yaw, yaw_rate, trans_drift = self.measure_spin()
            direction = 'CW' if self.iteration == 1 else 'CCW'
            self.results.append({
                'test': f'spin_{direction}',
                'total_yaw_deg': round(math.degrees(total_yaw), 2),
                'avg_yaw_rate_dps': round(yaw_rate, 2),
                'translational_drift_m': round(trans_drift, 4),
            })
            self.get_logger().info(
                f'  Yaw: {math.degrees(total_yaw):.1f}° ({yaw_rate:.1f}°/s)  '
                f'Drift: {trans_drift:.3f}m')
            self.state = 'SPIN_START'

        # ── PHASE 5: READ CURRENT PARAMS & APPLY ───────────

        elif self.state == 'READ_PARAMS':
            self.stop()
            self.get_logger().info('')
            self.get_logger().info('  Reading current SERVO parameters from Pixhawk...')
            self.servo_params = self.read_servo_params()
            self.state = 'APPLY_PARAMS'

        elif self.state == 'APPLY_PARAMS':
            new_params = self.compute_new_servo_params()

            self.get_logger().info('')
            self.get_logger().info('  Computed new SERVO parameters:')
            for name, val in sorted(new_params.items()):
                old = self.servo_params.get(name, '?')
                self.get_logger().info(f'    {name}: {old} → {val}')

            success = self.apply_servo_params(new_params)

            if success:
                self.get_logger().info('  ✓ Parameters applied to Pixhawk')
            else:
                self.get_logger().warn('  Some parameters failed to apply — check manually')

            self.state = 'SAVE_RESULTS'
            self.new_params = new_params

        elif self.state == 'SAVE_RESULTS':
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('  CALIBRATION COMPLETE')
            self.get_logger().info('=' * 50)
            self.get_logger().info(f'  Forward:  L={self.fwd_left_scale:.4f}  R={self.fwd_right_scale:.4f}')
            self.get_logger().info(f'  Reverse:  L={self.rev_left_scale:.4f}  R={self.rev_right_scale:.4f}')
            self.get_logger().info(f'  Reverse multiplier: {self.reverse_throttle_multiplier:.4f}')

            out_path = pathlib.Path.home() / 'usv_logs' / 'motor_calibration.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)

            calibration = {
                'scaling_factors': {
                    'forward': {
                        'left_scale': round(self.fwd_left_scale, 4),
                        'right_scale': round(self.fwd_right_scale, 4),
                    },
                    'reverse': {
                        'left_scale': round(self.rev_left_scale, 4),
                        'right_scale': round(self.rev_right_scale, 4),
                        'throttle_multiplier': round(self.reverse_throttle_multiplier, 4),
                    },
                },
                'servo_params': {
                    'original': self.servo_params,
                    'calibrated': self.new_params,
                },
                'qgc_summary': {
                    'description': 'These values have been auto-applied to the Pixhawk. '
                                   'Verify in QGC under Parameters if needed.',
                    **self.new_params,
                },
                'tests': self.results,
            }
            with open(out_path, 'w') as f:
                json.dump(calibration, f, indent=2)
            self.get_logger().info(f'  Saved to {out_path}')

            self.get_logger().info('')
            self.get_logger().info('╔' + '═' * 58 + '╗')
            self.get_logger().info('║  VALUES AUTO-APPLIED TO PIXHAWK (in RAM only)           ║')
            self.get_logger().info('║  Please verify and save permanently in QGroundControl    ║')
            self.get_logger().info('╠' + '═' * 58 + '╣')
            self.get_logger().info('║                                                          ║')
            self.get_logger().info('║  Open QGC → Parameters → search for each param below.   ║')
            self.get_logger().info('║  Confirm the values match, then click "Save to Vehicle"  ║')
            self.get_logger().info('║  to write them permanently to EEPROM.                    ║')
            self.get_logger().info('║                                                          ║')
            for name in sorted(self.new_params.keys()):
                old = self.servo_params.get(name, '?')
                new = self.new_params[name]
                line = f'║  {name:<14}  {old:>5}  →  {new:>5}'
                line = line.ljust(60) + '║'
                self.get_logger().info(line)
            self.get_logger().info('║                                                          ║')
            self.get_logger().info('║  These values are in RAM only and will be lost on        ║')
            self.get_logger().info('║  reboot unless you save them in QGC!                     ║')
            self.get_logger().info('║                                                          ║')
            self.get_logger().info('╚' + '═' * 58 + '╝')

            self.state = 'FINISHED'
            self.timer.cancel()

        elif self.state == 'FINISHED':
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MotorCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()