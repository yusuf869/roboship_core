#!/usr/bin/env python3
"""
Motor Auto-Calibration Node (Simplified)
─────────────────────────────────────────
Drives the USV forward and reverse, measures heading drift from
mocap, computes the percentage imbalance between left and right
motors, and pushes corrected SERVO parameters to the Pixhawk.

How it works:
  1. WAIT — Wait for user to arm + set MANUAL mode
  2. READ PARAMS — Read current SERVO9/10 MIN/MAX/TRIM from Pixhawk
  3. FORWARD RUN — Drive forward, measure heading drift
  4. REVERSE RUN — Drive reverse, measure heading drift
  5. COMPUTE & APPLY — Express drift as a percentage motor imbalance,
     scale the SERVO ranges accordingly, push to Pixhawk

The percentage logic:
  - Drive straight (steering centred) for a few seconds
  - Record start heading and end heading from mocap
  - Heading drift = end_heading − start_heading
  - If the boat drifts RIGHT, the LEFT motor is stronger
  - The drift angle over the run distance gives a percentage
    imbalance: correction = drift_deg / RUN_DURATION scaled to
    the SERVO range
  - That percentage is applied to reduce the stronger motor's
    SERVO range (MAX for forward, MIN for reverse)

Safety: The script does NOT arm or change modes automatically.
You must arm the vehicle and set MANUAL mode yourself.

All movement uses RC override on channel 0 (steering) and
channel 2 (throttle) — the same proven approach used by the
pure pursuit controller and all other working movement nodes.

Output: ~/usv_logs/motor_calibration.json + params applied to Pixhawk.

Prerequisites:
    - SYSID_MYGCS = 1, EKF3 sources = ExternalNav
    - Mocap pipeline running
    - Vehicle in water with room to drive ~1m in each direction

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
from mavros_msgs.msg import OverrideRCIn, State
from mavros_msgs.srv import ParamGet, ParamSet
from geographic_msgs.msg import GeoPointStamped


def yaw_from_quaternion(q):
    """Extract yaw from quaternion (ENU frame)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def normalise_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class MotorCalibration(Node):

    # ── Tunables ────────────────────────────────────────────────
    FWD_THROTTLE = 1560             # PWM for forward test
    REV_THROTTLE = 1440             # PWM for reverse test
    RUN_DURATION = 3.0              # seconds to drive per test
    SETTLE_TIME  = 2.0              # seconds to wait between runs
    HEADING_TOL_DEG = 2.0           # below this = "straight enough"

    # Max correction per run (prevents wild over-correction)
    MAX_CORRECTION_PERCENT = 15.0

    # Scaling: degrees-of-drift per second → percentage motor imbalance
    # This is a tuning knob. At ~0.2 m/s with 60µs PWM range,
    # 1°/s drift ≈ ~2% thrust imbalance. Adjust if corrections
    # are too aggressive or too timid.
    DRIFT_TO_PERCENT = 2.0

    # Which SERVO outputs drive left and right motors
    LEFT_SERVO  = 'SERVO9'
    RIGHT_SERVO = 'SERVO10'

    def __init__(self):
        super().__init__('motor_calibration')

        # ── State machine ──────────────────────────────────────
        self.state = 'WAIT_FOR_ARM'
        self.test_start_time = None
        self.start_yaw = None
        self.start_pos = None
        self.current_pose = None
        self.mavros_state = None
        self.last_wait_log = 0.0
        self.origin_set = False

        # Results log
        self.results = []
        self.servo_params = {}
        self.new_params = {}

        # Drift measurements (degrees)
        self.fwd_drift_deg = 0.0
        self.rev_drift_deg = 0.0
        self.fwd_distance  = 0.0
        self.rev_distance  = 0.0

        # ── Subscribers ────────────────────────────────────────
        # BEST_EFFORT QoS — matches mocap pipeline and MAVROS defaults
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(
            PoseStamped, '/mavros/vision_pose/pose', self.pose_cb, qos)
        self.create_subscription(
            State, '/mavros/state', self.state_cb, 10)

        # ── Publishers ─────────────────────────────────────────
        # RC override: channel 0 = steering, channel 2 = throttle
        # This is the proven working movement method
        self.rc_pub = self.create_publisher(
            OverrideRCIn, '/mavros/rc/override', 10)
        self.origin_pub = self.create_publisher(
            GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)

        # ── Services ───────────────────────────────────────────
        self.param_get_client = self.create_client(
            ParamGet, '/mavros/param/get')
        self.param_set_client = self.create_client(
            ParamSet, '/mavros/param/set')

        # ── Timers ─────────────────────────────────────────────
        self.timer = self.create_timer(0.05, self.loop)          # 20 Hz
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
        """Publish a dummy global origin so the EKF has a reference.
        Stops once we have a pose (origin accepted)."""
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

    # ── Motor commands (proven RC override pattern) ─────────────
    #
    #   Channel 0 = steering PWM   (1500 = centre)
    #   Channel 2 = throttle PWM   (1500 = neutral)
    #
    #   ArduRover FRAME_CLASS=2 mixer converts these into
    #   left/right motor PWM via SERVO9/10_FUNCTION = 73/74.

    def send_rc(self, throttle, steering=0):
        """Send RC override.
        throttle: raw PWM (>1500 = forward, <1500 = reverse)
        steering: offset from 1500 (+ = turn right, - = turn left)
        """
        rc = OverrideRCIn()
        rc.channels = [0] * 18
        rc.channels[0] = max(1000, min(2000, 1500 + int(steering)))
        rc.channels[2] = max(1000, min(2000, int(throttle)))
        self.rc_pub.publish(rc)

    def stop(self):
        """Neutral throttle, centred steering."""
        rc = OverrideRCIn()
        rc.channels = [0] * 18
        rc.channels[0] = 1500
        rc.channels[2] = 1500
        self.rc_pub.publish(rc)

    # ── Measurement helpers ─────────────────────────────────────

    def record_start(self):
        """Snapshot current heading and position at start of a run."""
        pos = self.current_pose.pose.position
        self.start_yaw = yaw_from_quaternion(self.current_pose.pose.orientation)
        self.start_pos = (pos.x, pos.y)
        self.test_start_time = self.get_clock().now().nanoseconds / 1e9

    def elapsed(self):
        return self.get_clock().now().nanoseconds / 1e9 - self.test_start_time

    def measure_drift(self):
        """Return (heading_drift_degrees, distance_travelled)."""
        pos = self.current_pose.pose.position
        yaw = yaw_from_quaternion(self.current_pose.pose.orientation)
        drift_deg = math.degrees(normalise_angle(yaw - self.start_yaw))
        dx = pos.x - self.start_pos[0]
        dy = pos.y - self.start_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return drift_deg, dist

    # ── Param read/write ───────────────────────────────────────

    def get_param(self, name):
        if not self.param_get_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Param get service not available')
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
            self.get_logger().error('Param set service not available')
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
        """Read current SERVO9/10 MIN, MAX, TRIM from Pixhawk."""
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
                    default = {'_MIN': 1000, '_MAX': 2000, '_TRIM': 1500}
                    params[name] = default[suffix]
        return params

    # ── Correction computation ──────────────────────────────────

    def compute_correction(self):
        """
        From the measured heading drift, compute new SERVO params.

        Logic:
          - Forward drift > 0 (drifted RIGHT) → left motor is stronger
            → reduce LEFT SERVO_MAX (shrink its forward range)
          - Forward drift < 0 (drifted LEFT) → right motor is stronger
            → reduce RIGHT SERVO_MAX

          - Reverse drift > 0 (drifted RIGHT) → right motor is stronger
            in reverse → reduce RIGHT SERVO_MIN range
          - Reverse drift < 0 (drifted LEFT) → left motor is stronger
            in reverse → reduce LEFT SERVO_MIN range

        The percentage correction is:
          drift_rate = drift_deg / RUN_DURATION  (degrees per second)
          correction_pct = drift_rate * DRIFT_TO_PERCENT
          clamped to MAX_CORRECTION_PERCENT
        """
        p = dict(self.servo_params)  # copy
        new = dict(p)

        # ── Forward correction ──────────────────────────────
        fwd_rate = self.fwd_drift_deg / self.RUN_DURATION
        fwd_pct = fwd_rate * self.DRIFT_TO_PERCENT
        fwd_pct = max(-self.MAX_CORRECTION_PERCENT,
                      min(self.MAX_CORRECTION_PERCENT, fwd_pct))

        self.get_logger().info(f'  Forward drift: {self.fwd_drift_deg:.1f}° '
                               f'over {self.RUN_DURATION}s '
                               f'= {fwd_rate:.2f}°/s '
                               f'→ {fwd_pct:.1f}% imbalance')

        if abs(fwd_pct) > 0.5:  # only correct if meaningful
            if fwd_pct > 0:
                # Drifted right → left motor too strong → shrink left forward range
                servo = self.LEFT_SERVO
            else:
                # Drifted left → right motor too strong → shrink right forward range
                servo = self.RIGHT_SERVO

            trim = p[f'{servo}_TRIM']
            old_max = p[f'{servo}_MAX']
            fwd_range = old_max - trim
            reduction = int(fwd_range * abs(fwd_pct) / 100.0)
            new_max = old_max - reduction
            new_max = max(trim + 1, new_max)  # don't go below trim
            new[f'{servo}_MAX'] = new_max

            self.get_logger().info(
                f'  → {servo}_MAX: {old_max} → {new_max} '
                f'(reduced by {reduction} PWM)')
        else:
            self.get_logger().info('  → Forward drift within tolerance, no correction')

        # ── Reverse correction ──────────────────────────────
        rev_rate = self.rev_drift_deg / self.RUN_DURATION
        rev_pct = rev_rate * self.DRIFT_TO_PERCENT
        rev_pct = max(-self.MAX_CORRECTION_PERCENT,
                      min(self.MAX_CORRECTION_PERCENT, rev_pct))

        self.get_logger().info(f'  Reverse drift: {self.rev_drift_deg:.1f}° '
                               f'over {self.RUN_DURATION}s '
                               f'= {rev_rate:.2f}°/s '
                               f'→ {rev_pct:.1f}% imbalance')

        if abs(rev_pct) > 0.5:
            if rev_pct > 0:
                # Drifted right in reverse → right motor stronger in reverse
                # → shrink right reverse range (increase MIN towards trim)
                servo = self.RIGHT_SERVO
            else:
                # Drifted left in reverse → left motor stronger in reverse
                servo = self.LEFT_SERVO

            trim = p[f'{servo}_TRIM']
            old_min = new.get(f'{servo}_MIN', p[f'{servo}_MIN'])
            rev_range = trim - old_min
            reduction = int(rev_range * abs(rev_pct) / 100.0)
            new_min = old_min + reduction
            new_min = min(trim - 1, new_min)  # don't go above trim
            new[f'{servo}_MIN'] = new_min

            self.get_logger().info(
                f'  → {servo}_MIN: {old_min} → {new_min} '
                f'(increased by {reduction} PWM)')
        else:
            self.get_logger().info('  → Reverse drift within tolerance, no correction')

        return new

    # ── State machine ──────────────────────────────────────────

    def loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        # ── WAIT FOR ARM + MANUAL ──────────────────────────

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
                        f'  Armed ✓ — Please switch to MANUAL mode '
                        f'(current: {mode})')

            if (self.mavros_state.connected
                    and self.mavros_state.armed
                    and self.mavros_state.mode == 'MANUAL'):
                self.get_logger().info('  ✓ Armed in MANUAL mode — detected!')
                self.get_logger().info('  Reading SERVO parameters...')
                self.state = 'READ_PARAMS'
            return

        # ── READ CURRENT SERVO PARAMS ──────────────────────

        if self.state == 'READ_PARAMS':
            self.servo_params = self.read_servo_params()
            self.get_logger().info('')
            self.get_logger().info('  Settling before forward run...')
            self.test_start_time = now
            self.state = 'PRE_FWD_SETTLE'
            return

        if self.state == 'PRE_FWD_SETTLE':
            self.stop()
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'FWD_START'
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

        # ── FORWARD RUN ────────────────────────────────────
        #
        #  Drive forward with centred steering. The skid-steer
        #  mixer in ArduRover converts throttle+steering to
        #  left/right motor PWM. If the motors aren't matched,
        #  the heading will drift.

        if self.state == 'FWD_START':
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('  FORWARD TEST')
            self.get_logger().info('=' * 50)
            self.record_start()
            self.state = 'FWD_RUN'

        elif self.state == 'FWD_RUN':
            if self.elapsed() < self.RUN_DURATION:
                # Straight forward — steering centred at 0
                self.send_rc(throttle=self.FWD_THROTTLE, steering=0)
            else:
                self.stop()
                self.fwd_drift_deg, self.fwd_distance = self.measure_drift()
                fwd_speed = self.fwd_distance / self.RUN_DURATION
                self.get_logger().info(
                    f'  Heading drift: {self.fwd_drift_deg:+.1f}°')
                self.get_logger().info(
                    f'  Distance:      {self.fwd_distance:.3f}m '
                    f'({fwd_speed:.3f} m/s)')
                self.results.append({
                    'test': 'forward',
                    'drift_deg': round(self.fwd_drift_deg, 2),
                    'distance_m': round(self.fwd_distance, 3),
                    'speed_ms': round(fwd_speed, 4),
                    'throttle_pwm': self.FWD_THROTTLE,
                })
                self.test_start_time = now
                self.state = 'FWD_SETTLE'

        elif self.state == 'FWD_SETTLE':
            self.stop()
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'REV_START'

        # ── REVERSE RUN ────────────────────────────────────

        elif self.state == 'REV_START':
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('  REVERSE TEST')
            self.get_logger().info('=' * 50)
            self.record_start()
            self.state = 'REV_RUN'

        elif self.state == 'REV_RUN':
            if self.elapsed() < self.RUN_DURATION:
                # Straight reverse — steering centred at 0
                self.send_rc(throttle=self.REV_THROTTLE, steering=0)
            else:
                self.stop()
                self.rev_drift_deg, self.rev_distance = self.measure_drift()
                rev_speed = self.rev_distance / self.RUN_DURATION
                self.get_logger().info(
                    f'  Heading drift: {self.rev_drift_deg:+.1f}°')
                self.get_logger().info(
                    f'  Distance:      {self.rev_distance:.3f}m '
                    f'({rev_speed:.3f} m/s)')
                self.results.append({
                    'test': 'reverse',
                    'drift_deg': round(self.rev_drift_deg, 2),
                    'distance_m': round(self.rev_distance, 3),
                    'speed_ms': round(rev_speed, 4),
                    'throttle_pwm': self.REV_THROTTLE,
                })
                self.test_start_time = now
                self.state = 'REV_SETTLE'

        elif self.state == 'REV_SETTLE':
            self.stop()
            if (now - self.test_start_time) >= self.SETTLE_TIME:
                self.state = 'COMPUTE'

        # ── COMPUTE & APPLY ────────────────────────────────

        elif self.state == 'COMPUTE':
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('  COMPUTING CORRECTIONS')
            self.get_logger().info('=' * 50)

            self.new_params = self.compute_correction()

            # Show what changed
            self.get_logger().info('')
            self.get_logger().info('  New SERVO parameters:')
            changed = False
            for name in sorted(self.new_params.keys()):
                old = self.servo_params.get(name, '?')
                new = self.new_params[name]
                marker = ' ✓' if old != new else ''
                self.get_logger().info(
                    f'    {name}: {old} → {new}{marker}')
                if old != new:
                    changed = True

            if not changed:
                self.get_logger().info('')
                self.get_logger().info('  No corrections needed — motors are '
                                       'within tolerance!')
                self.state = 'SAVE_RESULTS'
                return

            self.state = 'APPLY_PARAMS'

        elif self.state == 'APPLY_PARAMS':
            self.get_logger().info('')
            self.get_logger().info('  Applying to Pixhawk...')
            success = True
            for name, value in self.new_params.items():
                old = self.servo_params.get(name)
                if old != value:
                    if not self.set_param(name, value):
                        success = False

            if success:
                self.get_logger().info('  ✓ Parameters applied')
            else:
                self.get_logger().warn('  Some parameters failed — check manually')

            self.state = 'SAVE_RESULTS'

        # ── SAVE & FINISH ──────────────────────────────────

        elif self.state == 'SAVE_RESULTS':
            self.stop()

            # ── Summary ────────────────────────────────────
            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('  CALIBRATION COMPLETE')
            self.get_logger().info('=' * 50)
            self.get_logger().info(
                f'  Forward drift: {self.fwd_drift_deg:+.1f}° '
                f'over {self.fwd_distance:.2f}m')
            self.get_logger().info(
                f'  Reverse drift: {self.rev_drift_deg:+.1f}° '
                f'over {self.rev_distance:.2f}m')

            # ── Save JSON ──────────────────────────────────
            out_path = pathlib.Path.home() / 'usv_logs' / 'motor_calibration.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)

            calibration = {
                'measurements': {
                    'forward': {
                        'drift_deg': round(self.fwd_drift_deg, 2),
                        'distance_m': round(self.fwd_distance, 3),
                        'throttle_pwm': self.FWD_THROTTLE,
                        'run_duration_s': self.RUN_DURATION,
                    },
                    'reverse': {
                        'drift_deg': round(self.rev_drift_deg, 2),
                        'distance_m': round(self.rev_distance, 3),
                        'throttle_pwm': self.REV_THROTTLE,
                        'run_duration_s': self.RUN_DURATION,
                    },
                },
                'servo_params': {
                    'original': self.servo_params,
                    'calibrated': self.new_params,
                },
                'qgc_summary': {
                    'description':
                        'These values have been auto-applied to the Pixhawk '
                        '(RAM only). Verify in QGC under Parameters and '
                        'save permanently if happy.',
                    **{k: v for k, v in self.new_params.items()
                       if v != self.servo_params.get(k)},
                },
                'tests': self.results,
            }
            with open(out_path, 'w') as f:
                json.dump(calibration, f, indent=2)
            self.get_logger().info(f'  Saved to {out_path}')

            # ── QGC reminder box ───────────────────────────
            self.get_logger().info('')
            self.get_logger().info('╔' + '═' * 58 + '╗')
            self.get_logger().info(
                '║  VALUES AUTO-APPLIED TO PIXHAWK (in RAM only)           ║')
            self.get_logger().info(
                '║  Please verify and save permanently in QGroundControl    ║')
            self.get_logger().info('╠' + '═' * 58 + '╣')
            self.get_logger().info(
                '║                                                          ║')
            for name in sorted(self.new_params.keys()):
                old = self.servo_params.get(name, '?')
                new = self.new_params[name]
                if old != new:
                    line = f'║  {name:<14}  {old:>5}  →  {new:>5}'
                    line = line.ljust(60) + '║'
                    self.get_logger().info(line)
            self.get_logger().info(
                '║                                                          ║')
            self.get_logger().info(
                '║  These values are in RAM only and will be lost on        ║')
            self.get_logger().info(
                '║  reboot unless you save them in QGC!                     ║')
            self.get_logger().info(
                '╚' + '═' * 58 + '╝')

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