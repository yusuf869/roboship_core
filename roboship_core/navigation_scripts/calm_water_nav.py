"""
CalmWaterNav — Position Setpoint Version
=========================================
Publishes waypoints as SET_POSITION_TARGET_LOCAL_NED position setpoints
via /mavros/setpoint_raw/local. ArduRover's L1 controller handles all
heading control, speed management, and motor mixing.

This is a test to determine whether ArduRover actually responds to
position setpoints in GUIDED mode via MAVROS (see ArduPilot #4488).

IMPORTANT — MAVROS expects positions in ENU (ROS convention):
    x = East
    y = North
    z = Up
It converts to NED internally before sending to the Pixhawk.

Prerequisites before running:
    1. SYSID_MYGCS = 1
    2. EKF3 sources set to ExternalNav (6)
    3. Mocap pipeline running (qualisys → relay → /mavros/vision_pose/pose)
    4. Global origin set (see set_origin() below)
    5. Switch to GUIDED mode on transmitter
    6. Arm on transmitter or via MAVROS
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from geographic_msgs.msg import GeoPointStamped
from rclpy.qos import qos_profile_sensor_data
import math


class CalmWaterNav(Node):
    def __init__(self):
        super().__init__('calm_water_nav')

        self.current_state = State()
        self.current_pose = PoseStamped()
        self.have_pose = False
        self.origin_set = False

        # ── Subscribers ────────────────────────────────────────────────
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_cb, qos_profile_sensor_data)

        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_cb,
            qos_profile_sensor_data)

        # ── Publishers ─────────────────────────────────────────────────
        self.setpoint_pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10)

        # Publisher to set the EKF global origin (needed for local NED)
        self.origin_pub = self.create_publisher(
            GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)

        # ── Waypoints in ENU: (East, North) in metres ─────────────────
        self.waypoints = [
            (0.5, 0.5),
            (-0.5, 0.5),
            (-0.5, -0.5),
            (0.5, -0.5),
        ] * 3 # repeat 3 times
        self.current_wp_index = 0
        

        # ── Tuning ────────────────────────────────────────────────────
        self.reached_threshold = 0.2  # m — waypoint capture radius

        # type_mask 3580 = 0b0000_1101_1111_1000
        # Bits set (ignored): vx, vy, vz, ax, ay, az, yaw, yaw_rate
        # Bits clear (used):  x, y, z
        # This tells ArduRover: "I'm giving you position only"
        self.type_mask = 3580

        # MAV_FRAME_LOCAL_NED = 1
        # MAVROS converts our ENU values to NED before sending
        self.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # = 1

        # ── Timers ────────────────────────────────────────────────────
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.origin_timer = self.create_timer(1.0, self.set_origin)  # 1 Hz until set

        self.get_logger().info(
            'CalmWaterNav (position setpoint mode) started. '
            'Waiting for MAVROS...')

    # ── Callbacks ──────────────────────────────────────────────────────

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg
        self.have_pose = True

    # ── Set global origin ──────────────────────────────────────────────

    def set_origin(self):
        """Publish a fake global origin so the Pixhawk can map local NED
        to global coordinates. Uses Plymouth, UK as reference.
        Keeps publishing once per second until we have a pose (meaning
        the EKF has accepted the origin and is producing local position)."""
        if self.origin_set:
            return

        msg = GeoPointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position.latitude = 51.498356
        msg.position.longitude = -0.176894
        msg.position.altitude = 0.0
        self.origin_pub.publish(msg)

        if self.have_pose:
            self.origin_set = True
            self.origin_timer.cancel()
            self.get_logger().info('Global origin set and EKF has pose.')
        else:
            self.get_logger().info(
                'Publishing global origin, waiting for EKF pose...',
                throttle_duration_sec=2.0)

    # ── Main control loop ──────────────────────────────────────────────

    def control_loop(self):
        # Wait for connection and pose
        if not self.current_state.connected or not self.have_pose:
            self.get_logger().info(
                'Waiting for FCU and pose...',
                throttle_duration_sec=2.0)
            return

        # Wait for GUIDED mode
        if self.current_state.mode != "GUIDED":
            self.get_logger().info(
                f'In {self.current_state.mode}. Switch to GUIDED.',
                throttle_duration_sec=5.0)
            return

        # Wait for arming
        if not self.current_state.armed:
            self.get_logger().info(
                'GUIDED active. ARM to begin.',
                throttle_duration_sec=2.0)
            return

        # All waypoints done
        if self.current_wp_index >= len(self.waypoints):
            # Keep publishing last waypoint to hold position
            self.publish_position(self.waypoints[-1])
            self.get_logger().info(
                'All waypoints reached. Holding.',
                throttle_duration_sec=5.0)
            return

        # ── Active navigation ──────────────────────────────────────────
        target_e, target_n = self.waypoints[self.current_wp_index]

        # Current position from EKF (ENU: x=East, y=North)
        curr_e = self.current_pose.pose.position.x
        curr_n = self.current_pose.pose.position.y

        de = target_e - curr_e
        dn = target_n - curr_n
        dist = math.sqrt(de**2 + dn**2)

        self.get_logger().info(
            f'(Lap {self.current_wp_index // 4 + 1}/3)'
            f'WP {self.current_wp_index + 1}/{len(self.waypoints)} '
            f'target=({target_e}, {target_n}) dist={dist:.2f}m',
            throttle_duration_sec=1.0)

        # Check waypoint reached
        if dist < self.reached_threshold:
            self.get_logger().info(
                f'Waypoint {self.current_wp_index + 1} reached!')
            self.current_wp_index += 1

        # Publish target position — ArduRover handles everything else
        if self.current_wp_index < len(self.waypoints):
            self.publish_position(self.waypoints[self.current_wp_index])
        else:
            self.publish_position(self.waypoints[-1])

    def publish_position(self, waypoint):
        """Publish a position-only setpoint in ENU coordinates.
        MAVROS converts to NED internally."""
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = self.coordinate_frame
        msg.type_mask = self.type_mask

        # ENU: x = East, y = North, z = Up
        msg.position.x = float(waypoint[0])  # East
        msg.position.y = float(waypoint[1])  # North
        msg.position.z = 0.0                 # surface vehicle

        self.setpoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CalmWaterNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
