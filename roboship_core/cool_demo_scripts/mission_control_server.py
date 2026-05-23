"""
USV Mission Control Server
───────────────────────────
Extended version of usv_monitor.py that adds:
  - Click-to-set-waypoint from the web UI
  - Multi-waypoint path definition
  - Bounding box display (loads from mocap_boundary.json)
  - Publishes waypoints to ROS for CalmWaterNav or pure pursuit

The browser sends JSON commands over the websocket:
  {"cmd": "set_target", "x": 1.5, "y": 2.0}
  {"cmd": "set_waypoints", "waypoints": [[1,0],[1,1],[0,1]]}
  {"cmd": "clear_waypoints"}
  {"cmd": "start_mission"}
  {"cmd": "stop_mission"}

Usage:
    ros2 run roboship_core mission_control
    Then open mission_control.html in your browser.

Dependencies:
    pip install websockets --break-system-packages
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from mavros_msgs.msg import OverrideRCIn, State
from std_msgs.msg import String
from math import atan2
import json
import time
import asyncio
import threading
import pathlib
import websockets


WS_PORT = 8765
POSE_TOPIC = '/mavros/vision_pose/pose'
BOUNDARY_FILE = pathlib.Path.home() / 'usv_logs' / 'mocap_boundary.json'


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return atan2(siny, cosy)


class MissionControlServer(Node):

    def __init__(self):
        super().__init__('mission_control')

        # Current state
        self.latest_pose = None
        self.waypoints = []         # list of (x, y) from the web UI
        self.mission_active = False

        # Load bounding box if it exists
        self.boundary = self._load_boundary()

        # Pose subscriber
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PoseStamped, POSE_TOPIC, self._pose_cb, qos)
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)

        # Waypoint publisher — other nodes subscribe to this
        self.wp_pub = self.create_publisher(PoseArray, '/usv/waypoints', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/usv/target', 10)
        self.cmd_pub = self.create_publisher(String, '/usv/mission_cmd', 10)

        # Websocket
        self._clients = set()
        self._push_event = threading.Event()
        self._ws_thread = threading.Thread(target=self._start_ws_server, daemon=True)
        self._ws_thread.start()

        # Broadcast timer
        self.create_timer(0.1, self._broadcast_tick)  # 10 Hz

        self.get_logger().info(f'Mission control server started on ws://0.0.0.0:{WS_PORT}')
        if self.boundary:
            self.get_logger().info(
                f'Loaded boundary with {len(self.boundary)} points')

    # ── ROS callbacks ──────────────────────────────────────────

    def _pose_cb(self, msg):
        self.latest_pose = msg
        self._push_event.set()

    def _state_cb(self, msg):
        self.mavros_state = msg

    # ── Boundary ───────────────────────────────────────────────

    def _load_boundary(self):
        if BOUNDARY_FILE.exists():
            with open(BOUNDARY_FILE) as f:
                data = json.load(f)
            points = data.get('boundary_points', [])
            return [(p['x'], p['y']) for p in points]
        return None

    # ── Waypoint publishing ────────────────────────────────────

    def publish_waypoints(self):
        """Publish the current waypoint list as a PoseArray."""
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for wx, wy in self.waypoints:
            p = Pose()
            p.position.x = float(wx)
            p.position.y = float(wy)
            p.position.z = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)
        self.wp_pub.publish(msg)

    def publish_target(self, x, y):
        """Publish a single target position."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)

    # ── Handle commands from browser ───────────────────────────

    def handle_command(self, data):
        cmd = data.get('cmd', '')

        if cmd == 'set_target':
            x, y = data['x'], data['y']
            self.get_logger().info(f'Target set: ({x:.2f}, {y:.2f})')
            self.publish_target(x, y)

        elif cmd == 'set_waypoints':
            self.waypoints = [tuple(wp) for wp in data['waypoints']]
            self.get_logger().info(f'Received {len(self.waypoints)} waypoints from web UI')
            self.publish_waypoints()

        elif cmd == 'add_waypoint':
            x, y = data['x'], data['y']
            self.waypoints.append((x, y))
            self.get_logger().info(
                f'Waypoint added: ({x:.2f}, {y:.2f}) — total: {len(self.waypoints)}')
            self.publish_waypoints()

        elif cmd == 'clear_waypoints':
            self.waypoints = []
            self.get_logger().info('Waypoints cleared')
            self.publish_waypoints()

        elif cmd == 'start_mission':
            self.mission_active = True
            msg = String()
            msg.data = 'start'
            self.cmd_pub.publish(msg)
            self.get_logger().info('Mission start command sent')

        elif cmd == 'stop_mission':
            self.mission_active = False
            msg = String()
            msg.data = 'stop'
            self.cmd_pub.publish(msg)
            self.get_logger().info('Mission stop command sent')

        elif cmd == 'undo_waypoint':
            if self.waypoints:
                removed = self.waypoints.pop()
                self.get_logger().info(
                    f'Removed waypoint ({removed[0]:.2f}, {removed[1]:.2f})')
                self.publish_waypoints()

    # ── Websocket server ───────────────────────────────────────

    def _start_ws_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_main())

    async def _ws_main(self):
        async with websockets.serve(self._ws_handler, '0.0.0.0', WS_PORT):
            await asyncio.Future()

    async def _ws_handler(self, ws, path=None):
        self._clients.add(ws)
        self.get_logger().info(f'Browser connected ({len(self._clients)} clients)')

        # Send boundary on connect
        if self.boundary:
            await ws.send(json.dumps({
                'type': 'boundary',
                'points': [{'x': p[0], 'y': p[1]} for p in self.boundary],
            }))

        # Send current waypoints
        if self.waypoints:
            await ws.send(json.dumps({
                'type': 'waypoints',
                'points': [{'x': w[0], 'y': w[1]} for w in self.waypoints],
            }))

        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    self.handle_command(data)
                    # Broadcast updated waypoints to all clients
                    wp_msg = json.dumps({
                        'type': 'waypoints',
                        'points': [{'x': w[0], 'y': w[1]} for w in self.waypoints],
                    })
                    await asyncio.gather(
                        *[c.send(wp_msg) for c in self._clients],
                        return_exceptions=True)
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            self.get_logger().info(f'Browser disconnected ({len(self._clients)} clients)')

    # ── Broadcast pose to browsers ─────────────────────────────

    def _broadcast_tick(self):
        if self.latest_pose is None or not self._clients:
            return

        pos = self.latest_pose.pose.position
        yaw = yaw_from_quat(self.latest_pose.pose.orientation)

        payload = json.dumps({
            'type': 'pose',
            'x': round(pos.x, 4),
            'y': round(pos.y, 4),
            'yaw': round(yaw, 4),
            'mission_active': self.mission_active,
        })

        # Fire-and-forget broadcast
        dead = set()
        for ws in self._clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send(payload),
                    asyncio.get_event_loop())
            except Exception:
                dead.add(ws)
        self._clients -= dead


def main(args=None):
    rclpy.init(args=args)
    node = MissionControlServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()