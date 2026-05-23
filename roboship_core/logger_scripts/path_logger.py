#!/usr/bin/env python3
"""
Path Logger Node
────────────────
Subscribes to the USV pose topic, logs to CSV, and broadcasts
live position data over a WebSocket so you can visualise from
any browser on the same network.

Usage:
    ros2 run <your_pkg> path_logger          # if installed
    python3 path_logger.py                    # standalone

Dependencies (pip):
    pip install websockets --break-system-packages
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from math import atan2, degrees
import csv, json, time, threading, asyncio, pathlib

# ── TUNABLES ────────────────────────────────────────────────────
POSE_TOPIC = "/mavros/vision_pose/pose"   # change if using raw Qualisys topic
CSV_DIR    = pathlib.Path.home() / "usv_logs"
WS_PORT    = 8765                             # websocket port

# Waypoints (x, y) in the same frame as mocap / EKF local position
WAYPOINTS = [
    (1.0, 1.0),
    (4.0, 1.0),
    (4.0, 4.0),
    (1.0, 4.0),
]

# Tank boundary for the live viz (metres). Adjust to match your tank.
TANK_X_MIN, TANK_X_MAX = -0.5, 5.0
TANK_Y_MIN, TANK_Y_MAX = -0.5, 5.0
# ────────────────────────────────────────────────────────────────


def quat_to_yaw(q):
    """Extract yaw (degrees) from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return degrees(atan2(siny, cosy))


class PathLogger(Node):
    def __init__(self):
        super().__init__("path_logger")

        # ── CSV setup ───────────────────────────────────────────
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = CSV_DIR / f"usv_log_{stamp}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["t_sec", "x", "y", "z", "yaw_deg"])
        self.t0 = None
        self.get_logger().info(f"Logging to {self.csv_path}")

        # ── WebSocket state ─────────────────────────────────────
        self.ws_clients: set = set()
        self.latest_msg: str = ""
        self._start_ws_server()

        # ── ROS2 subscriber ─────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PoseStamped, POSE_TOPIC, self._pose_cb, qos)
        self.get_logger().info(f"Subscribed to {POSE_TOPIC}")
        self.get_logger().info(f"Open live_viz.html and connect to ws://<server_ip>:{WS_PORT}")

    # ── pose callback ───────────────────────────────────────────
    def _pose_cb(self, msg: PoseStamped):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        yaw = quat_to_yaw(msg.pose.orientation)

        # log to CSV
        self.csv_writer.writerow([f"{t:.4f}", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", f"{yaw:.2f}"])
        self.csv_file.flush()

        # broadcast to websocket clients
        self.latest_msg = json.dumps({
            "t": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "yaw": round(yaw, 2),
            "waypoints": WAYPOINTS,
            "tank": [TANK_X_MIN, TANK_X_MAX, TANK_Y_MIN, TANK_Y_MAX],
        })
        # non-blocking push — actual send happens in the asyncio thread
        self._push_event.set()

    # ── WebSocket server (runs in a background thread) ──────────
    def _start_ws_server(self):
        self._push_event = threading.Event()
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_ws_loop, daemon=True)
        t.start()

    def _run_ws_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_main())

    async def _ws_main(self):
        import websockets
        async with websockets.serve(self._ws_handler, "0.0.0.0", WS_PORT):
            self.get_logger().info(f"WebSocket server on port {WS_PORT}")
            while True:
                # wait for new data from the ROS callback
                await asyncio.get_event_loop().run_in_executor(None, self._push_event.wait)
                self._push_event.clear()
                if self.ws_clients and self.latest_msg:
                    await asyncio.gather(
                        *[c.send(self.latest_msg) for c in self.ws_clients],
                        return_exceptions=True,
                    )

    async def _ws_handler(self, ws):
        self.ws_clients.add(ws)
        self.get_logger().info(f"Client connected ({len(self.ws_clients)} total)")
        try:
            await ws.wait_closed()
        finally:
            self.ws_clients.discard(ws)
            self.get_logger().info(f"Client disconnected ({len(self.ws_clients)} total)")

    def destroy_node(self):
        self.csv_file.close()
        print(f"[path_logger] CSV saved: {self.csv_path}") 
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()