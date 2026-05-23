import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import csv, time

class MocapDecayLogger(Node):
    def __init__(self):
        super().__init__('mocap_decay_logger')
        self.sub = self.create_subscription(
            PoseStamped, '/mavros/vision_pose/pose', self.cb, 50)
        fname = f'mocap_decay_{int(time.time())}.csv'
        self.f = open(fname, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.t0 = None
        self.get_logger().info(f'Logging to {fname}')

    def cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        p, q = msg.pose.position, msg.pose.orientation
        self.w.writerow([f'{t:.4f}', f'{p.x:.5f}', f'{p.y:.5f}',
                         f'{p.z:.5f}', f'{q.x:.5f}', f'{q.y:.5f}',
                         f'{q.z:.5f}', f'{q.w:.5f}'])

    def destroy_node(self):
        self.f.close()
        super().destroy_node()

def main():
    rclpy.init()
    n = MocapDecayLogger()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node()
    rclpy.shutdown()