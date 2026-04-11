import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPointStamped
from mavros_msgs.srv import CommandHome
import random
import math

class FakeMocap(Node):
    def __init__(self):
        super().__init__('fake_mocap')

        self.pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        self.origin_pub = self.create_publisher(GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)
        self.set_home_client = self.create_client(CommandHome, '/mavros/cmd/set_home')

        self.home_set = False

        self.x = 0.0
        self.y = 0.0
        self.speed = 0.05

        self.waypoints = [(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)]
        self.wp_index = 0

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.init_timer = self.create_timer(1.0, self.send_origin_and_home)

    def send_origin_and_home(self):
        if self.home_set:
            self.init_timer.cancel()
            return

        origin_msg = GeoPointStamped()
        origin_msg.header.stamp = self.get_clock().now().to_msg()
        origin_msg.header.frame_id = "map"
        origin_msg.position.latitude  = 51.5074
        origin_msg.position.longitude = -0.1278
        origin_msg.position.altitude  = 0.0
        self.origin_pub.publish(origin_msg)

        if self.set_home_client.service_is_ready():
            req = CommandHome.Request()
            req.current_gps = False
            req.yaw       = 0.0
            req.latitude  = 51.5074
            req.longitude = -0.1278
            req.altitude  = 0.0
            self.set_home_client.call_async(req)
            self.home_set = True
            self.init_timer.cancel()

    def timer_callback(self):
        if self.wp_index < len(self.waypoints):
            target_x, target_y = self.waypoints[self.wp_index]
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.sqrt(dx**2 + dy**2)

            if dist > 0.01:
                step = self.speed * 0.1
                self.x += (dx / dist) * step
                self.y += (dy / dist) * step
            else:
                self.wp_index += 1

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = self.x + random.uniform(-0.001, 0.001)
        msg.pose.position.y = self.y + random.uniform(-0.001, 0.001)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeMocap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()