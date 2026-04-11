"""
Solves the VisOdom not healthy error
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MocapKeepalive(Node):
    def __init__(self):
        super().__init__('mocap_keepalive')
        self.pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        self.create_timer(0.1, self.cb)

    def cb(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MocapKeepalive())

if __name__ == '__main__':
    main()