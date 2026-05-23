import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from motion_capture_tracking_interfaces.msg import NamedPoseArray


class MocapRelay(Node):

    def __init__(self):
        super().__init__('mocap_relay')

        self.min_interval = 1.0 / 50.0  # 50 Hz max
        self.last_pub_time = 0.0

        self.declare_parameter('rigid_body_name', 'Roboship')
        self.declare_parameter('output_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('frame_id', 'map')

        self.body_name = self.get_parameter('rigid_body_name').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        self.pub = self.create_publisher(PoseStamped, output_topic, 10)
        self.sub = self.create_subscription(
            NamedPoseArray, '/poses', self.poses_callback, 10)

        self.get_logger().info(
            f'Mocap relay started — looking for "{self.body_name}" '
            f'on /poses, republishing to {output_topic}')

    def poses_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_pub_time) < self.min_interval:
            return
        for named_pose in msg.poses:
            if named_pose.name == self.body_name:
                out = PoseStamped()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = self.frame_id
                out.pose = named_pose.pose
                self.pub.publish(out)
                self.last_pub_time = now
                return

        self.get_logger().warn(
            f'Rigid body "{self.body_name}" not found in /poses '
            f'(received: {[p.name for p in msg.poses]})',
            throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = MocapRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()