import rclpy
from rclpy.node import Node

class StatusBroadcaster(Node):
    def __init__(self):
        super().__init__('system_status_node')
        self.get_logger().info(''Roboship Core Systems: ONLINE. Awaiting Pixhawk Connection...'')

def main(args=None):
    rclpy.init(args=args)
    node = StatusBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()