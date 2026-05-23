import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import csv
import time

class PixhawkLogger(Node):
    def __init__(self):
        super().__init__('pixhawk_logger')
        
        # 1. Define a QoS profile that matches MAVROS's Best Effort policy
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 2. Apply the custom QoS profile to the subscription
        self.subscription = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.listener_callback,
            sensor_qos_profile) # <-- This is the critical change
        
        self.csv_file = open('pixhawk_imu_log.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Pi_Timestamp", "Pixhawk_Hardware_Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Q_X", "Q_Y", "Q_Z", "Q_W"])
        
        self.get_logger().info('Listening to Pixhawk IMU (Best Effort QoS). Press Ctrl+C to stop.')

    def listener_callback(self, msg):
        pi_time = time.time()
        pix_time = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
        
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        
        self.csv_writer.writerow([pi_time, pix_time, ax, ay, az, gx, gy, gz, qx, qy, qz, qw])

    def __del__(self):
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = PixhawkLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()