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

        self.x = -3
        self.y = -1
        self.speed = 0.5

        self.waypoints = [(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)]
        self.wp_index = 0
        
        # --- KINEMATICS & NOISE ---
        self.dt = 0.1                       # Timer interval in seconds
        self.current_heading = 0.0          # Actual physical yaw of the USV (radians)
        self.max_turn_rate = 0.8            # Max radians the boat can turn per second (~45 deg/s)
        
        self.time_elapsed = 0.0
        self.drift_amplitude = 0.4          # Max noise angle
        self.drift_frequency = 0.8          # Speed of the meandering oscillation

        self.timer = self.create_timer(self.dt, self.timer_callback)
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
        self.time_elapsed += self.dt

        if self.wp_index < len(self.waypoints):
            target_x, target_y = self.waypoints[self.wp_index]
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.sqrt(dx**2 + dy**2)

            if dist > 0.15: 
                step = self.speed * self.dt
                
                # 1. Calculate the ideal line-of-sight heading
                ideal_heading = math.atan2(dy, dx)
                
                # 2. Calculate the shortest angular difference to the ideal heading
                # This ensures the boat turns left if the target is to the left, etc.
                angle_diff = (ideal_heading - self.current_heading + math.pi) % (2 * math.pi) - math.pi
                
                # 3. Limit the turn based on the boat's physical max turn rate
                max_turn_this_step = self.max_turn_rate * self.dt
                actual_turn = max(-max_turn_this_step, min(max_turn_this_step, angle_diff))
                
                # Apply the turn to the boat's actual heading
                self.current_heading += actual_turn
                
                # Keep heading normalized between -pi and pi
                self.current_heading = (self.current_heading + math.pi) % (2 * math.pi) - math.pi
                
                # 4. Introduce environmental drift (noise) ON TOP of the physical heading
                heading_noise = (math.sin(self.time_elapsed * self.drift_frequency) * self.drift_amplitude)
                movement_heading = self.current_heading + heading_noise
                
                # 5. Move the USV forward along the noisy movement vector
                self.x += math.cos(movement_heading) * step
                self.y += math.sin(movement_heading) * step
            else:
                self.wp_index = (self.wp_index + 1) % len(self.waypoints)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # Add slight GPS/Mocap position jitter
        msg.pose.position.x = self.x + random.uniform(-0.005, 0.005)
        msg.pose.position.y = self.y + random.uniform(-0.005, 0.005)
        msg.pose.position.z = 0.0
        
        # Convert our yaw (current_heading) into a Quaternion so visualizers show the boat rotating!
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(self.current_heading / 2.0)
        msg.pose.orientation.w = math.cos(self.current_heading / 2.0)
        
        self.pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeMocap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()