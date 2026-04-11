import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import math
from rclpy.qos import qos_profile_sensor_data

class CalmWaterNav(Node):
    def __init__(self):
        super().__init__('calm_water_nav')

        self.current_state = State()
        self.current_pose = PoseStamped()
        self.mission_started = False
        self.startup_counter = 0  # counts loops before arming attempt

        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_cb, qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos_profile_sensor_data)

        self.local_pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- Define your 4 waypoints (X, Y) in MoCap frame (metres) ---
        self.waypoints = [
            (1.0, 1.0),
            (4.0, 1.0),
            (4.0, 4.0),
            (1.0, 4.0),
        ]
        self.current_wp_index = 0
        self.reached_threshold = 0.3

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('USV Navigator initialised. Waiting for MAVROS connection...')

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg

    def get_current_pose_as_setpoint(self):
        """Returns the current position as a setpoint (used for hold / startup streaming)."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = self.current_pose.pose.position.x
        pose.pose.position.y = self.current_pose.pose.position.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = self.current_pose.pose.orientation
        return pose

    def build_waypoint_setpoint(self, x, y):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0 # Z is at zero (assuming very small vertical movment/ disturbances)
        return pose
        
    def control_loop(self):
        if not self.current_state.connected:
            self.get_logger().info('Waiting for FCU connection...', throttle_duration_sec=2.0)
            return

        # --- Phase 1: The Idle State (Waiting for SWC) ---
        # If SWC is NOT in the GUIDED position, the script does nothing but stream the current position.
        # (Streaming current position is required so ArduPilot lets you switch to GUIDED when you flip SWC)
        if self.current_state.mode != "GUIDED":
            self.local_pos_pub.publish(self.get_current_pose_as_setpoint())
            self.get_logger().info(f'Idling in {self.current_state.mode}. Flip SWC to GUIDED to start mission.', throttle_duration_sec=5.0)
            # Reset mission if we flick out of GUIDED mode (acts as a pause/reset)
            # self.current_wp_index = 0 # Uncomment if you want flipping the switch to restart the mission
            return

        # --- Phase 2: The Arming State (Waiting for SWD) ---
        # If SWC is GUIDED, but SWD is not Armed, wait for the user to arm it.
        if not self.current_state.armed:
            self.local_pos_pub.publish(self.get_current_pose_as_setpoint())
            self.get_logger().info('In GUIDED mode. Flip SWD to ARM and begin!', throttle_duration_sec=2.0)
            return

        # ==========================================
        # IF WE REACH HERE: SWD IS ARMED & SWC IS GUIDED. MISSION IS GO!
        # ==========================================
        
        # --- Phase 3: Mission Complete ---
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info('Mission complete. Holding position.', throttle_duration_sec=5.0)
            last_x, last_y = self.waypoints[-1]
            self.local_pos_pub.publish(self.build_waypoint_setpoint(last_x, last_y))
            return

        # --- Phase 4: Normal Waypoint Navigation ---
        target_x, target_y = self.waypoints[self.current_wp_index] # Go to waypoint 0 first.
        
        
        
        dist = math.sqrt(
            (self.current_pose.pose.position.x - target_x) ** 2 +
            (self.current_pose.pose.position.y - target_y) ** 2
        )
        self.get_logger().info(f'Navigating to waypoint {self.current_wp_index + 1}. Distance = {dist:.2f}m', throttle_duration_sec=5.0)
        if dist < self.reached_threshold: # If we're close enough to the target waypoint, move on to the next one.
            self.get_logger().info(f'Reached Waypoint {self.current_wp_index + 1} ({target_x}, {target_y})')
            self.current_wp_index += 1 # Increment waypoint index to move to the next waypoint
            if self.current_wp_index >= len(self.waypoints):
                return

        target_x, target_y = self.waypoints[self.current_wp_index] 
        self.local_pos_pub.publish(self.build_waypoint_setpoint(target_x, target_y))


def main(args=None):
    rclpy.init(args=args)
    node = CalmWaterNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
