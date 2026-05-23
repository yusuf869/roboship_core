import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import csv, json, time, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Global flag to keep both sensors perfectly synced
IS_RECORDING = False

# --- 1. The Pixhawk (ROS 2) Receiver ---
class PixhawkLogger(Node):
    def __init__(self):
        super().__init__('pixhawk_logger')
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.subscription = self.create_subscription(
            Imu, '/mavros/imu/data', self.listener_callback, sensor_qos_profile)
        
        self.csv_file = open('pixhawk_imu_log.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Pi_Timestamp", "Pixhawk_Hardware_Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Q_X", "Q_Y", "Q_Z", "Q_W"])

    def listener_callback(self, msg):
        global IS_RECORDING
        if not IS_RECORDING:
            return # Valve is closed; throw data away
            
        pi_time = time.time()
        pix_time = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
        self.csv_writer.writerow([
            pi_time, pix_time, 
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])

    def close(self):
        self.csv_file.close()

# --- 2. The iPhone (HTTP) Receiver ---
class SensorLoggerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global IS_RECORDING
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        if IS_RECORDING: # Valve is open; save data
            try:
                data = json.loads(post_data.decode('utf-8'))
                if 'payload' in data:
                    with open('iphone_imu_log.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        for reading in data['payload']:
                            x = reading['values'].get('x', reading['values'].get('qx', 0))
                            y = reading['values'].get('y', reading['values'].get('qy', 0))
                            z = reading['values'].get('z', reading['values'].get('qz', 0))
                            w = reading['values'].get('w', reading['values'].get('qw', 0)) 
                            writer.writerow([time.time(), reading.get('name', ''), reading.get('time', 0)/1e9, x, y, z, w])
                self.send_response(200)
            except Exception:
                self.send_response(400)
        else:
            self.send_response(200) # Tell phone we got it, but throw it away
            
        self.end_headers()
        self.wfile.write(b"Success")

    def log_message(self, format, *args): 
        pass # Mute annoying HTTP logs in the terminal

def run_iphone_server():
    with open('iphone_imu_log.csv', mode='w', newline='') as file:
        csv.writer(file).writerow(["Pi_Timestamp", "Sensor_Type", "Phone_Timestamp", "X", "Y", "Z", "W"])
    server = HTTPServer(('0.0.0.0', 5555), SensorLoggerHandler)
    server.serve_forever()

# --- 3. The Master Control Flow ---
def main(args=None):
    global IS_RECORDING
    rclpy.init(args=args)
    
    # Spin up the Pixhawk receiver
    pixhawk_node = PixhawkLogger()
    ros_thread = threading.Thread(target=rclpy.spin, args=(pixhawk_node,), daemon=True)
    ros_thread.start()
    
    # Spin up the iPhone receiver
    iphone_thread = threading.Thread(target=run_iphone_server, daemon=True)
    iphone_thread.start()
    
    print("\n--- MASTER LOGGER READY ---")
    print("1. Ensure Pixhawk is powered and publishing MAVROS.")
    print("2. Press 'Start Recording' in the iPhone app right now.")
    print("   (Data is streaming, but recording is paused).")
    input(">> PRESS ENTER TO OPEN THE VALVES AND SAVE DATA <<")
    
    # TRIGGER THE START
    IS_RECORDING = True
    print("\n[REC] Recording to CSVs! Do your physical tap sync, then tumble the boat.")

    input("\n>> PRESS ENTER TO STOP RECORDING <<")
    
    # TRIGGER THE STOP
    IS_RECORDING = False
    print("\n[STOP] Recording stopped.")
        
    pixhawk_node.close()
    rclpy.shutdown()
    print("Data saved to CSVs. You can now run the SLERP visualizer!")

if __name__ == '__main__':
    main()