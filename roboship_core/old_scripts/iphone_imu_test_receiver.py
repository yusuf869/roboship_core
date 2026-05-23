import json
import csv
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

class SensorLoggerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            if 'payload' in data:
                with open('iphone_imu_log.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for reading in data['payload']:
                        sensor_name = reading.get('name', '')
                        phone_time = reading.get('time', 0) / 1000000000.0  
                        values = reading.get('values', {})
                        
                        # Dynamically grab standard axes or quaternion axes
                        x = values.get('x', values.get('qx', 0))
                        y = values.get('y', values.get('qy', 0))
                        z = values.get('z', values.get('qz', 0))
                        w = values.get('w', values.get('qw', 0)) # The required 4th dimension
                        
                        writer.writerow([time.time(), sensor_name, phone_time, x, y, z, w])
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Success")
            
        except Exception as e:
            print(f"Error parsing data: {e}")
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def run(port=5555):
    # Notice the "W" at the end of the header row below!
    with open('iphone_imu_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pi_Timestamp", "Sensor_Type", "Phone_Timestamp", "X", "Y", "Z", "W"]) 

    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, SensorLoggerHandler)
    print(f"Listening for HTTP Push from iPhone on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped. Data saved.")
        httpd.server_close()

if __name__ == '__main__':
    run()