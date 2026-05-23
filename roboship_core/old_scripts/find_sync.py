import pandas as pd
import matplotlib.pyplot as plt

print("Loading data...")
df_phone = pd.read_csv('iphone_imu_log.csv')
df_hawk = pd.read_csv('pixhawk_imu_log.csv')

# Filter phone data for accelerometer only
df_phone_accel = df_phone[df_phone['Sensor_Type'] == 'accelerometer'].copy()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df_phone_accel['Pi_Timestamp'], df_phone_accel['Z'], label='iPhone Z-Accel (Wi-Fi)', alpha=0.8)
plt.plot(df_hawk['Pi_Timestamp'], df_hawk['Accel_Z'], label='Pixhawk Z-Accel (Serial)', alpha=0.8)

plt.xlabel('Pi Timestamp (Seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Find the "Thwack" Spike')
plt.legend()
plt.grid(True)

plt.savefig('sync_plot.png', dpi=300)
print("Saved to sync_plot.png! Open it in VSCode.")