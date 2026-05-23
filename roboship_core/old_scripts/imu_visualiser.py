import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# ──────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ──────────────────────────────────────────────

def quat_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

def make_box_vertices(sx=1.0, sy=0.6, sz=0.1):
    """Unit box centred at origin. sx/sy/sz are half-extents."""
    c = np.array([
        [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
        [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz],
    ])
    return c

BOX_FACES = [
    [0,1,2,3], [4,5,6,7],  # bottom / top
    [0,1,5,4], [2,3,7,6],  # front / back
    [0,3,7,4], [1,2,6,5],  # left / right
]
FACE_COLOURS = ['#4c72b0','#dd8452','#55a868','#c44e52','#8172b3','#937860']

def rotated_box(quat, half_extents=(1.0, 0.6, 0.1)):
    """Return (verts, faces) for a box rotated by quaternion."""
    R = quat_to_rotation_matrix(quat)
    verts = make_box_vertices(*half_extents) @ R.T
    faces = [[verts[i] for i in f] for f in BOX_FACES]
    return verts, faces

# ---------- Pixhawk ----------
pix = pd.read_csv('pixhawk_imu_log.csv')
pix['t'] = pix['Pi_Timestamp'] - pix['Pi_Timestamp'].iloc[0]  # seconds from start
pix_quats = pix[['Q_X','Q_Y','Q_Z','Q_W']].values
pix_accel_x = pix['Accel_X'].values
pix_time = pix['t'].values

# ---------- iPhone ----------
iphone_raw = pd.read_csv('iphone_imu_log.csv')

# Sensor Logger pushes multiple sensor types in the same CSV.
# Orientation rows give us quaternion-like data; accel rows give us X accel.
# Adapt these sensor-type strings if yours differ (print the CSV to check).
ORIENT_TYPES = {'orientation', 'attitude', 'deviceOrientation', 'gameRotation'}
ACCEL_TYPES  = {'accelerometer', 'totalAcceleration', 'acceleration'}

orient_mask = iphone_raw['Sensor_Type'].str.strip().isin(ORIENT_TYPES)
accel_mask  = iphone_raw['Sensor_Type'].str.strip().isin(ACCEL_TYPES)

iph_orient = iphone_raw[orient_mask].reset_index(drop=True)
iph_accel  = iphone_raw[accel_mask].reset_index(drop=True)

if iph_orient.empty:
    print("WARNING: No orientation rows found in iPhone CSV.")
    print(f"  Sensor types present: {iphone_raw['Sensor_Type'].unique()}")
    print("  Add the correct type name to ORIENT_TYPES and re-run.")
    # Fall back: use zeros so the script still runs
    iph_orient = pd.DataFrame({
        'Pi_Timestamp': pix['Pi_Timestamp'],
        'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'W': 1.0
    })

if iph_accel.empty:
    print("WARNING: No accel rows found in iPhone CSV.")
    print(f"  Sensor types present: {iphone_raw['Sensor_Type'].unique()}")
    iph_accel = pd.DataFrame({
        'Pi_Timestamp': pix['Pi_Timestamp'],
        'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'W': 0.0
    })

iph_orient['t'] = iph_orient['Pi_Timestamp'] - iph_orient['Pi_Timestamp'].iloc[0]
iph_accel['t']  = iph_accel['Pi_Timestamp']  - iph_accel['Pi_Timestamp'].iloc[0]

iph_quats   = iph_orient[['X','Y','Z','W']].values
iph_time_o  = iph_orient['t'].values
iph_accel_x = iph_accel['X'].values
iph_time_a  = iph_accel['t'].values

# ──────────────────────────────────────────────
# 2. TIME-SYNC: resample both to a common timeline
# ──────────────────────────────────────────────

FPS = 30
total_duration = min(pix_time[-1], iph_time_o[-1]) if len(iph_time_o) else pix_time[-1]
frame_times = np.arange(0, total_duration, 1.0 / FPS)
N_FRAMES = len(frame_times)

def interp_quats_nearest(times, src_times, src_quats):
    """Nearest-neighbour quaternion lookup (avoids naive lerp sign issues)."""
    idxs = np.searchsorted(src_times, times, side='right') - 1
    idxs = np.clip(idxs, 0, len(src_quats) - 1)
    return src_quats[idxs]

pix_q_sync = interp_quats_nearest(frame_times, pix_time, pix_quats)
iph_q_sync = interp_quats_nearest(frame_times, iph_time_o, iph_quats)

# ──────────────────────────────────────────────
# 3. BUILD THE FIGURE
# ──────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
fig.suptitle('IMU Comparison: iPhone vs Pixhawk', fontsize=14, fontweight='bold')

# Top row: two 3D axes
ax_iph  = fig.add_subplot(2, 2, 1, projection='3d')
ax_pix  = fig.add_subplot(2, 2, 2, projection='3d')
# Bottom row: single time-series axis spanning both columns
ax_plot = fig.add_subplot(2, 1, 2)

def style_3d(ax, title):
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12)
    ax.set_box_aspect([1,1,1])

style_3d(ax_iph, 'iPhone')
style_3d(ax_pix, 'Pixhawk')

# Bottom plot — full accel-x traces drawn once, with a moving cursor
ax_plot.plot(pix_time, pix_accel_x, label='Pixhawk Accel X', alpha=0.8, linewidth=0.8)
ax_plot.plot(iph_time_a, iph_accel_x, label='iPhone Accel X', alpha=0.8, linewidth=0.8)
ax_plot.set_xlabel('Time (s)')
ax_plot.set_ylabel('Accel X (m/s²)')
ax_plot.set_title('X-Axis Acceleration vs Time')
ax_plot.legend(loc='upper right')
ax_plot.grid(True, alpha=0.3)
cursor_line, = ax_plot.plot([], [], 'r-', linewidth=1.5, label='_nolegend_')

# ──────────────────────────────────────────────
# 4. ANIMATION LOOP
# ──────────────────────────────────────────────

def draw_box(ax, quat, colours=FACE_COLOURS, alpha=0.6):
    """Clear axis artists and draw a rotated box."""
    # Remove old poly collections
    while ax.collections:
        ax.collections[0].remove()
    _, faces = rotated_box(quat)
    poly = Poly3DCollection(faces, facecolors=colours, edgecolors='k',
                            linewidths=0.5, alpha=alpha)
    ax.add_collection3d(poly)

def update(frame):
    t = frame_times[frame]

    draw_box(ax_iph, iph_q_sync[frame])
    draw_box(ax_pix, pix_q_sync[frame])

    # Move the red cursor on the bottom graph
    cursor_line.set_data([t, t], [ax_plot.get_ylim()[0], ax_plot.get_ylim()[1]])

    if frame % FPS == 0:
        print(f'  frame {frame}/{N_FRAMES}  ({100*frame/N_FRAMES:.0f}%)')
    return []

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS, blit=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])

OUTPUT_FILE = 'imu_comparison.gif'
print(f'Rendering {N_FRAMES} frames to {OUTPUT_FILE} ...')
anim.save(OUTPUT_FILE, writer='pillow', fps=FPS)
print(f'Done! Saved to {OUTPUT_FILE}')