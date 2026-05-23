#!/usr/bin/env python3
"""
USV Post-Processing — Comprehensive Analysis
─────────────────────────────────────────────
Reads a CSV log from usv_monitor and generates 12 individual figures.

Usage:
    python3 post_process.py                              # latest log
    python3 post_process.py ~/usv_logs/usv_log_xxx.csv   # specific file

Requires:
    pip install matplotlib pandas numpy
"""

import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── TUNABLES ────────────────────────────────────────────────────
WAYPOINTS = np.array([
    [0.5, 0.5],
    [-0.5, 0.5],
    [-0.5, -0.5],
    [0.5, -0.5],
])
TANK_XLIM = (-2.0, 2.0)
TANK_YLIM = (-2.0, 2.0)

WP_ARRIVAL_RADIUS = 0.2
LOOPING = True
# ────────────────────────────────────────────────────────────────


def point_to_segment_dist(p, a, b):
    ab = b - a; ap = p - a
    t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest), closest, t

def cross_track_error_signed(p, a, b):
    ab = b - a; ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    return cross / (np.linalg.norm(ab) + 1e-12)

def segment_length(a, b):
    return np.linalg.norm(b - a)

def compute_speed(df):
    dx = np.diff(df["x"].values, prepend=df["x"].values[0])
    dy = np.diff(df["y"].values, prepend=df["y"].values[0])
    dt = np.diff(df["t_sec"].values, prepend=df["t_sec"].values[0])
    dt[dt == 0] = 1e-9
    return np.sqrt(dx**2 + dy**2) / dt

def compute_heading(df):
    dx = np.diff(df["x"].values, append=df["x"].values[-1])
    dy = np.diff(df["y"].values, append=df["y"].values[-1])
    
    # Calculate wrapped heading in radians
    heading_rad = np.arctan2(dy, dx)
    
    # Unwrap to remove the +/- 180 jumps
    heading_unwrapped = np.unwrap(heading_rad)
    
    # Convert to degrees
    return np.degrees(heading_unwrapped)

def assign_segments_and_laps(df, wps, arrival_r):
    n_wps = len(wps)
    xy = df[["x", "y"]].values
    times = df["t_sec"].values
    target_wp = 0
    segments = np.full(len(df), -1, dtype=int)
    laps = np.full(len(df), 0, dtype=int)
    current_lap = 0
    
    wp_arrivals = []
    wp_departures = []
    wp_overshoots = []
    
    in_wp_zone = False
    overshoot_tracking = False
    overshoot_max = 0.0
    overshoot_idx = 0
    u_in = np.array([0.0, 0.0])
    u_out = np.array([0.0, 0.0])
    segment_len = 1.0 # Default fallback
    
    for i in range(len(xy)):
        p = xy[i]
        prev_wp = (target_wp - 1) % n_wps
        segments[i] = prev_wp
        laps[i] = current_lap
        
        dist_to_target = np.linalg.norm(p - wps[target_wp])
        
        if dist_to_target < arrival_r:
            if not in_wp_zone:
                in_wp_zone = True
                wp_arrivals.append((target_wp, i, times[i]))
                
                if overshoot_tracking:
                    actual_euclidean = np.linalg.norm(xy[overshoot_idx] - wps[overshoot_wp])
                    wp_overshoots.append((overshoot_wp, actual_euclidean, overshoot_idx))
                    overshoot_tracking = False
        elif in_wp_zone:
            in_wp_zone = False
            wp_departures.append((target_wp, i, times[i]))
            
            overshoot_tracking = True
            overshoot_wp = target_wp
            overshoot_max = 0.0
            overshoot_idx = i
            
            # Incoming vector
            wp_prev_to_departed = (overshoot_wp - 1) % n_wps
            v_in = wps[overshoot_wp] - wps[wp_prev_to_departed]
            norm_v_in = np.linalg.norm(v_in)
            u_in = v_in / norm_v_in if norm_v_in > 0 else np.array([0.0, 0.0])
            
            # Outgoing vector and segment length
            next_wp = (overshoot_wp + 1) % n_wps
            v_out = wps[next_wp] - wps[overshoot_wp]
            segment_len = np.linalg.norm(v_out)
            u_out = v_out / segment_len if segment_len > 0 else np.array([0.0, 0.0])
            
            target_wp = next_wp
            if target_wp == 0:
                current_lap += 1
            laps[i] = current_lap
            
        if overshoot_tracking:
            progress_along_new_leg = np.dot(p - wps[overshoot_wp], u_out)
            
            # CRITICAL FIX: Give the boat 40% of the next straightaway to finish the corner
            if progress_along_new_leg > (segment_len * 0.4):
                actual_euclidean = np.linalg.norm(xy[overshoot_idx] - wps[overshoot_wp])
                wp_overshoots.append((overshoot_wp, actual_euclidean, overshoot_idx))
                overshoot_tracking = False
            else:
                d_past = np.dot(p - wps[overshoot_wp], u_in)
                if d_past > overshoot_max:
                    overshoot_max = d_past
                    overshoot_idx = i
                    
    if overshoot_tracking:
        actual_euclidean = np.linalg.norm(xy[overshoot_idx] - wps[overshoot_wp])
        wp_overshoots.append((overshoot_wp, actual_euclidean, overshoot_idx))
        
    return segments, laps, wp_arrivals, wp_departures, wp_overshoots
    
def compute_cte(df, wps, segments):
    n_wps = len(wps); xy = df[["x", "y"]].values
    cte = np.zeros(len(df)); cte_abs = np.zeros(len(df))
    for i in range(len(df)):
        seg = segments[i]
        if seg < 0: continue
        a = wps[seg]; b = wps[(seg + 1) % n_wps]
        cte[i] = cross_track_error_signed(xy[i], a, b)
        cte_abs[i], _, _ = point_to_segment_dist(xy[i], a, b)
    return cte, cte_abs


def compute_lap_stats(df, wps, laps, wp_arrivals, segments, cte_abs):
    n_wps = len(wps); xy = df[["x", "y"]].values; times = df["t_sec"].values
    ideal_lap_dist = sum(segment_length(wps[i], wps[(i+1) % n_wps]) for i in range(n_wps))
    lap_stats = []
    for lap_num in sorted(set(laps)):
        mask = laps == lap_num
        if mask.sum() < 2: continue
        lap_xy = xy[mask]; lap_t = times[mask]; lap_cte = cte_abs[mask]
        actual_dist = np.sum(np.linalg.norm(np.diff(lap_xy, axis=0), axis=1))
        duration = lap_t[-1] - lap_t[0]
        efficiency = (ideal_lap_dist / actual_dist * 100) if actual_dist > 0 else 0
        lap_stats.append({"lap": lap_num, "duration_s": round(duration, 2),
            "actual_dist_m": round(actual_dist, 3), "ideal_dist_m": round(ideal_lap_dist, 3),
            "efficiency_%": round(efficiency, 1), "mean_cte_m": round(np.mean(lap_cte), 4),
            "max_cte_m": round(np.max(lap_cte), 4),
            "mean_speed_ms": round(actual_dist / duration, 3) if duration > 0 else 0})
    return lap_stats, ideal_lap_dist


def find_latest_log():
    log_dir = pathlib.Path.home() / "usv_logs"
    logs = sorted(log_dir.glob("usv_log_*.csv"))
    if not logs: print(f"No logs found in {log_dir}"); sys.exit(1)
    return logs[-1]


def setup_matplotlib():
    plt.style.use("default")
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 8,
        "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.3,
        "grid.color": "gray", "grid.linestyle": "-",
        "xtick.color": "black", "ytick.color": "black",
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "text.color": "black", "legend.fontsize": 9,
        "legend.framealpha": 0.8, "legend.edgecolor": "black",
        "figure.figsize": (8, 6), "figure.dpi": 150,
        "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
        "lines.linewidth": 1.5, "font.size": 10,
    })


def new_figure(title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    return fig, ax


def plot_01_trajectory(df, wps):
    fig, ax = new_figure("2D Trajectory vs Ideal Path")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    if TANK_XLIM: ax.set_xlim(TANK_XLIM)
    if TANK_YLIM: ax.set_ylim(TANK_YLIM)
    ideal_x = list(wps[:, 0]) + [wps[0, 0]]; ideal_y = list(wps[:, 1]) + [wps[0, 1]]
    ax.plot(ideal_x, ideal_y, "k--", linewidth=1.5, alpha=0.5, label="Ideal path")
    sc = ax.scatter(df["x"], df["y"], c=df["t_sec"], cmap="jet", s=4, zorder=2, label="Actual path")
    plt.colorbar(sc, ax=ax, label="Time (s)", pad=0.02, shrink=0.85)
    ax.plot(df["x"].iloc[0], df["y"].iloc[0], "go", ms=10, label="Start")
    ax.plot(df["x"].iloc[-1], df["y"].iloc[-1], "rs", ms=10, label="End")
    for i, (wx, wy) in enumerate(wps):
        ax.plot(wx, wy, "D", color="darkorange", ms=9, zorder=5)
        ax.add_patch(Circle((wx, wy), WP_ARRIVAL_RADIUS, fill=False, ec="darkorange", ls="--", lw=1))
        ax.annotate(f"WP{i}", (wx, wy), textcoords="offset points", xytext=(8, 8), fontsize=9, color="darkorange", fontweight="bold")
    
    # Legend to the left, colorbar remains on the right
    ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
    fig.tight_layout()
    return fig


def plot_02_cte(df, wps, segments, cte_signed):
    fig, ax = new_figure("Cross-Track Error vs Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Absolute CTE (m)")
    n_wps = len(wps); colors = plt.cm.tab10(np.linspace(0, 1, n_wps))
    for seg_i in range(n_wps):
        mask = segments == seg_i
        if mask.any():
            ax.scatter(df["t_sec"].values[mask], cte_signed[mask], s=2, color=colors[seg_i],
                       label=f"WP{seg_i} → WP{(seg_i+1)%n_wps}", alpha=0.7)
    
    ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
    fig.tight_layout()
    return fig


def plot_03_speed(df, speed):
    fig, ax = new_figure("Speed Profile")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
    ax.plot(df["t_sec"], speed, linewidth=0.6, color="steelblue", alpha=0.7, label="Instantaneous")
    window = min(50, len(speed) // 4)
    if window > 1:
        speed_smooth = pd.Series(speed).rolling(window, center=True).mean()
        ax.plot(df["t_sec"], speed_smooth, linewidth=2, color="red", label=f"Rolling avg (n={window})")
        
    ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
    fig.tight_layout()
    return fig


def plot_04_heading(df, heading):
    fig, ax = new_figure("Heading vs Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading (deg)")
    
    ax.scatter(df["t_sec"], heading, s=2, alpha=0.5, color="steelblue")
    
    # Removed the ax.set_ylim(-180, 180) so it can scale freely
    # Added dynamic gridlines every 45 degrees based on the new data range
    min_hdg = int(np.floor(min(heading) / 45.0) * 45)
    max_hdg = int(np.ceil(max(heading) / 45.0) * 45)
    ax.set_yticks(np.arange(min_hdg, max_hdg + 1, 45))
    
    fig.tight_layout()
    return fig

def plot_05_overshoot(df, wps, wp_overshoots):
    fig, ax = new_figure("Waypoint Overshoot Trace")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    if TANK_XLIM: ax.set_xlim(TANK_XLIM)
    if TANK_YLIM: ax.set_ylim(TANK_YLIM)
    
    # How many seconds before and after the apex to draw the line
    TIME_WINDOW_SEC = 3
    
    ideal_x = list(wps[:, 0]) + [wps[0, 0]]; ideal_y = list(wps[:, 1]) + [wps[0, 1]]
    ax.plot(ideal_x, ideal_y, "k--", linewidth=1, alpha=0.4, label="Ideal path")
    
    n_wps = len(wps)
    wp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"][:n_wps]
    xy = df[["x", "y"]].values
    
    # Group overshoots by waypoint
    overshoot_by_wp = {}
    for wp_idx, os_dist, os_idx in wp_overshoots:
        overshoot_by_wp.setdefault(wp_idx, []).append((os_dist, os_idx))
        
    for i, (wx, wy) in enumerate(wps):
        c = wp_colors[i]
        ax.plot(wx, wy, "D", color=c, ms=10, zorder=5, label=f"WP{i} ({wx:.1f}, {wy:.1f})")
        ax.add_patch(Circle((wx, wy), WP_ARRIVAL_RADIUS, fill=False, ec=c, ls="--", lw=1, alpha=0.5))
        
        if i in overshoot_by_wp:
            entries = overshoot_by_wp[i]
            
            # Find the absolute maximum overshoot distance across all laps
            max_entry = max(entries, key=lambda e: e[0])
            max_os_dist, max_os_idx = max_entry
            
            # --- NEW TRACE LOGIC ---
            # Get the exact time the boat hit the apex of the turn
            t_apex = df["t_sec"].iloc[max_os_idx]
            
            # Slice the dataframe to get just the time window around the apex
            mask = (df["t_sec"] >= t_apex - TIME_WINDOW_SEC) & (df["t_sec"] <= t_apex + TIME_WINDOW_SEC)
            trace_x = df.loc[mask, "x"]
            trace_y = df.loc[mask, "y"]
            
            # Plot the continuous path snippet
            ax.plot(trace_x, trace_y, "-", color=c, lw=2.5, alpha=0.7, label=f"WP{i} Trace (±{TIME_WINDOW_SEC}s)")
            
            # Plot the single 'x' marker exactly at the furthest point
            ox, oy = xy[max_os_idx]
            ax.plot(ox, oy, "x", color=c, ms=10, mew=2.5, zorder=6)
            
            # Annotate the maximum distance
            ax.annotate(f"{max_os_dist:.2f} m", (ox, oy), textcoords="offset points",
                        xytext=(8, -8), ha='left', va='top', fontsize=8, color=c, fontweight="bold")
                
    # Place legend to the left
    ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5), fontsize=8)
    fig.tight_layout() 
    return fig


def plot_06_wp_table(wps, wp_arrivals, wp_overshoots, df):
    fig, ax = plt.subplots(figsize=(10, 3)); ax.set_title("Waypoint Arrival Stats", fontweight="bold"); ax.axis("off")
    n_wps = len(wps)
    arrival_by_wp = {}
    for wp_idx, si, t in wp_arrivals: arrival_by_wp.setdefault(wp_idx, []).append(t)
    overshoot_by_wp = {}
    for wp_idx, os_dist, _ in wp_overshoots: overshoot_by_wp.setdefault(wp_idx, []).append(os_dist)
    rows = []
    for i in range(n_wps):
        n_arr = len(arrival_by_wp.get(i, []))
        mean_os = np.mean(overshoot_by_wp[i]) if i in overshoot_by_wp else 0
        max_os = np.max(overshoot_by_wp[i]) if i in overshoot_by_wp else 0
        dists_i = np.linalg.norm(df[["x", "y"]].values - wps[i], axis=1)
        rows.append([f"WP{i} ({wps[i][0]:.1f}, {wps[i][1]:.1f})", f"{n_arr}", f"{dists_i.min():.4f}", f"{mean_os:.3f}", f"{max_os:.3f}"])
    col_labels = ["Waypoint", "Arrivals", "Min Dist (m)", "Mean Overshoot (m)", "Max Overshoot (m)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
    table.auto_set_column_width(list(range(len(col_labels))))
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        if key[0] == 0: cell.set_facecolor("#d9e2f3"); cell.set_text_props(fontweight="bold")
        else: cell.set_facecolor("white")
    return fig


def plot_07_lap_duration(lap_stats):
    fig, ax = new_figure("Lap Duration"); ax.set_xlabel("Lap"); ax.set_ylabel("Duration (s)")
    if lap_stats:
        ln = [s["lap"] for s in lap_stats]; dur = [s["duration_s"] for s in lap_stats]
        bars = ax.bar(ln, dur, color="steelblue", edgecolor="black", linewidth=0.8)
        for b, d in zip(bars, dur): ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{d:.1f} s", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(ln)
    else: ax.text(0.5, 0.5, "Not enough laps", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    fig.tight_layout()
    return fig


def plot_08_efficiency(lap_stats):
    fig, ax = new_figure("Path Efficiency per Lap"); ax.set_xlabel("Lap"); ax.set_ylabel("Efficiency (%)")
    if lap_stats:
        ln = [s["lap"] for s in lap_stats]; eff = [s["efficiency_%"] for s in lap_stats]
        bars = ax.bar(ln, eff, color="steelblue", edgecolor="black", linewidth=0.8)
        ax.axhline(100, color="red", ls="--", lw=1.5, label="100% (ideal)")
        for b, e in zip(bars, eff): ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{e:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(ln)
        ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
    else: ax.text(0.5, 0.5, "Not enough laps", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    fig.tight_layout()
    return fig


def plot_09_summary(df, wps, cte_abs, cte_signed, lap_stats, ideal_lap_dist, wp_overshoots, speed):
    fig, ax = plt.subplots(figsize=(7, 6)); ax.set_title("Overall Summary", fontweight="bold"); ax.axis("off")
    xy = df[["x", "y"]].values; total_time = df["t_sec"].iloc[-1] - df["t_sec"].iloc[0]
    total_dist = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    n_laps = len(lap_stats)
    mean_lap_time = np.mean([s["duration_s"] for s in lap_stats]) if lap_stats else 0
    mean_efficiency = np.mean([s["efficiency_%"] for s in lap_stats]) if lap_stats else 0
    mean_speed_val = total_dist / total_time if total_time > 0 else 0
    overshoot_vals = [os for _, os, _ in wp_overshoots]
    mean_os = np.mean(overshoot_vals) if overshoot_vals else 0
    max_os = np.max(overshoot_vals) if overshoot_vals else 0
    rows = [
        ["Total time", f"{total_time:.1f} s"], ["Total distance", f"{total_dist:.3f} m"],
        ["Laps completed", f"{n_laps}"], ["Mean lap time", f"{mean_lap_time:.1f} s"],
        ["Mean speed", f"{mean_speed_val:.4f} m/s"], ["Ideal lap distance", f"{ideal_lap_dist:.3f} m"],
        ["Mean path efficiency", f"{mean_efficiency:.1f}%"],
        ["Mean CTE (abs)", f"{np.mean(cte_abs):.4f} m"], ["Max CTE (abs)", f"{np.max(cte_abs):.4f} m"],
        ["RMS CTE", f"{np.sqrt(np.mean(cte_abs**2)):.4f} m"],
        ["95th percentile CTE", f"{np.percentile(cte_abs, 95):.4f} m"],
        ["Mean overshoot", f"{mean_os:.4f} m"], ["Max overshoot", f"{max_os:.4f} m"],
    ]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    table.auto_set_column_width([0, 1])
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        if key[0] == 0: cell.set_facecolor("#d9e2f3"); cell.set_text_props(fontweight="bold")
        else: cell.set_facecolor("white")
    return fig


def plot_10_cte_hist(cte_signed):
    fig, ax = new_figure("CTE Distribution"); ax.set_xlabel("CTE (m)"); ax.set_ylabel("Count")
    ax.hist(cte_signed, bins=60, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", ls="--", lw=1.5, label="Ideal (0)")
    ax.axvline(np.mean(cte_signed), color="red", ls="-", lw=1.5, label=f"Mean = {np.mean(cte_signed):.3f} m")
    
    ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
    fig.tight_layout()
    return fig


def plot_11_cte_boxplot(wps, segments, cte_abs):
    fig, ax = new_figure("CTE by Segment"); ax.set_ylabel("Absolute CTE (m)")
    n_wps = len(wps); seg_data = []; seg_labels = []
    for seg_i in range(n_wps):
        mask = segments == seg_i
        if mask.any(): seg_data.append(cte_abs[mask]); seg_labels.append(f"WP{seg_i}→{(seg_i+1)%n_wps}")
    if seg_data:
        bp = ax.boxplot(seg_data, tick_labels=seg_labels, patch_artist=True)
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
        for patch, c in zip(bp["boxes"], colors[:len(seg_data)]): patch.set_facecolor(c); patch.set_alpha(0.7)
    fig.tight_layout()
    return fig


def plot_12_cumulative(df, wps, segments):
    fig, ax = new_figure("Cumulative Distance: Actual vs Ideal")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
    xy = df[["x", "y"]].values; n_wps = len(wps)
    step_dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum_actual = np.concatenate([[0], np.cumsum(step_dists)])
    cum_ideal = np.zeros(len(df)); running_ideal = 0; prev_seg = segments[0]; seg_start_ideal = 0
    for i in range(len(df)):
        seg = segments[i]
        if seg != prev_seg and seg >= 0:
            running_ideal += segment_length(wps[prev_seg], wps[(prev_seg+1) % n_wps])
            prev_seg = seg; seg_start_ideal = running_ideal
        if seg >= 0:
            a = wps[seg]; b = wps[(seg + 1) % n_wps]
            _, _, t_param = point_to_segment_dist(xy[i], a, b)
            cum_ideal[i] = seg_start_ideal + t_param * segment_length(a, b)
        else: cum_ideal[i] = 0
    ax.plot(df["t_sec"], cum_actual, linewidth=1.5, color="steelblue", label="Actual")
    ax.plot(df["t_sec"], cum_ideal, linewidth=1.5, color="red", ls="--", label="Ideal (projected)")
    ax.fill_between(df["t_sec"], cum_actual, cum_ideal, alpha=0.15, color="red", label="Wasted distance")
    
    ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
    fig.tight_layout()
    return fig


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_log()
    df = pd.read_csv(path); print(f"Loaded {len(df)} samples from {path}")

    segments, laps, wp_arrivals, wp_departures, wp_overshoots = assign_segments_and_laps(df, WAYPOINTS, WP_ARRIVAL_RADIUS)
    cte_signed, cte_abs = compute_cte(df, WAYPOINTS, segments)
    speed = compute_speed(df); heading = compute_heading(df)
    lap_stats, ideal_lap_dist = compute_lap_stats(df, WAYPOINTS, laps, wp_arrivals, segments, cte_abs)

    print(f"\n{'═'*55}")
    print(f"  USV Run Analysis — {path.name}")
    print(f"{'═'*55}")
    print(f"  Samples:       {len(df)}")
    print(f"  Duration:      {df['t_sec'].iloc[-1]:.1f} s")
    print(f"  Laps:          {len(lap_stats)}")
    print(f"  RMS CTE:       {np.sqrt(np.mean(cte_abs**2)):.4f} m")
    print(f"  Mean CTE:      {np.mean(cte_abs):.4f} m")
    print(f"  Max CTE:       {np.max(cte_abs):.4f} m")
    print(f"  95th pct CTE:  {np.percentile(cte_abs, 95):.4f} m")
    if lap_stats:
        print(f"\n  Per-lap breakdown:")
        for ls in lap_stats:
            print(f"    Lap {ls['lap']}: {ls['duration_s']:.1f}s  eff={ls['efficiency_%']:.1f}%  mean_cte={ls['mean_cte_m']:.4f}m  speed={ls['mean_speed_ms']:.3f}m/s")
    if wp_overshoots:
        print(f"\n  Overshoot summary:")
        osbw = {}
        for wp_idx, os_dist, _ in wp_overshoots: osbw.setdefault(wp_idx, []).append(os_dist)
        for wp_idx in sorted(osbw):
            vals = osbw[wp_idx]; print(f"    WP{wp_idx}: mean={np.mean(vals):.3f}m  max={np.max(vals):.3f}m  n={len(vals)}")
    print(f"{'═'*55}\n")

    setup_matplotlib()
    stem = path.stem; out_dir = path.parent
    figures = [
        ("01_trajectory", plot_01_trajectory(df, WAYPOINTS)),
        ("02_cross_track_err", plot_02_cte(df, WAYPOINTS, segments, cte_abs)),
        ("03_speed_profile", plot_03_speed(df, speed)),
        ("04_heading", plot_04_heading(df, heading)),
        ("05_overshoot", plot_05_overshoot(df, WAYPOINTS, wp_overshoots)),
        ("06_wp_stats", plot_06_wp_table(WAYPOINTS, wp_arrivals, wp_overshoots, df)),
        ("07_lap_duration", plot_07_lap_duration(lap_stats)),
        ("08_path_efficiency", plot_08_efficiency(lap_stats)),
        ("09_summary", plot_09_summary(df, WAYPOINTS, cte_abs, cte_signed, lap_stats, ideal_lap_dist, wp_overshoots, speed)),
        ("10_cte_histogram", plot_10_cte_hist(cte_signed)),
        ("11_cte_boxplot", plot_11_cte_boxplot(WAYPOINTS, segments, cte_abs)),
        ("12_cumulative_dist", plot_12_cumulative(df, WAYPOINTS, segments)),
    ]
    for name, fig in figures:
        out_path = out_dir / f"{stem}_{name}.png"; fig.savefig(out_path); print(f"  Saved: {out_path.name}")
    print(f"\nSaved 12 figures to {out_dir}/")
    plt.show()


if __name__ == "__main__":
    main()