#!/usr/bin/env python3
"""
USV Post-Processing — Comprehensive Analysis (v2)
─────────────────────────────────────────────────
Reads a CSV log from usv_monitor OR ilos_guidance and generates
publication-quality figures.

Key upgrades over v1:
  • Savitzky–Golay smoothing of position/heading to remove mocap
    stair-stepping (mocap often runs slower than the log rate).
  • Heading is taken from mocap (`psi`) when available, then unwrapped
    and plotted as a continuous curve.
  • Cross-track error uses the ILOS controller's own `y_e` when
    available, plotted as a continuous curve coloured by lap number.
  • Laps are extracted from `wp_idx` (controller ground truth) when
    available, with a robust fall-back to geometric reconstruction.
  • Speed plot shows the rolling average only.
  • Extra plots: lap-by-lap overlay, heading tracking, heading error,
    control inputs, ILOS integral convergence, CTE per lap, CTE PSD.

Usage:
    python3 post_process.py                              # latest log
    python3 post_process.py ~/usv_logs/usv_log_xxx.csv   # specific file

Requires:
    pip install matplotlib pandas numpy scipy
"""

import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter, welch

# ── TUNABLES ────────────────────────────────────────────────────
WAYPOINTS = np.array([
    [ 0.5,  0.5],
    [-0.5,  0.5],
    [-0.5, -0.5],
    [ 0.5, -0.5],
])
TANK_XLIM = (-2.0, 2.0)
TANK_YLIM = (-2.0, 2.0)

WP_ARRIVAL_RADIUS = 0.3
LOOPING = True

# Smoothing — window is in *samples* and must be odd.
# For 20 Hz logs with ~1 Hz mocap a 1.5-s window works well.
SMOOTH_WINDOW = 31
SMOOTH_POLY   = 3
# ────────────────────────────────────────────────────────────────


# ── geometry helpers ────────────────────────────────────────────
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


# ── smoothing helpers ───────────────────────────────────────────
def safe_savgol(arr, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """Savitzky–Golay with automatic window shrinking for short signals."""
    n = len(arr)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        return np.asarray(arr, dtype=float)
    return savgol_filter(np.asarray(arr, dtype=float), w, poly)


# ── derived quantities ──────────────────────────────────────────
def compute_speed(df, smooth=True):
    """Speed from (smoothed) positions, expressed as m/s."""
    x = safe_savgol(df["x"].values) if smooth else df["x"].values
    y = safe_savgol(df["y"].values) if smooth else df["y"].values
    t = df["t_sec"].values
    dt = np.gradient(t)
    dt[dt == 0] = 1e-9
    return np.hypot(np.gradient(x), np.gradient(y)) / dt


def compute_heading(df, has_psi):
    """
    Unwrapped, smoothed heading in degrees.
    Uses mocap `psi` when available; otherwise reconstructs from path tangent.
    """
    if has_psi:
        psi = np.unwrap(df["psi"].values)
    else:
        dx = np.gradient(df["x"].values)
        dy = np.gradient(df["y"].values)
        psi = np.unwrap(np.arctan2(dy, dx))
    return np.degrees(safe_savgol(psi))


def compute_yaw_rate(df, has_psi):
    """Yaw rate in rad/s from smoothed, unwrapped heading."""
    if has_psi:
        psi_u = np.unwrap(df["psi"].values)
    else:
        dx = np.gradient(df["x"].values); dy = np.gradient(df["y"].values)
        psi_u = np.unwrap(np.arctan2(dy, dx))
    psi_s = safe_savgol(psi_u)
    t = df["t_sec"].values
    return np.gradient(psi_s, t)


# ── lap & segment assignment ────────────────────────────────────
def laps_from_wp_idx(wp_idx_series, n_wps):
    """Robustly count laps from controller's cumulative wp index, which may wrap."""
    wp_within = wp_idx_series % n_wps
    laps = np.zeros(len(wp_idx_series), dtype=int)
    current = 0
    for i in range(1, len(wp_idx_series)):
        if wp_within[i] < wp_within[i - 1] and wp_within[i - 1] == n_wps - 1:
            current += 1
        laps[i] = current
    return laps, (wp_within.astype(int))


def assign_segments_and_laps(df, wps, arrival_r, has_wp_idx):
    """
    Returns: segments, laps, wp_arrivals, wp_departures, wp_overshoots.
    Uses controller wp_idx when available; otherwise reconstructs geometrically.
    """
    n_wps = len(wps); xy = df[["x", "y"]].values; times = df["t_sec"].values

    if has_wp_idx:
        # `wp_idx` in the log is the *target* waypoint, so the segment the
        # boat is currently *on* is the one ending at wp_idx.
        laps, wp_within = laps_from_wp_idx(df["wp_idx"].values, n_wps)
        segments = (wp_within - 1) % n_wps
    else:
        # Geometric reconstruction (original logic, retained for fall-back).
        target_wp = 0
        segments = np.full(len(df), -1, dtype=int)
        laps = np.zeros(len(df), dtype=int)
        current_lap = 0
        in_zone = False
        for i in range(len(xy)):
            segments[i] = (target_wp - 1) % n_wps
            laps[i] = current_lap
            if np.linalg.norm(xy[i] - wps[target_wp]) < arrival_r:
                if not in_zone:
                    in_zone = True
            elif in_zone:
                in_zone = False
                target_wp = (target_wp + 1) % n_wps
                if target_wp == 0:
                    current_lap += 1
                laps[i] = current_lap

    # Detect arrivals / departures / overshoots from positions
    target_wp = int((segments[0] + 1) % n_wps)
    wp_arrivals, wp_departures, wp_overshoots = [], [], []
    in_zone = False
    overshoot_tracking = False
    overshoot_max = 0.0; overshoot_idx = 0; overshoot_wp = 0
    u_in = np.zeros(2); u_out = np.zeros(2); seg_len = 1.0

    for i in range(len(xy)):
        dist = np.linalg.norm(xy[i] - wps[target_wp])
        if dist < arrival_r:
            if not in_zone:
                in_zone = True
                wp_arrivals.append((target_wp, i, times[i]))
                if overshoot_tracking:
                    eucl = np.linalg.norm(xy[overshoot_idx] - wps[overshoot_wp])
                    wp_overshoots.append((overshoot_wp, eucl, overshoot_idx))
                    overshoot_tracking = False
        elif in_zone:
            in_zone = False
            wp_departures.append((target_wp, i, times[i]))
            overshoot_tracking = True
            overshoot_wp = target_wp
            overshoot_max = 0.0; overshoot_idx = i
            prev = (overshoot_wp - 1) % n_wps
            v_in = wps[overshoot_wp] - wps[prev]
            n_in = np.linalg.norm(v_in)
            u_in = v_in / n_in if n_in > 0 else np.zeros(2)
            nxt = (overshoot_wp + 1) % n_wps
            v_out = wps[nxt] - wps[overshoot_wp]
            seg_len = np.linalg.norm(v_out)
            u_out = v_out / seg_len if seg_len > 0 else np.zeros(2)
            target_wp = nxt

        if overshoot_tracking:
            progress = np.dot(xy[i] - wps[overshoot_wp], u_out)
            if progress > seg_len * 0.4:
                eucl = np.linalg.norm(xy[overshoot_idx] - wps[overshoot_wp])
                wp_overshoots.append((overshoot_wp, eucl, overshoot_idx))
                overshoot_tracking = False
            else:
                d_past = np.dot(xy[i] - wps[overshoot_wp], u_in)
                if d_past > overshoot_max:
                    overshoot_max = d_past
                    overshoot_idx = i

    if overshoot_tracking:
        eucl = np.linalg.norm(xy[overshoot_idx] - wps[overshoot_wp])
        wp_overshoots.append((overshoot_wp, eucl, overshoot_idx))

    return segments, laps, wp_arrivals, wp_departures, wp_overshoots


def compute_cte(df, wps, segments, has_ye):
    """Cross-track error (signed and absolute)."""
    if has_ye:
        cte = df["y_e"].values
        return cte, np.abs(cte)
    n_wps = len(wps); xy = df[["x", "y"]].values
    cte = np.zeros(len(df)); cte_abs = np.zeros(len(df))
    for i in range(len(df)):
        seg = segments[i]
        if seg < 0: continue
        a = wps[seg]; b = wps[(seg + 1) % n_wps]
        cte[i] = cross_track_error_signed(xy[i], a, b)
        cte_abs[i], _, _ = point_to_segment_dist(xy[i], a, b)
    return cte, cte_abs


def compute_lap_stats(df, wps, laps, cte_abs):
    n_wps = len(wps); xy = df[["x", "y"]].values; times = df["t_sec"].values
    ideal = sum(segment_length(wps[i], wps[(i + 1) % n_wps]) for i in range(n_wps))
    stats = []
    for ln in sorted(set(laps)):
        mask = laps == ln
        if mask.sum() < 2: continue
        lxy = xy[mask]; lt = times[mask]; lcte = cte_abs[mask]
        actual = np.sum(np.linalg.norm(np.diff(lxy, axis=0), axis=1))
        dur = lt[-1] - lt[0]
        eff = (ideal / actual * 100) if actual > 0 else 0
        rms = float(np.sqrt(np.mean(lcte ** 2)))
        stats.append({
            "lap": int(ln), "duration_s": round(dur, 2),
            "actual_dist_m": round(actual, 3), "ideal_dist_m": round(ideal, 3),
            "efficiency_%": round(eff, 1),
            "mean_cte_m": round(float(np.mean(lcte)), 4),
            "rms_cte_m":  round(rms, 4),
            "max_cte_m":  round(float(np.max(lcte)), 4),
            "mean_speed_ms": round(actual / dur, 3) if dur > 0 else 0,
        })
    return stats, ideal


# ── housekeeping ────────────────────────────────────────────────
def find_latest_log():
    log_dir = pathlib.Path.home() / "usv_logs"
    logs = sorted(log_dir.glob("usv_log_*.csv")) + sorted(log_dir.glob("ilos_*.csv"))
    if not logs:
        print(f"No logs found in {log_dir}")
        sys.exit(1)
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
        "legend.framealpha": 0.85, "legend.edgecolor": "black",
        "figure.figsize": (8, 6), "figure.dpi": 150,
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
        "lines.linewidth": 1.6, "font.size": 10,
    })


def new_figure(title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    return fig, ax


def lap_colormap(n_laps):
    """Discrete viridis sampling, one colour per lap."""
    if n_laps <= 1:
        return [plt.cm.viridis(0.5)]
    return [plt.cm.viridis(v) for v in np.linspace(0.15, 0.85, n_laps)]


def left_legend(ax, *args, **kwargs):
    kwargs.setdefault("loc", "center right")
    kwargs.setdefault("bbox_to_anchor", (-0.15, 0.5))
    ax.legend(*args, **kwargs)


# ───────────────────── PLOTS ────────────────────────────────────
def plot_01_trajectory(df, wps, x_s, y_s):
    fig, ax = new_figure("2D Trajectory (smoothed) vs Ideal Path")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    if TANK_XLIM: ax.set_xlim(TANK_XLIM)
    if TANK_YLIM: ax.set_ylim(TANK_YLIM)

    ideal_x = list(wps[:, 0]) + [wps[0, 0]]
    ideal_y = list(wps[:, 1]) + [wps[0, 1]]
    ax.plot(ideal_x, ideal_y, "k--", linewidth=1.5, alpha=0.5, label="Ideal path")

    # Faint raw points to show data density (and any mocap stair-stepping)
    ax.scatter(df["x"], df["y"], s=2, color="lightgray", alpha=0.4, zorder=1)

    pts = np.array([x_s, y_s]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(df["t_sec"].min(), df["t_sec"].max())
    lc = LineCollection(segs, cmap="jet", norm=norm, linewidth=2.2, zorder=3)
    lc.set_array(df["t_sec"].values[:-1])
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="Time (s)", pad=0.02, shrink=0.85)

    ax.plot(x_s[0],  y_s[0],  "go", ms=10, label="Start", zorder=4)
    ax.plot(x_s[-1], y_s[-1], "rs", ms=10, label="End",   zorder=4)
    for i, (wx, wy) in enumerate(wps):
        ax.plot(wx, wy, "D", color="darkorange", ms=9, zorder=5)
        ax.add_patch(Circle((wx, wy), WP_ARRIVAL_RADIUS, fill=False,
                            ec="darkorange", ls="--", lw=1))
        ax.annotate(f"WP{i}", (wx, wy), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="darkorange", fontweight="bold")

    handles = [Line2D([0], [0], color="k", ls="--", lw=1.5),
               Line2D([0], [0], color="blue", lw=2.2),
               Line2D([0], [0], color="lightgray", marker="o", lw=0, ms=4)]
    left_legend(ax, handles, ["Ideal path", "Smoothed track", "Raw samples"])
    fig.tight_layout()
    return fig


def plot_02_lap_overlay(df, wps, x_s, y_s, laps):
    fig, ax = new_figure("Lap-by-Lap Trajectory Overlay (Repeatability)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    if TANK_XLIM: ax.set_xlim(TANK_XLIM)
    if TANK_YLIM: ax.set_ylim(TANK_YLIM)

    ax.plot(list(wps[:, 0]) + [wps[0, 0]], list(wps[:, 1]) + [wps[0, 1]],
            "k--", linewidth=1.5, alpha=0.5, label="Ideal path")

    unique_laps = sorted(set(laps))
    colours = lap_colormap(len(unique_laps))
    for col, ln in zip(colours, unique_laps):
        mask = laps == ln
        if mask.sum() < 2:
            continue
        ax.plot(x_s[mask], y_s[mask], color=col, lw=2.0, alpha=0.85,
                label=f"Lap {ln}")

    for i, (wx, wy) in enumerate(wps):
        ax.plot(wx, wy, "D", color="black", ms=7, zorder=5)
        ax.annotate(f"WP{i}", (wx, wy), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold")
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_03_cte(df, cte_signed, laps):
    fig, ax = new_figure("Cross-Track Error vs Time (coloured by lap)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Signed CTE (m)")

    cte_smooth = safe_savgol(cte_signed)
    unique_laps = sorted(set(laps))
    colours = lap_colormap(len(unique_laps))

    # Faint raw line
    ax.plot(df["t_sec"], cte_signed, color="lightgray", lw=0.7, alpha=0.6,
            label="Raw")

    for col, ln in zip(colours, unique_laps):
        mask = laps == ln
        if mask.sum() < 2:
            continue
        ax.plot(df["t_sec"][mask], cte_smooth[mask], color=col, lw=2.0,
                label=f"Lap {ln}")

    ax.axhline(0, color="black", lw=1, alpha=0.6)
    rms = float(np.sqrt(np.mean(cte_signed ** 2)))
    ax.axhline( rms, color="red", ls=":", lw=1, alpha=0.7, label=f"±RMS = {rms:.3f} m")
    ax.axhline(-rms, color="red", ls=":", lw=1, alpha=0.7)

    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_04_cte_by_lap(cte_abs, laps):
    fig, ax = new_figure("Absolute CTE Distribution per Lap")
    ax.set_xlabel("Lap"); ax.set_ylabel("|CTE| (m)")

    unique_laps = sorted(set(laps))
    data = [cte_abs[laps == ln] for ln in unique_laps if (laps == ln).sum() > 1]
    labels = [f"Lap {ln}" for ln in unique_laps if (laps == ln).sum() > 1]
    if not data:
        ax.text(0.5, 0.5, "Not enough lap data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return fig

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color="red", lw=1.5),
                    medianprops=dict(color="black", lw=1.5))
    colours = lap_colormap(len(data))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.plot([], [], color="black", lw=1.5, label="Median")
    ax.plot([], [], color="red",   lw=1.5, ls="--", label="Mean")
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_05_overshoot(df, wps, wp_overshoots, x_s, y_s):
    fig, ax = new_figure("Waypoint Overshoot Trace")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    if TANK_XLIM: ax.set_xlim(TANK_XLIM)
    if TANK_YLIM: ax.set_ylim(TANK_YLIM)
    TIME_WINDOW_SEC = 3

    ax.plot(list(wps[:, 0]) + [wps[0, 0]], list(wps[:, 1]) + [wps[0, 1]],
            "k--", linewidth=1, alpha=0.4, label="Ideal path")
    n_wps = len(wps)
    wp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f"][:n_wps]

    overshoot_by_wp = {}
    for wp_idx, os_dist, os_idx in wp_overshoots:
        overshoot_by_wp.setdefault(wp_idx, []).append((os_dist, os_idx))

    for i, (wx, wy) in enumerate(wps):
        c = wp_colors[i]
        ax.plot(wx, wy, "D", color=c, ms=10, zorder=5,
                label=f"WP{i} ({wx:.1f}, {wy:.1f})")
        ax.add_patch(Circle((wx, wy), WP_ARRIVAL_RADIUS, fill=False,
                            ec=c, ls="--", lw=1, alpha=0.5))
        if i in overshoot_by_wp:
            max_dist, max_idx = max(overshoot_by_wp[i], key=lambda e: e[0])
            t_apex = df["t_sec"].iloc[max_idx]
            mask = (df["t_sec"] >= t_apex - TIME_WINDOW_SEC) & \
                   (df["t_sec"] <= t_apex + TIME_WINDOW_SEC)
            ax.plot(x_s[mask], y_s[mask], "-", color=c, lw=2.5, alpha=0.7,
                    label=f"WP{i} trace (±{TIME_WINDOW_SEC}s)")
            ax.plot(x_s[max_idx], y_s[max_idx], "x", color=c, ms=10, mew=2.5, zorder=6)
            ax.annotate(f"{max_dist:.2f} m", (x_s[max_idx], y_s[max_idx]),
                        textcoords="offset points", xytext=(8, -8),
                        ha="left", va="top", fontsize=8, color=c, fontweight="bold")
    left_legend(ax, fontsize=8)
    fig.tight_layout()
    return fig


def plot_06_wp_table(wps, wp_arrivals, wp_overshoots, df):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title("Waypoint Arrival Stats", fontweight="bold"); ax.axis("off")
    n_wps = len(wps)
    arrivals = {}
    for wp, _, _ in wp_arrivals: arrivals.setdefault(wp, 0); arrivals[wp] += 1
    overs = {}
    for wp, d, _ in wp_overshoots: overs.setdefault(wp, []).append(d)
    rows = []
    for i in range(n_wps):
        n_arr = arrivals.get(i, 0)
        mean_os = float(np.mean(overs[i])) if i in overs else 0.0
        max_os  = float(np.max(overs[i]))  if i in overs else 0.0
        min_dist = float(np.linalg.norm(df[["x", "y"]].values - wps[i], axis=1).min())
        rows.append([f"WP{i} ({wps[i][0]:.1f}, {wps[i][1]:.1f})",
                     f"{n_arr}", f"{min_dist:.4f}",
                     f"{mean_os:.3f}", f"{max_os:.3f}"])
    cols = ["Waypoint", "Arrivals", "Min Dist (m)", "Mean Overshoot (m)", "Max Overshoot (m)"]
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    tbl.auto_set_column_width(list(range(len(cols))))
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("gray")
        if key[0] == 0:
            cell.set_facecolor("#d9e2f3"); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
    return fig


def plot_07_speed(df, speed):
    fig, ax = new_figure("Speed Profile (rolling average)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
    window = min(50, max(5, len(speed) // 20))
    speed_smooth = pd.Series(speed).rolling(window, center=True, min_periods=1).mean()
    ax.plot(df["t_sec"], speed_smooth, lw=2, color="steelblue",
            label=f"Rolling avg (n={window})")
    mean_v = float(np.nanmean(speed_smooth))
    ax.axhline(mean_v, color="red", ls="--", lw=1.2,
               label=f"Mean = {mean_v:.3f} m/s")
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_08_heading(df, heading_deg, has_psi_d):
    fig, ax = new_figure("Heading: Actual vs Desired")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading (deg)")
    ax.plot(df["t_sec"], heading_deg, lw=1.8, color="steelblue", label="ψ actual")
    if has_psi_d:
        psi_d_deg = np.degrees(np.unwrap(df["psi_d"].values))
        ax.plot(df["t_sec"], psi_d_deg, lw=1.5, ls="--", color="red",
                label="ψ desired")
    # Adaptive gridlines (data may exceed ±180° after unwrapping)
    lo = float(np.nanmin(heading_deg))
    hi = float(np.nanmax(heading_deg))
    span = hi - lo
    step = 45 if span <= 360 else (90 if span <= 720 else 180)
    lo_t = int(np.floor(lo / step) * step)
    hi_t = int(np.ceil(hi / step) * step)
    ax.set_yticks(np.arange(lo_t, hi_t + 1, step))
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_09_heading_error(df, has_psi_err, heading_deg, has_psi_d):
    fig, ax = new_figure("Heading Error vs Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading error (deg)")
    if has_psi_err:
        # psi_err comes pre-wrapped in [-pi, pi]; convert to degrees.
        err_deg = np.degrees(df["psi_err"].values)
    elif has_psi_d:
        psi_d = np.degrees(np.unwrap(df["psi_d"].values))
        err_deg = ((psi_d - heading_deg + 180) % 360) - 180
    else:
        ax.text(0.5, 0.5, "No psi_err / psi_d available",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        return fig
    err_smooth = safe_savgol(err_deg)
    ax.plot(df["t_sec"], err_deg, color="lightgray", lw=0.7, alpha=0.6, label="Raw")
    ax.plot(df["t_sec"], err_smooth, color="darkorange", lw=1.8, label="Smoothed")
    ax.axhline(0, color="black", lw=1, alpha=0.6)
    rms = float(np.sqrt(np.mean(err_deg ** 2)))
    ax.axhline( rms, color="red", ls=":", lw=1, label=f"±RMS = {rms:.1f}°")
    ax.axhline(-rms, color="red", ls=":", lw=1)
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_10_yaw_rate(df, yaw_rate, has_r_cmd):
    fig, ax = new_figure("Yaw Rate: Actual vs Commanded")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Yaw rate (rad/s)")
    ax.plot(df["t_sec"], yaw_rate, color="steelblue", lw=1.8, label="r actual (from ψ̇)")
    if has_r_cmd:
        r_cmd_smooth = safe_savgol(df["r_cmd"].values, window=11, poly=2)
        ax.plot(df["t_sec"], r_cmd_smooth, color="red", ls="--", lw=1.5,
                label="r commanded")
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_11_control(df, has_u, has_r):
    fig, ax = new_figure("Control Commands")
    ax.set_xlabel("Time (s)")
    if has_u:
        ax.plot(df["t_sec"], df["U_cmd"], color="steelblue", lw=1.5,
                label="U_cmd (m/s)")
        ax.set_ylabel("Surge command (m/s)", color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue")
    if has_r:
        ax2 = ax.twinx()
        ax2.plot(df["t_sec"], df["r_cmd"], color="darkorange", lw=1.2, alpha=0.8,
                 label="r_cmd (rad/s)")
        ax2.set_ylabel("Yaw-rate command (rad/s)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.grid(False)
    if not (has_u or has_r):
        ax.text(0.5, 0.5, "No control commands logged",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    fig.tight_layout()
    return fig


def plot_12_ilos_integral(df, has_y_int, laps):
    fig, ax = new_figure("ILOS Integral State Convergence")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("y_int (m·s)")
    if not has_y_int:
        ax.text(0.5, 0.5, "No y_int column (not an ILOS log)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        return fig
    ax.plot(df["t_sec"], df["y_int"], color="purple", lw=1.6, label="ILOS integral")
    # Mark lap boundaries
    unique_laps = sorted(set(laps))
    for ln in unique_laps[1:]:
        idx = np.argmax(laps == ln)
        ax.axvline(df["t_sec"].iloc[idx], color="black", ls=":", alpha=0.4)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_13_lap_duration(lap_stats):
    fig, ax = new_figure("Lap Duration")
    ax.set_xlabel("Lap"); ax.set_ylabel("Duration (s)")
    if lap_stats:
        ln  = [s["lap"] for s in lap_stats]
        dur = [s["duration_s"] for s in lap_stats]
        bars = ax.bar(ln, dur, color=lap_colormap(len(lap_stats)),
                      edgecolor="black", lw=0.8)
        for b, d in zip(bars, dur):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{d:.1f} s", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(ln)
    else:
        ax.text(0.5, 0.5, "Not enough laps", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    fig.tight_layout()
    return fig


def plot_14_efficiency(lap_stats):
    fig, ax = new_figure("Path Efficiency per Lap")
    ax.set_xlabel("Lap"); ax.set_ylabel("Efficiency (%)")
    if lap_stats:
        ln  = [s["lap"] for s in lap_stats]
        eff = [s["efficiency_%"] for s in lap_stats]
        bars = ax.bar(ln, eff, color=lap_colormap(len(lap_stats)),
                      edgecolor="black", lw=0.8)
        ax.axhline(100, color="red", ls="--", lw=1.5, label="100% (ideal)")
        for b, e in zip(bars, eff):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{e:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(ln)
        left_legend(ax)
    else:
        ax.text(0.5, 0.5, "Not enough laps", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    fig.tight_layout()
    return fig


def plot_15_cumulative(df, wps, segments):
    fig, ax = new_figure("Cumulative Distance: Actual vs Ideal")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
    xy = df[["x", "y"]].values; n_wps = len(wps)
    step_d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum_actual = np.concatenate([[0], np.cumsum(step_d)])
    cum_ideal = np.zeros(len(df))
    running = 0; prev_seg = segments[0]; seg_start = 0
    for i in range(len(df)):
        seg = segments[i]
        if seg != prev_seg and seg >= 0:
            running += segment_length(wps[prev_seg], wps[(prev_seg + 1) % n_wps])
            prev_seg = seg; seg_start = running
        if seg >= 0:
            a = wps[seg]; b = wps[(seg + 1) % n_wps]
            _, _, t_param = point_to_segment_dist(xy[i], a, b)
            cum_ideal[i] = seg_start + t_param * segment_length(a, b)
    ax.plot(df["t_sec"], cum_actual, lw=1.7, color="steelblue", label="Actual")
    ax.plot(df["t_sec"], cum_ideal,  lw=1.7, color="red", ls="--", label="Ideal (projected)")
    ax.fill_between(df["t_sec"], cum_actual, cum_ideal,
                    alpha=0.15, color="red", label="Wasted distance")
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_16_cte_hist(cte_signed):
    fig, ax = new_figure("CTE Distribution (with Gaussian fit)")
    ax.set_xlabel("Signed CTE (m)"); ax.set_ylabel("Density")
    mu = float(np.mean(cte_signed)); sig = float(np.std(cte_signed))
    n, bins, _ = ax.hist(cte_signed, bins=60, density=True, color="steelblue",
                         edgecolor="black", lw=0.5, alpha=0.7,
                         label=f"Observed (n={len(cte_signed)})")
    # Gaussian fit overlay
    xs = np.linspace(bins[0], bins[-1], 300)
    gauss = (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sig) ** 2)
    ax.plot(xs, gauss, "r-", lw=2,
            label=f"N(μ={mu:.3f}, σ={sig:.3f})")
    ax.axvline(0,  color="black", ls="--", lw=1.2, label="Ideal (0)")
    ax.axvline(mu, color="red",   ls="-",  lw=1.2, alpha=0.6)
    left_legend(ax)
    fig.tight_layout()
    return fig


def plot_17_cte_psd(df, cte_signed):
    fig, ax = new_figure("CTE Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (m²/Hz)")
    # Estimate sample rate from time vector
    t = df["t_sec"].values
    fs = 1.0 / np.mean(np.diff(t))
    nperseg = min(256, len(cte_signed) // 4)
    if nperseg < 16:
        ax.text(0.5, 0.5, "Not enough samples for PSD",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        return fig
    f, Pxx = welch(cte_signed - np.mean(cte_signed), fs=fs, nperseg=nperseg)
    # Drop the DC bin so log-x doesn't choke
    f, Pxx = f[1:], Pxx[1:]
    ax.loglog(f, Pxx, color="steelblue", lw=1.5)
    ax.set_xlim(f[0], f[-1])
    if np.any(Pxx > 0):
        peak = f[np.argmax(Pxx)]
        ax.axvline(peak, color="red", ls="--", lw=1,
                   label=f"Peak ≈ {peak:.2f} Hz  ({1.0/peak:.1f} s period)")
        left_legend(ax)
    fig.tight_layout()
    return fig


def plot_18_summary(df, cte_abs, lap_stats, ideal_lap, wp_overshoots, speed,
                    heading, has_psi_err):
    fig, ax = plt.subplots(figsize=(7.5, 7)); ax.axis("off")
    ax.set_title("Overall Summary", fontweight="bold")
    xy = df[["x", "y"]].values
    total_t = df["t_sec"].iloc[-1] - df["t_sec"].iloc[0]
    total_d = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    n_laps = len(lap_stats)
    mean_lap_t = float(np.mean([s["duration_s"]   for s in lap_stats])) if lap_stats else 0
    mean_eff   = float(np.mean([s["efficiency_%"] for s in lap_stats])) if lap_stats else 0
    mean_v_global = float(np.nanmean(speed))
    os_vals = [d for _, d, _ in wp_overshoots]
    mean_os = float(np.mean(os_vals)) if os_vals else 0
    max_os  = float(np.max(os_vals))  if os_vals else 0
    heading_range = float(np.nanmax(heading) - np.nanmin(heading))

    rows = [
        ["Total time",            f"{total_t:.1f} s"],
        ["Total distance",        f"{total_d:.3f} m"],
        ["Laps completed",        f"{n_laps}"],
        ["Mean lap time",         f"{mean_lap_t:.1f} s"],
        ["Mean speed (avg)",      f"{mean_v_global:.4f} m/s"],
        ["Ideal lap distance",    f"{ideal_lap:.3f} m"],
        ["Mean path efficiency",  f"{mean_eff:.1f} %"],
        ["Mean |CTE|",            f"{np.mean(cte_abs):.4f} m"],
        ["Max  |CTE|",            f"{np.max(cte_abs):.4f} m"],
        ["RMS  CTE",              f"{np.sqrt(np.mean(cte_abs**2)):.4f} m"],
        ["95th percentile |CTE|", f"{np.percentile(cte_abs, 95):.4f} m"],
        ["Heading range (unwrapped)", f"{heading_range:.0f}°"],
        ["Mean overshoot",        f"{mean_os:.4f} m"],
        ["Max overshoot",         f"{max_os:.4f} m"],
    ]
    if has_psi_err:
        # Append heading-error stats if available
        err = np.degrees(df["psi_err"].values)
        rows.insert(11, ["Mean |ψ error|", f"{np.mean(np.abs(err)):.1f}°"])
        rows.insert(12, ["RMS ψ error",    f"{np.sqrt(np.mean(err**2)):.1f}°"])

    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    tbl.auto_set_column_width([0, 1])
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("gray")
        if key[0] == 0:
            cell.set_facecolor("#d9e2f3"); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
    return fig


# ─────────────────────── MAIN ───────────────────────────────────
def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_log()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")

    # ── CSV-format compatibility ────────────────────────────────
    if "t_sec" not in df.columns and "t" in df.columns:
        df.rename(columns={"t": "t_sec"}, inplace=True)
    if "t_sec" in df.columns:
        df["t_sec"] = df["t_sec"] - df["t_sec"].iloc[0]

    has_psi    = "psi"     in df.columns
    has_psi_d  = "psi_d"   in df.columns
    has_psi_err= "psi_err" in df.columns
    has_y_e    = "y_e"     in df.columns
    has_y_int  = "y_int"   in df.columns
    has_u_cmd  = "U_cmd"   in df.columns
    has_r_cmd  = "r_cmd"   in df.columns
    has_wp_idx = "wp_idx"  in df.columns
    fmt = "ILOS" if (has_psi and has_y_e and has_psi_d) else "generic"

    # ── Data quality diagnostics ────────────────────────────────
    n_unique_xy = df[["x", "y"]].drop_duplicates().shape[0]
    dur = df["t_sec"].iloc[-1] - df["t_sec"].iloc[0]
    log_rate    = len(df) / dur if dur > 0 else 0
    effective_mocap_rate = n_unique_xy / dur if dur > 0 else 0

    # ── Derived signals ─────────────────────────────────────────
    x_s = safe_savgol(df["x"].values)
    y_s = safe_savgol(df["y"].values)
    heading_deg = compute_heading(df, has_psi)
    yaw_rate    = compute_yaw_rate(df, has_psi)
    speed       = compute_speed(df, smooth=True)

    segments, laps, wp_arrivals, wp_departures, wp_overshoots = \
        assign_segments_and_laps(df, WAYPOINTS, WP_ARRIVAL_RADIUS, has_wp_idx)
    cte_signed, cte_abs = compute_cte(df, WAYPOINTS, segments, has_y_e)
    lap_stats, ideal_lap_dist = compute_lap_stats(df, WAYPOINTS, laps, cte_abs)

    # ── Console report ──────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  USV Run Analysis — {path.name}")
    print(f"{'═'*60}")
    print(f"  Format:                 {fmt}")
    print(f"  Samples:                {len(df)}")
    print(f"  Duration:               {dur:.1f} s")
    print(f"  Log rate:               {log_rate:.1f} Hz")
    print(f"  Effective mocap rate:   {effective_mocap_rate:.2f} Hz "
          f"({n_unique_xy} unique poses)")
    if effective_mocap_rate < 5 and log_rate > 10:
        print(f"  ⚠ Mocap update rate is much lower than log rate — "
              f"position data was held constant between mocap updates. "
              f"Smoothing applied for plotting.")
    print(f"  Laps:                   {len(lap_stats)}")
    print(f"  RMS CTE:                {np.sqrt(np.mean(cte_abs**2)):.4f} m")
    print(f"  Mean |CTE|:             {np.mean(cte_abs):.4f} m")
    print(f"  Max  |CTE|:             {np.max(cte_abs):.4f} m")
    print(f"  95th pct |CTE|:         {np.percentile(cte_abs, 95):.4f} m")
    if has_psi_err:
        err = np.degrees(df["psi_err"].values)
        print(f"  Mean |ψ error|:         {np.mean(np.abs(err)):.1f}°")
        print(f"  RMS  ψ error:           {np.sqrt(np.mean(err**2)):.1f}°")
    if lap_stats:
        print(f"\n  Per-lap breakdown:")
        for ls in lap_stats:
            print(f"    Lap {ls['lap']}: {ls['duration_s']:5.1f}s   "
                  f"eff={ls['efficiency_%']:5.1f}%   "
                  f"mean|CTE|={ls['mean_cte_m']:.4f}m   "
                  f"RMS={ls['rms_cte_m']:.4f}m   "
                  f"speed={ls['mean_speed_ms']:.3f}m/s")
    if wp_overshoots:
        print(f"\n  Overshoot summary:")
        osbw = {}
        for wp, d, _ in wp_overshoots: osbw.setdefault(wp, []).append(d)
        for wp in sorted(osbw):
            vals = osbw[wp]
            print(f"    WP{wp}: mean={np.mean(vals):.3f}m  "
                  f"max={np.max(vals):.3f}m  n={len(vals)}")
    print(f"{'═'*60}\n")

    # ── Plot generation ─────────────────────────────────────────
    setup_matplotlib()
    stem = path.stem; out_dir = path.parent

    figures = [
        ("01_trajectory",        plot_01_trajectory(df, WAYPOINTS, x_s, y_s)),
        ("02_lap_overlay",       plot_02_lap_overlay(df, WAYPOINTS, x_s, y_s, laps)),
        ("03_cross_track_err",   plot_03_cte(df, cte_signed, laps)),
        ("04_cte_by_lap",        plot_04_cte_by_lap(cte_abs, laps)),
        ("05_overshoot",         plot_05_overshoot(df, WAYPOINTS, wp_overshoots, x_s, y_s)),
        ("06_wp_stats",          plot_06_wp_table(WAYPOINTS, wp_arrivals, wp_overshoots, df)),
        ("07_speed_profile",     plot_07_speed(df, speed)),
        ("08_heading",           plot_08_heading(df, heading_deg, has_psi_d)),
        ("09_heading_error",     plot_09_heading_error(df, has_psi_err, heading_deg, has_psi_d)),
        ("10_yaw_rate",          plot_10_yaw_rate(df, yaw_rate, has_r_cmd)),
        ("11_control_inputs",    plot_11_control(df, has_u_cmd, has_r_cmd)),
        ("12_ilos_integral",     plot_12_ilos_integral(df, has_y_int, laps)),
        ("13_lap_duration",      plot_13_lap_duration(lap_stats)),
        ("14_path_efficiency",   plot_14_efficiency(lap_stats)),
        ("15_cumulative_dist",   plot_15_cumulative(df, WAYPOINTS, segments)),
        ("16_cte_histogram",     plot_16_cte_hist(cte_signed)),
        ("17_cte_psd",           plot_17_cte_psd(df, cte_signed)),
        ("18_summary",           plot_18_summary(df, cte_abs, lap_stats, ideal_lap_dist,
                                                  wp_overshoots, speed, heading_deg,
                                                  has_psi_err)),
    ]
    for name, fig in figures:
        out = out_dir / f"{stem}_{name}.png"
        fig.savefig(out)
        print(f"  Saved: {out.name}")
    print(f"\nSaved {len(figures)} figures to {out_dir}/")
    plt.show()


if __name__ == "__main__":
    main()