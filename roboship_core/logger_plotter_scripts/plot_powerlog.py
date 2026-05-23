#!/usr/bin/env python3
"""
plot_power_log.py – Generate figures from a power_logger CSV.

Usage:
    python3 plot_power_log.py ~/power_logs/power_log_20260507_143000.csv

Produces three plots saved as PNGs next to the CSV:
    1. Stacked: throttle (%) on top, current (A) on bottom
    2. Current vs signed throttle % (U-shape)
    3. Cumulative mAh vs time (standalone)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["throttle_us"])
    df["throttle_us"] = df["throttle_us"].astype(int)
    df["throttle_pct"] = (df["throttle_us"] - 1500) / 10.0
    return df


def label_on_hline(ax, value, label, color, fontsize=10):
    """
    Place a label centred horizontally on the plot, just below a dashed hline.
    Draws the hline too.
    """
    ax.axhline(value, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        0.5, value, label,
        transform=ax.get_yaxis_transform(),  # x in axes fraction, y in data
        fontsize=fontsize,
        color=color,
        ha="center",
        va="top",  # just below the line
        fontweight="bold",
        clip_on=False,
        bbox=dict(
            facecolor="white", edgecolor="none",
            alpha=0.85, pad=2,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Plot 1: stacked time series
# ─────────────────────────────────────────────────────────────
def plot_stacked_time_series(df: pd.DataFrame, out_dir: str):
    fig, (ax_thr, ax_cur) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
    )

    color_curr = "#e74c3c"
    color_thr = "#3498db"
    t = df["elapsed_s"]

    # ── top: throttle ──
    ax_thr.plot(t, df["throttle_pct"], color=color_thr, linewidth=1)
    ax_thr.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_thr.set_ylabel("Throttle (%)")
    ax_thr.set_xlabel("Time (s)")
    ax_thr.set_title("Throttle", fontsize=12)
    ax_thr.grid(True, alpha=0.2)
    thr_abs_max = max(abs(df["throttle_pct"].min()), abs(df["throttle_pct"].max()), 50)
    margin = thr_abs_max * 0.15
    ax_thr.set_ylim(-thr_abs_max - margin, thr_abs_max + margin)
    ax_thr.tick_params(labelbottom=True)

    fig.suptitle("Current & Throttle vs Time", fontsize=14, y=1.01)

    # ── bottom: current ──
    ax_cur.plot(t, df["current_A"], color=color_curr, linewidth=1)
    ax_cur.set_ylabel("Current (A)")
    ax_cur.set_xlabel("Time (s)")
    ax_cur.set_title("Current", fontsize=12)
    ax_cur.set_ylim(bottom=0)
    ax_cur.grid(True, alpha=0.2)
    ax_cur.set_xlim(t.iloc[0], t.iloc[-1])

    # idle current
    zero_mask = df["throttle_pct"].abs()
    closest_idx = zero_mask.idxmin()
    idle_a = df.loc[closest_idx, "current_A"]
    label_on_hline(
        ax_cur, idle_a,
        f"Idle current = {idle_a:.2f} A  (0% throttle)",
        "grey", fontsize=10,
    )

    # peak current
    peak_idx = df["current_A"].idxmax()
    peak_a = df.loc[peak_idx, "current_A"]
    peak_t = df.loc[peak_idx, "elapsed_s"]
    ax_cur.plot(peak_t, peak_a, "o", color=color_curr, markersize=4)
    label_on_hline(
        ax_cur, peak_a,
        r"$A_{\mathrm{peak}}$" + f" = {peak_a:.2f} A",
        color_curr, fontsize=10,
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "current_throttle_vs_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
# Plot 2: current vs signed throttle (U-shape)
# ─────────────────────────────────────────────────────────────
def plot_current_vs_throttle(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    dead_zone = 2.0
    active = df[df["throttle_pct"].abs() > dead_zone]

    ax.scatter(
        active["throttle_pct"],
        active["current_A"],
        color="#e74c3c",
        s=10,
        alpha=0.4,
    )

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Throttle (%)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Current vs Throttle")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # peak current label
    peak_a = active["current_A"].max()
    label_on_hline(
        ax, peak_a,
        r"$A_{\mathrm{peak}}$" + f" = {peak_a:.2f} A",
        "#e74c3c", fontsize=10,
    )

    # direction arrows below x-axis
    arrow_y = -0.08
    ax.annotate(
        "",
        xy=(0.05, arrow_y), xycoords="axes fraction",
        xytext=(0.42, arrow_y), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
        annotation_clip=False,
    )
    ax.text(
        0.22, arrow_y - 0.035, "Reverse",
        transform=ax.transAxes, ha="center", fontsize=10,
        color="#e67e22", fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(0.95, arrow_y), xycoords="axes fraction",
        xytext=(0.58, arrow_y), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
        annotation_clip=False,
    )
    ax.text(
        0.78, arrow_y - 0.035, "Forward",
        transform=ax.transAxes, ha="center", fontsize=10,
        color="#27ae60", fontweight="bold",
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    path = os.path.join(out_dir, "current_vs_throttle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
# Plot 3: cumulative mAh (standalone)
# ─────────────────────────────────────────────────────────────
def plot_cumulative_ah(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    color_energy = "#8e44ad"
    t = df["elapsed_s"]

    # ── compute cumulative mAh ──
    dt_hours = np.diff(t.values) / 3600.0
    avg_current = (df["current_A"].values[:-1] + df["current_A"].values[1:]) / 2.0
    cumulative_mah = np.concatenate([[0], np.cumsum(avg_current * dt_hours)]) * 1000

    ax.plot(t, cumulative_mah, color=color_energy, linewidth=1.5)
    ax.set_ylabel("Energy consumed (mAh)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cumulative Energy Consumed", fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(t.iloc[0], t.iloc[-1])

    # final mAh label
    final_mah = cumulative_mah[-1]
    final_t = t.iloc[-1]
    ax.plot(final_t, final_mah, "o", color=color_energy, markersize=4)
    label_on_hline(
        ax, final_mah,
        r"$E_{\mathrm{total}}$" + f" = {final_mah:.1f} mAh",
        color_energy, fontsize=11,
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "cumulative_energy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    duration = df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0]
    dt_hours = np.diff(df["elapsed_s"].values) / 3600.0
    avg_current = (df["current_A"].values[:-1] + df["current_A"].values[1:]) / 2.0
    total_mah = np.sum(avg_current * dt_hours) * 1000

    print(f"\n{'─' * 40}")
    print(f"  Duration:       {duration:.1f} s")
    print(f"  Peak current:   {df['current_A'].max():.2f} A")
    print(f"  Mean current:   {df['current_A'].mean():.2f} A")
    print(f"  Total consumed: {total_mah:.1f} mAh")
    print(f"  Throttle range: {df['throttle_pct'].min():.0f}% – {df['throttle_pct'].max():.0f}%")
    print(f"{'─' * 40}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_power_log.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = load(csv_path)
    out_dir = os.path.dirname(os.path.abspath(csv_path))

    print_summary(df)
    plot_stacked_time_series(df, out_dir)
    plot_current_vs_throttle(df, out_dir)
    plot_cumulative_ah(df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()