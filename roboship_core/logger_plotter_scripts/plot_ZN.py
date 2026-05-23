#!/usr/bin/env python3
"""
plot_zn.py - Plot yaw rate and surge velocity from ZN tuning logger CSV.

Filters to GUIDED+armed data only. Useful for finding K_cr and T_cr.

Usage:
  python3 plot_zn.py                          # plots most recent zn_tuning_*.csv
  python3 plot_zn.py ~/roboship_logs/zn_tuning_20260518_190000.csv
"""

import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def load_csv(filepath):
    t, r_actual, u = [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['mode'] == 'GUIDED' and row['armed'] == '1':
                t.append(float(row['t']))
                r_actual.append(float(row['r_actual']))
                u.append(float(row['u']))
    return t, r_actual, u


def main():
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        log_dir = Path.home() / 'roboship_logs'
        files = sorted(log_dir.glob('zn_tuning_*.csv'))
        if not files:
            print(f'No zn_tuning_*.csv files in {log_dir}')
            sys.exit(1)
        filepath = files[-1]
        print(f'Using most recent: {filepath.name}')

    t, r_actual, u = load_csv(filepath)

    if not t:
        print('No GUIDED+armed data found in CSV.')
        sys.exit(1)

    # Zero the time axis
    t0 = t[0]
    t = [ti - t0 for ti in t]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, r_actual, 'b-', linewidth=0.8)
    ax1.set_ylabel('Yaw rate r (rad/s)')
    ax1.set_title(f'ZN Tuning — {filepath.name}')
    ax1.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, u, 'r-', linewidth=0.8)
    ax2.set_ylabel('Surge velocity u (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = filepath.with_suffix('.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    plt.show()


if __name__ == '__main__':
    main()