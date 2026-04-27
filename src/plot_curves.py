"""Plot training loss curves for all model runs from their log.csv files.

Usage:
  python -m src.plot_curves --runs results/runs --out results/training_curves.png
  python -m src.plot_curves --runs results/runs --out results/training_curves.png --pattern "sp_*"
"""

import argparse
import glob
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(csv_path: str) -> tuple[list, list, list, list]:
    steps_train, losses_train = [], []
    steps_val, losses_val = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            tl = row.get("train_loss", "").strip()
            vl = row.get("val_loss", "").strip()
            if tl:
                try:
                    steps_train.append(step)
                    losses_train.append(float(tl))
                except ValueError:
                    pass
            if vl:
                try:
                    steps_val.append(step)
                    losses_val.append(float(vl))
                except ValueError:
                    pass
    return steps_train, losses_train, steps_val, losses_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/training_curves.png")
    ap.add_argument("--pattern", default="*", help="glob pattern for run dirs (e.g. 'sp_*')")
    ap.add_argument("--smooth", type=int, default=5, help="EMA smoothing window (steps)")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    logs = sorted(glob.glob(str(runs_dir / args.pattern / "log.csv")))
    if not logs:
        print(f"[plot_curves] no log.csv files found under {runs_dir}/{args.pattern}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))

    for log_path, color in zip(logs, colors):
        tag = Path(log_path).parent.name
        st, tl, sv, vl = load_log(log_path)
        if st:
            ax1.plot(st, tl, alpha=0.3, color=color, linewidth=0.8)
            # EMA smoothed
            if len(tl) >= args.smooth:
                w = args.smooth
                smoothed = np.convolve(tl, np.ones(w) / w, mode="valid")
                ax1.plot(st[w - 1:], smoothed, color=color, linewidth=1.5, label=tag)
            else:
                ax1.plot(st, tl, color=color, linewidth=1.5, label=tag)
        if sv:
            ax2.plot(sv, vl, "o-", color=color, linewidth=1.5, markersize=4, label=tag)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train loss")
    ax1.set_title("Training loss curves")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val loss")
    ax2.set_title("Validation loss during training")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[plot_curves] saved {args.out}")


if __name__ == "__main__":
    main()
