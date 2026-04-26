"""Fit power law L = a * N^{-alpha} + c to (param-count, val-loss) pairs.

Reads results/runs/<tag>/metrics.json for tags matching --pattern (e.g. "sp_*"
and "mup_*"), fits one curve per parameterization, saves plot + JSON, and
extrapolates to 10x the largest model with a bootstrap confidence interval.
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def power_law(N, a, alpha, c):
    return a * np.power(N, -alpha) + c


def fit(N, L, p0=(1.0, 0.1, 1.0)):
    popt, pcov = curve_fit(power_law, N, L, p0=p0, maxfev=20000,
                           bounds=([1e-6, 1e-4, 0.0], [1e6, 2.0, 10.0]))
    return popt, pcov


def bootstrap_extrapolate(N, L, N_target, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), size=len(N))
        try:
            popt, _ = fit(N[idx], L[idx])
            preds.append(power_law(N_target, *popt))
        except Exception:
            continue
    preds = np.array(preds)
    return float(np.median(preds)), float(np.percentile(preds, 2.5)), float(np.percentile(preds, 97.5))


def collect(runs_dir: Path, pattern: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(runs_dir / pattern / "metrics.json"))):
        m = json.load(open(p))
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/scaling.json")
    ap.add_argument("--plot", default="results/scaling.png")
    ap.add_argument("--sp-pattern", default="sp_*")
    ap.add_argument("--mup-pattern", default="mup_*")
    args = ap.parse_args()

    runs = Path(args.runs)
    fig, ax = plt.subplots(figsize=(7, 5))
    summary = {}

    for label, pat, color in [("SP", args.sp_pattern, "C0"), ("µP", args.mup_pattern, "C1")]:
        ms = collect(runs, pat)
        if not ms:
            print(f"[fit] no runs for {label} (pattern={pat})"); continue
        N = np.array([m["n_params"] for m in ms], dtype=float)
        L = np.array([m["final_val_loss"] for m in ms], dtype=float)
        order = np.argsort(N); N, L = N[order], L[order]
        try:
            popt, pcov = fit(N, L)
            a, alpha, c = popt
        except Exception as e:
            print(f"[fit] {label} fit failed: {e}"); continue
        N_max = N.max(); N_target = 10 * N_max
        med, lo, hi = bootstrap_extrapolate(N, L, N_target)
        Ng = np.geomspace(N.min() * 0.5, N_target * 1.2, 200)
        ax.scatter(N, L, color=color, label=f"{label} runs")
        ax.plot(Ng, power_law(Ng, *popt), color=color, linestyle="--",
                label=f"{label} fit: $L={a:.2f} N^{{-{alpha:.3f}}}+{c:.2f}$")
        ax.axvline(N_target, color=color, alpha=0.2)
        summary[label] = dict(
            N=N.tolist(), L=L.tolist(),
            a=float(a), alpha=float(alpha), c=float(c),
            cov=pcov.tolist(),
            extrapolation=dict(N=float(N_target), median=med, ci_low=lo, ci_high=hi),
            tags=[m["tag"] for m in ms],
        )
        print(f"[fit] {label}: alpha={alpha:.4f}, a={a:.3f}, c={c:.3f}; "
              f"L({N_target:.2e}) = {med:.3f} [95% {lo:.3f}, {hi:.3f}]")

    ax.set_xscale("log"); ax.set_xlabel("Parameters N"); ax.set_ylabel("Validation loss (1 epoch)")
    ax.set_title("SVG transformer scaling")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.plot, dpi=150)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fit] wrote {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
