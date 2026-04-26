"""LR sweep on the smallest model.

Runs train.py over a log-spaced LR grid and writes results/sweep_<tag>.json
listing (lr, val_loss). The minimum-val-loss LR is used as the transferable
learning rate for all larger models in Part 2 (SP) or Part 3 (µP).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--out-runs", default="results/runs")
    ap.add_argument("--out", default="results/sweep.json")
    ap.add_argument("--mup", action="store_true")
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap each sweep run; useful since LR sweep on Tiny is cheap "
                         "but you may want even shorter for very wide sweeps")
    ap.add_argument("--tag-prefix", default=None)
    args = ap.parse_args()

    prefix = args.tag_prefix or ("tiny_mup" if args.mup else "tiny_sp")
    results = []
    for lr in args.lrs:
        tag = f"{prefix}_lr{lr:.0e}"
        cmd = [sys.executable, "-m", "src.train",
               "--config", args.config, "--data", args.data,
               "--out", args.out_runs, "--tag", tag,
               "--lr", str(lr), "--epochs", "1"]
        if args.mup: cmd.append("--mup")
        if args.max_steps: cmd += ["--max-steps", str(args.max_steps)]
        print("[sweep] running:", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[sweep] FAILED at lr={lr}; recording NaN")
            results.append(dict(lr=lr, val_loss=float("nan"), tag=tag))
            continue
        m = json.load(open(Path(args.out_runs) / tag / "metrics.json"))
        results.append(dict(lr=lr, val_loss=m["final_val_loss"], tag=tag))

    finite = [r for r in results if np.isfinite(r["val_loss"])]
    best = min(finite, key=lambda r: r["val_loss"]) if finite else None
    out = dict(mup=args.mup, results=results, best=best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
