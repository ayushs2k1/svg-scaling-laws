"""Single-run trainer.

Usage:
  python -m src.train --config configs/tiny.yaml --data data/bin --lr 3e-4 \
      [--mup] [--epochs 1] [--tag tiny_sp_lr3e-4] [--max-steps N]

Writes results/runs/<tag>/{ckpt.pt, metrics.json, log.csv}.
metrics.json contains the fields scaling_fit.py and the report template consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import GPT, GPTConfig


def load_data(bin_dir: Path, split: str) -> np.ndarray:
    return np.load(bin_dir / f"{split}.npy", mmap_mode="r")


def get_batch(data: np.ndarray, seq_len: int, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - seq_len - 1, size=(batch,))
    x = np.stack([data[i:i + seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


def cosine_lr(step: int, warmup: int, total: int, peak_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    if step >= total:
        return peak_lr * min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_loss(model, data, seq_len, batch, device, n_batches=50) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--mup", action="store_true")
    ap.add_argument("--base-d-model", type=int, default=128, help="µP proxy width (Tiny)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    bin_dir = Path(args.data)
    meta = json.load(open(bin_dir / "meta.json"))
    vocab_size = meta["vocab_size"]
    train_tokens = meta["train"]["n_tokens"]

    out = Path(args.out) / args.tag; out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(
        vocab_size=vocab_size,
        seq_len=cfg_dict["seq_len"],
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        d_ff=cfg_dict["d_ff"],
        dropout=cfg_dict.get("dropout", 0.0),
        mup=args.mup,
        base_d_model=args.base_d_model,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    n_params_nonemb = model.num_params(non_embedding=True)
    print(f"[train] params: {n_params:,} (non-emb: {n_params_nonemb:,})")

    micro_batch = cfg_dict["micro_batch"]
    seq_len = cfg_dict["seq_len"]
    batch_tokens = cfg_dict["batch_tokens"]
    grad_accum = max(1, batch_tokens // (micro_batch * seq_len))
    print(f"[train] micro_batch={micro_batch} seq_len={seq_len} grad_accum={grad_accum} "
          f"-> {micro_batch*seq_len*grad_accum} tokens/step")

    train_data = load_data(bin_dir, "train")
    val_data = load_data(bin_dir, "val")

    tokens_per_step = micro_batch * seq_len * grad_accum
    steps_per_epoch = max(1, train_tokens // tokens_per_step)
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)
    warmup = cfg_dict["warmup_steps"]
    print(f"[train] steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    groups = model.param_groups(args.lr, weight_decay=cfg_dict["weight_decay"])
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))

    log_path = out / "log.csv"
    log_f = open(log_path, "w", newline="")
    log = csv.writer(log_f); log.writerow(["step", "lr", "train_loss", "val_loss", "tok_per_s", "gpu_mem_gb"])

    model.train()
    t_start = time.time()
    tokens_seen = 0
    train_loss_ema = None
    final_val = None

    for step in range(total_steps):
        # set per-group LR by base_lr_ratio (preserve µP per-group multipliers)
        lr_now = cosine_lr(step, warmup, total_steps, args.lr)
        for g in opt.param_groups:
            base_ratio = g.get("_base_lr", None)
            if base_ratio is None:
                # store the multiplier the model gave us at construction
                g["_base_lr"] = g["lr"] / args.lr
                base_ratio = g["_base_lr"]
            g["lr"] = lr_now * base_ratio

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        t0 = time.time()
        for _ in range(grad_accum):
            x, y = get_batch(train_data, seq_len, micro_batch, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_accum += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_dict["grad_clip"])
        opt.step()

        tokens_seen += tokens_per_step
        dt = time.time() - t0
        tps = tokens_per_step / max(dt, 1e-9)
        train_loss_ema = loss_accum if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * loss_accum

        if step % args.log_every == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            print(f"step {step:6d}/{total_steps} | lr {lr_now:.2e} | "
                  f"loss {loss_accum:.4f} (ema {train_loss_ema:.4f}) | {tps:,.0f} tok/s | mem {mem:.1f}GB")
            log.writerow([step, f"{lr_now:.6e}", f"{loss_accum:.6f}", "",
                          f"{tps:.1f}", f"{mem:.3f}"]); log_f.flush()

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            v = eval_loss(model, val_data, seq_len, micro_batch, device)
            print(f"  [val] step {step}: val_loss={v:.4f}")
            log.writerow([step, "", "", f"{v:.6f}", "", ""]); log_f.flush()

    # final val
    final_val = eval_loss(model, val_data, seq_len, micro_batch, device, n_batches=200)
    elapsed = time.time() - t_start
    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    metrics = dict(
        tag=args.tag,
        config=cfg_dict,
        mup=args.mup,
        lr=args.lr,
        n_params=n_params,
        n_params_nonemb=n_params_nonemb,
        vocab_size=vocab_size,
        steps=total_steps,
        tokens_seen=tokens_seen,
        wall_clock_s=elapsed,
        gpu_mem_gb=mem,
        tokens_per_s=tokens_seen / max(elapsed, 1e-9),
        final_train_loss=train_loss_ema,
        final_val_loss=final_val,
    )
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                "lr": args.lr, "mup": args.mup}, out / "ckpt.pt")
    log_f.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
