"""Evaluate a trained checkpoint and a directory of generated samples.

Computes:
  - test-set perplexity
  - XML validity rate    (lxml.etree)
  - SVG render rate      (CairoSVG)
  - Structural validity  (root tag is <svg>, has viewBox or width/height)
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig
from .train import load_data, get_batch


def perplexity(ckpt_path: str, data_dir: str, n_batches: int = 200) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    data = load_data(Path(data_dir), "test")
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, cfg.seq_len, 8, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            losses.append(loss.item())
    return float(math.exp(np.mean(losses)))


def check_xml(svg_str: str) -> bool:
    from lxml import etree
    try:
        etree.fromstring(svg_str.encode("utf-8"))
        return True
    except Exception:
        return False


def check_render(svg_str: str) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))
        return True
    except Exception:
        return False


def check_structural(svg_str: str) -> bool:
    from lxml import etree
    try:
        root = etree.fromstring(svg_str.encode("utf-8"))
        if etree.QName(root).localname != "svg":
            return False
        has_box = ("viewBox" in root.attrib) or ("width" in root.attrib and "height" in root.attrib)
        return has_box
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--samples", default="results/samples")
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    metrics = {}
    print("[eval] computing test perplexity...")
    metrics["test_perplexity"] = perplexity(args.ckpt, args.data)
    print(f"  test perplexity = {metrics['test_perplexity']:.3f}")

    svgs = sorted(Path(args.samples, "svg").glob("*.svg"))
    n = len(svgs)
    if n == 0:
        print("[eval] no samples found");
    xml_ok = render_ok = struct_ok = 0
    for p in svgs:
        s = p.read_text()
        if check_xml(s): xml_ok += 1
        if check_render(s): render_ok += 1
        if check_structural(s): struct_ok += 1
    metrics["n_samples"] = n
    metrics["xml_validity_rate"] = (xml_ok / n) if n else None
    metrics["render_rate"] = (render_ok / n) if n else None
    metrics["structural_validity_rate"] = (struct_ok / n) if n else None
    print(json.dumps(metrics, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
