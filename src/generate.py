"""Sample SVGs from a trained checkpoint.

Writes:
  out/svg/<i>.svg                raw SVG strings
  out/png/<i>.png                rendered PNGs (where CairoSVG succeeds)
  out/samples.json               metadata (prompt, params, validity flags)
"""

import argparse
import json
from pathlib import Path

import torch

from .model import GPT, GPTConfig


PREFIXES = [
    ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">',
     "blank canvas"),
    ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
     '<circle cx="12" cy="12" r="10" fill="none" stroke="black"/>'
     '<circle cx="9" cy="10" r="1" fill="black"/>',
     "partial face (head + one eye)"),
    ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
     '<path d="M3 12 L8 17',
     "open path"),
    ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
     '<g fill="none" stroke="currentColor" stroke-width="2">'
     '<rect x="4" y="4" width="6" height="6"/>',
     "group with one shape"),
    ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
     '<path d="',
     "raw path prefix"),
]


def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", default="data/tok")
    ap.add_argument("--out", default="results/samples")
    ap.add_argument("--n-uncond", type=int, default=10)
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 0.8, 1.0])
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.tok) / "tokenizer.json"))
    bos = tok.token_to_id("<bos>"); eos = tok.token_to_id("<eos>")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    torch.manual_seed(args.seed)

    out = Path(args.out); (out / "svg").mkdir(parents=True, exist_ok=True); (out / "png").mkdir(parents=True, exist_ok=True)
    samples = []
    idx_counter = 0

    def sample(prefix_str: str | None, label: str, temp: float):
        nonlocal idx_counter
        if prefix_str:
            ids = [bos] + tok.encode(prefix_str).ids
        else:
            ids = [bos]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids = model.generate(x, max_new_tokens=args.max_new, temperature=temp,
                                 top_k=args.top_k, top_p=args.top_p, eos_id=eos)
        gen_ids = out_ids[0].tolist()
        # strip leading bos and trailing past eos
        if bos in gen_ids: gen_ids = gen_ids[gen_ids.index(bos) + 1:]
        if eos in gen_ids: gen_ids = gen_ids[:gen_ids.index(eos)]
        text = tok.decode(gen_ids)
        path_svg = out / "svg" / f"{idx_counter:03d}.svg"
        path_svg.write_text(text)
        png_ok = False
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=text.encode("utf-8"),
                             write_to=str(out / "png" / f"{idx_counter:03d}.png"))
            png_ok = True
        except Exception:
            pass
        samples.append(dict(idx=idx_counter, label=label, prefix=prefix_str,
                            temperature=temp, top_k=args.top_k, top_p=args.top_p,
                            n_chars=len(text), rendered=png_ok, svg=text[:200]))
        idx_counter += 1

    # Unconditional × temperatures
    for t in args.temperatures:
        for _ in range(args.n_uncond):
            sample(None, "unconditional", t)

    # Prefix-conditioned × default temperature 0.8
    for prefix, label in PREFIXES:
        sample(prefix, label, 0.8)

    with open(out / "samples.json", "w") as f:
        json.dump(dict(n=len(samples), samples=samples), f, indent=2)
    print(f"[generate] wrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
