"""Download SVG datasets from HuggingFace, normalize/clean, and write splits.

Output layout:
  out/train.txt   one cleaned SVG per line (newlines escaped)
  out/val.txt
  out/test.txt
  out/stats.json  dataset statistics

Cleaning steps:
  - parse with lxml; drop SVGs that fail to parse
  - strip comments / processing instructions
  - drop metadata, title, desc tags
  - normalize whitespace
  - round numeric attributes / path coords to 1 decimal
  - drop SVGs shorter than --min-chars or longer than --max-tokens (char proxy)
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from lxml import etree
from tqdm import tqdm

NUM_RE = re.compile(r"-?\d+\.\d+")
WS_RE = re.compile(r"\s+")
NS_STRIP = re.compile(r"\s+xmlns(:\w+)?=\"[^\"]*\"")


def round_numbers(s: str, ndigits: int = 1) -> str:
    def _r(m):
        return f"{round(float(m.group(0)), ndigits)}"
    return NUM_RE.sub(_r, s)


def clean_svg(svg: str) -> str | None:
    """Return canonicalized SVG string, or None if invalid."""
    if not svg or "<svg" not in svg:
        return None
    try:
        # tolerant parser; remove comments & PIs
        parser = etree.XMLParser(remove_comments=True, remove_pis=True, recover=False)
        root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    except Exception:
        return None

    # strip metadata/title/desc anywhere in tree
    for tag in ("metadata", "title", "desc"):
        for el in root.iter():
            # match local name regardless of ns
            if etree.QName(el).localname == tag:
                el.getparent().remove(el) if el.getparent() is not None else None

    out = etree.tostring(root, encoding="unicode")
    out = WS_RE.sub(" ", out).strip()
    out = round_numbers(out, ndigits=1)
    return out


def iter_dataset(name: str, split: str, text_field_candidates=("Svg", "svg", "code", "text")):
    """Yield raw svg strings from a HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset(name, split=split, streaming=False)
    field = next((f for f in text_field_candidates if f in ds.column_names), None)
    if field is None:
        raise ValueError(f"No SVG text field found in {name} (cols={ds.column_names})")
    for ex in ds:
        yield ex[field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["starvector/svg-icons-simple",
                             "starvector/svg-emoji-simple"])
    ap.add_argument("--max-chars", type=int, default=8192,
                    help="char-length cap (rough proxy for token length)")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="cap items per dataset (for quick smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cleaned: list[str] = []
    raw_count = 0
    parse_fail = 0
    too_short = 0
    too_long = 0

    for ds_name in args.datasets:
        print(f"[prepare] loading {ds_name}")
        n_taken = 0
        for raw in tqdm(iter_dataset(ds_name, "train"), desc=ds_name):
            raw_count += 1
            if args.limit_per_dataset and n_taken >= args.limit_per_dataset:
                break
            c = clean_svg(raw)
            if c is None:
                parse_fail += 1
                continue
            if len(c) < args.min_chars:
                too_short += 1
                continue
            if len(c) > args.max_chars:
                too_long += 1
                continue
            cleaned.append(c)
            n_taken += 1

    print(f"[prepare] kept {len(cleaned)} / {raw_count}")
    random.shuffle(cleaned)
    n = len(cleaned)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test = cleaned[:n_test]
    val = cleaned[n_test:n_test + n_val]
    train = cleaned[n_test + n_val:]

    for split, items in (("train", train), ("val", val), ("test", test)):
        with open(out / f"{split}.txt", "w") as f:
            for s in items:
                # one SVG per line; \n inside content was already collapsed
                f.write(s.replace("\n", " ") + "\n")

    stats = dict(
        n_raw=raw_count, n_kept=len(cleaned), n_parse_fail=parse_fail,
        n_too_short=too_short, n_too_long=too_long,
        n_train=len(train), n_val=len(val), n_test=len(test),
        char_len_quantiles=_quantiles([len(s) for s in cleaned]),
        datasets=list(args.datasets),
    )
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def _quantiles(xs):
    if not xs: return {}
    xs = sorted(xs)
    def q(p): return xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}


if __name__ == "__main__":
    main()
