"""Tokenize the cleaned SVG splits and pack into uint16 binary shards.

Each SVG is wrapped <bos> ... <eos> and concatenated. Training samples a contiguous
seq_len window (no document masking; standard nanoGPT-style packing).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="filter SVGs whose token length exceeds this")
    args = ap.parse_args()

    inp = Path(args.inp); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(args.tok) / "tokenizer.json"))
    bos = tok.token_to_id("<bos>"); eos = tok.token_to_id("<eos>")
    vocab_size = tok.get_vocab_size()
    assert vocab_size < 2**16, "use uint32 if vocab >= 65536"

    counts = {}
    seq_lens_all = []
    for split in ("train", "val", "test"):
        path = inp / f"{split}.txt"
        if not path.exists(): continue
        ids_chunks = []
        kept = dropped = 0
        seq_lens = []
        with open(path) as f:
            for line in tqdm(f, desc=f"tok {split}"):
                line = line.rstrip("\n")
                if not line: continue
                enc = tok.encode(line).ids
                if len(enc) + 2 > args.seq_len:
                    dropped += 1
                    continue
                seq_lens.append(len(enc) + 2)
                ids_chunks.append([bos] + enc + [eos])
                kept += 1
        flat = np.fromiter(
            (i for chunk in ids_chunks for i in chunk),
            dtype=np.uint16,
            count=sum(len(c) for c in ids_chunks),
        )
        np.save(out / f"{split}.npy", flat)
        counts[split] = dict(n_files=kept, n_dropped=dropped, n_tokens=int(flat.size))
        if split == "train":
            seq_lens_all = seq_lens

    meta = dict(vocab_size=vocab_size, bos=bos, eos=eos, seq_len=args.seq_len, **counts)
    if seq_lens_all:
        arr = np.array(seq_lens_all)
        meta["train_seq_len"] = dict(
            mean=float(arr.mean()), p50=int(np.percentile(arr, 50)),
            p90=int(np.percentile(arr, 90)), p99=int(np.percentile(arr, 99)),
            max=int(arr.max()),
        )
        np.save(out / "train_seq_lens.npy", arr)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
