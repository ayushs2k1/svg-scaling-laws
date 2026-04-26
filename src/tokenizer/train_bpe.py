"""Train a byte-level BPE tokenizer on the cleaned SVG corpus.

Special tokens: <pad> <bos> <eos> <unk>
Each SVG file is wrapped in <bos>...<eos> when packed (see data/pack.py).
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="dir with train.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=4096)
    args = ap.parse_args()

    inp = Path(args.inp); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = [str(inp / "train.txt")]

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train(files, trainer)
    tok.save(str(out / "tokenizer.json"))

    meta = dict(
        vocab_size=tok.get_vocab_size(),
        specials={s: tok.token_to_id(s) for s in SPECIALS},
    )
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
